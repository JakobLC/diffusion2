import copy
import functools
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
from shutil import rmtree
import jlc
from fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from plot_utils import plot_forward_pass,make_loss_plot
from datasets import CatBallDataset, custom_collate_with_info
from nn import update_ema
from unet import create_unet_from_args
from cont_gaussian_diffusion import create_diffusion_from_args
from utils import dump_kvs,get_batch_metrics,write_args
#from utils import (make_loss_plot,make_cond_loss_plot,load_state_dict_loose,ReturnOnceOnNext,
#                    TemporarilyDeterministic,DummyWith,plot_fixed_images,dump_kvs)
#from datasets import get_random_points_images,get_noisy_bbox_images

#TODO
# forced xstart
# prepare_inputs as preprocessing step outside loop
# make seeding better in the sampler (one seed to define a whole diffusion process instead of 1000)
# implement all points in arguments
# debug save intermediate
# debug save raw samples

INITIAL_LOG_LOSS_SCALE = 20.0

class DiffusionModelTrainer:
    def __init__(self,args):
        self.args = args
        
        self.cgd = create_diffusion_from_args(args)
        self.model = create_unet_from_args(args)
        
        train_ds = CatBallDataset(max_classes=args.max_num_classes-1,dataset_len=1000,size=args.image_size)
        vali_ds = CatBallDataset(max_classes=args.max_num_classes-1,dataset_len=100,seed_translation=1000,size=args.image_size)
        eval_batch_size = args.eval_batch_size if args.eval_batch_size>0 else args.train_batch_size
        self.train_dl = jlc.DataloaderIterator(torch.utils.data.DataLoader(train_ds, 
                                                                           batch_size=args.train_batch_size, 
                                                                           shuffle=True,
                                                                           drop_last=True,
                                                                           collate_fn=custom_collate_with_info))
        self.vali_dl = jlc.DataloaderIterator(torch.utils.data.DataLoader(vali_ds, 
                                                                          batch_size=eval_batch_size, 
                                                                          shuffle=True,
                                                                          drop_last=True,
                                                                          collate_fn=custom_collate_with_info))
                                    
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.log("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        self.kvs_buffer = {}
        self.kvs_step_buffer = []
        
        _ = self.check_save_path()
        
        if Path(self.args.save_path).exists():
            possible_ckpts = list(Path(self.args.save_path).glob("ckpt_*.pt"))
            is_test_folder = Path(self.args.save_path).stem=="test"
            if len(possible_ckpts)==0 or is_test_folder:
                #nuke old folder if no ckpts are found
                new_training = True
                rmtree(self.args.save_path)
                os.makedirs(self.args.save_path)
            else:
                assert len(possible_ckpts)==1, "Multiple ckpts found. Please delete all but one."
                ckpt = torch.load(str(possible_ckpts[0]))
                new_training = False
        else:
            new_training = True
            os.makedirs(self.args.save_path)

        #init models, optimizers etc
        self.model = self.model.to(self.device)
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        
        if self.args.use_fp16:
            self.master_params = make_master_params(self.model_params)
            self.model.convert_to_fp16()
        
        betas = [float(x) for x in self.args.betas.split(",")]
        self.opt = AdamW(self.master_params, lr=self.args.lr, weight_decay=self.args.weight_decay,
                         betas=betas)
        
        self.ema_rate = [float(x) for x in self.args.ema_rate.split(",")]
        self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        
        assert len(self.master_params) == len(self.ema_params[0])
            
        
        if new_training:
            self.log("Starting new training run.")
            write_args(args, Path(args.save_path)/"args.json")
            self.step = 1
            self.log_loss_scale = INITIAL_LOG_LOSS_SCALE
            self.best_miou = 0.0
            self.fixed_batch = None
            self.log_kv_step("loss")
        else:
            self.log("Resuming training run.")
            self.step = ckpt["step"]+1
            self.log_loss_scale = ckpt["log_loss_scale"]
            self.best_miou = ckpt["best_miou"]
            self.fixed_batch = ckpt["fixed_batch"]
            self.opt.load_state_dict(ckpt["optimizer"])
            self.master_params = self._state_dict_to_master_params(ckpt["model"])
            self.model.load_state_dict(ckpt["model"])
            self.model_params = list(self.model.parameters())
            
            for i, ema_rate in enumerate(self.ema_rate):
                if "ema_"+str(ema_rate) in ckpt.keys():
                    self.ema_params[i] = self._state_dict_to_master_params(ckpt["ema_"+str(ema_rate)])

    def save_state_dict(self,ema_idx=None,delete_old=False,miou=None):
        if delete_old:
            rmtree(Path(self.args.save_path)/"state_dicts")
        if ema_idx is None:
            name = "model"
            params = self.master_params
        else:
            assert ema_idx<len(self.ema_rate), "ema_idx must be smaller than the number of ema rates"
            name = "ema"
            params = self.ema_params[ema_idx]
        
        if miou is None:
            miou = 0.0
        name += f"_step{self.step:06d}"
        name += f"_miou{miou:.4f}"
        if not (Path(self.args.save_path)/"state_dicts").exists():
            os.makedirs(Path(self.args.save_path)/"state_dicts")
        torch.save(self._master_params_to_state_dict(params), str(Path(self.args.save_path)/"state_dicts"/f"{name}.pt"))
        
    def save_train_ckpt(self,delete_old=True):
        save_dict = {"step": self.step,
                     "model": self._master_params_to_state_dict(self.master_params),
                     "optimizer": self.opt.state_dict(),
                     "log_loss_scale": self.log_loss_scale,
                     "best_miou": self.best_miou,
                     "fixed_batch": self.fixed_batch}
        for ema_rate, params in zip(self.ema_rate, self.ema_params):
            save_dict["ema_"+str(ema_rate)] = self._master_params_to_state_dict(params)
        torch.save(save_dict, str(Path(self.args.save_path)/f"ckpt_{self.step:06d}.pt"))
        if delete_old:
            possible_ckpts = list(Path(self.args.save_path).glob("ckpt_*.pt"))
            for ckpt in possible_ckpts:
                if int(ckpt.stem.split("_")[1])<self.step:
                    os.remove(ckpt)
    
    def get_kwargs(self, batch):
        x,info = batch
        x = x.to(self.device)
        model_kwargs = {"image": [],
                        "bbox": [],
                        "points": [],
                        "cond": []}
        bs = x.shape[0]
        for i in range(bs):
            if np.random.rand()<self.args.image_prob:
                model_kwargs["image"].append(info[i]["image"])
            else:
                model_kwargs["image"].append(None)
            if self.args.weak_signals:
                if np.random.rand()<self.args.weak_bbox_prob:
                    raise NotImplementedError("bbox not implemented")
                else:
                    model_kwargs["bbox"].append(None)
                    
                if np.random.rand()<self.args.weak_points_prob:
                    raise NotImplementedError("points not implemented")
                else:
                    model_kwargs["points"].append(None)
            if self.args.cond_type!="none":
                if np.random.rand()<self.args.cond_prob:
                    raise NotImplementedError("cond not implemented")
                    #model_kwargs["cond"] = info["get_cond"]()
                else:
                    model_kwargs["cond"].append(None)    
                
        return x,model_kwargs,info
    
    def run_train_step(self, batch):
        zero_grad(self.model_params)
        x,model_kwargs,info = self.get_kwargs(batch)
        output = self.cgd.train_loss_step(self.model,x,model_kwargs=model_kwargs)
        
        output = {**output,**model_kwargs}
        
        loss = output["loss"]
        
        if self.args.use_fp16:
            loss_scale = 2 ** self.log_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()
        if self.args.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        #logging
        self.log_kv_step(loss.item())
        metrics = get_batch_metrics(output)
        self.log_train_step(output,metrics)
        
        return output,metrics
    
    def evaluate_loop(self):
        with torch.no_grad():
            for vali_i in range(self.args.num_vali_batches):
                self.run_eval_step(next(self.vali_dl))
        self.dump_kvs()
        
    def run_eval_step(self, batch):
        x,model_kwargs,info = self.get_kwargs(batch)
        output = self.cgd.train_loss_step(self.model,x,model_kwargs=model_kwargs)
        metrics = get_batch_metrics(output)
        self.log_eval_step(output,metrics)
        
    def log_eval_step(self,output,metrics,prefix="vali_"):
        self.log_kv({prefix+"loss": output["loss"].item()})
        self.log_kv_step(output["loss"].item())
        self.log_kv({prefix+k:v for k,v in metrics.items()})
        
    def log_train_step(self,output,metrics):
        self.log_kv({"loss": output["loss"].item()})
        self.log_kv_step(output["loss"].item())
        self.log_kv(metrics)
    
    def _update_lr(self):
        frac_warmup = np.clip(self.step / (self.args.lr_warmup_steps+1e-14), 0.0, 1.0)
        frac_decay = np.clip((self.step - self.args.max_iter + self.args.lr_decay_steps) / (self.args.lr_decay_steps+1e-14), 0.0, 1.0)
        decay_func_dict = {"linear": lambda x: 1-x,
                           "cosine": lambda x: 0.5 * (1.0 + np.cos(np.pi * x))}
        warmup_mult = decay_func_dict[self.args.lr_warmup_type](1-frac_warmup)
        decay_mult = decay_func_dict[self.args.lr_decay_type](frac_decay)
        
        lr_new = self.args.lr * warmup_mult * decay_mult
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr_new
                
    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.log_loss_scale -= 1
            self.log(f"Found NaN, decreased log_loss_scale to {self.log_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_loss_scale))
        self._log_grad_norm()
        self._update_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.log_loss_scale += self.args.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._update_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        self.log_kv({"grad_norm": np.sqrt(sqsum)})
        
    def train_loop(self):
        self.log("Starting training loop...")
        pbar = tqdm()
        while self.step <= self.args.max_iter:
            
            self.model.train()
            
            batch = next(self.train_dl)
            
            output,metrics = self.run_train_step(batch)
            
            pbar.update(1)
                
            if self.step % self.args.save_interval == 0:
                self.save_train_ckpt()

            if self.step % self.args.log_vali_interval == 0:
                self.evaluate_loop()
            
            if self.step % self.args.update_foward_pass_plot_interval == 0:
                plot_forward_pass(Path(self.args.save_path)/f"forward_pass_{self.step:06d}.png",output,metrics,self.cgd.ab)
            
            if self.step % self.args.update_loss_plot_interval == 0:
                make_loss_plot(self.args.save_path)
                
            self.step += 1
            
        self.log("Training loop finished.")
        
        
    def log(self, msg, filename="log.txt"):
        """logs any string to a file"""
        filepath = Path(self.args.save_path)/filename
        with open(str(filepath), "a" if filepath.exists() else "w") as f:
            f.write(msg + "\n")
    
    def log_kv(self, d):
        """
        Saves key-value pairs in a buffer to be saved to a file later.
        Any values (lists or float/int) associated with a key will be 
        reduced to their mean when logged.
        """
        for key,value in d.items():
            if not isinstance(key,list):
                key = [key]
            if not isinstance(value,list):
                value = [value]
            if len(key)!=len(value):
                if len(key)==1:
                    key = key*len(value)
                else:
                    raise ValueError("key and value must have the same length")
            for k,v in zip(key,value):
                if k in self.kvs_buffer:
                    if isinstance(self.kvs_buffer[k],list):
                        if isinstance(v,list):
                            self.kvs_buffer[k].extend(v)
                        else:
                            self.kvs_buffer[k].append(v)
                else:
                    self.kvs_buffer[k] = [v]
    
    def log_kv_step(self, *values):
        """
        Saves values in a buffer to be saved to a file later.
        No reduction is applied to the values.
        """
        self.kvs_step_buffer.append(values)
        
    def dump_kvs(self, filename="progress.csv"):
        """
        Saves the kvs buffer and prints it, aswell as the kvs 
        step buffer to a file and then clears the buffers.
        """
        self.log_kv({"step": self.step})
        self.log_kv({"loss_scale": self.log_loss_scale})
        for k,v in self.kvs_buffer.items():
            if isinstance(v,list):
                self.kvs_buffer[k] = np.mean(v)
        self.fancy_print_kvs(self.kvs_buffer)
        dump_kvs(str(Path(self.args.save_path)/filename),self.kvs_buffer)
        self.kvs_buffer = {}
        
        with open(str(Path(self.args.save_path)/"progress_step.csv"), "a") as f:
            for row in self.kvs_step_buffer:
                f.write(",".join([str(v) for v in row]) + "\n")
        self.kvs_step_buffer = []
    
    def fancy_print_kvs(self, kvs, atmost_digits=5):
        """prints kvs in a nice format like
         |#########|########|
         | key1    | value1 |
              ...
         | keyN    | valueN |
         |#########|########|
        """
        values_print = []
        keys_print = []
        for k,v in kvs.items():
            if isinstance(v,float):
                v = f"{v:.{atmost_digits}g}"
            else:
                v = str(v)
            values_print.append(v)
            keys_print.append(k) 
        max_key_len = max([len(k) for k in keys_print])
        max_value_len = max([len(v) for v in values_print])
        print_str = ""
        print_str += "|" + "#"*(max_key_len+2) + "|" + "#"*(max_value_len+2) + "|\n"
        for k,v in zip(keys_print,values_print):
            print_str += "| " + k + " "*(max_key_len-len(k)+1) + "| " + v + " "*(max_value_len-len(v)+1) + "|\n"
        print_str += "|" + "#"*(max_key_len+2) + "|" + "#"*(max_value_len+2) + "|\n"
        self.log(print_str)

    def _master_params_to_state_dict(self, master_params):
        """converts a list of params (flattened list if fp16) 
        to a state dict based on the model's state dict"""
        if self.args.use_fp16:
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """converts a state dict to a list of params based on the model's state dict"""
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.args.use_fp16:
            return make_master_params(params)
        else:
            return params

    def check_save_path(self):
        """Check if the save path is a subfolder of the saves folder. 
        This is important since the training loop will delete the save folder 
        under some conditions."""
        saves_folder = Path(os.path.abspath(__file__)).parent.parent/"saves"
        save_path = Path(os.path.abspath(self.args.save_path))
        out = False
        num_parents = len(save_path.parents)
        parent = save_path.parent
        for _ in range(num_parents):
            is_saves_folder = parent==saves_folder
            if is_saves_folder:
                out = True
                break
            parent = parent.parent
        assert out, "The save path must be a subfolder of the saves folder. save_path: "+str(save_path)+", saves_folder: "+str(saves_folder)
        return out

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: train from scratch")
        from utils import SmartParser
        args = SmartParser().get_args()
        args.model_name = "test2"
        args.save_path = "./saves/test2/"
        trainer = DiffusionModelTrainer(args)
        trainer.train_loop()
    elif args.unit_test==1:
        print("UNIT TEST 1: continued training")
        from utils import SmartParser
        args = SmartParser().get_args(do_parse_args=False)
        args.model_name = "test_trained"
        args.save_path = "./saves/test_trained/"
        args.max_iter = 5002
        args.update_foward_pass_plot_interval = 2
        trainer = DiffusionModelTrainer(args)
        trainer.train_loop()
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()
import copy
import functools
import os
from pathlib import Path
import psutil
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from shutil import rmtree
import jlc
import traceback
import torch.nn.functional as F
import json
import argparse
from fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
import datetime
from argparse_utils import save_args, TieredParser, load_existing_args, overwrite_existing_args, load_defaults
from sampling import DiffusionSampler
from plot_utils import plot_forward_pass,make_loss_plot
from datasets import (CatBallDataset, custom_collate_with_info, 
                      SegmentationDataset, points_image_from_label)
from nn import update_ema
from unet import create_unet_from_args, unet_kwarg_to_tensor, get_sam_image_encoder
from cont_gaussian_diffusion import create_diffusion_from_args
from utils import (dump_kvs,get_all_metrics,MatplotlibTempBackend,
                   fancy_print_kvs,bracket_glob_fix,format_relative_path,
                   set_random_seed,is_infinite_and_not_none,get_time,
                   AlwaysReturnsFirstItemOnNext,format_save_path)

INITIAL_LOG_LOSS_SCALE = 20.0

class DiffusionModelTrainer:
    def __init__(self,args):
        self.exit_flag = False
        self.args = args

        self.seed = set_random_seed(args.seed)
        if self.args.debug_run=="seed":
            print("Seed: "+str(self.seed))
            print("numpy random state: "+str(np.random.get_state()))
            print("torch random state: "+str(torch.random.get_rng_state()))
            print("torch cuda random state: "+str(torch.cuda.get_rng_state()))
            print("torch cuda random state (manual_seed): "+str(torch.cuda.manual_seed(self.seed)))
            exit()
        args.seed = self.seed

        self.cgd = create_diffusion_from_args(args)
        self.model = create_unet_from_args(args)
        n_trainable = jlc.num_of_params(self.model,print_numbers=False)[0]
        
        if self.args.mode=="new":
            if self.args.save_path=="":
                self.args.save_path = format_save_path(self.args)
            self.log("Starting new training run.")
        elif self.args.mode=="load":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            ckpt = torch.load(self.args.ckpt_name)
            if self.args.save_path=="":
                self.args.save_path = format_save_path(self.args)
            self.log("Starting new training run with loaded ckpt from: "+self.args.ckpt_name)
        elif self.args.mode=="cont":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            if self.exit_flag:
                return
            ckpt = torch.load(self.args.ckpt_name)
            if self.args.save_path=="":
                self.args.save_path = str(Path(self.args.ckpt_name).parent)
            self.log("Continuing training run.")
        elif self.args.mode=="gen":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            ckpt = torch.load(self.args.ckpt_name)
            if self.args.save_path=="":
                self.args.save_path = str(Path(self.args.ckpt_name).parent)
            self.log("Setting up generation.")
        else:
            raise ValueError("Unknown mode: "+self.args.mode+", must be one of ['new','load','cont','gen']")
        if torch.cuda.is_available():
            self.log("CUDA available. Using GPU.")
            self.device = torch.device("cuda")
        else:
            self.log("WARNING: CUDA not available. Using CPU.")
            raise NotImplementedError("CPU not implemented")
            self.device = torch.device("cpu")
        
        self.log("Number of trainable parameters: "+str(n_trainable))

        self.log("Saving to: "+self.args.save_path)

        if self.args.cat_ball_data:
            raise NotImplementedError("cat_ball_data not implemented")
            """train_ds = CatBallDataset(max_num_classes=args.max_num_classes,dataset_len=10000,size=args.image_size)
            vali_ds = CatBallDataset(max_num_classes=args.max_num_classes,dataset_len=1000,seed_translation=len(train_ds),size=args.image_size)"""
        if args.save_best_ckpt:
            n_setups = len(args.gen_setups.split(","))
            assert -n_setups < args.best_ckpt_gen_setup_idx < n_setups, "save_best_ckpt_setup must be in gen_setups"
            
        if self.args.mode in ["new","load","cont"]:
            self.create_datasets(["train","vali"])
        if self.args.debug_run=="no_dl":
            self.train_dl=AlwaysReturnsFirstItemOnNext(self.train_dl)

        self.kvs_buffer = {}
        self.kvs_gen_buffer = {}
        self.kvs_step_buffer = []            
        self.num_nan_losses = 0

        #init models, optimizers etc
        self.model = self.model.to(self.device)
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        
        if self.args.image_encoder!="none":
            self.image_encoder = get_sam_image_encoder(self.args.image_encoder,device=self.device)

        if self.args.use_fp16:
            self.master_params = make_master_params(self.model_params)
            self.model.convert_to_fp16()
        
        betas = [float(x) for x in self.args.betas.split(",")]
        self.opt = AdamW(self.master_params, lr=self.args.lr, weight_decay=self.args.weight_decay,
                         betas=betas)
        
        self.ema_rates = [float(x) for x in self.args.ema_rate.split(",")]
        self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rates]
        
        assert len(self.master_params) == len(self.ema_params[0])
            
        if self.args.mode in ["cont","load","gen"]:
            self.log_loss_scale = ckpt["log_loss_scale"]
            self.best_metric = (ckpt["best_metric"] if ckpt["best_metric"] is not None else 0.0) if "best_metric" in ckpt.keys() else 0.0
            self.fixed_batch = ckpt["fixed_batch"]
            self.opt.load_state_dict(ckpt["optimizer"])
            self.model.load_state_dict(ckpt["model"])
            self.master_params = self._state_dict_to_master_params(ckpt["model"])
            self.model_params = list(self.model.parameters())
            for i, ema_rate in enumerate(self.ema_rates):
                if "ema_"+str(ema_rate) in ckpt.keys():
                    self.ema_params[i] = self._state_dict_to_master_params(ckpt["ema_"+str(ema_rate)])

        self.list_of_sample_opts = []
        for gen_setup in self.args.gen_setups.split(","):
            sample_opts = TieredParser("sample_opts").get_args(alt_parse_args=[],modified_args={"gen_setup": gen_setup})
            self.list_of_sample_opts.append(sample_opts)

        if self.args.mode in ["load","new"]:
            self.step = 0
            self.log_loss_scale = INITIAL_LOG_LOSS_SCALE
            self.best_metric = 0.0
            self.fixed_batch = None
            self.log_kv_step(self.args.log_train_metrics.split(","))
            self.args.step_and_time = [[self.step,datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")]]
            save_args(args)
        elif self.args.mode in ["cont","gen"]:
            self.step = ckpt["step"]
            #if sample_opts file exists, replace the gen_ids with the ones from the file
            id_dict = TieredParser("sample_opts").load_and_format_id_dict()
            for i in range(len(self.list_of_sample_opts)):
                #look through existing sample_opts and make sure it is associated with our model, and has the same gen_setup
                gen_id_matched = None
                for k,v in id_dict.items():
                    if Path(v["name_match_str"]).parent.absolute()==Path(self.args.save_path).absolute():
                        if v["gen_setup"]==self.list_of_sample_opts[i].gen_setup:
                            gen_id_matched = k
                            break
                if gen_id_matched is not None:
                    self.list_of_sample_opts[i] = load_existing_args(gen_id_matched,name_key="sample_opts",use_loaded_dynamic_args=False)
                    self.list_of_sample_opts[i].gen_id = gen_id_matched
            if self.args.mode=="cont":
                if isinstance(self.args.step_and_time,list):
                    self.args.step_and_time.append([self.step,get_time()])
                else:
                    self.args.step_and_time = [[self.step,get_time()]]
                overwrite_existing_args(self.args)
        self.log("Init complete.")

    def create_datasets(self,split_list=["train","vali"],args=None):
        if self is None:
            self = argparse.Namespace()
            pure_gen_dataset_mode = True
            self.args = dummy_dataset_args()
            if args is not None:
                assert isinstance(args,(dict,argparse.Namespace)), "args must be dict or argparse.Namespace or None"
                self.args.__dict__.update(args.__dict__ if isinstance(args,argparse.Namespace) else args)
            
            if self.args.seed is None:
                self.args.seed = set_random_seed(self.args.seed)
            self.seed = self.args.seed
        else:
            if args is not None:
                assert self.args.mode=="gen", "args can only be specified in gen mode"
                assert isinstance(args,(dict,argparse.Namespace)), "args must be dict or argparse.Namespace or None"
                self.args.__dict__.update(args.__dict__ if isinstance(args,argparse.Namespace) else args)
            pure_gen_dataset_mode = False
        
        if isinstance(split_list,str):
            split_list = [split_list]
        split_ratio = [float(item) for item in self.args.split_ratio.split(",")]
        for split in split_list:
            dataset = SegmentationDataset(split=split,
                                        split_ratio=split_ratio,
                                        image_size=self.args.image_size,
                                        datasets=self.args.datasets,
                                        min_label_size=self.args.min_label_size,
                                        max_num_classes=self.args.max_num_classes,
                                        shuffle_zero=self.args.shuffle_zero,
                                        geo_aug_p=self.args.geo_aug_prob,
                                        crop_method=self.args.crop_method,
                                        label_padding_val=255 if self.args.ignore_padded else 0,
                                        split_method=self.args.split_method)
            bs = {"train": self.args.train_batch_size,
                  "vali": self.args.vali_batch_size if self.args.vali_batch_size>0 else self.args.train_batch_size,
                  "test": self.args.train_batch_size}[split]
            if pure_gen_dataset_mode:
                sampler = dataset.get_gen_dataset_sampler(self.args.datasets,self.seed)
            else:
                sampler = dataset.get_sampler(self.seed) if hasattr(dataset,"get_sampler") else None
            dataloader = jlc.DataloaderIterator(torch.utils.data.DataLoader(dataset,
                                        batch_size=bs,
                                        sampler=sampler,
                                        shuffle=(sampler is None),
                                        drop_last=not pure_gen_dataset_mode,
                                        collate_fn=custom_collate_with_info,
                                        num_workers=self.args.dl_num_workers))
            if self.args.debug_run=="no_dl":
                #self.tr = tracker.SummaryTracker()
                dataloader = AlwaysReturnsFirstItemOnNext(dataloader)
            elif self.args.debug_run=="spam_getitem":
                for i in range(1000):
                    batch = next(dataloader)
                    if i%10==0:
                        mem_usage = psutil.virtual_memory().percent
                        print(f"i: {i}, mem_usage: {mem_usage}")
        if pure_gen_dataset_mode:
            return dataloader
        else:
            setattr(self,split+"_dl",dataloader)

    def load_ckpt(self, ckpt_name, check_valid_args=False):
        if ckpt_name=="":
            ckpt_name = "./saves/ver-*/*"+self.args.model_name+"*/ckpt_*.pt"
        ckpt_matches = list(Path(os.path.abspath(__file__)).parent.parent.glob(bracket_glob_fix(ckpt_name)))
        if len(ckpt_matches)==0:
            self.exit_flag = True
            self.log("WARNING: No ckpts found. Please specify a valid ckpt_name. ckpt_name: "+ckpt_name)
            return ""
        elif len(ckpt_matches)>1:
            raise ValueError("Multiple ckpts found. Please specify a more specific ckpt_name. ckpt_name: "+ckpt_name+", ckpt_matches: "+str(ckpt_matches))
        else:
            ckpt_name = ckpt_matches[0]
        if check_valid_args:
            ckpt_args = json.loads((ckpt_name.parent/"args.json").read_text())
            #check all ckpt args are the same as the current args
            for k,v in ckpt_args.items():
                if k not in ["save_path","ckpt_name","mode"]:
                    assert k in self.args.__dict__.keys(), f"could not find key {k} from ckpt_args in current args"
                    assert self.args.__dict__[k]==v, f"ckpt args and current args differ for key {k}: {v} vs {self.args.__dict__[k]}"
        return format_relative_path(ckpt_name)
        
    def save_train_ckpt(self,delete_old=True,name_str="ckpt_",additional_str="",only_keep_keys=None):
        save_dict = {"step": self.step,
                     "model": self._master_params_to_state_dict(self.master_params),
                     "optimizer": self.opt.state_dict(),
                     "log_loss_scale": self.log_loss_scale,
                     "best_metric": self.best_metric,
                     "fixed_batch": self.fixed_batch}
        for ema_rate, params in zip(self.ema_rates, self.ema_params):
            save_dict["ema_"+str(ema_rate)] = self._master_params_to_state_dict(params)
        save_name = str(Path(self.args.save_path)/f"{name_str}{additional_str}{self.step:06d}.pt")
        if only_keep_keys is not None:
            for k in only_keep_keys:
                assert k in save_dict.keys(), f"key {k} not found in save_dict"
            save_dict = {k:v for k,v in save_dict.items() if k in only_keep_keys}
        torch.save(save_dict, save_name)
        if delete_old:
            possible_ckpts = list(Path(self.args.save_path).glob(bracket_glob_fix(f"{name_str}*.pt")))
            for ckpt_name in possible_ckpts:
                if str(ckpt_name)!=save_name:
                    os.remove(ckpt_name)
    
    def get_ema_model(self,ema_idx):
        assert ema_idx < len(self.ema_rates), "ema_idx must be smaller than the number of ema rates"
        tmp = make_master_params(copy.deepcopy(self.model_params))
        master_params_to_model_params(self.model_params, self.ema_params[ema_idx])

        self.ema_params[ema_idx] = tmp
        del tmp
        swap_pointers_func = lambda: self.get_ema_model(ema_idx)

        return self.model, swap_pointers_func

    def get_kwargs(self, batch, gen=False):
        x,info = batch
        if self.args.image_encoder!="none":
            x = F.interpolate(x.float(),(self.args.image_size,self.args.image_size),mode="nearest-exact").to(self.device).long()
        else:
            x = x.to(self.device)
        model_kwargs = {"image": [],
                        "bbox": [],
                        "points": [],
                        "cond": [],
                        "classes": []}
        to_dev = lambda x: x.to(self.device) if x is not None else None
        bs = x.shape[0]
        for i in range(bs):
            if np.random.rand()<self.args.image_prob or gen:
                model_kwargs["image"].append(info[i]["image"])
            else:
                model_kwargs["image"].append(None)
            if self.args.weak_signals:
                if self.args.weak_bbox_prob>0:
                    if np.random.rand()<=self.args.weak_bbox_prob:
                        raise NotImplementedError("bbox not implemented")
                    else:
                        model_kwargs["bbox"].append(None)
                if self.args.weak_points_prob>0:
                    if np.random.rand()<=self.args.weak_points_prob:
                        model_kwargs["points"].append(points_image_from_label(x[i]))
                    else:
                        model_kwargs["points"].append(None)
            if self.args.cond_type!="none":
                if np.random.rand()<self.args.cond_prob:
                    raise NotImplementedError("cond not implemented")
                    #model_kwargs["cond"] = info["get_cond"]()
                else:
                    model_kwargs["cond"].append(None)
            if self.args.class_type!="none":
                if self.args.class_type=="num_classes" and self.args.classes_prob>0:
                    if np.random.rand()<self.args.classes_prob:
                        model_kwargs["classes"].append(torch.tensor(info[i]["num_classes"],dtype=torch.long))
                    else:
                        model_kwargs["classes"].append(None)
                else:
                    raise NotImplementedError(f"class_type={self.args.class_type} not implemented")
        if self.args.image_encoder!="none":
            assert self.args.crop_method=="sam_big", "image_encoder requires sam_big crop_method"
            image_idx = [i for i in range(bs) if model_kwargs["image"][i] is not None]
            if len(image_idx)>0:
                with torch.no_grad():
                    image_features = self.image_encoder(to_dev(unet_kwarg_to_tensor([item for item in model_kwargs["image"] if item is not None])))
                model_kwargs["image_features"] = torch.zeros(bs,*image_features.shape[1:],device=self.device)
                model_kwargs["image_features"][image_idx] = image_features
                model_kwargs["image"] = F.avg_pool2d(unet_kwarg_to_tensor(model_kwargs["image"]),1024//self.args.image_size)

        model_kwargs = {k: to_dev(unet_kwarg_to_tensor(v)) for k,v in model_kwargs.items()}
        return x,model_kwargs,info
    
    def get_self_cond(self,info):
        if (not self.args.self_cond) or (self.args.self_cond_prob==0):
            self_cond = False
        else:
            if self.args.self_cond_batched:
                if self.args.self_cond_prob>=np.random.rand():
                    self_cond = True
                else:
                    self_cond = False
            else:
                self_cond = []
                for _ in range(len(info)):
                    self_cond.append(self.args.self_cond_prob>=np.random.rand())
        return self_cond

    def run_train_step(self, batch):
        zero_grad(self.model_params)
        x,model_kwargs,info = self.get_kwargs(batch)
        self_cond = self.get_self_cond(info)

        output = self.cgd.train_loss_step(self.model,x,model_kwargs=model_kwargs,self_cond=self_cond)

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
        metrics = get_all_metrics(output,ab=self.cgd.ab)
        self.log_train_step(output,metrics)
        
        return output,metrics
    
    def evaluate_loop(self):
        with torch.no_grad():
            for vali_i in range(self.args.num_vali_batches):
                self.run_vali_step(next(self.vali_dl))
        self.dump_kvs()
        self.kvs_buffer = {}
        
    def run_vali_step(self, batch):
        x,model_kwargs,info = self.get_kwargs(batch)
        output = self.cgd.train_loss_step(self.model,x,model_kwargs=model_kwargs)
        metrics = get_all_metrics(output,ab=self.cgd.ab)
        self.log_vali_step(output,metrics)
        
    def log_vali_step(self,output,metrics,prefix="vali_"):
        self.log_kv({prefix+"loss": output["loss"].item()})
        self.log_kv({prefix+k:v for k,v in metrics.items()})
        
    def log_train_step(self,output,metrics):
        self.log_kv({"loss": output["loss"].item()})
        kvs_step = {}
        if "loss" in self.args.log_train_metrics.split(","):
            kvs_step["loss"] = output["loss"].item()
            if np.isnan(kvs_step["loss"]):
                self.num_nan_losses += 1
                if self.num_nan_losses>20:
                    self.log("Too many NaN losses, stopping training.")
                    self.exit_flag = True
            else:
                self.num_nan_losses = 0
        if "grad_norm" in self.args.log_train_metrics.split(","):
            kvs_step["grad_norm"] = self.last_grad_norm
        if "clip_ratio" in self.args.log_train_metrics.split(","):
            kvs_step["clip_ratio"] = self.last_clip_ratio
        if "mem_usage" in self.args.log_train_metrics.split(","):
            mem_usage = psutil.virtual_memory().percent
            kvs_step["mem_usage"] = mem_usage
            self.log_kv({"mem_usage": mem_usage})
        #reorder to match the order in args.log_train_metrics
        kvs_step = [kvs_step[k] for k in self.args.log_train_metrics.split(",")]
        self.log_kv_step(kvs_step)
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
        if any(is_infinite_and_not_none(p.grad) for p in self.model_params):
            self.log_loss_scale -= 1
            self.last_grad_norm = -1.0
            self.last_clip_ratio = -1.0 
            self.log(f"Found NaN, decreased log_loss_scale to {self.log_loss_scale}")
            if self.log_loss_scale <= -20:
                self.log("Loss scale has gotten too small, stopping training.")
                self.exit_flag = True
            return
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_loss_scale))
        self._log_grad_norm()
        if self.args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.master_params, self.args.clip_grad_norm)
        self._update_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rates, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.log_loss_scale += self.args.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        if self.args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.master_params, self.args.clip_grad_norm)
        self._update_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rates, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        ratio_count = 0
        num_total_params = 0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
            ratio_count += (p.grad.abs() > self.args.clip_grad_norm).sum().item()
            num_total_params += p.numel()
        
        self.last_clip_ratio = ratio_count / num_total_params
        self.last_grad_norm = np.sqrt(sqsum)
        self.log_kv({"grad_norm": self.last_grad_norm})
        self.log_kv({"clip_ratio": self.last_clip_ratio})
    
    def train_loop(self):
        try:
            self.train_loop_inner()
        except Exception as e:
            self.log("Exception in training loop:\n" + traceback.format_exc())
            self.log("Exiting training loop.")

    def train_loop_inner(self):
        if self.exit_flag:
            self.log("Training loop stopped due to exit flag.")
            return
        self.step += 1
        self.log("Starting training loop...")
        #pbar = tqdm(unit='ims', unit_scale=self.args.train_batch_size)
        pbar = tqdm()
        while self.step <= self.args.max_iter:
            
            self.model.train()
            
            batch = next(self.train_dl)
            
            output,metrics = self.run_train_step(batch)
            pbar.update(1)

            if self.step % self.args.log_vali_interval == 0 and self.args.log_vali_interval>0:
                self.evaluate_loop()
                
            if self.step % self.args.update_forward_pass_plot_interval == 0 and self.args.update_forward_pass_plot_interval>0:
                with MatplotlibTempBackend(backend="agg"):
                    plot_forward_pass(Path(self.args.save_path)/f"forward_pass_{self.step:06d}.png",
                                      output,metrics,ab=self.cgd.ab,sample_names=batch[-1])
            
            if self.step % self.args.gen_interval == 0 and self.args.gen_interval>0:
                self.generate_samples()
            
            if self.step % self.args.update_loss_plot_interval == 0 and self.args.update_loss_plot_interval>0:
                with MatplotlibTempBackend(backend="agg"):
                    make_loss_plot(self.args.save_path,self.step)

            if ((self.step % self.args.save_interval) == 0 
                and self.args.save_interval>0 
                and self.num_nan_losses==0):
                self.save_train_ckpt()

            if str(self.step) in self.args.save_ckpt_steps.split(","):
                self.save_train_ckpt(delete_old=False,name_str="savesteps_ckpt_",)

            self.step += 1
            if self.exit_flag:
                break
        if self.exit_flag:
            self.log("Training loop stopped due to exit flag.")
        else:
            self.log("Training loop finished.")
        
    
        
    def log(self, msg, filename="log.txt", also_print=True):
        """logs any string to a file"""
        filepath = Path(self.args.save_path)/filename
        if (not self.exit_flag) or self.args.save_path!="":
            if not filepath.exists():
                self.check_save_path(self.args.save_path)
                os.makedirs(self.args.save_path, exist_ok=True)
                with open(str(filepath), "w") as f:
                    f.write(msg + "\n")
            else:
                with open(str(filepath), "a") as f:
                    try:
                        f.write(msg + "\n")
                    except UnicodeEncodeError:
                        msg_hat = msg.encode("utf-8", "ignore").decode("utf-8")
                        print(f"UnicodeEncodeError: {msg} -> {msg_hat}")
                        raise ValueError("UnicodeEncodeError")
        if also_print:
            print(msg)
    
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
    
    def log_kv_step(self, values):
        """
        Saves values in a buffer to be saved to a file later.
        No reduction is applied to the values.
        """
        self.kvs_step_buffer.append(values)
    
    def dump_kvs_gen(self, filename="logging_gen.csv"):
        self.kvs_gen_buffer["step"] = self.step
        for k,v in self.kvs_gen_buffer.items():
            if isinstance(v,list):
                self.kvs_gen_buffer[k] = np.mean(v)
        fancy_print_str = fancy_print_kvs(self.kvs_gen_buffer,s="Ø")
        self.log(fancy_print_str)
        dump_kvs(str(Path(self.args.save_path)/filename),self.kvs_gen_buffer)
        self.kvs_gen_buffer = {}

    def dump_kvs(self, filename="logging.csv"):
        """
        Saves the kvs buffer and prints it, aswell as the kvs 
        step buffer to a file and then clears the buffers.
        """
        self.log_kv({"step": self.step})
        self.log_kv({"loss_scale": self.log_loss_scale})
        for k,v in self.kvs_buffer.items():
            if isinstance(v,list):
                self.kvs_buffer[k] = np.mean(v)
        fancy_print_str = fancy_print_kvs(self.kvs_buffer)
        self.log(fancy_print_str)
        dump_kvs(str(Path(self.args.save_path)/filename),self.kvs_buffer)
        self.kvs_buffer = {}
        
        with open(str(Path(self.args.save_path)/"logging_step.csv"), "a") as f:
            for row in self.kvs_step_buffer:
                f.write(",".join([str(v) for v in list(row)]) + "\n")
        self.kvs_step_buffer = []
    
    
        ###self.log(print_str)

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

    def check_save_path(self,save_path):
        """Check if the save path is a subfolder of the saves folder."""
        saves_folder = Path(os.path.abspath(__file__)).parent.parent/"saves"
        save_path = Path(os.path.abspath(save_path))
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
    
    def generate_samples(self, list_of_sample_opts=None, max_reduction_measures=["hiou","ari"]):
        """
        Inputs:
            gen_tuples: list of tuples (gen_setup, modified_args) with types (str,dict).
            max_reduction_measures: list of strings with the names of the metrics to reduce to their max as well as mean reduction.
        """
        if list_of_sample_opts is None:
            for i in range(len(self.list_of_sample_opts)):
                self.list_of_sample_opts[i].name_match_str = format_relative_path(Path(self.args.save_path)/f"ckpt_{self.step:06d}.pt")
            list_of_sample_opts = self.list_of_sample_opts
        gen_setup_idx = 0
        for opts in list_of_sample_opts:
            sampler = DiffusionSampler(trainer=self, opts=opts)
            _, metric_dict = sampler.sample()
            metric_kvs = {}
            for k,v in metric_dict.items():
                metric_kvs[k] = sum(v,[]) if isinstance(v,list) else v
            for m in max_reduction_measures:
                metric_kvs["max_"+m] = [max(v) for v in metric_dict[m]]
            self.kvs_gen_buffer.update(metric_kvs)
            
            #maybe save best ckpt
            if (self.args.mode!="gen" and 
                self.args.save_best_ckpt and 
                self.args.best_ckpt_gen_setup_idx==gen_setup_idx):
                metric = self.args.best_ckpt_metric
                new_best_metric = np.array(self.kvs_gen_buffer[metric]).mean().item()
                if new_best_metric>self.best_metric:
                    self.best_metric = new_best_metric
                    model_key = f"ema_{self.ema_rates[sampler.opts.ema_idx]}" if sampler.opts.ema_idx>=0 else "model"
                    gen_setup = self.args.gen_setups.split(",")[gen_setup_idx]
                    self.save_train_ckpt(delete_old=True,
                                        name_str="best_ckpt_",
                                        additional_str=f"{gen_setup}_{metric}={self.best_metric:.4f}_",
                                        only_keep_keys=None if self.args.best_ckpt_full else [model_key])
            self.dump_kvs_gen()
            gen_setup_idx += 1

def dummy_dataset_args():
    args = argparse.Namespace(datasets="ade20k",
                                split_ratio="0.8,0.1,0.1",
                                image_size=64,
                                min_label_size=0.0,
                                max_num_classes=64,
                                shuffle_zero=True,
                                geo_aug_prob=0.0,
                                crop_method="sam_big",
                                ignore_padded=True,
                                split_method="native",
                                train_batch_size=8,
                                vali_batch_size=-1,
                                dl_num_workers=4,
                                seed=-1,
                                debug_run="")
    return args

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
        args = SmartParser().get_args(alt_parse_args=[])
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
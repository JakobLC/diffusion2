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
import time
from pprint import pprint
from source.utils.fp16_utils import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
import datetime
from source.utils.argparse_utils import (save_args, TieredParser,load_existing_args, 
                            overwrite_existing_args,get_ckpt_name)
from source.sampling import DiffusionSampler
from source.utils.plot_utils import plot_forward_pass,make_loss_plot,mask_overlay_smooth
from source.utils.data_utils import (CatBallDataset, custom_collate_with_info, 
                      SegmentationDataset, points_image_from_label,
                      bbox_image_from_label,get_dataset_from_args)
from source.models.nn import update_ema
from source.models.unet import create_unet_from_args, get_sam_image_encoder
from source.cont_gaussian_diffusion import create_diffusion_from_args
from source.utils.mixed_utils import (dump_kvs,fancy_print_kvs,bracket_glob_fix,
                   format_relative_path,set_random_seed,is_infinite_and_not_none,
                   get_time,AlwaysReturnsFirstItemOnNext,format_save_path,
                   shaprint,to_dev,model_arg_is_trivial,nice_split,imagenet_preprocess,
                   sam_resize_index, prettify_classname,fix_clip_matrix_in_state_dict,
                   format_model_kwargs,unet_kwarg_to_tensor)
from jlc import load_state_dict_loose, MatplotlibTempBackend
from source.utils.metric_and_loss_utils import get_all_metrics
from torchvision.transforms.functional import resize
from source.models.cond_vit import (fancy_vit_from_args, get_opt4_from_cond_vit_setup, 
                                    dynamic_image_keys,cond_kwargs_int2bit,
                                    num_tokens_from_token_info, is_valid_cond_vit_setup,
                                    ModelInputKwargs,all_input_keys,unet_vit_inputs_from_args)

INITIAL_LOG_LOSS_SCALE = 20.0
VALID_DEBUG_RUNS = ["print_model_name_and_exit","no_dl","anomaly","only_dl","cond_vit_info",
                    "dummymodel","unet_print","token_info_overview","unet_input_dict",
                    "vit_input_dict","restart_step2","no_kwargs","no_train_step"]

class DiffusionModelTrainer:
    def __init__(self,args):
        self.exit_flag = False
        self.restart_flag = False
        self.args = args
        self.prev_time = time.time()
        self.init()

    def init(self):
        if self.args.debug_run not in VALID_DEBUG_RUNS:
            assert self.args.debug_run=="", "debug_run must be one of "+str(VALID_DEBUG_RUNS)+", found: "+self.args.debug_run
        
        if self.args.debug_run=="print_model_name_and_exit":
            print(self.args.model_name)
            self.exit_flag = True
            return
        
        if self.args.debug_run=="restart_step2":
            self.restart_step2 = True
        else:
            self.restart_step2 = False

        if not self.restart_flag:
            self.args.seed = set_random_seed(self.args.seed)
        self.cgd = create_diffusion_from_args(self.args)
        if not self.args.mode=="data":
            self.model = create_unet_from_args(self.args)
            n_trainable_all = jlc.num_of_params(self.model,print_numbers=False)[0]
            if self.model.vit is not None:
                n_trainable_vit = jlc.num_of_params(self.model.vit,print_numbers=False)[0]
            else:
                n_trainable_vit = 0
            n_trainable_unet = n_trainable_all - n_trainable_vit
            #add space every 3rd number from the back
            n_trainable_unet, n_trainable_vit = str(n_trainable_unet), str(n_trainable_vit)
            n_trainable_unet = " ".join([n_trainable_unet[::-1][i:i+3] for i in range(0,len(n_trainable_unet),3)])[::-1]
            n_trainable_vit = " ".join([n_trainable_vit[::-1][i:i+3] for i in range(0,len(n_trainable_vit),3)])[::-1]
        else:
            n_trainable_unet = "N/A"
            n_trainable_vit = "N/A"
        if self.args.mode=="new":
            if self.args.save_path=="":
                self.args.save_path = format_save_path(self.args)
            self.log("Starting new training run.")
        elif self.args.mode=="load":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            if self.exit_flag:
                return
            ckpt = torch.load(self.args.ckpt_name,weights_only=False)
            if self.args.save_path=="":
                self.args.save_path = format_save_path(self.args)
            self.log("Starting new training run with loaded ckpt from: "+self.args.ckpt_name)
        elif self.args.mode=="cont":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            if self.exit_flag:
                return
            ckpt = torch.load(self.args.ckpt_name,weights_only=False)
            if self.args.save_path=="":
                self.args.save_path = str(Path(self.args.ckpt_name).parent)
            self.log("Continuing training run.")
        elif self.args.mode=="gen":
            self.args.ckpt_name = self.load_ckpt(self.args.ckpt_name)
            if self.exit_flag:
                return
            ckpt = torch.load(self.args.ckpt_name,weights_only=False)
            if self.args.save_path=="":
                self.args.save_path = str(Path(self.args.ckpt_name).parent)
            self.log("Setting up generation.")
        elif self.args.mode=="data":
            pass
        else:
            raise ValueError("Unknown mode: "+self.args.mode+", must be one of ['new','load','cont','gen']")
        if torch.cuda.is_available():
            self.log("CUDA available. Using GPU.")
            self.device = torch.device("cuda")
        else:
            self.log("WARNING: CUDA not available. Using CPU.")
            raise NotImplementedError("CPU not implemented")
            self.device = torch.device("cpu")
        
        self.log("Number of trainable parameters (UNet): "+n_trainable_unet)
        self.log("Number of trainable parameters (ViT): "+n_trainable_vit)

        self.log("Saving to: "+self.args.save_path)

        if self.args.save_best_ckpt:
            n_setups = len(self.args.gen_setups.split(","))
            assert -n_setups < self.args.best_ckpt_gen_setup_idx < n_setups, "save_best_ckpt_setup must be in gen_setups"
            
        if self.args.mode in ["new","load","cont"]:
            self.create_datasets(["train","vali"])
        
        self.kvs_buffer = {}
        self.kvs_gen_buffer = {}
        self.kvs_step_buffer = []            
        self.num_nan_losses = 0

        if self.args.image_encoder!="none":
            self.image_encoder = get_sam_image_encoder(self.args.image_encoder,device=self.device)

        #init models, optimizers etc
        
        if not self.model.has_unet:
            opt4 = get_opt4_from_cond_vit_setup(self.args.cond_vit_setup)
            assert opt4 in ["c","d"], "opt4 must be one of ['c','d'] when using no_unet. found: "+opt4

        if self.args.mode != "data":
            self.model = self.model.to(self.device)
            self.model_params = list(self.model.parameters())
            self.master_params = self.model_params
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
            if self.args.load_state_dict_loose and self.args.mode=="load":
                _ = load_state_dict_loose(self.model,ckpt["model"])
                self.master_params = self._state_dict_to_master_params(self.model.state_dict())
                #warn if there are no warmup steps
                if self.args.lr_warmup_steps==0:
                    self.log("WARNING: self.args.lr_warmup_steps=0. Warmup is recommended with loose model loading as it does not work for the optimizer.")
            else:
                self.opt.load_state_dict(ckpt["optimizer"])
                ckpt["model"] = fix_clip_matrix_in_state_dict(ckpt["model"],self.model)
                self.model.load_state_dict(ckpt["model"])
                self.master_params = self._state_dict_to_master_params(ckpt["model"])
                for i, ema_rate in enumerate(self.ema_rates):
                    if "ema_"+str(ema_rate) in ckpt.keys():
                        self.ema_params[i] = self._state_dict_to_master_params(ckpt["ema_"+str(ema_rate)])
            self.model_params = list(self.model.parameters())
            

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
            save_args(self.args)
            self.update_training_history(f"event={self.args.mode}, step={self.step}, time={get_time()}")
        elif self.args.mode in ["cont","gen"]:
            self.step = ckpt["step"]
            self.args.model_id = json.loads((Path(self.args.save_path)/"args.json").read_text())[0]["model_id"]
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
                event = f"event={self.args.mode}, step={self.step}, time={get_time()}"
                if isinstance(self.args.training_history,list):
                    self.args.training_history.append(event)
                else:
                    self.args.training_history = [event]
                overwrite_existing_args(self.args)
        elif self.args.mode=="data":
            self.log("Data mode, no training loop.")
            self.step = 0
            self.fixed_batch = None

        self.restart_flag = False
        self.log("Init complete.")

    def create_datasets(self,split_list=["train","vali"]):
        if isinstance(split_list,str):
            split_list = [split_list]
        for split in split_list:
            dataloader = get_dataset_from_args(self.args,split,mode="training")
            if self.args.debug_run=="no_dl":
                dataloader = AlwaysReturnsFirstItemOnNext(dataloader)
                setattr(self,split+"_dl",dataloader)
            elif self.args.debug_run=="anomaly":
                torch.autograd.set_detect_anomaly(True)
            elif self.args.debug_run=="only_dl":
                for _ in tqdm(dataloader):
                    pass
            elif self.args.debug_run=="cond_vit_info" and split=="vali":
                pprint(is_valid_cond_vit_setup(self.args.cond_vit_setup,long_names_instead=1))
                print(ModelInputKwargs(self.args,construct_args=True).kwarg_table)
                self.exit_flag = True
            elif self.args.debug_run=="unet_input_dict" and split=="vali":
                print(self.model.unet_input_dict)
                self.exit_flag = True
            elif self.args.debug_run=="vit_input_dict" and split=="vali":
                print(self.model.vit.input_dict)
                self.exit_flag = True
            else:
                setattr(self,split+"_dl",dataloader)

    def load_ckpt(self, ckpt_name, check_valid_args=False):
        if ckpt_name=="":
            ckpt_name = "ver-*/*"+self.args.model_name+"*/ckpt_*.pt"
        try:
            ckpt_name = get_ckpt_name(ckpt_name,return_multiple_matches=False)
        except ValueError as e:
            self.log("WARNING: "+str(e))
            self.exit_flag = True
            self.log("Exiting due too many/few ckpt matches.")
            return ""
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
            possible_ckpts = self.list_existing_ckpts(name_str)
            for ckpt_name in possible_ckpts:
                if str(ckpt_name)!=save_name:
                    os.remove(ckpt_name)

    def list_existing_ckpts(self,name_str="ckpt_"):
        return list(Path(self.args.save_path).glob(bracket_glob_fix(f"{name_str}*.pt")))
    
    def get_ema_model(self,ema_idx):
        assert ema_idx < len(self.ema_rates), "ema_idx must be smaller than the number of ema rates"
        tmp = make_master_params(copy.deepcopy(self.model_params))
        master_params_to_model_params(self.model_params, self.ema_params[ema_idx])

        self.ema_params[ema_idx] = tmp
        del tmp
        swap_pointers_func = lambda: self.get_ema_model(ema_idx)

        return self.model, swap_pointers_func

    def get_kwargs(self, batch, gen=False, del_none=True):
        x,info = batch
        if self.args.image_encoder!="none":
            if self.args.image_size!=x.shape[-1]:
                raise ValueError("image_size must be equal to the image size of the image encoder. Found "+str(self.args.image_size)+" vs "+str(x.shape[-1]))
                x = F.interpolate(x.float(),(self.args.image_size,self.args.image_size),mode="nearest-exact").to(self.device).long()
        else:
            x = x.to(self.device)
        unet_inputs,vit_inputs,used_inputs = unet_vit_inputs_from_args(self.args)
        model_kwargs = {k: [] for k in used_inputs}
        bs = x.shape[0]
        for i in range(bs):
            if self.args.debug_run=="no_kwargs":
                break
            if np.random.rand()<=self.args.p_image or gen:
                model_kwargs["image"].append(info[i]["image"])
            else:
                model_kwargs["image"].append(None)

            if "bbox" in used_inputs:
                if np.random.rand()<=self.args.p_bbox or gen:
                    model_kwargs["bbox"].append(bbox_image_from_label(x[i]))
                else:
                    model_kwargs["bbox"].append(None)

            if "points" in used_inputs:
                if np.random.rand()<=self.args.p_points or gen:
                    model_kwargs["points"].append(points_image_from_label(x[i]))
                else:
                    model_kwargs["points"].append(None)

            if "self_cond" in used_inputs:
                val = np.random.rand()<=self.args.p_self_cond or gen
                model_kwargs["self_cond"].append(torch.tensor(val,dtype=torch.long))
                    
            if "num_labels" in used_inputs:
                if np.random.rand()<=self.args.p_num_labels or gen:
                    model_kwargs["num_labels"].append(torch.tensor(info[i]["num_labels"],dtype=torch.long))
                else:
                    model_kwargs["num_labels"].append(None)
                    
            if "class_names" in used_inputs:
                if np.random.rand()<=self.args.p_class_names or gen:
                    class_names = list(info[i]["idx_to_class_name"].values())
                    if "padding" in class_names:
                        class_names.remove("padding")
                    model_kwargs["class_names"].append(
                        [prettify_classname(cn,dataset_name=info[i]["dataset_name"]) for cn in list(class_names)]
                        )
                else:
                    model_kwargs["class_names"].append(None)
            
            for k in dynamic_image_keys:
                if k in used_inputs:
                    model_kwargs[k].append(info[i].get("cond",{}).get(k,None))
                
            if "semantic" in used_inputs:
                if np.random.rand()<=self.args.p_semantic:
                    model_kwargs["semantic"].append(torch.tensor([info[i]["semantic"]+1],dtype=torch.long))
                else:
                    model_kwargs["semantic"].append(None)
        #end of bs loop
        if self.args.image_encoder!="none":
            model_kwargs["image_features"] = self.get_image_features(model_kwargs,bs)
        model_kwargs = cond_kwargs_int2bit(model_kwargs,ab=self.cgd.ab)
        model_kwargs = format_model_kwargs(model_kwargs,del_none=del_none,dev=self.device,list_instead=True)
        return x,model_kwargs,info
    
    def get_image_features(self,model_kwargs,bs):
        assert self.args.crop_method in ["sam_big","sam_small"], "image_encoder requires sam_big or sam_small crop_method"
        bs = len(model_kwargs["image"])
        image_idx = [i for i in range(bs) if model_kwargs["image"][i] is not None]
        if len(image_idx)>0:
            image = unet_kwarg_to_tensor([item for item in model_kwargs["image"] if item is not None])
            if not image.shape[-1]==image.shape[-2]==1024:
                image = resize(image,(1024,1024),antialias=True)
            with torch.no_grad():
                image_features = self.image_encoder(to_dev(image))
            model_kwargs["image_features"] = [None if i not in image_idx else image_features[j] 
                                              for j,i in enumerate(range(bs))]
            if self.args.crop_method=="sam_big":
                model_kwargs["image"] = F.avg_pool2d(unet_kwarg_to_tensor(model_kwargs["image"]),1024//self.args.image_size)
        return image_features

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
            if np.isnan(kvs_step["loss"]) or (self.restart_step2 and self.step>2):
                self.num_nan_losses += 1
                if self.num_nan_losses>20 or (self.restart_step2 and self.step>2):
                    self.log("Too many NaN losses, stopping training.")
                    self.restart_flag = True
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
        if "num_tokens" in nice_split(self.args.log_train_metrics):
            if hasattr(self.model,"last_token_info"):
                last_token_info = self.model.last_token_info
                num_tokens = num_tokens_from_token_info(last_token_info)
            else:
                num_tokens = 0
            kvs_step["num_tokens"] = num_tokens
        if hasattr(self.model,"last_token_info"):
            last_token_info = self.model.last_token_info
            num_tokens = num_tokens_from_token_info(last_token_info)
            metrics["num_tokens"] = num_tokens
        #reorder to match the order in args.log_train_metrics
        kvs_step = [kvs_step[k] for k in nice_split(self.args.log_train_metrics)]
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
                self.restart_flag = True
            return
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_loss_scale))
        if "grad_norm" in nice_split(self.args.log_train_metrics):
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
        if "grad_norm" in nice_split(self.args.log_train_metrics):
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
            if p.grad is not None:
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
        except KeyboardInterrupt:
            self.log("Exception in training loop:\n" + traceback.format_exc())
            self.log("Exiting training loop.")
            exit()
        except Exception as e:
            self.log("Exception in training loop:\n" + traceback.format_exc())
            self.log("Dumping last step kvs")
            self.dump_kvs(only_steps=True) if self.kvs_step_buffer else None
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
            if self.args.debug_run=="no_train_step":
                if self.step==1:
                    output,metrics = self.run_train_step(batch)
                    self.no_train_step_output = output,metrics
                else:
                    output,metrics = self.no_train_step_output
            else:
                output,metrics = self.run_train_step(batch)
            pbar.update(1)

            if self.step % self.args.log_vali_interval == 0 and self.args.log_vali_interval>0:
                self.evaluate_loop()
                
            if self.step % self.args.update_forward_pass_plot_interval == 0 and self.args.update_forward_pass_plot_interval>0:
                with MatplotlibTempBackend(backend="agg"):
                    plot_forward_pass(Path(self.args.save_path)/f"forward_pass_{self.step:06d}.png",
                                      output,metrics,ab=self.cgd.ab,sample_names=batch[-1],
                                      imagenet_stats=self.args.crop_method.startswith("sam"))
            
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
            if self.exit_flag or self.restart_flag:
                break
        if self.exit_flag:
            self.log("Training loop stopped due to exit flag.")
            self.update_training_history(f"event=exit, step={self.step}, time={get_time()}")
        elif self.restart_flag:
            restart_event = [s.find("event=restart")>=0 for s in self.args.training_history]
            num_restarts = sum(restart_event)
            if self.args.max_training_restarts>num_restarts:
                self.update_training_history(f"event=restart, step={self.step}, time={get_time()}")
                self.log(f"Restarting training loop, restart {num_restarts+1} of {self.args.max_training_restarts}.")
                ckpt_exists = self.step>=self.args.save_interval
                if ckpt_exists:
                    self.args.ckpt_name = ""
                    self.args.mode = "cont"
                self.restart_processing(num_restarts)
                self.init()
                self.train_loop()
            else:
                self.log(f"Exceeded max_training_restarts={self.args.max_training_restarts}, stopping training.")

        else:
            self.update_training_history(f"event=finished, step={self.step}, time={get_time()}")
            self.log("Training loop finished.")
    
    def restart_processing(self,num_restarts):
        restart_folder = Path(self.args.save_path)/"restart_logs"/f"restart_{num_restarts+1}"
        if not restart_folder.exists():
            os.makedirs(restart_folder, exist_ok=True)
        files_to_move = ["log.txt","logging.csv","logging_step.csv"]
        for file in files_to_move:
            file_old = restart_folder.parent.parent/file
            if file_old.exists():
                os.rename(file_old,restart_folder.parent/file)

    def update_training_history(self,event,do_nothing=False):
        if not do_nothing:
            if not isinstance(self.args.training_history,list):
                self.args.training_history = []
            self.args.training_history.append(event)
            overwrite_existing_args(self.args)
        
    def log(self, msg, filename="log.txt", also_print=True):
        """logs any string to a file"""
        filepath = Path(self.args.save_path)/filename
        if self.args.save_path!="" and self.args.mode!="data":
            if not filepath.exists():
                create_save = not self.exit_flag
                if create_save:
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
                        print(f"e1: {msg_hat}")
                        print(f"e2: {msg}")
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

    def dump_kvs(self, filename="logging.csv", only_steps=False):
        """
        Saves the kvs buffer and prints it, aswell as the kvs 
        step buffer to a file and then clears the buffers.
        """
        if not only_steps:
            self.log_kv({"time": time.time()-self.prev_time})
            self.prev_time = time.time()
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
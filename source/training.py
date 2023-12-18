import copy
import functools
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image

from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
#from .sampling_util import DiffusionSampler,plot_forward_pass
from .utils import (make_loss_plot,make_cond_loss_plot,load_state_dict_loose,ReturnOnceOnNext,
                    TemporarilyDeterministic,DummyWith,plot_fixed_images,dump_kvs)
from .datasets import get_random_points_images,get_noisy_bbox_images

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class DiffusionModelTrainer:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_dl,
        vali_dl,
        args,
        lg_loss_scale=None
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_dl = train_dl
        self.vali_dl = vali_dl
        self.args = args
        
        #init variables
        self.step = 0
        self.kvs = {}
        
    def train_loop(self):
        for _ in range(10):
            self.step += 1
            self.log_kv({"step":self.step,"loss": 0.01})
            self.log_kv_step({"step":self.step,"loss": 0.01})
    
    def log(self, msg, filename="log.txt"):
        """logs any string to a file"""
        with open(str(Path(self.args.save_path)/filename), "a") as f:
            f.write(msg + "\n")
    
    def log_kv(self, key, value):
        """
        Saves key-value pairs in a buffer to be saved to a file later.
        Any values (lists or float/int) associated with a key will be 
        reduced to their mean when logged.
        """
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
            if k in self.kvs:
                if isinstance(self.kvs[k],list):
                    if isinstance(v,list):
                        self.kvs[k].extend(v)
                    else:
                        self.kvs[k].append(v)
            else:
                self.kvs[k] = [v]
    
    def log_kv_step(self, key, value):
        """
        Saves key-value pairs in a buffer to be saved to a file later.
        No reduction is applied to the values. Only 1 value can be 
        used per key per training step.
        """
        
    def dump_kvs(self, filename="progress.csv"):
        for k,v in self.kvs:
            if isinstance(v,list):
                self.kvs[k] = np.mean(v)
        dump_kvs(str(Path(self.save_path)/filename),self.kvs)
        self.kvs = {}
        
    def get_logdir(self):
        return os.path.abspath(os.path.join(self.args.logs_folder,self.args.model_name_long))




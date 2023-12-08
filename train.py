import shutil
import argparse
import datetime
import json
import os
from pathlib import Path
import git
#from improved_diffusion.dist_mod import MPI,dist_util
import torch
from source import logger#dist_util,
from source.script_util import SmartParser, model_specific_args, write_args
#from improved_diffusion.train_util import TrainLoop
#from improved_diffusion.utils import set_random_seed, num_of_params
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from pprint import pprint
from source.datasets import CatBallDataset, custom_collate_with_info
import numpy as np

#TODO
#make seeding better in the sampler (one seed to define a whole diffusion process instead of 1000)
#test parser
#implement all points in arguments
#add version looping that automatically finds name and saves
#make sure loading trained models works

def main(**modified_args):
    args = SmartParser().get_args(modified_args)    
    if isinstance(args,tuple):
        args, modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    
    new_log = True
    logs_root = Path(args.logs_folder)
    print("logs_root",logs_root)
    
    lg_loss_scale = None
    if args.resume_from_step and len(args.resume_ckpt_name)>0:
        #find out if ckpt is available
        ckpt_match = list(Path(logs_root).glob(f"*{args.model_name}/{args.model_str}.pt"))
        if len(ckpt_match)==1:
            new_log = False
            args.resume_ckpt_path = str(ckpt_match[0])
            log_path = logs_root / ckpt_match[0].parent.name
            
            filename = str(log_path / "progress.csv")
            data = np.genfromtxt(filename, delimiter=",")[1:]
            if len(data.shape)==1:
                data = np.expand_dims(data,0)
            data = data[~np.any(np.isinf(data),axis=1)]
            column_names = open(filename).readline().strip().split(",")
            lg_loss_scale = data[-1,column_names.index("lg_loss_scale")]
        else:
            available_ckpts = [str(p) for p in ckpt_match]
            raise ValueError(f"Could not find ckpt for {args.model_name} in {logs_root} with model_str={args.model_str}. Available ckpts: {available_ckpts}")
            
    save_path = logs_root / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{args.model_name}"
    
    pprint(args.__dict__)
    
    if new_log:
        os.makedirs(save_path, exist_ok=True)
        write_args(args, save_path / "args.json")
    
if __name__ == "__main__":
    main()


import sys, os
sys.path.append(os.path.abspath('./source/'))
from source.training import DiffusionModelTrainer
import warnings
from pathlib import Path
warnings.filterwarnings('ignore', category=DeprecationWarning)
from source.utils.argparse_utils import TieredParser, load_existing_args, get_ckpt_name
import json

def main(**modified_args):
    sample_opts = TieredParser("sample_opts").get_args(modified_args=modified_args)
    if isinstance(sample_opts,list):
        #sample_opts, modified_args_list = sample_opts
        modified_args_list = sample_opts
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    ckpt_name = get_ckpt_name(sample_opts.name_match_str,return_multiple_matches=False)
    print("\nckpt_name:",ckpt_name)
    if len(ckpt_name)==0:
        print("No ckpt found")
        return
    print(str(Path(ckpt_name).parent / "args.json"))
    model_id = load_existing_args(str(Path(ckpt_name).parent / "args.json"),"args",verify_keys=False).model_id
    print("\nmodel_id:",model_id)
    args = load_existing_args(model_id,"args",verify_keys=True)
    if sample_opts.seed>=0:
        args.seed = sample_opts.seed
    args.mode = "gen"
    args.ckpt_name = ckpt_name
    trainer = DiffusionModelTrainer(args)
    trainer.generate_samples([sample_opts])
    
if __name__ == "__main__":
    main()


import sys, os
sys.path.append(os.path.abspath('./source/'))
from source.training import DiffusionModelTrainer, trainer_from_sample_opts
import warnings
from pathlib import Path
warnings.filterwarnings('ignore', category=DeprecationWarning)
from source.utils.argparsing import TieredParser, load_existing_args, get_ckpt_name

def main(**modified_args):
    sample_opts = TieredParser("sample_opts").get_args(modified_args=modified_args)
    if isinstance(sample_opts,list):
        #sample_opts, modified_args_list = sample_opts
        modified_args_list = sample_opts
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    trainer = trainer_from_sample_opts(sample_opts,verbose=True)
    trainer.generate_samples([sample_opts])
    
if __name__ == "__main__":
    main()
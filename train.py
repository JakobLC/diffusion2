

import sys, os
sys.path.append(os.path.abspath('./source/'))

from source.utils import SmartParser
from source.training import DiffusionModelTrainer
import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main(**modified_args):    
    args = SmartParser().get_args(modified_args)    
    if isinstance(args,tuple):
        args, modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    if args.model_name.startswith("test"):
        save_path = "./saves/" + args.model_name
    else:
        save_path = str(Path("./saves/") / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{args.model_name}")
    args.save_path = save_path
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()
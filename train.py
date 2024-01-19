

import sys, os
sys.path.append(os.path.abspath('./source/'))

from source.utils import SmartParser
from source.training import DiffusionModelTrainer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


#TODO
# implement all points in arguments
# forced xstart
# make seeding better in the sampler (one seed to define a whole diffusion process instead of 1000)
# make sample setup strings
# make sample.py
# remove lines from dead runs in the logging files when continuing training
# make nuke.py to remove all dead runs from the logging files
# add SAM image features or image embedding model
# write pretty_point() function
def main(**modified_args):    
    args = SmartParser().get_args(modified_args)    
    if isinstance(args,tuple):
        args, modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
        
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()
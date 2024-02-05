

import sys, os
sys.path.append(os.path.abspath('./source/'))

from source.utils import SmartParser
from source.training import DiffusionModelTrainer
from pathlib import Path
#import warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)


#TODO
# implement all points in arguments
# implement forced xstart
# implement fixed batches for long training runs
# make nuke.py to remove all dead runs from the logging files
# add SAM image features or image embedding model
# implement dummy diffusion
# implement timestep delta from bit diffusion
# implement better corrupt from bit diffusion
# add option to save sampling results to sample_info.json
# make background is not special
# loop over all data, save better masks as bitmaps
# make num points depend on num labels
def main(**modified_args):
    args = SmartParser().get_args(modified_args=modified_args)
    if isinstance(args,list):
        modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
        
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()
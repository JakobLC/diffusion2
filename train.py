

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
# add SAM image features or image embedding model
# implement dummy diffusion
# implement timestep delta from bit diffusion
# implement better corrupt from bit diffusion
# add option to save sampling results to sample_info.json
# make num points depend on num labels
# add data: VISOR, Totalsegmenter data, UVO
# add https://github.com/lilohuang/PyTurboJPEG to load jpegs
# add logsnr_wrap
# write dataset+id for each sample in forwardpass/grid

#model TODO
# loop over lr
# loop over input scaling and image sizes
# 

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
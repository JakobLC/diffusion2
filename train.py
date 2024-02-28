

import sys, os
sys.path.append(os.path.abspath('./source/'))

#from source.utils import SmartParser
from source.argparse_utils import TieredParser, save_args, load_existing_args
from source.training import DiffusionModelTrainer

#import warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)


#TODO [priority]
# [1] make K-means clustering of SAM features for comparison
# [2] implement dummy diffusion
# [2] make so if you do --model_name m1;m1 --some_other_arg arg1;arg2, it will also loop when followed up with --mode cont
# [3] fix SAM augmentations
# [3] add model_name[ver1;ver2] argparsing
# [3] try binary sampled and thresholded stepper
# [3] implement forced xstart
# [3] implement fixed batches for long training runs
# [4] make MSE plot in forward pass show multiplied with loss weight but "MSE=" not multiplied
# [4] add split_idx to datasets lacking them
# [4] implement timestep delta from bit diffusion
# [4] implement better corrupt from bit diffusion
# [4] make num points depend on num labels
# [4] add data: VISOR, Totalsegmenter data, UVO
# [5] make continuing training easier
# [5] add time, step saving to args
# [5] implement all points in arguments
# [5] check multiple workers works
# [5] add num classes to image renderings
# [5] resume training by model_id
# [5] add continue training on nan losses, max resumed trainings argument

#training TODO
# loop over lr
# loop over input scaling and image sizes
# train model with low logsnr_max and v prediction (logsnr_max=5)

def main(**modified_args):
    args = TieredParser().get_args(modified_args=modified_args)
    if isinstance(args,list):
        modified_args_list = args
        for modified_args in modified_args_list:
            main(**modified_args)
        return
    trainer = DiffusionModelTrainer(args)
    trainer.train_loop()
    
if __name__ == "__main__":
    main()
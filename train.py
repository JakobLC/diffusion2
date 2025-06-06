

import sys, os
sys.path.append(os.path.abspath('./source/'))

#from source.utils import SmartParser
from source.utils.argparsing import TieredParser, save_args, load_existing_args
from source.training import DiffusionModelTrainer

#import warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)


#TODO [priority]
# [2] implement dummy diffusion
# [2] make so if you do --model_name m1;m1 --some_other_arg arg1;arg2, it will also loop when followed up with --mode cont
# [3] fix SAM augmentations
# [3] add model_name[ver1;ver2] argparsing
# [3] implement forced xstart
# [3] implement fixed batches for long training runs
# [3] make so weights are created in order that allow loose state dict loading
# [4] make MSE plot in forward pass show multiplied with loss weight but "MSE=" not multiplied
# [4] implement timestep delta from bit diffusion
# [4] implement better corrupt from bit diffusion
# [4] make num points depend on num labels, size of label
# [5] implement all points in arguments
# [5] check multiple workers works
# [4] make i.e. 2x,4x imsize compared to diff sample 
# [5] resume training by model_id
# [7] add half precision to vit
# [5] add parameter to NOT give timesteps to ViT

# [9] decouple get_kwargs from trainer
# [10] add sam encoder to dataloader
# [10] make all probabilities get sampled with the dataset (with option for generation mode)'
# [10] make unfified framework for probabilities

# [11] fix sam encoder
# [12] add support for non-agnostic onehot
# [12] add support for non-agnostic multi dataset with analog bits
# [14] unet add shared dynamic inputs channels and class embed instead
# [15] fix restart logs so you keep steps up until ckpt loading

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
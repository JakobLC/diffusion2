import os,sys
sys.path.append('./source/')
import matplotlib.pyplot as plt
from source.utils.mixed import set_random_seed, apply_mask
from source.utils.dataloading import get_dataset_from_args
from source.utils.plot import (visualize_dataset_with_labels, 
                              visualize_batch,visualize_cond_batch,imagenet_preprocess)
from source.utils.dataloading import custom_collate_with_info
from source.utils.argparsing import TieredParser
import numpy as np
from PIL import Image
import jlc

modified_args={"diff_channels": 6,
                "dl_num_workers": 0,
                "p_semantic": 0,
                "semantic_dl_prob": 0,
                "seed": 2,
                "image_size": 256,
                "datasets": "entityseg",
                "train_batch_size": 20}
args = TieredParser().get_args(alt_parse_args=["--model_name","default"],modified_args=modified_args)

dli = get_dataset_from_args(args,split="all",return_type="ds")
indices = []
for i in range(2800):
    if 86 in dli.items[i]["classes"]:
        indices.append(i)
print(f"Found {len(indices)} images with class 86")
indices = indices[:20]
batch = custom_collate_with_info([dli[i] for i in indices])
print(batch[1][0].keys())
print([batch[1][i]["i"] for i in range(len(batch[1]))])
im = visualize_batch(batch,figsize_per_pixel=0.02,class_text_size=10,
                     with_class_names=1,with_text_didx=1,alpha_mask=0.4,
                     show_border=1,crop=1,n_col=5,return_im=1,imshow=0)
im = np.clip(im*255,0,255).astype(np.uint8)
Image.fromarray(im).save("tmp3.png")

#good index from entityseg: 41, 942
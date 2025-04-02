import os,sys
sys.path.append('/home/jloch/Desktop/diff/diffusion2/')
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
                "image_size": 1000,
                "datasets": "entityseg",
                "train_batch_size": 1}
args = TieredParser().get_args(alt_parse_args=["--model_name","default"],modified_args=modified_args)

dli = get_dataset_from_args(args,split="all",return_type="ds")
mask,info = dli[52]
print(info["imshape"])
image = imagenet_preprocess(info["image"].permute(1,2,0).numpy(),inv=True,dim=2)[:800]
mask = mask.permute(1,2,0).numpy()[:800]
class_names = info["idx_to_class_name"]
#print(jlc.mask_overlay_smooth.__code__.co_varnames)
im = jlc.mask_overlay_smooth(image,mask,alpha_mask=0.5,class_names=class_names,show_border=1,fontsize=20)
#im = np.clip(im*255,0,255).astype(np.uint8)
Image.fromarray(im).save("tmp/tmp4.png")
Image.fromarray(np.clip(image*255,0,255).astype(np.uint8)).save("tmp/tmp4_image.png")
im_no_names = jlc.mask_overlay_smooth(image,mask,alpha_mask=0.5,show_border=1)
for uq in np.unique(mask):
    #save transparent pngs for each bbox of a class
    name = f"{class_names[uq]}_{uq}.png"
    bbox = (np.where(mask==uq)[1].min(),np.where(mask==uq)[0].min(),
            np.where(mask==uq)[1].max(),np.where(mask==uq)[0].max())
    mask_im = im_no_names[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    mask_mask = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    #add transparency channel as the mask itself
    mask_im = np.dstack([mask_im,np.where(mask_mask==uq,255,0)]).astype(np.uint8)
    Image.fromarray(mask_im).save("./tmp/masks/"+name)

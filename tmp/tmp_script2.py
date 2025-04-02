import torch
import jlc
from PIL import Image
import numpy as np
i = 0
loaded = torch.load("tmp/tmp.pth")

#jlc.shaprint(loaded)

#print(jlc.mask_overlay_smooth.__code__.co_varnames)
image = loaded["hd"][i]["image"].numpy()
mask = loaded["hd"][i]["pred_int"].numpy()
im = jlc.mask_overlay_smooth(image,mask,alpha_mask=0.5,show_border=1)
for uq in np.unique(mask):
    #save transparent pngs for each bbox of a class
    name = f"{uq}.png"
    bbox = (np.where(mask==uq)[1].min(),np.where(mask==uq)[0].min(),
            np.where(mask==uq)[1].max(),np.where(mask==uq)[0].max())
    mask_im = im[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    mask_mask = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    #add transparency channel as the mask itself
    mask_im = np.dstack([mask_im,np.where(mask_mask==uq,255,0)]).astype(np.uint8)
    Image.fromarray(mask_im).save("./tmp/masks/"+name)

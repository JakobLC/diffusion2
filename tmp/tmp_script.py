import os,sys
sys.path.append('./')
from source.sam import evaluate_sam, all_sam_setups

from source.eval_and_plotting import SavedSamples, SavedSamplesManager
import argparse

from PIL import Image
import torch
"""
import matplotlib.pyplot as plt
import numpy as np
import jlc"""

save_data = True
save_img = True
postprocess = False

generator_kwargs = all_sam_setups["sam2_1"]
generator_kwargs.update({"points_per_side": 32,
                         "points_per_batch": 64,
                        "pred_iou_thresh": 0.5,
                        "stability_score_thresh": 0.7,
                        "stability_score_offset": 0.7,
                        "crop_n_layers": 2,
                        "crop_n_points_downscale_factor": 2,
                        "use_m2m": False})
#defaults below
"""sam2_setups =  {1: {"points_per_side": 64,
                    "points_per_batch": 128,
                    "pred_iou_thresh": 0.7,
                    "stability_score_thresh": 0.92,
                    "stability_score_offset": 0.7,
                    "crop_n_layers": 1,
                    "box_nms_thresh": 0.7,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 25.0,
                    "use_m2m": True}"""


#generator_kwargs[""] = 0

eval_sam_kwargs = argparse.Namespace(datasets="entityseg",
                                    model_type=3,
                                    num_return_segments=2,
                                    split="all",
                                    ratio_of_dataset=2,
                                    generator_kwargs=generator_kwargs,
                                    pri_didx=["entityseg/52","entityseg/41"],
                                    longest_side_resize=1000,
                                    batch_size=2,
                                    postprocess_kwargs=None,
                                    full_resolution_decoder=True)
metrics_mean, light_data, heavy_data = evaluate_sam(**vars(eval_sam_kwargs))

sam_samples = SavedSamples(light_data=light_data,heavy_data=heavy_data)
if postprocess:
    sam_samples.postprocess({"mode": "min_area", "min_area": 0.005})

didx = sam_samples.didx#[sam_samples.didx[i] for i in [5,13,14,0]]
print(didx)
ssm = SavedSamplesManager(sam_samples)
im = ssm.plot_qual_seg(didx=didx,resize_width=1024,transpose=1,add_text_axis=1,num_images=2)
print(im.shape)
if save_img:
    Image.fromarray(im).save("tmp/tmp.png")
if save_data:
    torch.save({"hd": heavy_data, "ld": light_data},"tmp/tmp.pth")
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import torch
import argparse
from jlc import shaprint
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling.sam import Sam

#add source if we are running this file directly
if __name__=="__main__":
    sys.path.append("/home/jloch/Desktop/diff/diffusion2")

from source.utils.metric_and_loss_utils import get_segment_metrics
from source.utils.argparse_utils import TieredParser
from source.utils.mixed_utils import postprocess_list_of_segs
from source.training import DiffusionModelTrainer
from source.utils.data_utils import get_dataset_from_args
import warnings 
import tqdm

#add sam2 to path
sys.path.append("/home/jloch/Desktop/diff/segment-anything-2/")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.modeling.sam2_base import SAM2Base
sam12_info = {  "names":      ["sam1_b", "sam1_l" ,"sam1_h"  ,"sam2_t", "sam2_s", "sam2_b+"  , "sam2_l" ],
                "ckpt_names": ["01ec64", "0b3195" ,"4b8939"  ,"tiny"  , "small" , "base_plus", "large"  ],
                "model_cfgs": [ None   , None     , None     ,"t"     , "s"     , "b+"       , "l"      ],
                "num_params": [89670912, 308278272, 637026048,38945986, 46043842, 80833666   , 224430130]}

def get_sam12(name_or_idx,
              apply_postprocessing=False,
              device="cuda"):
    if isinstance(name_or_idx, str):
        assert name_or_idx in sam12_info['names'], f"Invalid name {name_or_idx}. Valid names are {sam12_info['names']}"
        idx = sam12_info['names'].index(name_or_idx)
    else:
        assert isinstance(name_or_idx, int), "name_or_idx must be an integer or a string"
        assert 0 <= name_or_idx < len(sam12_info['ckpt_names']), f"name_or_idx out of range, idx={name_or_idx}"
        idx = name_or_idx
    is_sam1 = sam12_info['names'][idx].startswith("sam1")

    if is_sam1:
        name = sam12_info['names'][idx]
        ckpt_name = name.replace("sam1","sam_vit")+f"_{sam12_info['ckpt_names'][idx]}"
        model_type = sam12_info['names'][idx].replace("sam1","vit")

        sam1_checkpoint = "../segment-anything/segment_anything/checkpoint/"+ckpt_name+".pth"
        sam = sam_model_registry[model_type](checkpoint=sam1_checkpoint)
        sam.to(device=device)
    else:
        sam2_checkpoint = f"../segment-anything-2/checkpoints/sam2_hiera_{sam12_info['ckpt_names'][idx]}.pt"
        model_cfg = f"sam2_hiera_{sam12_info['model_cfgs'][idx]}.yaml"
        sam = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=apply_postprocessing)
    return sam

def to_cpu_if_torch(x):
    if torch.is_tensor(x):
        return x.cpu()
    else:
        return x

def to_numpy_if_torch(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        assert isinstance(x,np.ndarray), "expected x to be a numpy array or a torch tensor, found "+str(type(x))
        return x

class Sam2AgnosticGenerator(SAM2AutomaticMaskGenerator):
    def __init__(self, sam, **kwargs):
        super().__init__(sam, **kwargs)

    def batched_generate(self, images, image_features=None):
        if image_features is not None:
            raise NotImplementedError("image_features not supported for sam2")
        masks = []
        for i in range(len(images)):
            image = images[i]
            mask = self.generate(image)
            masks.append(mask)
        segmentations = [get_segmentation(mask) for mask in masks]
        return segmentations
    
class SamAgnosticGenerator(SamAutomaticMaskGenerator):
    def __init__(self, sam, **kwargs):
        super().__init__(sam, **kwargs)
        self.data_root = os.path.abspath("./data")

        self.predictor = SamAgnosticPredictor(sam)

    def batched_generate(self, images, image_features=None):
        if image_features is not None:
            self.predictor.store_image_batch(image_features)
        masks = []
        for i in range(len(images)):
            image = images[i]
            mask = self.generate(image)
            masks.append(mask)
        segmentations = [get_segmentation(mask) for mask in masks]
        return segmentations

    def batched_generate_raw(self, model_kwargs,info):
        if "image_features" in model_kwargs.keys():
            self.store_image_batch(model_kwargs["image_features"])
        images = []
        gts = []
        for i in range(len(info)):
            image_path = os.path.join(self.data_root,info[i]["dataset_name"],info[i]["image_path"])
            label_path = os.path.join(self.data_root,info[i]["dataset_name"],info[i]["label_path"])
            images.append(np.array(Image.open(image_path)))
            gts.append(np.array(Image.open(label_path)))
        bs = len(info)
        masks = []
        for i in range(bs):
            image = images[i]
            mask = self.generate(image)
            masks.append(mask)
        return masks, images, gts

class Sam2AgnosticPredictor(SamPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SamAgnosticPredictor(SamPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_stored_features = False

    @torch.no_grad()
    def set_torch_image(self,transformed_image,original_image_size):
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()
        
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        if self.has_stored_features:
            self.features = self.restore_next()
        else:
            self.features = self.model.image_encoder(self.model.preprocess(transformed_image))
        self.is_image_set = True

    def restore_next(self):
        self.features = self.stored_features[self.stored_idx].unsqueeze(0)
        self.stored_idx += 1
        if self.stored_idx == len(self.stored_features):
            self.has_stored_features = False
        return self.features

    def store_image_batch(self, features):
        self.stored_features = features
        self.has_stored_features = True
        self.stored_idx = 0


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def get_segmentation(anns):
    if len(anns) == 0:
        warnings.warn("No annotations found")
        return np.zeros((1,1))
    h, w = anns[0]['segmentation'].shape
    segment = np.zeros((h,w), dtype=np.uint8)
    for k, ann in enumerate(anns):
        segment[ann['segmentation']] = k+1
        if k == 255:
            warnings.warn("More than 255 segments found, only the first 255 will be shown.")
            break
    return segment

def evaluate_sam(datasets="ade20k",
                 model_type=0,
                 model_name_for_dataloader="qual[T0]",
                 num_return_segments=0,
                 split="vali",
                 ratio_of_dataset=1.0,
                 generator_kwargs={},
                 longest_side_resize=0,
                 pri_didx=None,
                 progress_bar=True,
                 device="cuda",
                 batch_size=4,
                 full_resolution_decoder=False,
                 return_heavy_keys = ["pred_int","image","gt"],
                 postprocess_kwargs=None,
                 keep_info_keys=["split_idx","i","dataset_name","num_classes"]):
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "datasets must be a string or a list of strings"
        datasets = datasets.split(",")
    if isinstance(split,int):
        assert split in [0,1,2,3], "split must be one of [0,1,2,3] or one of [train,vali,test,all]"
        split = ["train","vali","test","all"][split]
    assert num_return_segments<=64, "num_return_segments must be less than or equal to 64, due to memory constraints"
    assert isinstance(longest_side_resize,int), "longest_side_resize must be an integer"
    
    args = TieredParser().get_args(alt_parse_args=["--model_name", model_name_for_dataloader,
                                                    "--ckpt_name", "*"+model_name_for_dataloader+"*",
                                                    "--mode","data",
                                                    "--train_batch_size",str(batch_size)])
    
    #this does not matter unless features from the dataloader are used:
    if longest_side_resize>0 and not longest_side_resize==1024:
        args.image_size = longest_side_resize
        args.crop_method = "sam_small"

    lsr = 0 if full_resolution_decoder else longest_side_resize
    #args.image_encoder = ['sam_vit_b','sam_vit_l','sam_vit_h'][model_type] if model_type<3 else ""
    args.datasets = datasets
    dli = get_dataset_from_args(args,prioritized_didx=pri_didx,return_type="dli")
    load_raw_image_label = dli.dataloader.dataset.load_raw_image_label
    
    sam = get_sam12(model_type,apply_postprocessing=False,device=device)
    if isinstance(sam,Sam):
        sam_agnostic = SamAgnosticGenerator(sam,**generator_kwargs)
    elif isinstance(sam,SAM2Base):
        sam_agnostic = Sam2AgnosticGenerator(sam,**generator_kwargs)
    else:
        raise NotImplementedError(f"Unknown model type {type(sam)}")
    n = len(dli)
    if isinstance(ratio_of_dataset,float) and ratio_of_dataset<=1:
        n_batches = np.ceil(n*ratio_of_dataset).astype(int)
    else:
        assert ratio_of_dataset>=batch_size, "ratio_of_dataset must be greater than batch_size"
        n_batches = np.ceil(ratio_of_dataset/batch_size).astype(int)
    n_batches = min(n_batches,n)
    assert n_batches>0, "no data found"
    
    heavy_data = []
    light_data = []
    print(f"evaluating {n_batches} batches")
    for batch_num in tqdm.tqdm(range(n_batches),disable=not progress_bar):
        info = next(dli)[-1]
        bs = len(info)
        images = []
        gts = []
        for i in range(bs):
            image,gt = load_raw_image_label(info[i],longest_side_resize=lsr)
            images.append(image)
            gts.append(gt)

        segmentations = sam_agnostic.batched_generate(images)
        if postprocess_kwargs is not None:
            segmentations = postprocess_list_of_segs(segmentations,postprocess_kwargs)
        n_heavy = max(min(num_return_segments-len(heavy_data),bs),0)
        if n_heavy:
            for i in range(n_heavy):
                hd = {"pred_int": segmentations[i],
                      "image": images[i],
                      "gt": gts[i]}
                heavy_data.append({k: to_cpu_if_torch(hd[k]) for k in return_heavy_keys})
        #extend with None
        heavy_data.extend([None for _ in range(bs-n_heavy)])
        metrics = [get_segment_metrics(segmentations[i][None],info[i]) for i in range(bs)]
        light_data_batch = []
        for i in range(bs):
            light_data_batch.append({"info": {k: v for k,v in info[i].items() if k in keep_info_keys},
                                     "metrics": metrics[i]})

        light_data.extend(light_data_batch)
    metrics_mean = {k: np.mean([ld["metrics"][k] for ld in light_data if ld is not None]) for k in light_data[0]["metrics"].keys()}
    return metrics_mean, light_data, heavy_data

sam1_setups =  {1: {},
                2: {"points_per_side": 32,
                    "pred_iou_thresh": 0.86,
                    "stability_score_thresh": 0.92,
                    "crop_n_layers": 1,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 100}, # 100 pixels, fixed regardless of image size
                3: {"points_per_side": 32,
                    "pred_iou_thresh": 0.86,
                    "stability_score_thresh": 0.92,
                    "crop_n_layers": 1,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 128**2*0.005} # 0.5% of the image area
                }
sam2_setups =  {1: {"points_per_side": 64,
                    "points_per_batch": 128,
                    "pred_iou_thresh": 0.7,
                    "stability_score_thresh": 0.92,
                    "stability_score_offset": 0.7,
                    "crop_n_layers": 1,
                    "box_nms_thresh": 0.7,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 25.0,
                    "use_m2m": True}
                }
all_sam_setups = {**{f"sam1_{k}": v for k,v in sam1_setups.items()},
                  **{f"sam2_{k}": v for k,v in sam2_setups.items()}}

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",type=str,default="ade20k")
    parser.add_argument("--model_type",default=0)
    parser.add_argument("--setup",type=str,default="sam1_1")
    
    args = parser.parse_args()
    postprocess_kwargs = None
    """postprocess_kwargs = {"mode": "min_area", "min_area": 0.005}
    generator_kwargs = {"points_per_side": 32,
                        "pred_iou_thresh": 0.86,
                        "stability_score_thresh": 0.92,
                        "crop_n_layers": 1,
                        "crop_n_points_downscale_factor": 2,
                        "min_mask_region_area": 100}"""
    generator_kwargs = all_sam_setups[args.setup]
    metrics_mean, light_data, heavy_data = evaluate_sam(datasets=args.datasets,
                                                        model_type=args.model_type,
                                                        num_return_segments=16,
                                                        split="vali",
                                                        ratio_of_dataset=16,
                                                        generator_kwargs=generator_kwargs,
                                                        pri_didx=None,
                                                        longest_side_resize=1024,
                                                        batch_size=4,
                                                        postprocess_kwargs=postprocess_kwargs)
    print(metrics_mean)

if __name__=="__main__":
    main()
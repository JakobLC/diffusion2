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

from source.utils.metric_and_loss import get_segment_metrics
from source.utils.argparsing import TieredParser
from source.utils.mixed import postprocess_list_of_segs
from source.training import DiffusionModelTrainer
from source.utils.dataloading import get_dataset_from_args
import warnings
import tqdm

#add sam2 to path
#sys.path.append("/home/jloch/Desktop/diff/segment-anything-2/") old setup, using direct repository import. now using pip install
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.modeling.sam2_base import SAM2Base

sam12_info = {  "names":      ["sam1_b"  , "sam1_l"   ,"sam1_h"    ,"sam2_t"  , "sam2_s"  , "sam2_b+"  , "sam2_l" ],
                "ckpt_names": ["01ec64"  , "0b3195"   ,"4b8939"    ,"tiny"    , "small"   , "base_plus", "large"  ],
                "model_cfgs": [ None     , None       , None       ,"t"       , "s"       , "b+"       , "l"      ],
                "num_params": [89_670_912, 308_278_272, 637_026_048,38_945_986, 46_043_842, 80_833_666 , 224_430_130]}

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
        print(f"Evaluating with SAM1 model: {name} ({ckpt_name})")

        sam1_checkpoint = "../segment-anything/segment_anything/checkpoint/"+ckpt_name+".pth"
        sam = sam_model_registry[model_type](checkpoint=sam1_checkpoint)
        sam.to(device=device)
    else:
        sam2_checkpoint = f"../segment-anything-2/checkpoints/sam2_hiera_{sam12_info['ckpt_names'][idx]}.pt"
        model_cfg = f"sam2_hiera_{sam12_info['model_cfgs'][idx]}.yaml"
        sam = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=apply_postprocessing)
        print(f"Evaluating with SAM2 model: {sam12_info['names'][idx]} ({sam2_checkpoint})")
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
        Hs = [image.shape[0] for image in images]
        Ws = [image.shape[1] for image in images]
        segmentations = [get_segmentation(mask,h,w) for mask,h,w in zip(masks,Hs,Ws)]
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
        Hs = [image.shape[0] for image in images]
        Ws = [image.shape[1] for image in images]
        segmentations = [get_segmentation(mask,h,w) for mask,h,w in zip(masks,Hs,Ws)]
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

def get_segmentation(anns,h,w):
    segment = np.zeros((h,w), dtype=np.uint8)
    if len(anns) == 0:
        warnings.warn("No annotations found")
    #sort by size:
    size_order = np.argsort([a['area'] for a in anns])[::-1]
    if len(anns) > 255:
        warnings.warn("More than 255 segments found, only the largest 255 are included.")
    for k, i in enumerate(size_order):
        ann = anns[i]
        assert [h,w] == list(ann['segmentation'].shape), f"segmentation shape {ann['segmentation'].shape} does not match image shape ({h,w})"
        segment[ann['segmentation']] = k+1
        if k == 255:
            break
    return segment

def evaluate_sam(datasets="ade20k",
                 model_type=0,
                 model_name_for_dataloader="default",
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
    dli = get_dataset_from_args(args,prioritized_didx=pri_didx,return_type="dli", split=split,
                                mode="pri_didx" if pri_didx is not None else "training")
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

"""
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.
        """

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
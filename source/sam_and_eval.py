import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os,sys
import argparse
from training import DiffusionModelTrainer
from argparse_utils import TieredParser
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from plot_utils import mask_overlay_smooth,index_dict_with_bool
from collections import defaultdict
from utils import get_segment_metrics_np
import tqdm
from pathlib import Path
from datasets import AnalogBits,load_raw_image_label
import copy
from utils import imagenet_preprocess

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
    h, w = anns[0]['segmentation'].shape
    segment = np.zeros((h,w), dtype=np.uint8)
    for k, ann in enumerate(anns):
        segment[ann['segmentation']] = min(255, k+1)
    return segment

class SamAgnosticPredictor(SamPredictor):
    def __init__(self, sam, generator_kwargs={}):
        super().__init__(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam,**generator_kwargs)
        self.has_stored_features = False
        self.data_root = os.path.abspath("./data")
        
    def store_image_batch(self, features):
        self.stored_features = features
        self.has_stored_features = True
        self.stored_idx = 0

    def restore_next(self):
        self.features = self.stored_features[self.stored_idx]
        self.stored_idx += 1
        if self.stored_idx == len(self.stored_features):
            self.has_stored_features = False

    def batched_generate(self, images, gts, image_features=None):
        if image_features is not None:
            self.store_image_batch(image_features)
        masks = []
        for i in range(len(images)):
            image = images[i]
            mask = self.mask_generator.generate(image)
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
            mask = self.mask_generator.generate(image)
            masks.append(mask)
        return masks, images, gts
    
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
            input_image = self.model.preprocess(transformed_image)
            self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

def evaluate_sam(datasets="ade20k",
                 model_type=0,
                 model_name_for_dataloader="sam[64_native_tanh]",
                 num_return_segments=0,
                 return_idx=None,
                 split="vali",
                 ratio_of_dataset=1.0,
                 generator_kwargs={},
                 longest_side_resize=0,
                 progress_bar=True,
                 device="cuda",
                 batch_size=4,
                 full_resolution_decoder=False):
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "datasets must be a string or a list of strings"
        datasets = datasets.split(",")
    if isinstance(model_type,int):
        assert model_type in [0,1,2], "model_type must be one of [0,1,2] or one of [vit_b, vit_l, vit_h]"
        model_type = ["vit_b","vit_l","vit_h"][model_type]
    if isinstance(split,int):
        assert split in [0,1,2,3], "split must be one of [0,1,2,3] or one of [train,vali,test,all]"
        split = ["train","vali","test","all"][split]
    assert num_return_segments<=64
    if return_idx is not None:
        assert len(datasets.split(","))==1, "return_idx can only be used with a single dataset"
        if len(return_idx)>num_return_segments:
            return_idx = return_idx[:num_return_segments]
    assert isinstance(longest_side_resize,int), "longest_side_resize must be an integer"

    args = TieredParser().get_args(alt_parse_args=["--model_name", model_name_for_dataloader,
                                                    "--model_version","1.0.0",
                                                    "--ckpt_name", "*"+model_name_for_dataloader+"*",
                                                    "--mode","data",
                                                    "--train_batch_size",str(batch_size)])
    if longest_side_resize>0 and not longest_side_resize==1024:
        args.image_size = longest_side_resize
        args.crop_method = "sam_small"
    trainer = DiffusionModelTrainer(args)
    trainer.create_datasets("vali",args={"datasets": "ade20k"})
    load_raw_image_label = getattr(trainer,split+"_dl").dataloader.dataset.load_raw_image_label
    ckpt_type_dict = {"vit_b": "sam_vit_b_01ec64.pth",#89 670 912 #params
                  "vit_l": "sam_vit_l_0b3195.pth",#308 278 272 #params
                  "vit_h": "sam_vit_h_4b8939.pth"}#637 026 048 #params
    sam_checkpoint = "../segment-anything/segment_anything/checkpoint/"+ckpt_type_dict[model_type]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_agnostic = SamAgnosticPredictor(sam,generator_kwargs=generator_kwargs)
    n = len(getattr(trainer,split+"_dl"))
    n_batches = np.ceil(n*ratio_of_dataset).astype(int)
    n_batches = min(n_batches,n)
    assert n_batches>0, "no data found"
    seg_return = []
    metrics_all = defaultdict(list)
    print(f"evaluating {n_batches} batches")
    for i in tqdm.tqdm(range(n_batches),disable=not progress_bar):
        batch = next(getattr(trainer,split+"_dl"))
        x, model_kwargs, info = trainer.get_kwargs(batch,gen=True)
        features = model_kwargs["image_features"]
        bs = len(batch[-1])
        images = []
        gts = []
        for i in range(bs):
            image,gt = load_raw_image_label(info[i],longest_side_resize=0 if full_resolution_decoder else longest_side_resize)
            images.append(image)
            gts.append(gt)
        segmentations = sam_agnostic.batched_generate(images,gts,features)
        if return_idx is None:
            seg_return.extend(segmentations[:min(num_return_segments-len(seg_return),bs)])
        else:
            idx = [i for i in range(bs) if info[i]["i"] in return_idx]
            seg_return.extend([segmentations[i] for i in idx])
        metrics = [get_segment_metrics_np(gt,seg) for gt,seg in zip(gts,segmentations)]
        for k,v in metrics[0].items():
            metrics_all[k].extend([m[k] for m in metrics])

    metrics_mean = {k: np.mean(v) for k,v in metrics_all.items()}
    metrics_all = dict(metrics_all)
    return metrics_all, metrics_mean, seg_return

def concat_dict_list(d_list,
                    num_recursions=1,
                    ignore_weird_values=False,
                    raise_error_on_recursion_overflow=False,
                    modify_input=False):
    """
    Concatenates a list of dictionaries. If the values of are iterable, then
    the values are concatenated along the first axis. Works with torch.Tensor,
    np.ndarray, and lists. If the values are dictionaries, then the function
    is called recursively. Raises an error if the values are not the same or if
    the structure of the dictionaries are not the same (as d_list[0])
    """
    concat_dict = {}
    d_keys = d_list[0].keys()
    for k in d_keys:
        v = d_list[0][k]
        if isinstance(v,dict):
            if num_recursions>0:
                concat_dict[k] = concat_dict_list([d[k] for d in d_list],num_recursions-1)
            elif raise_error_on_recursion_overflow:
                raise ValueError(f"Recursion overflow at key {k}")
        else:
            if torch.is_tensor(v):
                concat_dict[k] = torch.cat([d[k] for d in d_list],dim=0)
            elif isinstance(v,np.ndarray):
                concat_dict[k] = np.concatenate([d[k] for d in d_list],axis=0)
            elif isinstance(v,list):
                concat_dict[k] = sum([d[k] for d in d_list],[])
            else:
                if ignore_weird_values:
                    concat_dict[k] = [d[k] for d in d_list]
                else:
                    raise ValueError(f"Expected v to be None, torch.Tensor, np.ndarray, or list, found {type(v)}")
        if modify_input:
            #save memory
            for d in d_list:
                del d[k]
    return concat_dict

class RawSampleLoader():
    def __init__(self,
                 foldername_or_gen_id,
                 glob_str="raw_sample_batch*.pt",
                 mem_threshold=4e9):
        id_dict = TieredParser("sample_opts").load_and_format_id_dict()
        if foldername_or_gen_id in id_dict.keys():
            self.gen_id = foldername_or_gen_id
        else:
            foldername = foldername_or_gen_id
        self.foldername = Path(foldername)
        assert self.foldername.exists(), f"{self.foldername} does not exist"
        self.batch_files = sorted(list(self.foldername.glob(glob_str)))
        assert len(self.batch_files)>0, f"no files found with glob string {glob_str}"
        self.num_batches = len(self.batch_files)
        self.mem_per_batch = os.path.getsize(self.batch_files[0])
        self.mem_all = self.mem_per_batch*self.num_batches
        self.mem_threshold = mem_threshold
        self.allow_full_load = self.mem_all<self.mem_threshold

    def load_all_data(self):
        if not self.allow_full_load:
            raise ValueError(f"memory usage of {self.mem_all} exceeds threshold of {self.mem_threshold}")
        return concat_dict_list([torch.load(f) for f in self.batch_files])
    
    def load_batch(self,idx):
        assert -self.num_batches<=idx<self.num_batches, f"idx must be in range [-{self.num_batches},{self.num_batches-1}]"
        return torch.load(self.batch_files[idx])
    
    def save_all_data(self,filename="raw_samples_all.pt",remove_old=False):
        if not self.allow_full_load:
            raise ValueError(f"memory usage of {self.mem_all} exceeds threshold of {self.mem_threshold}")
        concat_data = self.load_all_data()
        if not filename.endswith(".pt"):
            filename += ".pt"
        save_path = self.foldername/filename
        torch.save(concat_data,save_path)
        if remove_old:
            for f in self.batch_files:
                f.unlink()
    
    def load_and_match_light_stats(self,filename=None):
        if filename is None:
            self.foldername.parent.glob("light_stats*.json")

def extract_from_samples(samples,
                         ab=None,
                         extract=["pred_int","pred_prob","gt","image","raw_image","raw_gt"],
                         sam_reshape=True,
                         inv_imagenet=True,
                         raw_longest_side_resize=0):
    if ab is None:
        ab = AnalogBits(num_bits=samples["pred"].shape[1],shuffle_zero=True)
    extracted = {}
    bs = samples["pred"].shape[0]
    valid_keys = ["pred_int","pred_prob","gt","image","raw_image","raw_gt"]
    for k in extract:
        assert k in valid_keys, f"expected k to be one of {valid_keys}, found {k}"
    if "pred_int" in extract:
        extracted["pred_int"] = ab.bit2int(samples["pred"])
    if "pred_prob" in extract:
        extracted["pred_prob"] = ab.bit2prob(samples["pred"])
    if "gt" in extract:
        extracted["gt"] = ab.bit2int(samples["x"])
    if "image" in extract:
        image_found = False
        if "model_kwargs" in samples.keys():
            if "image" in samples["model_kwargs"].keys():
                image_found = True
                extracted["image"] = samples["model_kwargs"]["image"]
        if "info" in samples.keys() and not image_found:
            if "image" in samples["info"][0].keys():
                image_found = True
                extracted["image"] = torch.stack([info["image"] for info in samples["info"]])
        if image_found:
            if inv_imagenet:
                extracted["image"] = imagenet_preprocess(extracted["image"],inv=True,dim=1)
        else:
            raise ValueError("No image found in samples")
    if ("raw_image" in extract) or ("raw_gt" in extract):
        extracted["raw_image"] = []
        extracted["raw_gt"] = []
        for i in range(bs):
            image,label = load_raw_image_label(samples["info"][i],longest_side_resize=raw_longest_side_resize)
            extracted["raw_image"].append(image)
            extracted["raw_gt"].append(label)
        if "raw_image" not in extract:
            del extracted["raw_image"]
        if "raw_gt" not in extract:
            del extracted["raw_gt"]
    for k in extracted.keys():
        if torch.is_tensor(extracted[k]):
            extracted[k] = extracted[k].permute(0,2,3,1).cpu().numpy()
        if not isinstance(extracted[k],list):
            extracted[k] = [extracted[k][i] for i in range(bs)]
    resize=samples["pred"].shape[-1]
    if sam_reshape:
        for i in range(bs):
            imshape = samples["info"][i]["imshape"]
            h,w = imshape[0],imshape[1]
            new_h,new_w = sam_resize_index(h,w,resize=resize)
            for k in extracted.keys():
                if extracted[k][i].shape[0]==resize and extracted[k][i].shape[1]==resize:
                    extracted[k][i] = extracted[k][i][:new_h,:new_w]
    return extracted

def sam_resize_index(h,w,resize=64):
    if h>w:
        new_h = resize
        new_w = np.round(w/h*resize).astype(int)
    else:
        new_w = resize
        new_h = np.round(h/w*resize).astype(int)
    return new_h,new_w

from plot_utils import mask_overlay_smooth

def plot_qual_seg(ims,preds,names=None,
                      transposed=False,
                      resize_width=128,
                      border=0,
                      show_image_alone=False,
                      alpha_mask=0.6,
                      text=None,
                      text_color=[1,0,0]):
    """
    Function for plotting columns of different to compare segmentations.
    Ground truth is also considered a prediction.
    """
    ims = copy.deepcopy(ims)
    preds = copy.deepcopy(preds)
    if isinstance(preds,dict):
        assert names is None, "expected names to be None if preds is a dict"
        names = list(preds.keys())
        preds = list(preds.values())
    if isinstance(text,dict):
        text = list(text.values())
    if transposed:
        ims = [np.rot90(im) for im in ims]
        preds = [[np.rot90(p) for p in pred] for pred in preds]
        big_image = plot_qual_seg(ims,preds,names=names,
                                transposed=False,
                                resize_width=resize_width,
                                border=border,
                                show_image_alone=show_image_alone,
                                alpha_mask=alpha_mask,
                                text=text)
        big_image = np.rot90(big_image,-1)
    else:
        n_methods = len(preds)
        n_samples = len(ims)
        if names is None:
            names = [f"Method {i}" for i in range(n_methods)]
        imsizes = []
        rw = resize_width
        for i in range(n_samples):
            h,w = ims[i].shape[:2]
            imsize = [np.round(rw*h/w).astype(int),rw]
            imsizes.append(imsize)
        
        n_columns = n_methods+int(show_image_alone)
        W = n_columns*(rw+border*2)
        list_of_h = [s[0]+2*border for s in imsizes]
        H = sum(list_of_h)
        big_image = np.zeros((H,W,3))
        for i in range(n_samples):
            im = cv2.resize(ims[i],tuple(imsizes[i][::-1]),interpolation=cv2.INTER_AREA)
            h_slice = slice(sum(list_of_h[:i])+border,sum(list_of_h[:i+1])-border)
            if show_image_alone:
                w_slice = slice(border,rw+border)
                big_image[h_slice,w_slice] = im
                j = 1
            else:
                j = 0
            for _ in range(n_methods):
                jj = j-int(show_image_alone)
                pred = cv2.resize(preds[jj][i],tuple(imsizes[i][::-1]),interpolation=cv2.INTER_NEAREST)
                pred = mask_overlay_smooth(im,pred,alpha_mask=alpha_mask)
                w_slice = slice(j*(rw+border*2)+border,(j+1)*(rw+border*2)-border)
                big_image[h_slice,w_slice] = pred
                j += 1
    return big_image
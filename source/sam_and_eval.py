import json
import numpy as np
import scipy.ndimage as nd
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os,sys
import argparse
from training import DiffusionModelTrainer
from argparse_utils import TieredParser
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from plot_utils import mask_overlay_smooth,index_dict_with_bool,render_text_gridlike
from collections import defaultdict
from utils import get_segment_metrics_np, shaprint, load_json_to_dict_list,sam_resize_index, postprocess_list_of_segs
import tqdm
from pathlib import Path
from datasets import AnalogBits,load_raw_image_label,load_raw_image_label_from_didx,longest_side_resize_func
import copy
from utils import imagenet_preprocess, get_mse_metrics
from plot_utils import mask_overlay_smooth, darker_color, get_matplotlib_color
import enum
from collections import OrderedDict
import pandas as pd

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

def evaluate_sam(datasets="ade20k",
                 model_type=0,
                 model_name_for_dataloader="sam[64_native_tanh]",
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
                 postprocess_kwargs=None):
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "datasets must be a string or a list of strings"
        datasets = datasets.split(",")
    ckpt_type_dict = {"vit_b": "sam_vit_b_01ec64.pth",#89 670 912 #params
                  "vit_l": "sam_vit_l_0b3195.pth",#308 278 272 #params
                  "vit_h": "sam_vit_h_4b8939.pth"}#637 026 048 #params
    if isinstance(model_type,int):
        assert model_type in [0,1,2], "model_type must be one of [0,1,2] or one of [vit_b, vit_l, vit_h]"
        model_type = ["vit_b","vit_l","vit_h"][model_type]
    model_idx = list(ckpt_type_dict.keys()).index(model_type)
    if isinstance(split,int):
        assert split in [0,1,2,3], "split must be one of [0,1,2,3] or one of [train,vali,test,all]"
        split = ["train","vali","test","all"][split]
    assert num_return_segments<=64
    assert isinstance(longest_side_resize,int), "longest_side_resize must be an integer"

    args = TieredParser().get_args(alt_parse_args=["--model_name", model_name_for_dataloader,
                                                    "--model_version","1.0.0",
                                                    "--ckpt_name", "*"+model_name_for_dataloader+"*",
                                                    "--mode","data",
                                                    "--train_batch_size",str(batch_size)])
    if longest_side_resize>0 and not longest_side_resize==1024:
        args.image_size = longest_side_resize
        args.crop_method = "sam_small"
    if pri_didx is not None:
        args.pri_didx = pri_didx
    args.image_encoder = ['sam_vit_b','sam_vit_l','sam_vit_h'][model_idx]
    args.datasets = datasets
    trainer = DiffusionModelTrainer(args)
    trainer.create_datasets("vali",args=args)
    load_raw_image_label = getattr(trainer,split+"_dl").dataloader.dataset.load_raw_image_label
    
    sam_checkpoint = "../segment-anything/segment_anything/checkpoint/"+ckpt_type_dict[model_type]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_agnostic = SamAgnosticPredictor(sam,generator_kwargs=generator_kwargs)
    n = len(getattr(trainer,split+"_dl"))
    if isinstance(ratio_of_dataset,float) and ratio_of_dataset<=1:
        n_batches = np.ceil(n*ratio_of_dataset).astype(int)
    else:
        n_batches = np.ceil(ratio_of_dataset/batch_size).astype(int)
    n_batches = min(n_batches,n)
    assert n_batches>0, "no data found"
    
    heavy_data = []
    light_data = []
    print(f"evaluating {n_batches} batches")
    for i in tqdm.tqdm(range(n_batches),disable=not progress_bar):
        batch = next(getattr(trainer,split+"_dl"))
        _, model_kwargs, info = trainer.get_kwargs(batch,gen=True)
        features = model_kwargs["image_features"]
        bs = len(batch[-1])
        images = []
        gts = []
        for i in range(bs):
            image,gt = load_raw_image_label(info[i],longest_side_resize=0 if full_resolution_decoder else longest_side_resize)
            images.append(image)
            gts.append(gt)
        segmentations = sam_agnostic.batched_generate(images,gts,features)
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
        metrics = [get_segment_metrics_np(gt,seg) for gt,seg in zip(gts,segmentations)]
        light_data_batch = []
        for i in range(bs):
            light_data_batch.append({"info": {k: v for k,v in info[i].items() if k in ["split_idx","i","dataset_name","num_classes"]},
                            "model_kwargs_abs_sum": {k: (v[i].abs().sum().item() if torch.is_tensor(v) else 0) for k,v in model_kwargs.items()},
                            "metrics": metrics[i]})

        light_data.extend(light_data_batch)
    metrics_mean = {k: np.mean([ld["metrics"][k] for ld in light_data if ld is not None]) for k in light_data[0]["metrics"].keys()}
    return metrics_mean, light_data, heavy_data

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
    assert all([isinstance(d,dict) for d in d_list]), "expected all of d_list to be dictionaries, found "+str([type(d) for d in d_list])
    concat_dict = {}
    d_keys = d_list[0].keys()
    for k in d_keys:
        v = d_list[0][k]
        if isinstance(v,dict):
            if num_recursions>0:
                concat_dict[k] = concat_dict_list([d[k] for d in d_list],num_recursions-1,
                                                  ignore_weird_values=ignore_weird_values,
                                                    raise_error_on_recursion_overflow=raise_error_on_recursion_overflow,
                                                    modify_input=modify_input)
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

def seperate_dict_to_list(d):
    bs = None
    d_keys = d.keys()
    for k in d_keys:
        if hasattr(d[k],"__len__"):
            bs = len(d[k])
    if bs is None:
        raise ValueError("no iterable found in d")
    d_list = []
    for i in range(bs):
        d_i = {k: d[k][i] if hasattr(d[k],"__len__") else None for k in d_keys}
        d_list.append(d_i)
    return d_list

def didx_from_info(list_of_info):
    assert len(list_of_info)>0, "list_of_info must have length > 0"
    assert isinstance(list_of_info[0],dict), "list_of_info must contain dictionaries"
    if "info" in list_of_info[0].keys():
        list_of_info = [info["info"] for info in list_of_info]
    data_idx = [f"{info['dataset_name']}/{info['i']}" for info in list_of_info]
    return data_idx

def is_saved_samples(x):
    if isinstance(x,SavedSamples):
        out = True
    elif type(x).__name__.endswith("SavedSamples"):
        out = True 
    else:
        out = False
    return out

class SavedSamplesManager:
    def __init__(self):
        self.saved_samples = []
    
    def __len__(self):
        return len(self.saved_samples)
        
    def add_saved_samples(self,saved_samples):
        if isinstance(saved_samples,list):
            assert all([is_saved_samples(ss) for ss in saved_samples]), "expected all elements in a list to be instances of SavedSamples, found "+str([type(ss) for ss in saved_samples])
            self.saved_samples.extend(saved_samples)
        assert is_saved_samples(saved_samples), f"expected samples to be an instance of SavedSamples or a list of SavedSamples, found {type(saved_samples)}"
        self.saved_samples.append(saved_samples)

    def union_didx(self,heavy_only=False,ss_idx=None):
        didx = []
        ss_idx = list(range(len(self.saved_samples))) if ss_idx is None else ss_idx
        for k,ss in enumerate(self.saved_samples):
            if k in ss_idx:
                didx = ss.union_didx(didx,heavy_only=heavy_only)
        return didx
    
    def intersection_didx(self,heavy_only=False,ss_idx=None):
        didx = []
        ss_idx = list(range(len(self.saved_samples))) if ss_idx is None else ss_idx
        for k,ss in enumerate(self.saved_samples):
            if k in ss_idx:
                if k==ss_idx[0]:
                    didx = ss.union_didx(didx,heavy_only=heavy_only)
                else:
                    didx = ss.intersection_didx(didx,heavy_only=heavy_only)
        return didx

    def get_image_gt(self,didx,ss_idx=None,try_saved_samples=True):
        if len(didx)==0:
            return [],[]
        else:
            didx0 = didx[0]
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))
        image_keys = ["image","raw_image"]
        gt_keys = ["gt","raw_gt"]
        if try_saved_samples:
            ss_has_image_gt = []
            num_pixels_in_image = []
            for k in ss_idx:
                ss = self.saved_samples[k]
                has_image = any([k in ss.heavy_keys() for k in image_keys])
                has_gt = any([k in ss.heavy_keys() for k in gt_keys])
                ss_has_image_gt.append(has_image and has_gt)
                if has_image:
                    for k2 in image_keys:
                        if k2 in ss.heavy_keys():
                            num_pixels = len(ss.heavy_data[ss.didx_to_idx[didx0]][k2].flatten())
                            num_pixels_in_image.append(num_pixels)
                            break
                else:
                    num_pixels_in_image.append(0)
            ss_has_image_gt = []
            new_load = not any(ss_has_image_gt)
        else:
            new_load = True
        ims = []
        gts = []
        if new_load:
            for d in didx:
                im, gt = load_raw_image_label({"dataset_name": d.split("/")[0], "i": int(d.split("/")[1])})
                ims.append(im)
                gts.append(gt)
        else:
            max_num_pixels_idx = np.argmax(num_pixels_in_image)
            heavy_data = self.saved_samples[ss_idx[max_num_pixels_idx]].get_heavy_data(didx)
            image_key = [k for k in image_keys if k in heavy_data[0].keys()][0]
            gt_key = [k for k in gt_keys if k in heavy_data[0].keys()][0]
            for d in didx:
                ims.append(heavy_data[d][image_key])
                gts.append(heavy_data[d][gt_key])
        return ims,gts
    
    def plot_qual_seg(self,
                    didx=None,
                    num_images=4,
                    resize_width=None,
                    alpha_mask=0.6,
                    random=False,
                    image_gt_from_saved_samples=True,
                    text_measures="ari",
                    text_color_inside="white",
                    ss_idx=None,
                    plot_qual_seg_kwargs={}):
        if num_images is None:
            assert isinstance(didx,list), "If num_images is None, idx must be a list"
            num_images = len(didx)
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))
        heavy_avail = self.intersection_didx(heavy_only=True,ss_idx=ss_idx)
        if len(heavy_avail)==0:
            raise ValueError("No overlapping didx found")
        elif len(heavy_avail)<num_images:
            raise ValueError(f"Only {len(heavy_avail)} overlapping didx found but num_imagesnum_images={num_images} requested")
        if isinstance(random,bool):
            if random:
                didx_plot = np.random.choice(heavy_avail,num_images,replace=False)
            else:
                didx_plot = heavy_avail[:num_images]
        else:
            #random is a seed
            assert isinstance(random,int)
            didx_plot = np.random.RandomState(random).choice(heavy_avail,num_images,replace=False)
        if isinstance(didx_plot,np.ndarray):
            didx_plot = didx_plot.tolist()
        preds = []
        metrics = []
        ims,gts = self.get_image_gt(didx_plot,ss_idx=ss_idx,try_saved_samples=image_gt_from_saved_samples)
        if not isinstance(text_measures,list):
            text_measures = [text_measures]
        metrics_keys = [["metrics",m] for m in text_measures]
        ss_names = []
        for k in ss_idx:
            ss = self.saved_samples[k]
            ss_names.append(ss.name)

            metrics_k = ss.get_light_data(didx_plot,keys=metrics_keys,return_type="list")
            metrics_k = concat_dict_list(metrics_k,ignore_weird_values=True)["metrics"]
            preds_k = ss.get_segmentations(didx_plot)
            preds.append(preds_k)
            metrics.append(metrics_k)
        if resize_width is not None:
            assert isinstance(resize_width,int), "expected resize_width to be an int"
            plot_qual_seg_kwargs["resize_width"] = resize_width
        if alpha_mask is not None:
            assert isinstance(alpha_mask,float), "expected alpha_mask to be a float"
            plot_qual_seg_kwargs["alpha_mask"] = alpha_mask
        big_image = plot_qual_seg(ims,preds,gts,**plot_qual_seg_kwargs)
        transposed = plot_qual_seg_kwargs.get("transposed",False)
        text_inside = [[] for _ in range(len(ss_idx))]
        for i in range(len(didx_plot)):
            for j in range(len(ss_idx)):
                dict_ij = {k: v[i] for k,v in metrics[j].items()}
                text_inside[j].append("\n".join([f"{k}={v:.2f}" for k,v in dict_ij.items()]))
        
        text_outside = ((["Raw\n Image"] if plot_qual_seg_kwargs.get("show_image_alone",True) else [])
                        +(["Ground\n Truth"] if gts is not None else [])
                        +(ss_names))
        
        num_empty_text = len(text_outside)-len(text_inside)
        if num_empty_text>0:
            text_inside = [["" for _ in range(len(didx_plot))] for _ in range(num_empty_text)]+text_inside
        if transposed:
            y_sizes=[1 for _ in range(len(text_outside))]
            x_sizes=[im.shape[1]/im.shape[0] for im in ims]
            text_pos_kwargs={"left": text_outside, "top": didx_plot, "xtick_kwargs": {"fontsize": 10}}
            text_kwargs = {"color": text_color_inside,"fontsize": 1,"verticalalignment":"top","horizontalalignment":"left"}
        else:
            y_sizes=[im.shape[0]/im.shape[1] for im in ims]
            x_sizes=[1 for _ in range(len(text_outside))]
            text_pos_kwargs={"top": text_outside, "left": didx_plot, "xtick_kwargs": {"fontsize": 10}}
            text_kwargs = {"color": text_color_inside,"fontsize": 1,"verticalalignment":"top","horizontalalignment":"left"}
        big_image_w_text = render_text_gridlike(big_image,
                                x_sizes=x_sizes,
                                y_sizes=y_sizes,
                                text_inside=text_inside,
                                transpose_text_inside=transposed,
                                anchor_image=(0.05,0.05),
                                text_kwargs=text_kwargs,
                                text_pos_kwargs=text_pos_kwargs,
                                border_width_inside=0.2)
        return big_image,big_image_w_text
    
    def hist(self,
            metric_names="ari",
            ss_idx=None,
            intersection_only=True,
            mean_lines=False,
            mean_lines_text=True,
            subplot_per_metric=False,
            subplot_per_ss=False,
            transposed=False,
            figsize=(12,6),
            lines_instead_of_hist=False):
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,ss_idx=ss_idx,intersection_only=intersection_only)
        ncol = 1 if subplot_per_metric else len(metric_names)
        nrow = 1 if subplot_per_ss else len(ss_idx)
        if transposed:
            ncol,nrow = nrow,ncol
        plt.figure(figsize=figsize)
        
        for i in range(len(ss_idx)):
            for j in range(len(metric_names)):
                
                if nrow==1 and ncol==1:
                    subplot_index = 1
                elif nrow==1 and ncol>1:
                    subplot_index = 1+j
                elif nrow>1 and ncol==1:
                    subplot_index = i+1
                else:
                    subplot_index = j*nrow+i+1 if transposed else i*ncol+j+1
                plt.subplot(nrow,ncol,subplot_index)

                m = metric_names[j]
                y = metrics[i][m]
                if lines_instead_of_hist:
                    std_kernel = np.std(y)/10
                    
        return 
    
    def metrics_for_plotting(self,metric_names=None,ss_idx=None,intersection_only=True,transpose_metric_ss=False):
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))
        if intersection_only:
            intersection_didx = self.intersection_didx(heavy_only=False,ss_idx=ss_idx)
            if len(intersection_didx)==0:
                raise ValueError("No overlapping didx found. Consider setting intersection_only=False")
        else:
            intersection_didx = None
        uq,uq_c = np.unique(sum([self.saved_samples[i].get_metric_names() for i in ss_idx],[]),return_counts=True)
        valid_metrics = uq[uq_c==len(ss_idx)].tolist()
        if len(valid_metrics)==0:
            raise ValueError("No metrics present in all ss_idx found")
        if metric_names is None:
            metric_names = valid_metrics
        else:
            if not isinstance(metric_names,list):
                assert isinstance(metric_names,str), "expected either None, a list of strings or a str as metric. found type: "+str(type(metric_names))
                metric_names = [metric_names]
        assert all([m in valid_metrics for m in metric_names]), "expected all metrics to be present in all ss_idx, valid_metrics "+str(valid_metrics)+", found "+str(metric_names)
            
        if not isinstance(metric_names,list):
            assert isinstance(metric_names,str), "expected either None, a list of strings or a str as metric. found type: "+str(type(metric_names))
            metric_names = [metric_names]
        metrics = []
        keyslist = [["metrics",m]for m in metric_names]
        for k in ss_idx:
            ss = self.saved_samples[k]
            metrics_k = ss.get_light_data(intersection_didx,return_type="list")
            metrics_k = [index_with_keylist(item,keyslist) for item in metrics_k]
            metrics_k = {m: np.array([item[j] for item in metrics_k]) for j,m in enumerate(metric_names)}
            metrics.append(metrics_k)
        return metrics,metric_names,ss_idx

    def split_ss_by_cond(self,condition,ss_idx=0,remove_original_ss=True):
        """
        Splits a single instance of SavedSamples into multiple instances 
        based on a condition. The condition should return the same value
        for all light_data to be split into the same group.
        """
        conditions = {"dataset_name": (lambda ld: ld["dataset_name"])}
        if condition in conditions:
            condition = conditions[condition]
        original_ss = self.saved_samples[ss_idx]
        id_per_ld = [condition(ld) for ld in original_ss.light_data]
        uq_id, uq_inv = np.unique(id_per_ld,return_inverse=True)
        for i in range(len(uq_id)):
            idx = np.flatnonzero(uq_inv==i)
            new_didx = [original_ss.didx[i] for i in idx]
            new_ld = [original_ss.light_data[i] for i in idx]
            new_hd = [original_ss.heavy_data[i] for i in idx]
            new_name = original_ss.name+f"_{uq_id[i]}"
            self.add_saved_samples(SavedSamples(light_data=new_ld,
                                                heavy_data=new_hd,
                                                didx=new_didx,
                                                name=new_name)
                                                )
        if remove_original_ss:
            self.saved_samples.pop(ss_idx)

    def bar(self,
            metric_names=None,
            ss_idx=None,
            subplots=False,
            seperated_by_metrics=True,
            text=False,
            error_bars=False,
            intersection_only=True,
            figsize=(12,6),
            subplot_horz=True):
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,ss_idx=ss_idx,intersection_only=intersection_only)
        #metrics structure:
        #metrics[ss_idx][metric_name] = list of values where mean is the bar height (instead of mean value, so we can get std)
        assert len(metric_names)>0, "expected at least one metric to be present in all ss_idx"
        assert len(ss_idx)>0, "expected at least oneseperated_by_metrics ss_idx to be present"
        fancy_bar_args = {"text": text, 
                          "error_bars": error_bars}
        ss_names = [f"{self.saved_samples[i].name}" for i in ss_idx]
        width = 0.8/len(metric_names) if seperated_by_metrics else 0.8/len(ss_idx)
        ymax = max([max([np.mean(metrics[i][m]).item() for m in metric_names]) for i in range(len(ss_idx))])
        plt.figure(figsize=figsize)
        if subplots:
            if seperated_by_metrics:
                n_subplots = len(metric_names)
                
                for i in range(len(metric_names)):
                    plt.subplot(1,n_subplots,i+1) if subplot_horz else plt.subplot(n_subplots,1,i+1)
                    m = metric_names[i]
                    fancy_bar(fancy_bar_args,ss_names,[metrics[j][m] for j in range(len(ss_idx))])
                    plt.ylabel(m)
            else:
                n_subplots = len(ss_idx)
                
                for i in range(len(ss_idx)):
                    plt.subplot(1,n_subplots,i+1) if subplot_horz else plt.subplot(n_subplots,1,i+1)
                    fancy_bar(fancy_bar_args,metric_names,[metrics[i][m] for m in metric_names])
                    plt.title(f"Metrics for {ss_names[i]}")
                    plt.ylim(0,ymax)
        else:
            if seperated_by_metrics:
                width = 0.8/len(ss_idx)
                for i in range(len(ss_idx)):
                    fancy_bar(fancy_bar_args,[j+width*i for j in range(len(metric_names))],[metrics[i][m] for m in metric_names],width,label=ss_names[i])
                plt.xticks([i+width*(len(ss_idx)-1)/2 for i in range(len(metric_names))], metric_names)

            else:
                width = 0.8/len(metric_names)
                for m in metric_names:
                    fancy_bar(fancy_bar_args,[i+width*metric_names.index(m) for i in range(len(ss_idx))],[met[m] for met in metrics],width,label=m)
                plt.xticks([i+width*(len(metric_names)-1)/2 for i in range(len(ss_idx))], ss_names)
            plt.legend() 
        plt.tight_layout()
        plt.show()
        return metrics
    
    def mean_metric_table(self,
                          metric_names=None,
                          ss_idx=None,
                          intersection_only=False,
                          to_df=True,
                          significant_digits=3,convert_to_pct=False):
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,ss_idx=ss_idx,intersection_only=intersection_only)
        ss_names = [self.saved_samples[i].name for i in ss_idx]
        mean_metrics = {ss_name: {m: np.mean(metrics[i][m]).item() for m in metric_names} for i,ss_name in enumerate(ss_names)}
        if to_df:
            mean_metrics = pd.DataFrame(mean_metrics)
            if convert_to_pct:
                mean_metrics = mean_metrics.applymap(lambda x: f"{x*100:.{significant_digits}f}%")
            else:
                mean_metrics = mean_metrics.applymap(lambda x: f"{x:.{significant_digits}f}")
        else:
            if convert_to_pct:
                mean_metrics = {k: {kk: f"{vv*100:.{significant_digits}f}%" for kk,vv in v.items()} for k,v in mean_metrics.items()}
            else:
                mean_metrics = {k: {kk: f"{vv:.{significant_digits}f}" for kk,vv in v.items()} for k,v in mean_metrics.items()}

        return mean_metrics

    def scatter(self,metric1="ari",metric2=None,
                ss_idx1=None,ss_idx2=None,
                plot_x_y=True,
                full_lims=False,
                buffer=0.1,
                didx_of_extremes=0):
        assert metric1 is not None
        if metric2 is None:
            metric2 = metric1
        if ss_idx1 is None:
            ss_idx1 = 0
        assert len(self.saved_samples)>=ss_idx1, "Only found "+str(len(self.saved_samples))+" saved samples, too high for ss_idx1="+str(ss_idx1)
        if ss_idx2 is None:
            if metric1!=metric2:
                ss_idx2 = ss_idx1
            else:
                if len(self.saved_samples)>1:
                    ss_idx2 = 1
                else:
                    ss_idx2 = ss_idx1
        assert len(self.saved_samples)>=ss_idx2, "Only found "+str(len(self.saved_samples))+" saved samples, too high for ss_idx2="+str(ss_idx2)
        intersection_didx = self.intersection_didx(heavy_only=False,ss_idx=[ss_idx1,ss_idx2])
        if len(intersection_didx)==0:
            raise ValueError("No overlapping didx found")
        vals1 = self.saved_samples[ss_idx1].get_light_data(intersection_didx,keys=[["metrics",metric1]],return_type="list")
        vals1 = np.array(concat_dict_list(vals1,ignore_weird_values=True)["metrics"][metric1])
        vals2 = self.saved_samples[ss_idx2].get_light_data(intersection_didx,keys=[["metrics",metric2]],return_type="list")
        vals2 = np.array(concat_dict_list(vals2,ignore_weird_values=True)["metrics"][metric2])
        
        plt.scatter(vals1,vals2)
        plt.xlabel(f"{metric1} from {self.saved_samples[ss_idx1].name}")
        plt.ylabel(f"{metric2} from {self.saved_samples[ss_idx2].name}")
        out = None

        if didx_of_extremes>0:
            #assumes we are looking at a roughly oval scatter plot, uses mahalanobis distance
            dist = mahalanobis_distance(np.array([vals1,vals2]).T)
            if isinstance(didx_of_extremes,int):
                num_extremes = didx_of_extremes
            else:
                assert isinstance(didx_of_extremes,float), "expected didx_of_extremes to be an int or a float"
                num_extremes = int(didx_of_extremes*len(dist))
            didx_extremes = np.argsort(dist)[-num_extremes:]
            for i in didx_extremes:
                plt.text(vals1[i],vals2[i],intersection_didx[i])
            out = [intersection_didx[i] for i in didx_extremes]
        if plot_x_y:
                plt.plot([0,1],[0,1],color="black")
        if full_lims:
            plt.xlim(0,1)
            plt.ylim(0,1)
        else:
            m1 = max(min(vals1+vals2)-buffer,0)
            m2 = min(max(vals1+vals2)+buffer,1)
            plt.xlim(m1,m2)
            plt.ylim(m1,m2)
        plt.show()
        return out
    
def fancy_bar(fancy_bar_args,x,y,*args,**kwargs):

    assert len(x)==len(y), "expected x and y to have the same length, found len(x)="+str(len(x))+" and len(y)="+str(len(y))

    fancy_bar_args2 = {"text": False, 
                      "error_bars": False,
                      "error_bar_n": None,
                      "error_bar_std": None,
                      "formatter": lambda x: f"{x:.2f}"}
    for k in fancy_bar_args.keys():
        assert k in fancy_bar_args2.keys(), f"expected k to be one of {fancy_bar_args2.keys()}, found {k}"
        fancy_bar_args2[k] = fancy_bar_args[k]

    fancy_bar_args = fancy_bar_args2
    y = [np.array(y_i).flatten() for y_i in y]
    if all([len(y_i)>1 for y_i in y]):
        n = [len(y_i) for y_i in y]
        std = [y_i.std() for y_i in y]
        y = [y_i.mean() for y_i in y]
    else:
        if fancy_bar_args["error_bars"]:
            assert fancy_bar_args["error_bar_n"] is not None, "error_bar_n has to be explicitly set if error_bars is True and the given y (height) is already reduced to its means"
            assert fancy_bar_args["error_bar_std"] is not None, "error_bar_std has to be explicitly set if error_bars is True and the given y (height) is already reduced to its means"
            n = fancy_bar_args["error_bar_n"]
            std = fancy_bar_args["error_bar_std"]
    plt.bar(x,y,*args,**kwargs)

    if fancy_bar_args["error_bars"]:
        yerr = 1.96*np.array(std)/np.sqrt(n)
        caps = plt.errorbar(x,y,yerr=yerr,capsize=10,fmt="none",ecolor="black",barsabove=True)[1]
        for c in caps:
            c.set_markeredgewidth(1)
    if fancy_bar_args["text"]:
        ha = "center"
        fmt = fancy_bar_args["formatter"]
        for i in range(len(x)):
            t = fmt(y[i])
            if fancy_bar_args["error_bars"]:
                t += "±"+fmt(yerr[i])[int(yerr[i]<1.0):] 
            plt.text(x[i],y[i],t,ha=ha,va="bottom")

def is_nontrivial_list(x):
    if not isinstance(x,list):
        return False
    else:
        return len(x)>0
def get_nested_dict(d,keys):
    item = d
    for k in keys:
        assert k in item.keys(), "expected nested key structure, did not find the last key: d['"+("']['".join(keys))+"']"
        item = item[k]
    return item

def set_nested_dict(d,keys,value,create_intermediate=True):
    item = d
    for k in keys[:-1]:

        if not k in item.keys():
            if create_intermediate:     
                item[k] = {}
            else:
                raise ValueError("expected nested key structure, did not find the last key: d['"+("']['".join(keys))+"']")
        item = item[k]
    item[keys[-1]] = value

def extract_from_dict(d,keys,collapse_to_list=False):
    """
    Extracts keys from a dictionary. If a key is a list, then the 
    dictionary is indexed as a nested dictionary.
    """
    if collapse_to_list:
        out = []
    else:
        out = {}
    for k in keys:
        if isinstance(k,str):
            k = [k]
        assert isinstance(k,list), "expected k to be a string or a list of strings"
        item = get_nested_dict(d,k)
        if collapse_to_list:
            out.append(item)
        else:
            set_nested_dict(out,k,item) 
    return out

def extract_from_dict_list(d_list,keys):
    """
    Extracts keys from a list of dictionaries. If a key is a list, then the 
    dictionary is indexed as a nested dictionary.
    """
    out = []
    for d in d_list:
        assert isinstance(d,dict), "expected d_list to contain dictionaries. found "+str(type(d))
        out.append(extract_from_dict(d,keys))
    return out

def lists_of_dicts_as_type(didx,lists_of_dicts,return_type,add_didx_to_list=True,keys=None):
    assert return_type.lower() in ["ordereddict","list", "dict"], f"expected return_type to be one of ['ordereddict','list', 'dict', 'array'], found {return_type}"
    return_type = copy.copy(return_type).lower()
    if return_type == "ordereddict":
        out = OrderedDict()
        for i in range(len(didx)):
            out[didx[i]] = lists_of_dicts[i]
    elif return_type == "list":
        if add_didx_to_list:
            out = [{"didx": didx[i], **lists_of_dicts[i]} for i in range(len(didx))]
        else:
            out = lists_of_dicts
    elif return_type == "dict":
        out = {didx[i]: lists_of_dicts[i] for i in range(len(didx))}
    elif return_type == "array":
        if keys is None:
            keys = get_keys_recursive(lists_of_dicts[0])
        out = np.zeros((len(didx),len(keys)))
        for i in range(len(didx)):
            out[i] = index_with_keylist(lists_of_dicts[i],keys)
    return out

def index_with_keylist(d,keylist):
    if not isinstance(keylist,list):
        keylist = [keylist]
    out = []
    for k in keylist:
        item = d
        for i in range(len(k)):
            item = item[k[i]]
        out.append(item)
    return out

def get_keys_recursive(d,num_recursions=5,recursion_keys=[]):
    keys = []
    for k,v in d.items():
        if isinstance(v,dict):
            if num_recursions>0:
                keys += get_keys_recursive(v,num_recursions-1,recursion_keys+[k])
            else:
                keys.append(recursion_keys+[k])
        else:
            keys.append(recursion_keys+[k])
    return keys

class SavedSamples:
    def __init__(self,
                light_data = None,
                heavy_data = None,
                didx = None,
                name = None,
                mem_threshold=4e9,
                segment_key="pred_int"):
        assert all([(x is None) or is_nontrivial_list(x) for x in [light_data,heavy_data,didx]]), "expected all of [light_data,heavy_data,didx] to be None or a non-empty list"
        self.reset()
        if any([x is not None for x in [light_data,heavy_data,didx]]):
            self.add_samples(didx=didx,light_data=light_data,heavy_data=heavy_data)
            self.mem_all = sys.getsizeof(self)
            self.mem_threshold = mem_threshold
        self.segment_key = segment_key
        if name is not None:
            self.name = name

    def downscale_heavy_data(self,longest_side_resize=128,keys=["gt","pred_int","image"]):
        if not isinstance(keys,list):
            keys = [keys]
        for i in range(len(self.heavy_data)):
            hd = self.heavy_data[i]
            if hd is not None:
                for k in keys:
                    is_label = k!="image"
                    if k in hd.keys():
                        self.heavy_data[i][k] = longest_side_resize_func(self.heavy_data[i][k],is_label,longest_side_resize)

    def get_segmentations(self,didx=None,return_type="list",only_segments_if_list=True):
        if didx is None:
            didx = [d for d in self.didx if self.heavy_available[d]=="pos_loaded"]
        else:
            didx = [d for d in didx if self.heavy_available[d]=="pos_loaded"]
        out = self.get_heavy_data(didx,return_type=return_type,keys=[self.segment_key])
        if only_segments_if_list and return_type=="list":
            out = [o[self.segment_key] for o in out]
        return out

    def save(self,save_path):
        save_dict = {"name": self.name,
                    "heavy_data": self.heavy_data,
                    "light_data": self.light_data,
                    "didx": self.didx,
                    "postprocess_kwargs": self.postprocess_kwargs,
                    "segment_key": self.segment_key,
                    "mem_threshold": self.mem_threshold}
        torch.save(save_dict,save_path)

    def load(self,save_path):
        load_dict = torch.load(save_path)
        self.reset()
        self.name = load_dict["name"]
        self.segment_key = load_dict["segment_key"]
        self.mem_threshold = load_dict["mem_threshold"]
        self.postprocess_kwargs = load_dict["postprocess_kwargs"]
        self.add_samples(didx=load_dict["didx"],light_data=load_dict["light_data"],heavy_data=load_dict["heavy_data"])
        

    def reset(self):
        self.name = "unnamed"
        self.heavy_data = []
        self.light_data = []
        self.didx = []
        self.postprocess_kwargs = None
        self.didx_to_idx = defaultdict(lambda: -1)
        self.heavy_available = defaultdict(lambda: "unknown")
        #self.has_heavy_data = []
        self.mem_all = sys.getsizeof(self)

    def union_didx(self,other,heavy_only=False):
        didx_other, _ = self.normalize_indexer(other,err_if_not_found=False)
        if heavy_only:
            didx_other = [d for d in didx_other if self.heavy_available[d] in ["pos_loaded","pos_not_loaded"]]
        didx_other = [d for d in didx_other if d not in self.didx]
        return self.didx+didx_other

    def intersection_didx(self,other,heavy_only=False):
        didx_other, _ = self.normalize_indexer(other,err_if_not_found=False)
        if heavy_only:
            didx_other = [d for d in didx_other if self.heavy_available[d] in ["pos_loaded","pos_not_loaded"]]
        didx_other = [d for d in didx_other if d in self.didx]
        return didx_other

    def join_saved_samples(self,other,err_on_duplicates=True,duplicate_join_mode="both"):
        assert duplicate_join_mode in ["both","new","old"], f"expected duplicate_join_mode to be one of ['both','new','old'], found {duplicate_join_mode}"
        assert isinstance(other,SavedSamples), "expected other to be an instance of SavedSamples"
        if err_on_duplicates:
            assert len(self.intersection_didx(other))==0, "expected no overlap between didx"
        if duplicate_join_mode=="both":
            new_didx = self.didx+other.didx
            new_light_data = self.light_data+other.light_data
            new_heavy_data = self.heavy_data+other.heavy_data
            self.add_samples(didx=new_didx,light_data=new_light_data,heavy_data=new_heavy_data,concat=False)
        elif duplicate_join_mode=="new":
            new_didx = other.didx
        elif duplicate_join_mode=="old":
            new_didx = self.didx
        
    def mean_metrics(self,condition=lambda x: True):
        if not "metrics" in self.light_keys():
            raise ValueError("no metrics found in light_data")
        first = True
        for ld in self.light_data:
            if first:
                metric_keys = list(ld["metrics"].keys())
                mean_metrics = {k: [] for k in metric_keys}
                first = False
            if condition(ld):
                for k in metric_keys:
                    mean_metrics[k].append(ld["metrics"][k])
        return {k: np.mean(v) for k,v in mean_metrics.items()}

    def normalize_indexer(self,indexer=None,err_if_not_found=True):
        if indexer is None:
            indexer = list(range(len(self.didx)))
        if not isinstance(indexer,list):
            indexer = [indexer]
        indexer = copy.deepcopy(indexer)
        idx_ints = []
        didx = []
        if isinstance(indexer,SavedSamples):
            didx = indexer.didx
            idx_ints = [self.didx.find(d) for d in didx]
        else:
            for i in range(len(indexer)):
                idx = indexer[i]
                if isinstance(idx,int):
                    assert 0<=idx<len(self.didx), f"expected idx to be in range [0,{len(self.didx)-1}], found {idx}"
                    idx_ints.append(idx)
                    didx.append(self.didx[idx])
                elif isinstance(idx,str):
                    #assert idx in self.didx, f"expected idx to be in didx, found {idx}"
                    didx.append(idx)
                    idx_ints.append(self.didx_to_idx[didx[-1]])
                elif isinstance(idx,dict):
                    didx.append(didx_from_info([idx])[0])
                    idx_ints.append(self.didx_to_idx[didx[-1]])
                else:
                    raise ValueError(f"expected idx to be int or str or dict, found {type(idx)}")
        if err_if_not_found:
            found = [i>=0 for i in idx_ints]
            assert all(found), f"did not find didx={didx[idx_ints.index(-1)]}"
        return didx,idx_ints
    
    def get_light_data(self,indexer,return_type="ordereddict",keys=None):
        didx,idx = self.normalize_indexer(indexer)
        light_data = [self.light_data[i] for i in idx]
        if keys is not None:
            light_data = extract_from_dict_list(light_data,keys)
        return lists_of_dicts_as_type(didx,light_data,return_type,add_didx_to_list=keys is None)
    
    def get_heavy_data(self,indexer,include_light_data=False,return_type="ordereddict",keys=None):
        didx,idx = self.normalize_indexer(indexer)
        if include_light_data:
            heavy_data = [{**self.heavy_data[i],**self.light_data[i]} for i in range(idx)]
        else:
            heavy_data = [self.heavy_data[i] for i in idx]
        if keys is not None:
            heavy_data = extract_from_dict_list(heavy_data,keys)
        return lists_of_dicts_as_type(didx,heavy_data,return_type,add_didx_to_list=keys is None)
            
    def add_samples(self,didx=None,light_data=None,heavy_data=None):
        assert any([is_nontrivial_list(x) for x in [didx,light_data,heavy_data]]), "expected at least one of [didx,light_data,heavy_data] to be a non-empty list"
        lengths = [len(x) for x in [didx,light_data,heavy_data] if is_nontrivial_list(x)]
        assert len(set(lengths))==1, f"expected all of [didx,light_data,heavy_data] to have the same length, found {lengths}"
        if didx is None:
            didx = didx_from_info(light_data if light_data is not None else heavy_data)
        if light_data is None:
            light_data = [None for _ in range(len(didx))]
        assert len(light_data)==len(didx), f"expected light_data to have length {len(didx)}, found {len(light_data)}"
        if heavy_data is None:
            heavy_data = [None for _ in range(len(didx))]
        assert len(heavy_data)==len(didx), f"expected heavy_data to have length {len(didx)}, found {len(heavy_data)}"
        for d,l,h in zip(didx,light_data,heavy_data):
            find_idx = self.didx_to_idx[d]
            if find_idx>=0:
                if l is not None:
                    self.light_data[find_idx] = l
                if h is not None:
                    self.heavy_data[find_idx] = h
                    self.heavy_available[d] = "pos_loaded"
            else:
                self.didx.append(d)
                self.light_data.append(l)
                self.heavy_data.append(h)
                self.didx_to_idx[d] = len(self.didx)-1
                if h is not None:
                    self.heavy_available[d] = "pos_loaded"
                else:
                    self.heavy_available[d] = "unknown"

    def heavy_keys(self,sub_keys=False,search_all=False):
        if search_all:
            out = set()
            for i in range(len(self.heavy_data)):
                if self.heavy_data[i] is not None:
                    out = out.union(set(self.heavy_data[i].keys()))
                    if sub_keys:
                        for k in self.heavy_data[i].keys():
                            if isinstance(self.heavy_data[i][k],dict):
                                out = out.union(set([[k,k2] for k2 in self.heavy_data[i][k].keys()]))
        else:
            is_not_none_idx = [i for i in range(len(self.heavy_data)) if self.heavy_data[i] is not None]
            if len(is_not_none_idx)==0:
                out = []
            else:
                i = is_not_none_idx[0]
                out = list(self.heavy_data[i].keys())
                if sub_keys:
                    for k in self.heavy_data[i].keys():
                        if isinstance(self.heavy_data[i][k],dict):
                            out = out.union(set([[k,k2] for k2 in self.heavy_data[i][k].keys()]))
        return out

    def get_metric_names(self):
        metric_names = []
        for k in self.light_keys(sub_keys=True):
            if isinstance(k,list):
                if len(k)==2:
                    if k[0]=="metrics":
                        metric_names.append(k[1])
        return metric_names

    def light_keys(self,sub_keys=False,search_all=False):
        if search_all:
            out = []
            for i in range(len(self.light_data)):
                if self.light_data[i] is not None:
                    out.extend(list(self.light_data[i].keys()))
                    if sub_keys:
                        for k in self.light_data[i].keys():
                            if isinstance(self.light_data[i][k],dict):
                                out.extend([[k,k2] for k2 in self.light_data[i][k].keys()])
        else:
            is_not_none_idx = [i for i in range(len(self.light_data)) if self.light_data[i] is not None]
            if len(is_not_none_idx)==0:
                out = []
            else:
                i = is_not_none_idx[0]
                out = list(self.light_data[i].keys())
                if sub_keys:
                    for k in self.light_data[i].keys():
                        if isinstance(self.light_data[i][k],dict):
                            out.extend([[k,k2] for k2 in self.light_data[i][k].keys()])
        out = reduce_to_unique(out)
        return out
    
    def __len__(self):
        return len(self.didx)

    def load_all_data(self):
        self.load_light_data()
        self.load_heavy_data()
    
    def load_light_data(self):
        has_read_light_data = hasattr(self,"read_light_data")
        assert has_read_light_data, "expected read_light_data to be implemented in a subclass"
        output_of_read_light_data = self.read_light_data()
        assert len(output_of_read_light_data)==3, f"expected read_light_data to return a tuple of length 3 representing [didx,light_data,heavy_data], found {len(output_of_read_light_data)}"
        self.add_samples(*output_of_read_light_data)
    
    def load_heavy_data(self):
        has_read_heavy_data = hasattr(self,"read_heavy_data")
        assert has_read_heavy_data, "expected read_heavy_data to be implemented in a subclass"
        output_of_read_heavy_data = self.read_heavy_data()
        if output_of_read_heavy_data is None:
            return
        assert len(output_of_read_heavy_data)==3, f"expected read_heavy_data to return a tuple of length 3 representing [didx,light_data,heavy_data], found {len(output_of_read_heavy_data)}"
        self.add_samples(*output_of_read_heavy_data)

    def clone(self,new_name=None):
        copy_of_self = copy.deepcopy(self)
        if new_name is not None:
            copy_of_self.name = new_name
        return copy_of_self

    def postprocess(self,postprocess_kwargs={},recompute_metrics=True):
        if self.postprocess_kwargs is not None:
            print("WARNING: The samples were already postprocessed, reprocessing with new postprocess_kwargs")
        heavy_data = self.get_heavy_data(self.didx,return_type="list")
        didx = [hd["didx"] for hd in heavy_data]
        if "gt" in self.heavy_keys():
            gts = [hd["gt"] for hd in heavy_data]
        elif "raw_gt" in self.heavy_keys():
            gts = [hd["raw_gt"] for hd in heavy_data]
        else:
            _,gts = load_raw_image_label_from_didx(didx)
            #add the gts to heavy_data
            for i in range(len(heavy_data)):
                idx = self.didx_to_idx[didx[i]]
                heavy_data[idx]["gt"] = gts[i]
        segments = [hd[self.segment_key] for hd in heavy_data]
        if postprocess_kwargs is None:
            segments_pp = segments
        else:
            segments_pp = postprocess_list_of_segs(segments,seg_kwargs=postprocess_kwargs)
        if recompute_metrics:
            metrics = [get_segment_metrics_np(seg,gt) for seg,gt in zip(segments_pp,gts)]
        for i in range(len(segments)):
            idx = self.didx_to_idx[didx[i]]
            self.light_data[idx]["metrics"] = metrics[i]
            self.heavy_data[idx][self.segment_key] = segments_pp[i]
        self.postprocess_kwargs = postprocess_kwargs

    def read_heavy_data(self):
        raise NotImplementedError("expected read_heavy_data to be implemented in a subclass")
    
    def read_light_data(self):
        raise NotImplementedError("expected read_light_data to be implemented in a subclass")

def reduce_to_unique(x):
    out = []
    for i in range(len(x)):
        if x[i] not in out:
            out.append(x[i])
    return out

class DiffSamples(SavedSamples):
    def __init__(self,
                gen_id,
                mem_threshold=4e9,
                load_heavy=False,
                load_light=True,
                glob_str="raw_sample_batch*.pt"
                ):
        super().__init__(mem_threshold=mem_threshold)
        id_dict = TieredParser("sample_opts").load_and_format_id_dict()
        assert gen_id in id_dict.keys(), f"gen_id {gen_id} not found in id_dict"
        self.gen_id = gen_id
        self.name = gen_id
        self.sample_opts = id_dict[gen_id]
        if len(self.sample_opts["raw_samples_folder"])>0:
            self.raw_samples_files = sorted(list(Path(self.sample_opts["raw_samples_folder"]).glob(glob_str)))
            self.mem_per_batch = os.path.getsize(self.raw_samples_files[0])
            self.mem_all = self.mem_per_batch*len(self.raw_samples_files)
        else:
            self.raw_samples_files = []
        if load_light:
            self.load_light_data()
        if load_heavy:
            self.load_heavy_data()

    def read_heavy_data(self,read_didx=None,extract=True):
        if not (len(self.sample_opts["raw_samples_folder"])>0 and self.sample_opts["save_raw_samples"]):
            print("Warning: no raw_samples_folder found or save_raw_samples is False")
            return None
        heavy_data = []
        didx = []
        for i in range(len(self.raw_samples_files)):
            batch = torch.load(self.raw_samples_files[i])
            batch_didx = didx_from_info(batch["info"])
            bs = len(batch["info"])
            for b in range(bs):
                
                if read_didx is None:
                    append_b = True
                else:
                    append_b = batch_didx[b] in read_didx
                if append_b:
                    heavy_data.append(index_dict_with_bool(copy.deepcopy(batch),bool_iterable=np.arange(bs)==b))
                    didx.append(batch_didx[b])
                    self.heavy_available[batch_didx[b]] = "pos_loaded"
                else:
                    self.heavy_available[batch_didx[b]] = "pos_not_loaded"
        if read_didx is None:
            #set all existences which are unknown to negative
            for d in self.didx:
                if self.heavy_available[d]=="unknown":
                    self.heavy_available[d] = "negative"
        if extract:
            heavy_data = extract_from_sample_list(heavy_data)
        return didx,None,heavy_data
    
    def load_heavy_image_gt(self,didx_load=None):
        if didx_load is None:
            didx_load = [self.didx[i] for i in range(len(self.didx)) if self.heavy_data[i] is not None]
        for didx_i in didx_load:
            assert didx_i in self.didx, f"expected didx to be in self.didx, found {didx_i}"
            x  = {"dataset_name": didx_i.split("/")[0], "i": int(didx_i.split("/")[1])}
            image,gt = load_raw_image_label(x)
            i = self.didx_to_idx(didx_i)
            if not isinstance(self.heavy_data[i],dict):
                self.heavy_data[i] = {}
            self.heavy_data[i]["raw_image"] = image
            self.heavy_data[i]["raw_gt"] = gt

    def read_light_data(self,read_didx=None):
        assert len(self.sample_opts["light_stats_filename"])>0, "no light_stats_filename found"
        light_data = load_json_to_dict_list(self.sample_opts["light_stats_filename"])
        didx = didx_from_info(light_data)
        if read_didx is not None:
            light_data = [light_data[i] for i in range(len(didx)) if didx[i] in read_didx]
            didx = didx_from_info(light_data)
        for d,ld in zip(didx,light_data):
            if "has_raw_sample" in ld.keys():
                if self.heavy_available[d]=="unkown":
                    self.heavy_available[d] = "pos_not_loaded" if ld["has_raw_sample"] else "negative"
        return didx,light_data,None
    
    def concat_and_save_heavy_data(self,filename="raw_samples_all.pt",remove_old=False):
        heavy_data = self.load_heavy_data()
        if self.mem_all>self.mem_threshold:
            raise ValueError(f"memory usage of {self.mem_all} exceeds threshold of {self.mem_threshold}")
        if not filename.endswith(".pt"):
            filename += ".pt"
        save_path = self.foldername/filename
        torch.save(heavy_data,save_path)
        if remove_old:
            for f in self.batch_files:
                f.unlink()
    
    

def mahalanobis_distance(data):
    #calculate mahalanobis distance for a given dataset, each row is a sample
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    inv_covmat = np.linalg.inv(cov) 
    left = np.dot(data-mean, inv_covmat)
    mahal = np.dot(left, (data-mean).T)
    return mahal.diagonal() 


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

def extract_from_sample_list(sample_list,**kwargs):
    assert isinstance(sample_list,list), "expected sample_list to be a list"
    return seperate_dict_to_list(extract_from_samples(concat_dict_list(sample_list),**kwargs))

def plot_qual_seg(ims,preds,gts=None,names=None,
                      transposed=False,
                      resize_width=128,
                      border=0,
                      show_image_alone=True,
                      alpha_mask=0.6):
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
    if names is None:
        names = [f"Method {i}" for i in range(len(preds))]
    if gts is not None:
        names = ["GT"]+names
        preds = [gts]+preds
    if transposed:
        ims = [np.rot90(im,1) for im in ims[::-1]]
        preds = [[np.rot90(p,1) for p in pred[::-1]] for pred in preds]
        big_image = plot_qual_seg(ims,preds,names=names,
                                transposed=False,
                                resize_width=resize_width,
                                border=border,
                                show_image_alone=show_image_alone,
                                alpha_mask=alpha_mask)
        big_image = np.rot90(big_image,3)
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
                pred = preds[jj][i]
                pred = cv2.resize(pred,tuple(imsizes[i][::-1]),interpolation=cv2.INTER_NEAREST)
                pred = mask_overlay_smooth(im,pred,alpha_mask=alpha_mask)
                w_slice = slice(j*(rw+border*2)+border,(j+1)*(rw+border*2)-border)
                big_image[h_slice,w_slice] = pred
                j += 1
    return big_image

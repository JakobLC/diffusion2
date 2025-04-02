import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os,sys
import itertools

#add source if we are running this file directly
if __name__=="__main__":
    sys.path.append("/home/jloch/Desktop/diff/diffusion2")

from source.utils.argparsing import TieredParser, get_closest_matches,load_defaults
from collections import defaultdict
import tqdm
from pathlib import Path
from source.sam import (sam12_info, all_sam_setups,evaluate_sam)
from source.utils.dataloading import (load_raw_image_label,
                      load_raw_image_label_from_didx,
                      longest_side_resize_func)
import copy
from source.utils.metric_and_loss import get_segment_metrics, get_ambiguous_metrics
from source.utils.mixed import (imagenet_preprocess, ambiguous_info_from_fn,
                         load_json_to_dict_list,sam_resize_index, 
                         postprocess_list_of_segs,wildcard_match,postprocess_batch)
from source.utils.plot import (mask_overlay_smooth, darker_color, 
                              get_matplotlib_color,index_dict_with_bool,
                              render_text_gridlike)
from source.utils.analog_bits import ab_bit2prob
import enum
from collections import OrderedDict
import pandas as pd
import warnings
from jlc import shaprint
import argparse

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
    return any([isinstance(x,SavedSamples),
                type(x).__name__.endswith("SavedSamples"),
                type(x).__name__.endswith("DiffSamples")])

class SavedSamplesManager:
    def __init__(self,saved_samples=None):
        self.saved_samples = []
        if saved_samples is None:
            pass  
        elif isinstance(saved_samples,list):
            self.add_saved_samples(saved_samples)
        else:
            assert is_saved_samples(saved_samples), f"expected samples to be an instance of SavedSamples or a list of SavedSamples, found {type(saved_samples)}"
            self.add_saved_samples([saved_samples])
    
    def __len__(self):
        return len(self.saved_samples)
    
    def rename_by_opts_key(self,opts_keys=None,ss_idx=None,include_renamed_keys=False):
        """
        Renames all saved samples (or subset by ss_idx) by a key in the opts dictionary.
        If opts_key=None, then the key(s) are determined by the opts_keys attribute of the saved samples which differs.
        """
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))

        if (opts_keys is None) or is_empty_list(opts_keys):
            ignore_keys = load_defaults(filename="jsons/sample_opts_default.json",
                                                return_special_argkey="dynamic").keys()
            values_per_key = {}
            for i in ss_idx:
                if hasattr(self.saved_samples[i],"sample_opts"):
                    for k,v in self.saved_samples[i].sample_opts.items():
                        if k not in ignore_keys:
                            if k not in values_per_key.keys():
                                values_per_key[k] = []
                            values_per_key[k].append(v)
            opts_keys = [k for k,v in values_per_key.items() if len(set(v))>1]
        elif isinstance(opts_keys,str):
            opts_keys = [opts_keys]
        else:
            assert isinstance(opts_keys,list), "expected opts_keys to be a list, string or None, found "+str(type(opts_keys))
        if include_renamed_keys:
            new_name_func = lambda x: "_".join([f"{k}={str(x[k])}" for k in opts_keys])
        else:
            new_name_func = lambda x: "_".join([str(x[k]) for k in opts_keys])
        for i in ss_idx:
            if hasattr(self.saved_samples[i],"sample_opts"):
                self.saved_samples[i].name = new_name_func(self.saved_samples[i].sample_opts)

    def save(self,save_path):
        list_of_save_dicts = [ss.save(save_path=None,return_instead_of_save=True) for ss in self.saved_samples]
        torch.save(list_of_save_dicts,save_path)
    
    def load(self,load_path, verbose=False):
        list_of_save_dicts = torch.load(load_path)
        ss_list = []
        for save_dict in list_of_save_dicts:
            ss = SavedSamples()
            ss.load(save_dict)
            ss_list.append(ss)
        if verbose:
            print(f"Loaded {len(ss_list)} saved samples")
            #print number of heavy keys
            maxnamelen = max([len(ss.name) for ss in ss_list])
            for ss in ss_list:
                print(f"{ss.name:{maxnamelen}}: {len(ss.heavy_keys()):4} heavy, {len(ss.light_data):4} light")
        self.add_saved_samples(ss_list)

    def load_by_ids(self,model_id_match_str="*",gen_id_match_str="*",load_heavy=False,load_light=True):
        """
        Finds the (model_ids,gen_ids) pairs that match the given strings and loads them
        """
        id_pairs = get_all_id_pairs(only_with_light_stats=True)
        assert len(id_pairs)>0, "No saved samples found with light stats"
        gen_id_loads = []
        for model_id,gen_id in id_pairs:
            if (wildcard_match(pattern=model_id_match_str,text=model_id) and 
                wildcard_match(pattern=gen_id_match_str,  text=gen_id)):
                gen_id_loads.append(gen_id)
        if len(gen_id_loads)==0:
            print(f"model_id_match_str: {model_id_match_str}")
            print(f"gen_id_match_str: {gen_id_match_str}")
            warning_msg = "No saved samples found with the given match strings."
            warning_msg += f" Matched model_ids: {str([model_id for model_id,_ in id_pairs if wildcard_match(pattern=model_id_match_str,text=model_id)])}"
            warning_msg += f" Matched gen_ids: {str([gen_id for _,gen_id in id_pairs if wildcard_match(pattern=gen_id_match_str,text=gen_id)])}"
            warnings.warn(warning_msg)
        for gen_id in gen_id_loads:
            self.add_saved_samples(DiffSamples(gen_id=gen_id,load_heavy=load_heavy,load_light=load_light))
    
    def sort_saved_samples(self,key=None):
        if key is None:
            key = lambda x: x.name
        self.saved_samples = sorted(self.saved_samples,key=key)

    def add_saved_samples(self,saved_samples):
        if isinstance(saved_samples,list):
            assert all([is_saved_samples(ss) for ss in saved_samples]), "expected all elements in a list to be instances of SavedSamples, found "+str([type(ss) for ss in saved_samples])
            self.saved_samples.extend(saved_samples)
        else:
            assert is_saved_samples(saved_samples), f"expected samples to be an instance of SavedSamples or a list of SavedSamples, found {type(saved_samples)}"
            self.saved_samples.append(saved_samples)

    def clear_all_heavy_data(self):
        for ss in self.saved_samples:
            ss.clear_heavy_data()

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
                    seed=None,
                    image_gt_from_saved_samples=True,
                    text_measures="ari",
                    text_color_inside="white",
                    ss_idx=None,
                    plot_qual_seg_kwargs={},
                    add_text_axis=True,
                    add_text_inside=True,
                    add_spaces_left=2,
                    transpose=None,
                    pixel_mult=1):
        self.raise_error_on_no_ss()
        if num_images is None:
            assert isinstance(didx,list), "If num_images is None, idx must be a list"
            num_images = len(didx)
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))
        heavy_avail = self.intersection_didx(heavy_only=True,ss_idx=ss_idx)
        if len(heavy_avail)==0:
            raise ValueError("No overlapping didx found")
        elif len(heavy_avail)<num_images:
            raise ValueError(f"Only {len(heavy_avail)} overlapping didx found but num_images={num_images} requested")
        if didx is None:
            if seed is None:
                didx_plot = heavy_avail[:num_images]
            elif seed<0:
                didx_plot = np.random.choice(heavy_avail,num_images,replace=False)
            else:
                assert isinstance(seed,int), "expected seed to be an int (use seed), None (fixed, first images) or a negative int (random seed)"
                didx_plot = np.random.RandomState(seed).choice(heavy_avail,num_images,replace=False)
            if isinstance(didx_plot,np.ndarray):
                didx_plot = didx_plot.tolist()
        else:
            assert isinstance(didx,list), "expected didx to be a list"
            assert all([d in heavy_avail for d in didx]), "expected all didx to be in heavy_avail"
            didx_plot = didx
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

            metrics_k = ss.get_light_data(didx_plot,keys=metrics_keys)
            metrics_k = concat_dict_list(metrics_k,ignore_weird_values=True)["metrics"]
            preds_k = ss.get_segmentations(didx_plot)
            preds.append([preds_k_i.cpu().numpy() for preds_k_i in preds_k])
            metrics.append(metrics_k)
        if resize_width is not None:
            assert isinstance(resize_width,int), "expected resize_width to be an int"
            plot_qual_seg_kwargs["resize_width"] = resize_width
        if alpha_mask is not None:
            assert isinstance(alpha_mask,float), "expected alpha_mask to be a float"
            plot_qual_seg_kwargs["alpha_mask"] = alpha_mask
        if transpose is not None:
            plot_qual_seg_kwargs["transposed"] = transpose
        big_image = plot_qual_seg(ims,preds,gts,**plot_qual_seg_kwargs).astype(np.uint8)
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
            sp = " "*add_spaces_left
            text_outside = [sp+text+sp for text in text_outside]
            text_pos_kwargs={"left": text_outside, 
                             "top": didx_plot, 
                             "xtick_kwargs": {"fontsize": 10*pixel_mult}}
            text_inside = [[text_inside[j][i] for j in range(len(text_inside))] for i in range(len(text_inside[0]))]
        else:
            y_sizes=[im.shape[0]/im.shape[1] for im in ims]
            x_sizes=[1 for _ in range(len(text_outside))]
            sp = " "*add_spaces_left
            didx_plot = [sp+d+sp for d in didx_plot]
            text_pos_kwargs={"top": text_outside, 
                             "left": didx_plot, 
                             "xtick_kwargs": {"fontsize": 10*pixel_mult}}
        text_kwargs = {"color": text_color_inside,
                       "fontsize": 8*pixel_mult,
                       "verticalalignment": "top",
                       "horizontalalignment": "left"}
        if add_text_axis or add_text_inside:
            big_image = render_text_gridlike(big_image,
                                    x_sizes=x_sizes,
                                    y_sizes=y_sizes,
                                    text_inside=text_inside if add_text_inside else [],
                                    anchor_image=(0.05,0.05),
                                    text_kwargs=text_kwargs,
                                    text_pos_kwargs=text_pos_kwargs if add_text_axis else {},
                                    border_width_inside=2,
                                    pixel_mult=pixel_mult)
        return big_image
    
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
            lines_instead_of_hist=False,
            bins=None, 
            unit_xlim=True):
        self.raise_error_on_no_ss()
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,ss_idx=ss_idx,intersection_only=intersection_only)
        ncol =  len(metric_names) if subplot_per_metric else 1
        nrow = len(ss_idx) if subplot_per_ss else 1
        if transposed:
            ncol,nrow = nrow,ncol
        plt.figure(figsize=figsize)
        ymax = {}
        for i in range(len(ss_idx)):
            for j in range(len(metric_names)):
                n = self.saved_samples[ss_idx[i]].name
                m = metric_names[j]
                y = metrics[i][m]

                if nrow==1 and ncol==1:
                    subplot_index = 1
                    label = f"{n} {m}"
                    title = ""
                elif nrow==1 and ncol>1:
                    subplot_index = 1+j
                    label = n if subplot_per_metric else m
                    title = m if subplot_per_metric else n
                elif nrow>1 and ncol==1:
                    subplot_index = i+1
                    label = n if subplot_per_metric else m
                    title = m if subplot_per_metric else n
                else:
                    subplot_index = j*nrow+i+1 if transposed else i*ncol+j+1
                    label = ""
                    title = f"{n} {m}"

                plt.subplot(nrow,ncol,subplot_index)

                if lines_instead_of_hist:
                    if isinstance(bins,int):
                        bin_width = 1/bins
                    elif isinstance(bins,(list,np.ndarray)):
                        bin_width = bins[1]-bins[0]
                    else:
                        assert bins is None, "expected bins to be an int, list, np.ndarray or None"
                        bin_width =  np.std(y)/10
                    std_kernel = bin_width
                    x = np.linspace(0,1,500)
                    y_curve = []
                    for x_i in x:
                        y_curve.append(np.sum(np.exp(-(y-x_i)**2/(2*std_kernel**2))))
                    y_curve = np.array(y_curve)
                    y_curve = y_curve/y_curve.sum()*len(y_curve)
                    #plot between the curve and zero
                    plt.plot(x,y_curve,label=label)
                    plt.fill_between(x,y_curve,0,alpha=0.3)
                    ymax[(nrow,ncol)] = max(ymax.get((nrow,ncol),0),max(y_curve))
                else:
                    hist_out = plt.hist(y,bins=bins,label=label,density=True)
                    ymax[(nrow,ncol)] = max(ymax.get((nrow,ncol),0),max(hist_out[0]))
                if mean_lines:
                    mean = np.mean(y)
                    plt.axvline(mean,color="red")
                    if mean_lines_text:
                        plt.text(mean,0,f"{mean:.2f}",rotation=90)
                last_subplot_visit = True
                if last_subplot_visit:
                    plt.title(title)
                    if ncol==1 or nrow==1:
                        plt.legend()
                    if unit_xlim:
                        plt.xlim(0,1)
                    plt.ylim(0,ymax[(nrow,ncol)]*1.2+(1 if ymax[(nrow,ncol)]<1e10 else 0))
        plt.tight_layout()
        return 
    
    def get_concat_ss(self,ss_idx=None,duplicate_join_mode="err"):
        assert duplicate_join_mode in ["both","new","old","err"], f"expected duplicate_join_mode to be one of ['both','new','old'], found {duplicate_join_mode}"
        self.raise_error_on_no_ss()
        if ss_idx is None:
            ss_idx = list(range(len(self.saved_samples)))
        assert len(ss_idx)>0, "expected at least one ss_idx to be present"
        ss = copy.deepcopy(self.saved_samples[ss_idx[0]])
        for i in ss_idx[1:]:
            ss = ss.join_saved_samples(self.saved_samples[i],duplicate_join_mode=duplicate_join_mode)
        return ss

    def metrics_for_plotting(self,metric_names=None,ss_idx=None,intersection_only=True):
        self.raise_error_on_no_ss()
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
        keyslist = [["metrics",m] for m in metric_names]
        for k in ss_idx:
            ss = self.saved_samples[k]
            metrics_k = ss.get_light_data(intersection_didx)
            metrics_k = [index_with_keylist(item,keyslist) for item in metrics_k]
            metrics_k = {m: np.array(maybe_flatten([item[j] for item in metrics_k])) for j,m in enumerate(metric_names)}
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

    def raise_error_on_no_ss(self):
        assert len(self.saved_samples)>0, "No saved samples found. Cannot construct plots"

    def bar(self,
            metric_names=None,
            ss_idx=None,
            subplots=False,
            seperated_by_metrics=True,
            text=False,
            error_bars=False,
            intersection_only=True,
            figsize=(12,6),
            subplot_horz=True,
            add_max_reduction=False):
        self.raise_error_on_no_ss()
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,
                                ss_idx=ss_idx,
                                intersection_only=intersection_only)
        #metrics structure:
        #metrics[ss_idx][metric_name] = list of values where mean is the bar height (instead of mean value, so we can get std)
        assert len(metric_names)>0, "expected at least one metric to be present in all ss_idx"
        assert len(ss_idx)>0, "expected at least one seperated_by_metrics ss_idx to be present"
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
                    y = [metrics[j][m] for j in range(len(ss_idx))]
                    fancy_bar(fancy_bar_args,ss_names,y)
                    plt.ylabel(m)
            else:
                n_subplots = len(ss_idx)
                
                for i in range(len(ss_idx)):
                    plt.subplot(1,n_subplots,i+1) if subplot_horz else plt.subplot(n_subplots,1,i+1)
                    y = [metrics[i][m] for m in metric_names]
                    fancy_bar(fancy_bar_args,metric_names,y)
                    plt.title(f"Metrics for {ss_names[i]}")
                    plt.ylim(0,ymax)
        else:
            if seperated_by_metrics:
                width = 0.8/len(ss_idx)
                for i in range(len(ss_idx)):
                    x = [j+width*i for j in range(len(metric_names))]
                    y = [metrics[i][m] for m in metric_names]
                    
                    if add_max_reduction:
                        y_max = [y_i.max(1) for y_i in y]
                        bar_kwargs = {"color": f"C{i}",
                                      "edgecolor": "k",
                                      "linewidth": 1,
                                      "linestyle": "--",
                                      "alpha": 0.4}
                        fancy_bar({},x,y_max,width,**bar_kwargs)
                    fancy_bar(fancy_bar_args,x,y,width,label=ss_names[i],color=f"C{i}")
                plt.xticks([i+width*(len(ss_idx)-1)/2 for i in range(len(metric_names))], metric_names)

            else:
                width = 0.8/len(metric_names)
                for m in metric_names:
                    x = [i+width*metric_names.index(m) for i in range(len(ss_idx))]
                    y = [met[m] for met in metrics]
                    fancy_bar(fancy_bar_args,x,y,width,label=m)
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
                          significant_digits=3,
                          convert_to_pct=False,
                          opts_keys_for_table=[],
                          add_max_reduction=False):
        self.raise_error_on_no_ss()
        metrics,metric_names,ss_idx = self.metrics_for_plotting(metric_names=metric_names,ss_idx=ss_idx,intersection_only=intersection_only)
        ss_names = [self.saved_samples[i].name for i in ss_idx]
        #this line is short, but has no max reduction
        #mean_metrics = {ss_name: {m: np.mean(metrics[i][m]).item() for m in metric_names} for i,ss_name in enumerate(ss_names)}
        mean_metrics = {}
        for i,ss_name in enumerate(ss_names):
            mean_metrics[ss_name] = {}
            for m in metric_names:
                mean_metrics[ss_name][m] = np.mean(metrics[i][m]).item()
                if add_max_reduction:
                    mean_metrics[ss_name]["max_"+m] = np.max(metrics[i][m]).item()

        if len(opts_keys_for_table)>0:
            for k in opts_keys_for_table:
                list_of_vals = []
                for i in range(len(ss_idx)):
                    if hasattr(self.saved_samples[ss_idx[i]],"sample_opts"):
                        list_of_vals.append(self.saved_samples[ss_idx[i]].sample_opts.get(k,float("nan")))
                    else:
                        list_of_vals.append(float("nan"))
                for ss_name,v in zip(ss_names,list_of_vals):
                    mean_metrics[ss_name][k] = v
        if to_df:
            mean_metrics = pd.DataFrame(mean_metrics)
            def conversion(x):
                try:
                    if convert_to_pct:
                        return f"{float(x)*100:.{significant_digits}f}"
                    else:
                        return f"{float(x):.{significant_digits}}"
                except:
                    return x
            mean_metrics = mean_metrics.applymap(conversion)
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
        self.raise_error_on_no_ss()
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
        vals1 = self.saved_samples[ss_idx1].get_light_data(intersection_didx,keys=[["metrics",metric1]])
        vals1 = np.array(concat_dict_list(vals1,ignore_weird_values=True)["metrics"][metric1])
        vals2 = self.saved_samples[ss_idx2].get_light_data(intersection_didx,keys=[["metrics",metric2]])
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

def is_empty_list(x):
    if isinstance(x,list):
        out = len(x)==0
    else:
        out = False
    return out

def get_all_id_pairs(only_with_light_stats=False):
    """
    Returns all (model_id,gen_id) pairs
    """
    out = []
    id_dict = TieredParser("sample_opts").load_and_format_id_dict()
    for v in id_dict.values():
        gen_id = v["gen_id"]
        model_id = v.get("model_id","")
        if only_with_light_stats:
            add = False
            if "light_stats_filename" in v.keys():
                if v["light_stats_filename"] is not None:
                    if len(v["light_stats_filename"])>0:
                        if Path(v["light_stats_filename"]).exists():
                            add = True
        else:
            add = True
        if add:
            out.append((model_id,gen_id))
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
                segment_key="pred_int",
                is_ambiguous=False,
                np_to_torch=True):
        assert all([(x is None) or is_nontrivial_list(x) for x in [light_data,heavy_data,didx]]), "expected all of [light_data,heavy_data,didx] to be None or a non-empty list"
        self.reset()
        if any([x is not None for x in [light_data,heavy_data,didx]]):
            self.add_samples(didx=didx,light_data=light_data,heavy_data=heavy_data)
            self.mem_all = sys.getsizeof(self)
            self.mem_threshold = mem_threshold
        self.segment_key = segment_key
        self.is_ambiguous = is_ambiguous
        if np_to_torch:
            self.np_to_torch()
        if name is not None:
            self.name = name
        else:
            self.name = "unnamed"

    def load_heavy_image_gt(self,didx_load=None,resize=True):
        if didx_load is None:
            didx_load = [self.didx[i] for i in range(len(self.didx)) if self.heavy_data[i] is not None]
        if resize:
            imsize = self.get_image_size()
        else:
            imsize = None
        for didx_i in didx_load:
            assert didx_i in self.didx, f"expected didx to be in self.didx, found {didx_i}"
            x = {"dataset_name": didx_i.split("/")[0], "i": int(didx_i.split("/")[1])}
            image,gt = load_raw_image_label(x,longest_side_resize=imsize)
            i = self.didx_to_idx[didx_i]
            if not isinstance(self.heavy_data[i],dict):
                self.heavy_data[i] = {}
            self.heavy_data[i]["image"] = image
            self.heavy_data[i]["gt"] = gt
            if self.is_ambiguous:
                ambiguous_gts, gts_didx = get_ambiguous_gts(didx_i,imsize=imsize)
                self.heavy_data[i]["gt"] = ambiguous_gts
                self.light_data[i]["gts_didx"] = gts_didx

    def clear_heavy_data(self):
        self.heavy_data = [None for _ in range(len(self.heavy_data))]
        old_to_new = {"unkown": "unknown", 
                      "pos_loaded": "pos_not_loaded",
                      "negative": "negative",
                      "pos_not_loaded": "pos_not_loaded"}
        self.heavy_available = {k: old_to_new[v] for k,v in self.heavy_available.items()}

    def get_image_size(self):
        has_heavy = [self.heavy_available[d]=="pos_loaded" for d in self.didx]
        assert any(has_heavy), "expected atlaest some heavy data to be loaded to determine image size"
        hd = self.heavy_data[has_heavy.index(True)]
        assert self.segment_key in hd.keys(), f"expected segment_key={self.segment_key} to be present in heavy_data.keys()={hd.keys()}"
        s = hd[self.segment_key].shape
        assert len(s)>=3, f"expected at least 3 dimensions in the image shape, found shape={s}"

        out = max(s[-2:])
        #assert out in [2**i for i in range(4,11)], f"expected image size to be a power of 2 between 16 and 1024, found {out}"

        return out

    def crop_padding(self):
        """Iterates through and crops all heavy data which is
          matching the dimensions of segmentation.
        """
        for didx_i in range(len(self.didx)):
            if self.heavy_data[didx_i] is not None:
                hd = self.heavy_data[didx_i]
                ld = self.light_data[didx_i]
                segmentation = hd[self.segment_key]
                h,w = segmentation.shape[-2:]
                assert h==w, "Expected segmentation to be square, found h="+str(h)+" and w="+str(w)
                h_new,w_new = sam_resize_index(*ld["info"]["imshape"][:2],w)
                for k in hd.keys():
                    if isinstance(hd[k],np.ndarray) or torch.is_tensor(hd[k]):
                        h_k,w_k = hd[k].shape[-2:]
                        if h_k==h and w_k==w:
                            hd[k] = hd[k][...,:h_new,:w_new]

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
            didx = self.didx
        didx = [d for d in didx if self.heavy_available[d]=="pos_loaded"]
        out = self.get_heavy_data(didx,return_type=return_type,keys=[self.segment_key])
        if only_segments_if_list and return_type=="list":
            out = [o[self.segment_key] for o in out]
        return out

    def save(self,save_path,return_instead_of_save=False):
        save_dict = {"name": self.name,
                    "heavy_data": self.heavy_data,
                    "light_data": self.light_data,
                    "didx": self.didx,
                    "postprocess_kwargs": self.postprocess_kwargs,
                    "segment_key": self.segment_key,
                    "mem_threshold": self.mem_threshold}
        if return_instead_of_save:
            return save_dict
        else:
            torch.save(save_dict,save_path)

    def np_to_torch(self):
        """Converts all heavy data to torch tensors"""
        for i in range(len(self.heavy_data)):
            if self.heavy_data[i] is not None:
                for k,v in self.heavy_data[i].items():
                    if isinstance(v,np.ndarray):
                        self.heavy_data[i][k] = torch.from_numpy(v)
    
    def torch_to_np(self):
        """Converts all heavy data to numpy arrays"""
        for i in range(len(self.heavy_data)):
            if self.heavy_data[i] is not None:
                for k,v in self.heavy_data[i].items():
                    if isinstance(v,torch.Tensor):
                        self.heavy_data[i][k] = v.cpu().numpy()

    def load(self,save_path_or_dict):
        if isinstance(save_path_or_dict,dict):
            load_dict = save_path_or_dict
        else:
            load_dict = torch.load(save_path_or_dict)
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
        self.didx_to_idx = {}
        self.heavy_available = {}
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

    def join_saved_samples(self,other,duplicate_join_mode="err"):
        assert duplicate_join_mode in ["both","new","old","err"], f"expected duplicate_join_mode to be one of ['both','new','old'], found {duplicate_join_mode}"
        assert isinstance(other,SavedSamples), "expected other to be an instance of SavedSamples"
        if duplicate_join_mode=="err":
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
    
    def get_light_data(self,indexer,return_type="list",keys=None):
        didx,idx = self.normalize_indexer(indexer)
        light_data = [self.light_data[i] for i in idx]
        if keys is not None:
            light_data = extract_from_dict_list(light_data,keys)
        return lists_of_dicts_as_type(didx,light_data,return_type,add_didx_to_list=keys is None)
    
    def get_heavy_data(self,indexer,include_light_data=False,return_type="list",keys=None):
        didx,idx = self.normalize_indexer(indexer)
        if include_light_data:
            heavy_data = [{**self.heavy_data[i],**self.light_data[i]} for i in idx]
        else:
            heavy_data = [self.heavy_data[i] for i in idx]
        if keys is not None:
            heavy_data = extract_from_dict_list(heavy_data,keys)
        return lists_of_dicts_as_type(didx,heavy_data,return_type,add_didx_to_list=keys is None)
            
    def add_samples(self,didx=None,light_data=None,heavy_data=None, assert_heavy_format=True):
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
            find_idx = self.didx_to_idx.get(d,-1)
            if find_idx>=0:
                if l is not None:
                    self.light_data[find_idx] = l
                if h is not None:
                    self.heavy_data[find_idx] = h
                    self.heavy_available[d] = "pos_loaded"
                    if assert_heavy_format:
                        assert isinstance(h,dict), f"expected heavy_data to be a dict, found {type(h)}"
                        assert self.segment_key in h.keys(), f"expected segment_key={self.segment_key} to be in heavy_data.keys()={h.keys()}"
                        seg = h[self.segment_key]
                        assert torch.is_tensor(seg), f"expected segmentations to be torch tensors, found {type(seg)}"
                        seg_shape = seg.shape
                        assert len(seg_shape)==4, f"expected segmentations to have 3 dimensions, found seg_shape={seg_shape}"
                        assert seg_shape[1] == 1, f"expected segmentations to have 1 channel, found seg_shape={seg_shape}"
                        if self.is_ambiguous:
                            assert min(seg_shape[0],*seg_shape[2:])==seg_shape[0], f"expected first dimension of segmentations to be the smallest, found {seg_shape}"
                        else:
                            pass #assert seg_shape[0]==1, f"expected first dimension of segmentations to be 1, found {seg_shape}"
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
    
    def load_heavy_data(self,**kwargs):
        has_read_heavy_data = hasattr(self,"read_heavy_data")
        assert has_read_heavy_data, "expected read_heavy_data to be implemented in a subclass"
        output_of_read_heavy_data = self.read_heavy_data(**kwargs)
        if output_of_read_heavy_data is None:
            return
        assert len(output_of_read_heavy_data)==3, f"expected read_heavy_data to return a tuple of length 3 representing [didx,light_data,heavy_data], found {len(output_of_read_heavy_data)}"
        self.add_samples(*output_of_read_heavy_data)
    
    def reduce_to_only_heavy(self):
        didx_w_heavy = []
        for k,v in self.heavy_available.items():
            if v=="pos_loaded":
                didx_w_heavy.append(k)
        return self.reduce_by_indexer(didx_w_heavy)
    
    def reduce_by_indexer(self,indexer):
        didx,idx = self.normalize_indexer(indexer)
        light_data = [self.light_data[i] for i in idx]
        heavy_data = [self.heavy_data[i] for i in idx]
        new_ss = SavedSamples(light_data=light_data,heavy_data=heavy_data,didx=didx,name=self.name)
        return new_ss

    def clone(self,new_name=None,didx=None):
        new_name = self.name if new_name is None else new_name
        #remove all samples not in didx
        if didx is None:
            new_ss = copy.deepcopy(self)
            new_ss.name = new_name
        else:
            didx,idx = self.normalize_indexer(didx)
            light_data = [self.light_data[i] for i in idx]
            heavy_data = [self.heavy_data[i] for i in idx]

            new_ss = SavedSamples(light_data=light_data,
                                  heavy_data=heavy_data,
                                  didx=didx,
                                  name=new_name)
        return new_ss

    def recompute_metrics(self,tqdm_recompute=False):
        self.postprocess(postprocess_kwargs=None,recompute_metrics=True,tqdm_recompute=tqdm_recompute)

    def postprocess(self,postprocess_kwargs={},recompute_metrics=True,tqdm_recompute=False):
        if self.postprocess_kwargs is not None:
            warnings.warn("The samples were already postprocessed, reprocessing with new postprocess_kwargs")
        didx = [d for d in self.didx if self.heavy_available[d]=="pos_loaded"]
        if len(didx)<len(self.didx):
            warnings.warn(f"only {len(didx)} of {len(self.didx)} samples are postprocessed")
        heavy_data = self.get_heavy_data(didx,return_type="list")
        light_data = self.get_light_data(didx,return_type="list")
        if self.is_ambiguous:
            gts_in_light_data = np.all(["gts_didx" in ld for ld in light_data])
            gts_in_heavy_data = np.all(["gts_didx" in hd for hd in heavy_data])
            gts_in_info_ld = np.all(["gts_didx" in ld.get("info",{}) for ld in light_data])
            gts_in_info_hd = np.all(["gts_didx" in hd.get("info",{}) for hd in heavy_data])
            if gts_in_light_data:
                gts = [ld["gts_didx"] for ld in light_data]
            elif gts_in_heavy_data:
                gts = [hd["gts_didx"] for hd in heavy_data]
            elif gts_in_info_ld:
                gts = [ld["info"]["gts_didx"] for ld in light_data]
            elif gts_in_info_hd:
                gts = [hd["info"]["gts_didx"] for hd in heavy_data]
            else:
                gts = [hd["gt"] for hd in heavy_data]
                #assert atleast 3 non-trivial dims in first gt
                assert len(gts[0].squeeze().shape)>=3, f"expected at least 3 non-trivial dimensions in gt, found {gts[0].shape}"
        elif "gt" in self.heavy_keys():
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
            if self.is_ambiguous:
                segments_pp = [postprocess_batch(s_i,seg_kwargs=postprocess_kwargs) for s_i in segments]
            else:
                #raise NotImplementedError("Probably doesn't work, uncomment and try. Fix shapes and type")
                num_votes = segments[0].shape[0]
                if num_votes>1:
                    warnings.warn("only the first vote is currently implemented for postprocessing")
                segments = [s_i[0,0] for s_i in segments]
                segments_pp = postprocess_list_of_segs(segments,seg_kwargs=postprocess_kwargs)
        if recompute_metrics:
            metrics = []
            tqdm_hat = tqdm.tqdm if tqdm_recompute else lambda x: x
            for i in tqdm_hat(range(len(didx))):
                didx_i = didx[i]
                seg = segments_pp[i]
                gt = gts[i]
                if self.is_ambiguous:
                    metrics.append(get_ambiguous_metrics(seg.permute(1,2,0).cpu().numpy(),gt))
                else:
                    metrics.append(get_segment_metrics(seg[None],didx_i)) 
        for i in range(len(segments)):
            idx = self.didx_to_idx[didx[i]]
            self.light_data[idx]["metrics"] = metrics[i]
            self.heavy_data[idx][self.segment_key] = segments_pp[i]
        self.postprocess_kwargs = postprocess_kwargs

    def add_gt_image_to_heavy_data(self,didx=None,longest_side_resize=0,process_pred=True):
        if didx is None:
            didx = self.didx
        for d in didx:
            i = self.didx_to_idx[d]
            if "gt" in self.heavy_data[i].keys():
                continue
            x  = {"dataset_name": d.split("/")[0], "i": int(d.split("/")[1])}
            image,gt = load_raw_image_label(x,longest_side_resize=longest_side_resize)
            self.heavy_data[i]["gt"] = gt
            self.heavy_data[i]["image"] = image
            if "pred_int" in self.heavy_data[i].keys():
                if process_pred:
                    self.heavy_data[i]["pred_int"] = self.heavy_data[i]["pred_int"][0].permute(1,2,0).cpu().numpy()

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
                glob_str="raw_sample_batch*.pt",
                is_ambiguous=None,
                ):
        super().__init__(mem_threshold=mem_threshold)
        id_dict = TieredParser("sample_opts").load_and_format_id_dict()
        assert gen_id in id_dict.keys(), f"gen_id {gen_id} not found in id_dict. Closest matches are {get_closest_matches(gen_id,id_dict.keys())}"
        self.gen_id = gen_id
        self.name = gen_id
        if is_ambiguous is not None:
            self.is_ambiguous = is_ambiguous
        self.sample_opts = id_dict[gen_id]
        if len(self.sample_opts["raw_samples_folder"])>0:
            self.raw_samples_files = sorted(list(Path(self.sample_opts["raw_samples_folder"]).glob(glob_str)))
            if len(self.raw_samples_files)==0:
                warnings.warn(f"no files found in {self.sample_opts['raw_samples_folder']} with glob_str={glob_str}")
            self.mem_per_batch = os.path.getsize(self.raw_samples_files[0])
            self.mem_all = self.mem_per_batch*len(self.raw_samples_files)
        else:
            self.raw_samples_files = []
        if load_light:
            self.load_light_data()
        if load_heavy:
            self.load_heavy_data()

    def read_heavy_data(self,read_didx=None,extract=True,amb_cat_key="pred_int"):
        if not (len(self.sample_opts["raw_samples_folder"])>0 and self.sample_opts["save_raw_samples"]):
            warnings.warn("no raw_samples_folder found or save_raw_samples is False")
            return None
        heavy_data = []
        didx = []
        ack = amb_cat_key
        for i in range(len(self.raw_samples_files)):
            batch = torch.load(self.raw_samples_files[i])
            batch_didx = didx_from_info(batch["info"])
            bs = len(batch["info"])
            for b in range(bs):
                didx_i = batch_didx[b]
                if (read_didx is None) or (didx_i in read_didx):
                    item = index_dict_with_bool(copy.deepcopy(batch),bool_iterable=np.arange(bs)==b)
                    if self.heavy_available[didx_i]=="pos_loaded":
                        j = didx.index(didx_i)
                        #assert self.is_ambiguous, "Found repeat votes, but self.is_ambiguous is False for didx_i="+didx_i
                        assert ack in heavy_data[j].keys(), "expected pred_int to be in heavy_data[j].keys(), found "+str(heavy_data[j].keys())
                        heavy_data[j][ack] = torch.cat([heavy_data[j][ack],item[ack]],dim=0)
                    else:
                        didx.append(didx_i)
                        heavy_data.append(item)
                    self.heavy_available[didx_i] = "pos_loaded"
                else:
                    self.heavy_available[didx_i] = "pos_not_loaded"
        if read_didx is None:
            #set all existences which are unknown to negative
            for d in self.didx:
                if self.heavy_available[d]=="unknown":
                    self.heavy_available[d] = "negative"
        if extract:
            if "pred" in heavy_data[0].keys():
                heavy_data = extract_from_sample_list(heavy_data)
        return didx,None,heavy_data

    def read_light_data(self,read_didx=None):
        assert len(self.sample_opts["light_stats_filename"])>0, "no light_stats_filename found"
        light_data = load_json_to_dict_list(self.sample_opts["light_stats_filename"])
        didx = didx_from_info(light_data)
        if read_didx is not None:
            light_data = [light_data[i] for i in range(len(didx)) if didx[i] in read_didx]
            didx = didx_from_info(light_data)
        for d,ld in zip(didx,light_data):
            if "has_raw_sample" in ld.keys():
                if d not in self.heavy_available.keys():
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

def get_ambiguous_gts(x,imsize=128):
    if isinstance(x,str):
        x = {"dataset_name": x.split("/")[0], "i": int(x.split("/")[1])}
    else:
        assert isinstance(x,dict), "expected x to be a string or a dict"
        assert "dataset_name" in x.keys(), "expected x to have key 'dataset_name'"
        assert "i" in x.keys(), "expected x to have key 'i'"
    valid_multivote_datasets = ["lidc4","lidc15096"]
    assert x["dataset_name"] in valid_multivote_datasets, f"expected dataset_name to be one of {valid_multivote_datasets}, found {x['dataset_name']}"
    gts_didx = [f"{x['dataset_name']}/{x['i']+j}" for j in range(4)]
    gts = []
    h,w = imsize,imsize
    for didx in gts_didx:
        gts.append(torch.tensor(load_raw_image_label(didx,longest_side_resize=imsize)[1]).permute(2,0,1))
    gts = torch.stack(gts,axis=0)
    return gts, gts_didx

def mahalanobis_distance(data):
    """calculate mahalanobis distance for a given dataset, each row is a sample"""
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
                         raw_longest_side_resize=0,
                         ab_kw={}):
    assert "pred" in samples.keys(), "expected samples to have key 'pred'. Found keys: "+str(samples.keys())
    if ab is None:
        ab = AnalogBits(num_bits=samples["pred"].shape[1])
    extracted = {}
    bs = samples["pred"].shape[0]
    valid_keys = ["pred_int","pred_prob","gt","image","raw_image","raw_gt"]
    for k in extract:
        assert k in valid_keys, f"expected k to be one of {valid_keys}, found {k}"
    if "pred_int" in extract:
        extracted["pred_int"] = samples["pred_int"]
    if "pred_prob" in extract:
        if "pred_bit" in samples.keys():
            #unsafe addition of kwargs here, since passing the ab_kw to ab_bit2prob is bothersome
            if "num_bits" not in ab_kw.keys():
                ab_kw["num_bits"] = samples["pred_bit"].shape[1]
            extracted["pred_prob"] = ab_bit2prob(samples["pred_bit"],**ab_kw)
    if "gt" in extract:
        extracted["gt"] = samples["gt_int"]
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

def plot_qual_seg(ims,preds,
                  gts=None,
                  names=None,
                transposed=False,
                resize_width=128,
                border=0,
                show_image_alone=True,
                alpha_mask=0.6):
    """
    Function for plotting columns of different to compare segmentations.
    Ground truth is also considered a prediction.
    """
    assert isinstance(ims,list), "expected ims to be a list"
    assert all([isinstance(im,np.ndarray) for im in ims]), "expected all ims to be numpy arrays"
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
        assert all([len(x)==n_samples for x in preds]), "expected all preds to have the same length as ims"
        assert all([isinstance(x,np.ndarray) for pred_i in preds for x in pred_i]), "expected all preds to be numpy arrays"
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

def kernel_hist(y,std_mult=0.1,n_points=200,minval=None,maxval=None,kernel="gaussian"):
    sigma = np.std(y)*std_mult*(0.1443375 if kernel=="uniform" else 1)
    if minval is None:
        minval = np.min(y)-5*sigma
    if maxval is None:
        maxval = np.max(y)+5*sigma
    assert kernel in ["gaussian","uniform"]
    if kernel=="gaussian":
        coef = 1/(np.sqrt(2*np.pi)*sigma)/len(y)
        kernel_func = lambda t: np.exp(-0.5*(t/sigma)**2)
    elif kernel=="uniform":
        coef = 1/(2*sigma)/len(y)
        kernel_func = lambda t: (np.abs(t)<sigma).astype(float)
    else:
        raise ValueError("expected kernel to be one of ['gaussian','uniform'], found "+kernel)
    t = np.linspace(minval,maxval,n_points)
    h = []
    for ti in t:
        h.append(np.sum(kernel_func(y-ti))*coef)
    return t,h

def maybe_flatten(x):
    """Takes a list as input, if the list is a list of lists then it concatenates each list into a single list"""
    assert isinstance(x,list), "expected x to be a list"
    if all([isinstance(xi,list) for xi in x]):
        return sum(x,[])
    else:
        return x

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",type=str,default="ade20k")
    parser.add_argument("--model_type",default=0)
    parser.add_argument("--setup",type=str,default="sam1_1")
    parser.add_argument("--num_return_segments",type=int,default=64)
    parser.add_argument("--num_samples",type=int,default=256)
    
    args = parser.parse_args()
    verbose = True
    postprocess = False
    names = sam12_info["names"][4:]
    eval_sam_kwargs = argparse.Namespace(datasets="ade20k",
                                        model_type=args.model_type,
                                        num_return_segments=args.num_return_segments,
                                        split="vali",
                                        ratio_of_dataset=args.num_samples,
                                        generator_kwargs=all_sam_setups[args.setup],
                                        pri_didx=None,
                                        longest_side_resize=1024,
                                        batch_size=4,
                                        postprocess_kwargs=None,
                                        full_resolution_decoder=False)

    for name in names:
        eval_sam_kwargs.__dict__.update({"model_type":name})
        metrics_mean, light_data, heavy_data = evaluate_sam(**vars(eval_sam_kwargs))
        if verbose:
            print(f"\n\n Mean metrics for {name}:")
            print(metrics_mean)
        save_path = f"saves/sam_eval/sam12_comp/{name}_{args.num_samples}_{args.datasets}.pt"
        sam_samples = SavedSamples(light_data=light_data,heavy_data=heavy_data,name=name)
        sam_samples.save(save_path)
        if postprocess:
            sam_samples.postprocess({"mode": "min_area", "min_area": 0.005})
            sam_samples.save(save_path.replace(".pt","_areapost.pt"))

if __name__=="__main__":
    main()
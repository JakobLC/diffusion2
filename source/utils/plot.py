import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import jlc
import matplotlib.cm as cm
import os
import glob
from pathlib import Path
from PIL import Image
from source.utils.argparsing import TieredParser
from source.utils.mixed import (bracket_glob_fix, save_dict_list_to_json, 
                                imagenet_preprocess, 
                                load_json_to_dict_list, wildcard_match,
                                sam_resize_index,unet_kwarg_to_tensor,model_arg_is_trivial,
                                didx_from_info)
from source.utils.analog_bits import ab_bit2prob
from source.utils.metric_and_loss import get_likelihood
from source.utils.dataloading import get_dataset_from_args, load_raw_image_label
from source.models.unet import dynamic_image_keys
import cv2
import pandas as pd
import scipy.ndimage as nd
#from models.cond_vit import dynamic_image_keys
from jlc import (RenderMatplotlibAxis, darker_color,
                 distance_transform_edt_border,mask_overlay_smooth,
                 get_mask,render_axis_ticks,darker_color,get_matplotlib_color,
                 add_text_axis_to_image,to_xy_anchor,render_text_gridlike,
                 item_to_rect_lists)

import warnings
try:
    from sklearn.cluster import KMeans
except:
    warnings.warn("Could not import KMeans from sklearn.cluster")

def collect_gen_table(gen_id_patterns="all_ade20k[ts_sweep]*",
                   model_id_patterns="*",
                   save=False,
                   return_table=True,
                   save_name="",
                   verbose=True,
                   make_pretty_table=True,
                   pretty_digit_limit=5,
                   search_gen_setups_instead=False,
                   include_mode="last",
                   record_from_sample_opts=[],
                   record_from_args=[],
                   sort_by_key=["save_path"],
                   do_map_to_float=True,
                   round_digits=3):
    if isinstance(record_from_sample_opts,str):
        record_from_sample_opts = [record_from_sample_opts]
    if isinstance(record_from_args,str):
        record_from_args = [record_from_args]
    assert include_mode in ["last","last_per_gen_id","all"], f"expected include_mode to be one of ['last','last_per_gen_id','all'], found {include_mode}"
    if isinstance(gen_id_patterns,str):
        gen_id_patterns = [gen_id_patterns]
    if isinstance(model_id_patterns,str):
        model_id_patterns = [model_id_patterns]
    model_id_dict = TieredParser().load_and_format_id_dict()
    gen_id_dict = TieredParser("sample_opts").load_and_format_id_dict()
    save_paths = []
    table = pd.DataFrame()
    for model_id,v in model_id_dict.items():
        matched = False
        for model_id_pattern in model_id_patterns:
            if wildcard_match(model_id_pattern,model_id):
                if verbose: 
                    print(f"Matched pattern {model_id_pattern} with model_id {model_id}")
                matched = True
                break
        if matched:
            fn = Path(v["save_path"])/"logging_gen.csv"
            if fn.exists():
                with open(str(fn),"r") as f:
                    column_names = f.readline()[:-1].split(",")
                data = np.genfromtxt(str(fn), dtype=str, delimiter=",")[1:]
                if data.size==0:
                    continue
                if search_gen_setups_instead:
                    file_gen_ids = data[:,column_names.index("gen_setup")].astype(str)
                else:
                    file_gen_ids = data[:,column_names.index("gen_id")].astype(str)
                match_idx = set()
                
                for idx,fgi in enumerate(file_gen_ids):
                    for gen_id_pattern in gen_id_patterns:
                        if wildcard_match(gen_id_pattern,fgi):
                            if verbose: 
                                print(f"Matched pattern {gen_id_pattern} with gen_id {fgi} from model_id {model_id}")
                            match_idx.add(idx)
                            break
                if len(match_idx)==0:
                    continue
                if include_mode=="last":
                    match_idx = [max(match_idx)]
                    if verbose and len(match_idx)>1:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                elif include_mode=="all":
                    match_idx = list(match_idx)
                elif include_mode=="last_per_gen_id":
                    len_before = len(match_idx)
                    match_idx = list(match_idx)
                    match_idx = [max([i for i in match_idx if file_gen_ids[i]==file_gen_ids[j]]) for j in match_idx]
                    if verbose and len(match_idx)<len_before:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                else:
                    match_idx = list(match_idx)
                match_data_s = data[match_idx]
                if len(record_from_args)>0:
                    args = load_json_to_dict_list(str(Path(v["save_path"])/"args.json"))
                    for rfa in record_from_args:
                        assert rfa in args[0].keys(), f"expected record_from_args to be in args, found {rfa}"
                        match_data_s = np.concatenate([match_data_s,np.array([args[0][rfa] for _ in range(match_data_s.shape[0])]).reshape(-1,1)],axis=1)
                        column_names.append(rfa)
                if len(record_from_sample_opts)>0:
                    column_names += record_from_sample_opts
                    empty_array = np.array(["" for _ in range(match_data_s.shape[0])]).reshape(-1,1).repeat(len(record_from_sample_opts),axis=1)
                    match_data_s = np.concatenate([match_data_s,empty_array],axis=1)
                    gen_id_list = match_data_s[:,column_names.index("gen_id")].tolist()
                    for mds_i,gen_id in enumerate(gen_id_list):
                        sample_opts = gen_id_dict[gen_id]
                        for rfso in record_from_sample_opts:
                            match_data_s[mds_i,column_names.index(rfso)] = sample_opts[rfso]
                table = pd.concat([table,pd.DataFrame(match_data_s,columns=column_names)],axis=0)
                save_paths.extend([v["save_path"] for _ in range(len(match_idx))])
            else:
                pass#warnings.warn(f"Could not find file {fn}")
    if table.shape[0]==0:
        warnings.warn("Gen table is empty")
        if return_table:
            return table
        else:
            return
    else:
        if do_map_to_float:
            table = table.map(map_to_float)
        if round_digits>0:
            table = table.round(round_digits)
    table["save_path"] = save_paths
    if isinstance(sort_by_key,str):
        sort_by_key = sort_by_key.split(",")
    table = table.sort_values(by=sort_by_key)
    table = table.loc[:, (table != "").any(axis=0)]
    table_pd = table.copy()
    table = {k: table[k].tolist() for k in table.keys()}
    if make_pretty_table:
        buffer = 2
        pretty_table = ["" for _ in range(len(table["save_path"])+2)] 
        for k in table.keys():
            pretty_col = ["" for _ in range(len(table["save_path"])+2)]
            
            if (isinstance(table[k][0],str)
                and table[k][0].replace(".","").isdigit() 
                and table[k][0].find(".")>=0):
                idx = slice(pretty_digit_limit+2)
            else:
                idx = slice(None)
            max_length = max(max([len(str(x)[idx]) for x in table[k]]),len(k))+buffer
            pretty_col[0] = k+" "*(max_length-len(k))
            pretty_col[1] = "#"*max_length
            pretty_col[2:] = [str(x)[idx]+" "*(max_length-len(str(x)[idx])-2)+", " for x in table[k]]
            if k=="model_name":
                pretty_col[0] = "model_name"+" "*(max_length-len("model_name")-1)+"# "
                pretty_col[1] = "#"*(max_length+1)
                pretty_col[2:] = [s.replace(","," #") for s in pretty_col[2:]]
                pretty_table = [pretty_col[i]+pretty_table[i] for i in range(len(pretty_table))]
            else:
                pretty_table = [pretty_table[i]+pretty_col[i] for i in range(len(pretty_table))]
        table["pretty_table"] = pretty_table
    if save:
        save_dict_list_to_json(table,save_name,append=True)
    if return_table:
        return table_pd

def map_to_float(x):
    try:
        return float(x)
    except:
        return x

def get_dtype(vec):
    vec0 = vec[0]
    assert isinstance(vec0,(str,int,float,bytes)), f"expected vec0 to be a str, int, float, or bytes, found {type(vec0)}"
    try:
        int(vec0)
        return int
    except:
        pass
    try:
        float(vec0)
        return float
    except:
        pass
    return str

def pretty_point(im,footprint=None,radius=0.05):
    if torch.is_tensor(im):
        was_tensor = True
        device = im.device
        im = im.cpu().detach().permute(1,2,0).numpy()
    else:
        was_tensor = False
    assert isinstance(im,np.ndarray), "im must be a numpy array or torch.tensor"
    assert len(im.shape)==2 or len(im.shape)==3, "im must be a 2D or 3D torch.tensor or numpy array"
    if footprint is None:
        #make star-shaped footprint
        min_sidelength = min(im.shape[0],im.shape[1])
        rad1 = np.ceil(radius*min_sidelength*0.66666).astype(int)
        rad2 = radius*min_sidelength*0.33333
        rad3 = np.ceil(rad1+rad2).astype(int)
        footprint = np.ones((2*rad3+1,2*rad3+1))
        #make cross
        footprint[rad3,rad3-rad1:rad3+rad1+1] = 0
        footprint[rad3-rad1:rad3+rad1+1,rad3] = 0
        footprint = nd.distance_transform_edt(footprint,return_indices=False)
        footprint = (footprint<=rad2).astype(int)
    else:
        assert isinstance(footprint,np.ndarray), "footprint must be a numpy array or None"
    if len(im.shape)==2:
        im = im[:,:,np.newaxis]
    if len(footprint.shape)==2:
        footprint = footprint[:,:,np.newaxis]
    #convolve image with footprint
    conv = nd.convolve(im,footprint,mode='constant',cval=0.0)
    conv_num = nd.convolve((np.abs(im)>1e-10).astype(float),footprint,mode='constant',cval=0.0)
    # Same as pretty_point_image = conv/conv_num, but avoiding 0/0
    pretty_point_image = conv
    pretty_point_image[conv_num>0] = conv[conv_num>0]/conv_num[conv_num>0]
    if was_tensor:
        pretty_point_image = torch.tensor(pretty_point_image).permute(2,0,1).to(device)
    return pretty_point_image

def make_loss_plot(save_path,
                   step,
                   save=True,
                   show=False,
                   fontsize=14,
                   figsize_per_subplot=(8,2),
                   remove_old=True,
                   is_ambiguous=False):
    filename = os.path.join(save_path,"logging.csv")
    filename_gen = os.path.join(save_path,"logging_gen.csv")
    filename_step = os.path.join(save_path,"logging_step.csv")
    filenames = [filename_gen,filename_step,filename]
    #helpers
    #get logging index
    gli = lambda s: [s.startswith(p) for p in ["gen_","step_",""]].index(True)
    #get logging string
    gls = lambda s: ["gen_","step_",""][gli(s)]

    all_logging = {}
    for i in range(len(filenames)):
        fn = filenames[i]
        if not os.path.exists(fn):
            continue
        with open(fn,"r") as f:
            column_names = f.readline()[:-1].split(",")
        data = np.genfromtxt(fn, dtype=object, delimiter=",")[1:]
        data[data==b''] = b'nan'
        if data.size==0:
            continue
        if len(data.shape)==1:
            data = np.expand_dims(data,0)
        #inf_mask = np.logical_and(~np.any(np.isinf(data),axis=1),~np.all(np.isnan(data),axis=1))
        #data = data[inf_mask]
        
        if filename_step==fn:
            column_names.append("step")
            data = np.concatenate([data,np.arange(1,len(data)+1).reshape(-1,1)],axis=1)
        for j,k in enumerate(column_names):
            try:
                all_logging[["gen_","step_",""][i]+k] = data[:,j].astype(get_dtype(data[:,j]))
            except:
                print(f"Data shape {data.shape}, j={j}, k={k}, i={i}, column_names={column_names}")
                
    if len(all_logging.keys())==0:
        return
    plot_columns = [["loss","vali_loss"],
                    ["mse_x","vali_mse_x"],
                    ["mse_eps","vali_mse_eps"],
                    ["iou","vali_iou"],
                    ["gen_GED"] if is_ambiguous else ["gen_hiou","gen_max_hiou"],
                    ["gen_iou"] if is_ambiguous else ["gen_ari","gen_max_ari"],
                    ["step_loss"],
                    ["likelihood","vali_likelihood"]]
    plot_columns_new = []
    #remove non-existent columns
    for i in range(len(plot_columns)):
        plot_columns_i = []
        for s in plot_columns[i]:
            if s in all_logging.keys():
                plot_columns_i.append(s)
        if len(plot_columns_i)>0:
            plot_columns_new.append(plot_columns_i)
    plot_columns = plot_columns_new
    n = len(plot_columns)
    #at most 4 plots per column
    n1 = min(4,n)
    n2 = int(max(1,np.ceil(n/4)))
    
    if "gen_gen_setup" in all_logging.keys():
        plot_gen_setups = np.unique(all_logging["gen_gen_setup"])
    
    figsize = (figsize_per_subplot[0]*n2,figsize_per_subplot[1]*n1)
    fig = plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(n1,n2,i+1)
        Y = []
        for j in range(len(plot_columns[i])):
            name = plot_columns[i][j]
            y = all_logging[name]
            x = all_logging[gls(name)+"step"]
            nan_or_inf_mask = np.logical_or(np.isnan(y),np.isinf(y))
            if gls(name)=="gen_":
                for k,gen_setup in enumerate(plot_gen_setups):
                    setup_mask = np.array(all_logging["gen_gen_setup"])==gen_setup
                    #gen_id_mask = np.array(all_logging["gen_gen_id"])==""
                    setup_mask = np.logical_and(setup_mask,~nan_or_inf_mask)
                    y2 = y[setup_mask]
                    x2 = x[setup_mask]
                    if len(y2)>0:
                        Y.append(y2)
                        plot_kwargs = get_plot_kwargs(gen_setup+"_"+name,idx=k,y=y2)
                        plt.plot(x2,y2,**plot_kwargs)
            else:
                y = y[~nan_or_inf_mask]
                x = x[~nan_or_inf_mask]
                if len(y)>0:
                    Y.append(y)
                    plot_kwargs = get_plot_kwargs(name,idx=None,y=y)
                    plt.plot(x,y,**plot_kwargs)
        if len(Y)>0:
            plt.legend()
            plt.grid()
            xmax = x.max()
            plt.xlim(0,xmax)
            Y = np.array(sum([y.flatten().tolist() for y in Y],[]))
            if any(np.isinf(Y)):
                print("Warning: inf found in Y at plot_columns[i]=",plot_columns[i])
            ymin,ymax = Y.min(),Y.max()
            ymax += 0.1*(ymax-ymin)+1e-14
            if name.find("loss")>=0 or name.find("grad_norm")>=0:
                plt.yscale("log")
                if ymin<1e-8:
                    ymin = 1e-8
            else:
                ymin -= 0.1*(ymax-ymin)
            plt.ylim(ymin,ymax)
            plt.xlim(0,xmax*1.05)
            plt.xlabel("steps")
    plt.tight_layout()
    if show:
        plt.show()
    save_name = os.path.join(save_path, f"loss_plot_{step:06d}.png")
    if save:
        fig.savefig(save_name)
    if remove_old:
        clean_up(save_name)    
    plt.close(fig)

def get_plot_kwargs(name,idx,y):
    plot_kwargs = {"color": None,
                   "label": name}
    if name.find("gen_")>=0:
        if idx is not None:
            plot_kwargs["color"] = f"C{idx}"
    if name.find("max_")>=0:
        plot_kwargs["linestyle"] = "--"
    if len(y)<=25:
        plot_kwargs["marker"] = "o"
    return plot_kwargs

def gaussian_filter_xy(x,y,sigma):
    assert len(x)==len(y)
    if not isinstance(x,np.ndarray):
        x = np.array(x)
    if not isinstance(y,np.ndarray):
        y = np.array(y)
    gaussian = lambda x: np.exp(-x**2/(2*sigma**2))
    y2 = y.copy()
    for i in range(len(x)):
        filtered_weight = gaussian(x-x[i])
        filtered_weight /= filtered_weight.sum()+1e-14
        y2[i] = (y*filtered_weight).sum()
    return y2

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]

"""def analog_bits_on_image(x_bits,im,ab):
    assert isinstance(x_bits,torch.Tensor), "analog_bits_on_image expects a torch.Tensor"
    x_int = ab.bit2int(x_bits.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
    magnitude = np.minimum(torch.min(x_bits.abs(),0)[0].cpu().detach().numpy(),1)
    mask = np.zeros((im.shape[0],im.shape[1],2**ab.num_bits))
    for i in range(2**ab.num_bits):
        mask[:,:,i] = (x_int==i)*magnitude
    return mask_overlay_smooth(im,mask,alpha_mask=1.0)
"""

def mean_dim0(x):
    assert isinstance(x,torch.Tensor), "mean_dim2 expects a torch.Tensor"
    return (x*0.5+0.5).clamp(0,1).mean(0).cpu().detach().numpy()

def replace_nan_inf(x,replace_nan=0,replace_inf=0):
    if torch.is_tensor(x):
        x = x.clone()
        x[torch.isnan(x)] = replace_nan
        x[torch.isinf(x)] = replace_inf
    elif isinstance(x,np.ndarray):
        x = x.copy()
        x[np.isnan(x)] = replace_nan
        x[np.isinf(x)] = replace_inf
    else:
        raise ValueError(f"expected x to be a torch.Tensor or np.ndarray, found {type(x)}")
    return x

def error_image(x):
    return cm.RdBu(replace_nan_inf(255*mean_dim0(x)).astype(np.uint8))[:,:,:3]

def contains_nontrivial_key_val(key,dictionary,ignore_none=True):
    has_key = key in dictionary.keys()
    if has_key:
        if ignore_none:
            has_key = dictionary[key] is not None
    return has_key

def concat_inter_plots(foldername,concat_filename,num_timesteps,remove_children=True,remove_old=True):
    images = []
    filenames = sorted([str(f) for f in list(Path(foldername).glob("intermediate_*.png"))])
    batch_size = len(filenames)
    for filename in filenames:
        im = np.array(Image.open(filename))
        images.append(im)
    images = np.concatenate(images,axis=0)
    images = Image.fromarray(images)
    images.save(concat_filename)
    left = ["gt_bit","final pred_","image"]*batch_size
    right = ["x_t","pred_bit","pred_eps"]*batch_size
    t_vec = np.array(range(num_timesteps, 0, -1))/num_timesteps
    top = bottom = ["","t="]+[f"{t_vec[j]:.2f}" for j in range(num_timesteps)]
    top[1] = "points"
    _ = add_text_axis_to_image(concat_filename,save_filename=concat_filename,
                           left=left,top=top,right=right,bottom=bottom,xtick_kwargs={"fontsize":20})
    if remove_children:
        for filename in filenames:
            os.remove(filename)
        if len(os.listdir(foldername))==0:
            os.rmdir(foldername)
    if remove_old:
        clean_up(concat_filename)

def normal_image(x,imagenet_stats=True): 
    if imagenet_stats:
        x2 = imagenet_preprocess(x.unsqueeze(0),inv=True)
        return x2.squeeze(0).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    else:
        return (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()

def plot_inter(foldername,sample_output,model_kwargs,save_i_idx=None,plot_text=False,imagenet_stats=True,ab_kw={}):
    t = sample_output["inter"]["t"]
    num_timesteps = len(t)
    
    if save_i_idx is None:
        batch_size = sample_output["pred_bit"].shape[0]
        save_i_idx = np.arange(batch_size)
    else:
        assert isinstance(save_i_idx,list), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx)}"
        assert len(save_i_idx)>0, f"expected save_i_idx to be a list of ints or bools, found {save_i_idx}"
        assert isinstance(save_i_idx[0],(bool,int)), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx[0])}"
        batch_size = len(save_i_idx)
        if isinstance(save_i_idx[0],bool):
            save_i_idx = np.arange(batch_size)[save_i_idx]
        batch_size = len(save_i_idx)
    image_size = sample_output["pred_bit"].shape[-1]
    
    """im = np.zeros((batch_size,image_size,image_size,3))+0.5
    aboi = lambda x,i: mask_overlay_smooth(im[i],ab_bit2prob(x.unsqueeze(0),**ab_kw)[0].permute(1,2,0).cpu().numpy(),alpha_mask=1.0)
    points_aboi = lambda x,i: aboi(pretty_point(x),i)
    normal_image2 = lambda x,i: normal_image(x,imagenet_stats=imagenet_stats)
    map_dict = {"x_t": aboi,
                "pred_bit": aboi,
                "gt_bit": aboi,
                "pred_eps": aboi,
                "image": normal_image2,
                "points": points_aboi}"""
    map_dict = get_map_dict(imagenet_stats,ab_kw)
    zero_image = np.zeros((image_size,image_size,3))
    has_classes = contains_nontrivial_key_val("classes",model_kwargs)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filenames = []
    num_inter_exists = len(glob.glob(bracket_glob_fix(os.path.join(foldername,"intermediate_*.png"))))
    for i in range(batch_size):
        ii = save_i_idx[i]
        images = [[map_dict["gt_bit"](sample_output["gt_bit"][ii])],
                  [map_dict["pred_bit"](sample_output["pred_bit"][ii])],
                  [map_dict["image"](model_kwargs["image"][ii])] if contains_nontrivial_key_val("image",model_kwargs) else [zero_image]]
        images[0].append(map_dict["points"](model_kwargs["points"][ii]) if contains_nontrivial_key_val("points",model_kwargs) else zero_image)
        images[1].append(zero_image)
        images[2].append(zero_image)
        text = [["gt_bit"],["final pred_bit"],["image"]]
        for k_i,k in enumerate(["x_t","pred_bit","pred_eps"]):
            for j in range(num_timesteps):
                if k in sample_output["inter"].keys():
                    images[k_i].append(map_dict[k](sample_output["inter"][k][j][i]))
                    text_j = ("    t=" if j==0 else "")+f"{t[j]:.2f}" if k_i==0 else ""
                    text[k_i].append(text_j)
                else:
                    images[k_i].append(zero_image)
                    text[k_i].append("")
        filename = os.path.join(foldername,f"intermediate_{i+num_inter_exists:03d}.png")
        filenames.append(filename)
        images = sum(images,[])
        text = sum(text,[])
        if not plot_text:
            text = ["" for _ in range(len(text))]
        if has_classes:
            text[num_timesteps+3] = f"class={model_kwargs['classes'][ii].item()}"
        jlc.montage_save(save_name=filename,
                        show_fig=False,
                        arr=images,
                        padding=1,
                        n_col=num_timesteps+2,
                        text=text,
                        text_color="red",
                        pixel_mult=max(1,128//image_size),
                        text_size=12)

def get_sample_names_from_info(info,newline=True):
    dataset_names = [d["dataset_name"] for d in info]
    datasets_i = [d["i"] for d in info]
    newline = "\n" if newline else ""
    sample_names = [f"{dataset_names[i]}/{newline}{datasets_i[i]}" for i in range(len(datasets_i))]
    return sample_names

def plot_grid(filename,
              output,
              max_images=32,
              remove_old=False,
              measure='ari',
              text_inside=False,
              sample_names=None,
              imagenet_stats=True,
              show_keys=dynamic_image_keys+["image","gt_bit","pred_bit","points"],
              ab_kw={}):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    k0 = "pred_bit"
    assert k0 in output.keys(), f"expected output to have key {k0}, found {output.keys()}"
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    map_dict = get_map_dict(imagenet_stats,ab_kw)
    for k in list(show_keys):
        if model_arg_is_trivial(output.get(k,None)):
            show_keys.remove(k)
            continue
        if isinstance(output[k],list):
            output[k] = unet_kwarg_to_tensor(output[k])
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
        assert k in map_dict.keys(), f"No plotting method found in map_dict for key {k}"
    has_classes = False
    if "classes" in output.keys():
        if output["classes"] is not None:
            has_classes = True
    num_votes = output[k0].shape[1]
    images = []
    text = []
    for k in show_keys:
        if k in output.keys():
            if k==k0:
                for j in range(num_votes):
                    images.extend([map_dict[k](output[k][i][j]) for i in range(bs)])
                    text1 = [k if text_inside else ""]+[""]*(bs-1)
                    #text2 = ([f"\n{output[measure][i][j]*100:0.1f}" for i in range(bs)]) if measure in output.keys() else (["" for i in range(bs)])
                    text2 = ["" for i in range(bs)]
                    if j==0:
                        text1[0] = f"{measure}="+text1[0]
                    text.extend([t1+t2 for t1,t2 in zip(text1,text2)])
            else:
                text.extend([k if text_inside else ""]+[""]*(bs-1))
                images.extend([map_dict[k](output[k][i]) for i in range(bs)])
                if k=="points" and text_inside:
                    pass#text[-1] += f"\nclass={output['classes'][i].item()}" if has_classes else ""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    jlc.montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=bs,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,64//image_size),
                    text_size=12)
    if sample_names is None:
        sample_names = ["s#"+str(i) for i in range(bs)]
    else:
        pass
    if not text_inside:
        idx = show_keys.index("pred_bit")
        show_keys2 = show_keys[:idx]+["pred_bit\n#"+str(i) for i in range(num_votes)]+show_keys[idx+1:]
        if has_classes:
            bottom_names = [f"class={output['classes'][i].item()}" for i in range(bs)]
        else:
            bottom_names = ["" for i in range(bs)]
        add_text_axis_to_image(filename,
                            save_filename=filename,
                            top=sample_names,
                            bottom=bottom_names,
                            left=show_keys2,
                            right=show_keys2)
    if remove_old:
        clean_up(filename)

def likelihood_image(x):
    return cm.inferno(replace_nan_inf(255*mean_dim0(x*2-1)).astype(np.uint8))[:,:,:3]

def get_zero_im(x):
    return np.zeros((x.shape[-2],x.shape[-1],3))+0.5

def bit_to_np(x,ab_kw):
    return ab_bit2prob(x.unsqueeze(0),**ab_kw)[0].permute(1,2,0).cpu().numpy()

def get_map_dict(imagenet_stats,ab_kw):
    imgn_s = imagenet_stats
    nb = ab_kw.get("num_bits",6)
    if nb==1 or nb==3:
        aboi = lambda x: normal_image(x,imagenet_stats=False)
    else:
        aboi = lambda x: mask_overlay_smooth(get_zero_im(x),bit_to_np(x,ab_kw),alpha_mask=1.0)
    aboi_split = lambda x: mask_overlay_smooth(normal_image(x[-3:],imgn_s),bit_to_np(x[:-3],ab_kw),alpha_mask=0.6)
    points_aboi = lambda x: aboi(pretty_point(x))
    err_im = lambda x: error_image(x)
    lik_im = lambda x: likelihood_image(x)
    normal_image2 = lambda x: normal_image(x,imagenet_stats=imagenet_stats)
    aboi_keys = "x_t,pred_bit,pred_eps,gt_bit,gt_eps,self_cond".split(",")
    map_dict = {"image": normal_image2,
                "err_x": err_im,
                "err_eps": err_im,
                "points": points_aboi,
                "likelihood": lik_im}
    for k in aboi_keys:
        map_dict[k] = aboi
    for k in dynamic_image_keys:
        map_dict[k] = aboi_split
    return map_dict

def plot_forward_pass(filename,
                      output,
                      metrics,
                      max_images=32,
                      remove_old=True,
                      text_inside=False,
                      sort_samples_by_t=True,
                      sample_names=None,
                      imagenet_stats=True,
                      show_keys=["image","gt_bit","pred_bit","err_x","likelihood","pred_eps","gt_eps","x_t",
                                 "self_cond","points"]+dynamic_image_keys,
                      ab_kw={}):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    k0 = "x_t" #key which determines batch size and image size
    bs = output[k0].shape[0]
    if bs>max_images:
        bs = max_images
    image_size = output[k0].shape[-1]

    map_dict = get_map_dict(imagenet_stats,ab_kw)
    for k in list(show_keys):
        if k in ["err_x","likelihood"]:
            continue
        if model_arg_is_trivial(output.get(k,None)):
            show_keys.remove(k)
            continue
        if isinstance(output[k],list):
            output[k] = unet_kwarg_to_tensor(output[k])
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert len(output[k])==bs, f"expected output[{k}].shape[0] to be {bs}, found {output[k].shape[0]}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        assert k in map_dict.keys(), f"No plotting method found in map_dict for key {k}"
        output[k] = output[k][:bs]
    mask = (output["loss_mask"].to(output["gt_bit"].device) if "loss_mask" in output.keys() else 1.0)
    output["err_x"] = (output["pred_bit"]-output["gt_bit"])*mask
    if "mse_x" not in metrics.keys():
        metrics["mse_x"] = torch.mean(output["err_x"]**2,dim=[1,2,3]).tolist()
    
    if sort_samples_by_t:
        perm = torch.argsort(output["t"]).tolist()
    else:
        perm = torch.arange(bs).tolist()
    images = []
    for k in show_keys:
        is_not_none = [output[k][i] is not None for i in range(bs)]
        assert all(is_not_none), f"expected output[{k}] to be not None for all samples, found {is_not_none}"
        images.append([map_dict[k](output[k][i]) for i in perm])
    text = sum([[k if text_inside else ""]+[""]*(bs-1) for k in show_keys],[])

    if text_inside:
        if "err_idx" in show_keys:
            err_idx = show_keys.index("err_x")*bs
            for i in perm:
                text[i+err_idx] += f"\nmse={metrics['mse_x'][i]:.3f}"
        if "x_t" in show_keys:
            x_t_idx = show_keys.index("x_t")*bs
            for i in perm:
                text[i+x_t_idx] += f"\nt={output['t'][i].item():.3f}"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    jlc.montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=bs,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,128//image_size),
                    text_size=12)
    if sample_names is None:
        sample_names = ["s#"+str(i) for i in perm]
    else:
        sample_names = [sample_names[i] for i in perm]
    if not text_inside:
        t_and_mse = [f"t={output['t'][i].item():.3f}\nmse={metrics['mse_x'][i]:.3f}" for i in perm]
        if not model_arg_is_trivial(output.get("num_labels",None)):
            nl_pretty = [int(v) if v is not None else None for v in output['num_labels']]
            t_and_mse = [f"#labels={nl_pretty[i]}\n"+t for i,t in zip(perm,t_and_mse)]
        add_text_axis_to_image(filename,
                            save_filename=filename,
                            top=sample_names,
                            bottom=t_and_mse,
                            left=show_keys,
                            right=show_keys)
    if remove_old:
        clean_up(filename)
        
def clean_up(filename,verbose=False):
    """
    Removes all files in the same folder as filename that have 
    the same name and format except for the last part of the name
    seperated by an underscore. For example, if filename is
    "folder_name/loss_plot_000000.png", then this function will
    remove all files in folder_name that have the same name and
    format except for the last part of the name seperated by an
    underscore. For example, "folder_name/loss_plot_000001.png"
    """
    assert "_" in Path(filename).name, f"filename {filename} does not contain an underscore, which is assumed for clean_up."
    safe_filename = Path(filename)
    glob_str = "_".join(safe_filename.name.split("_")[:-1])+"_*"+safe_filename.suffix
    old_filenames = list(safe_filename.parent.glob(bracket_glob_fix(glob_str)))
    for old_filename in old_filenames:
        if old_filename!=safe_filename:
            if verbose:
                print("\nRemoving old file:",old_filename,", based on from safe file: ",safe_filename.parent)
            os.remove(old_filename)

def index_dict_with_bool(d,bool_iterable,keys=[],num_recursions=1,
                         raise_error_on_wrong_bs=True,
                         ignore_weird_values=False,
                         raise_error_on_recursion_overflow=False):
    bool_kwargs = {"raise_error_on_wrong_bs": raise_error_on_wrong_bs,
              "ignore_weird_values": ignore_weird_values,
              "raise_error_on_recursion_overflow": raise_error_on_recursion_overflow}
    assert isinstance(d,dict), "expected d to be a dict"
    for k,v in d.items():
        if isinstance(v,dict):
            if num_recursions>0:
                d[k] = index_dict_with_bool(v,bool_iterable,keys=keys+[k],num_recursions=num_recursions-1,**bool_kwargs)
            elif raise_error_on_recursion_overflow:
                raise ValueError(f"Recursion overflow at key {k}")
        else:
            d[k] = index_w_bool(v,bool_iterable,keys+[k],**bool_kwargs)
    return d

def index_w_bool(item,bool_iterable,keys,raise_error_on_wrong_bs=True,ignore_weird_values=False,raise_error_on_recursion_overflow=None):
    bs = len(bool_iterable)
        
    if item is not None:
        bs2 = len(item)
        if bs2!=bs:
            if raise_error_on_wrong_bs:
                raise ValueError(f"Expected len(item)={bs}, found {bs2}. type(item)={type(item)}. Keys={keys}")
            else:
                item = None
        if torch.is_tensor(item):
            out = torch.stack([item[i] for i in range(bs) if bool_iterable[i]],dim=0)
        elif isinstance(item,np.ndarray):
            out = np.concatenate([item[i][None] for i in range(bs) if bool_iterable[i]],axis=0)
        elif isinstance(item,list):
            out = [item[i] for i in range(len(item)) if bool_iterable[i]]
        else:
            if ignore_weird_values:
                out = item
            else:
                raise ValueError(f"Expected item to be None, torch.Tensor, np.ndarray, or list, found {type(item)}")
    else:
        out = None
    return out

def plot_class_sims(info_list,dataset_name,num_show_neighbours=4,num_roots=4,longest_side_resize=256):
    idx_to_class_filename = f"./data/{dataset_name}/idx_to_class.json"
    idx_to_class = load_json_to_dict_list(idx_to_class_filename)[0]
    
    kwargs = {"show_border": 1,
        "border_color": "black",
        "alpha_mask": 0.5,
        "pixel_mult": 1,
        "set_lims": True,
        "fontsize": 12,
        "text_alpha": 1.0,
        "text_border_instead_of_background": True,
        }
    lsr = longest_side_resize
    image_overlays = []
    text = []
    for info in info_list[:num_roots]:
        image,label = load_raw_image_label(info,longest_side_resize=lsr)
        
        class_names = {i: idx_to_class[str(idx)] for i,idx in enumerate(info["classes"])}
        image_overlay = mask_overlay_smooth(image,label,class_names=class_names,**kwargs)
        image_overlays.append([image_overlay])
        text.append([f"{info['dataset_name']}/{info['i']}"])
        idx = info["conditioning"]["same_classes"]
        for i,idx_i in zip(range(num_show_neighbours),idx):
            info_i = info_list[idx_i]
            image_i,label_i = load_raw_image_label(info_i,longest_side_resize=lsr)
            class_names_i = {i: idx_to_class[str(idx)] for i,idx in enumerate(info_i["classes"])}
            text[-1].append(f"idx={idx_i}")
            image_overlay_i = mask_overlay_smooth(image_i,label_i,class_names=class_names_i,**kwargs)
            image_overlays[-1].append(image_overlay_i)
    jlc.montage(image_overlays,text=text,text_color="red")
    jlc.zoom()
    return image_overlays

def visualize_batch(batch,with_text_didx=False,class_text_size=12,with_class_names=True,
                    imagenet_inv=True,crop=True,alpha_mask=0.9,show_border=1,**kwargs):
    bs = len(batch[-1])
    
    images = [b["image"].permute(1,2,0).numpy() for b in batch[-1]]
    if imagenet_inv:
        images = [imagenet_preprocess(im,inv=True,dim=2) for im in images]
    labels = [b.permute(1,2,0).numpy() for b in batch[0]]
    if crop:
        for i in range(bs):
            b = batch[-1][i]
            h,w = sam_resize_index(*b["imshape"][:2],b["image"].shape[-1])
            images[i] = images[i][:h,:w]
            labels[i] = labels[i][:h,:w]

    didx = [f"{b['dataset_name']}/{b['i']}" for b in batch[-1]]
    if with_text_didx:
        kwargs["text"] = didx
    if with_class_names:
        class_names = [info["idx_to_class_name"] for info in batch[-1]]
    else:
        class_names = [None]*bs
    out = jlc.montage([
        mask_overlay_smooth(im,lab,fontsize=class_text_size,alpha_mask=alpha_mask,class_names=class_names.pop(0),show_border=show_border) 
        for im,lab in zip(images,labels)],**kwargs)
    return out

def visualize_dataset_with_labels(dataset_name="totseg",num_images=12,create_figure=False,
                    overlay_kwargs = {            
            "border_color": "black",
            "alpha_mask": 0.5,
            "pixel_mult": 1,
            "set_lims": True,
            "fontsize": 12,
            "text_alpha": 1.0,
            "text_border_instead_of_background": True,
            }
            ):
    image_overlays = []
    info_jsonl_path = f"./data/{dataset_name}/info.jsonl"
    idx_to_class = load_json_to_dict_list(f"./data/{dataset_name}/idx_to_class.json")[0]
    info_list = load_json_to_dict_list(info_jsonl_path)
    info_list = [{**item,"dataset_name": dataset_name} for item in info_list]
    for k in range(num_images):
        idx = np.random.randint(0,len(info_list))
        info = info_list[idx]
        image,label = load_raw_image_label(info,longest_side_resize=512)

        class_names = {i: idx_to_class[str(idx)] for i,idx in enumerate(info["classes"])}
        image_overlay = mask_overlay_smooth(image,label,class_names=class_names,**overlay_kwargs)
        image_overlays.append(image_overlay)
    jlc.montage(image_overlays,create_figure=create_figure)
    jlc.zoom()

def visualize_latent_vec(dim,cmap="inferno",
                         transparent_bg=True,
                         edgecolor_circle=[0.2,0.2,0.2],
                         edgecolor_vector=[0.5,0.5,0.5],
                         fillcolor_vector=[0.8,0.8,0.8],
                         y_figsize = 5,
                         linewidth=3,
                         markerwidth=1.0,
                         num_vectors=1,
                         delta_x=1,
                         r = 0.4,
                         shared_vals=False):
    #makes a figure of a latent vec as a visualization of circles in a rounded rectangle using matplotlib
    
    figsize=(2+(num_vectors-1)*delta_x,dim+1)
    divby = figsize[1]/y_figsize
    figsize=(figsize[0]/divby,figsize[1]/divby)
    s = markerwidth*30000*(dim**-2)
    t1,t2 = np.linspace(0,np.pi),np.linspace(np.pi,2*np.pi)
    vals = np.random.rand(dim)
    plt.figure(figsize=figsize)
    for i in range(num_vectors):
        if not shared_vals:
            vals = np.random.rand(dim)
        x_container = (r*np.cos(t1)+i*delta_x).tolist() + (r*np.cos(t2)+i*delta_x).tolist()
        y_container = (r*np.sin(t1)+dim-1).tolist() + (r*np.sin(t2)).tolist()
        x_container += [x_container[0]]
        y_container += [y_container[0]]
        plt.plot(x_container,y_container,linewidth=linewidth,color=edgecolor_vector)
        plt.fill_between(x_container,y_container,0,color=fillcolor_vector)
        plt.scatter(np.ones(dim)*i*delta_x,np.arange(dim),c=vals,cmap=cmap,edgecolor=edgecolor_circle,linewidth=linewidth,s=s)

    plt.xlim(-1,1+(num_vectors-1)*delta_x)
    plt.ylim(-1,dim)
    if transparent_bg:
        plt.gca().set_facecolor("none")
    plt.axis("off")
    plt.show()

default_overlay_kwargs = {            
            "border_color": "black",
            "alpha_mask": 0.5,
            "pixel_mult": 1,
            "set_lims": True,
            "fontsize": 10,
            "text_alpha": 1.0,
            "text_border_instead_of_background": True
            }

#unfortunately has to be in training.py due to circular imports  TODO: fix this
def visualize_cond_batch(dli,
                         num_images=4,
                         overlay_kwargs = default_overlay_kwargs,
                         montage_kwargs = {},
                         crop=False,
                         add_class_names=True,
                         add_text_axis=True,
                         overlay_fontsize=None,
                         axis_fontsize=None,
                         text_inside_fontsize=None,
                         didx_text_inside=True):
    if overlay_fontsize is not None:
        overlay_kwargs["fontsize"] = overlay_fontsize
    all_image_keys = ["image"]+dynamic_image_keys
    n_col = len(dynamic_image_keys)+1
    n_row = num_images
    image_key_to_index = {"image": 0, **{k: i+1 for i,k in enumerate(dynamic_image_keys)}}
    image_overlays = []
    x,info = next(dli)
    didx = [["" for _ in range(n_row)] for _ in range(n_col)]
    cond_dicts = []
    zero_image = None#np.zeros((image_size,image_size,3))
    for i in range(num_images):
        image_overlays.append([zero_image for _ in range(n_col)])
        image_dict = {"image": [x[i],info[i]["image"],info[i]], **info[i].get("cond",{})}
        class_names = info[i]["idx_to_class_name"]
        cond_dicts.append(info[i]["conditioning"])
        for k in all_image_keys:
            if k in image_dict.keys():
                label,image,info_i = image_dict[k]
                didx[image_key_to_index[k]][i] = didx_from_info(info_i)
                label,image = label.permute(1,2,0).numpy(),image.permute(1,2,0).numpy()
                image = imagenet_preprocess(image,inv=True,dim=2)
                image_overlay = mask_overlay_smooth(image,label,
                                                    class_names=class_names if add_class_names else None,
                                                    **overlay_kwargs)
                image_overlays[-1][image_key_to_index[k]] = image_overlay
    
    if crop:
        for i in range(len(image_overlays)):
            h,w = sam_resize_index(*info[i]["imshape"][:2],info[i]["image"].shape[-1])
            for j in range(len(image_overlays[i])):
                if image_overlays[i][j] is not None:
                    image_overlays[i][j] = image_overlays[i][j][:h,:w]
    
    montage_kwargs = {"return_im": True, "imshow": False, **montage_kwargs}
    #if didx_text_inside:
    #    montage_kwargs["text"] = didx
    montage_im = jlc.montage(image_overlays,n_col=n_col,n_row=n_row,**montage_kwargs)
    if didx_text_inside:
        text_kwargs = {"color":"red","fontsize": 10,"verticalalignment":"top","horizontalalignment":"left"}
        if text_inside_fontsize is not None:
            text_kwargs["fontsize"] = text_inside_fontsize
        montage_im = render_text_gridlike(montage_im,
                                          x_sizes=n_col,
                                          y_sizes=n_row,
                                          text_inside=didx,
                                          border_width_inside=text_kwargs["fontsize"]//10,
                                          text_kwargs=text_kwargs)
    if add_text_axis:
        xtick_kwargs = {"fontsize": axis_fontsize} if axis_fontsize is not None else {}
        left_text = ["\n".join([f"{k}: {len(v)}," for k,v in cond_dict.items()])[:-1] for cond_dict in cond_dicts]
        bottom_text = ["image"]+dynamic_image_keys
        bottom_text = [b+"\n" for b in bottom_text]
        montage_im = jlc.add_text_axis_to_image(montage_im,n_horz=n_col,n_vert=n_row,
                                                left=left_text,
                                                bottom=bottom_text,
                                                xtick_kwargs=xtick_kwargs)
    return montage_im

def visualize_tensor(x,
                     k=3,
                     color_method="RGB",
                     use_mask=1,
                     sim_func=None,
                     quantile_normalize=0.05):
    """
    Creates RGB images with shape (B,3,H,W) from a tensor with 
    shape (B,C,H,W) by the k-means clustering of the C channels.
    """
    if sim_func is None:
        sim_func = lambda vectors,center: np.dot(vectors,center)/(np.linalg.norm(vectors)*np.linalg.norm(center))
    assert color_method in ["RGB","nc","random"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        was_torch = 1
    else:
        was_torch = 0
    assert len(x.shape)==4, "Input tensor must have shape (B,C,H,W)"
    B,C,H,W = x.shape
    if color_method == "RGB":
        colors = np.array([[1,0,0],[0,1,0],[0,0,1]])
    elif color_method == "nc":
        colors = jlc.nc.large_colors/255
    elif color_method == "random":
        colors = np.random.rand(k,3)
    else:
        raise ValueError("Invalid color_method: "+color_method)
    #repeat colors if k is larger than the number of colors
    colors = np.tile(colors,(k//len(colors)+1,1))[:k]
    out_image = np.zeros((B,3,H,W))
    for i in range(B):
        vectors = x[i].reshape((C,H*W)).T
        kmeans = KMeans(n_clusters=k, random_state=0).fit(vectors)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        for j in range(k):
            if use_mask:
                mask = labels==j
            else:
                mask = 1
            #add corr wrt. center
            sim = sim_func(vectors,centers[j])
            corr_j = (sim*mask).reshape((H,W))
            if quantile_normalize>0:
                corr_j = jlc.quantile_normalize(corr_j,quantile_normalize)
            for c_i in range(3):
                out_image[i,c_i] += corr_j*colors[j,c_i]
    if quantile_normalize>0:
        out_image = jlc.quantile_normalize(out_image,quantile_normalize)
    if was_torch:
        out_image = torch.tensor(out_image)
    return out_image

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0:")
        raise ValueError("This test is not implemented")
    elif args.unit_test==1:
        print("UNIT TEST 1: make_loss_plot")
        raise ValueError("This test is not implemented")
        save_path = "/home/jloch/Desktop/diff/diffusion2/saves/2024-02-19-17-16-55-965746_sam[128]"
        make_loss_plot(save_path,11,remove_old=False)
    elif args.unit_test==2:
        print("UNIT TEST 2: collect_gen_table")
        collect_gen_table(gen_id="many_ema")
    elif args.unit_test==3:
        print("UNIT TEST 2: collect_gen_table")
        collect_gen_table(gen_id="eval2",name_match_strings="auto")
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()
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
from source.utils.argparse_utils import TieredParser
from source.utils.utils import (bracket_glob_fix, save_dict_list_to_json, 
                                imagenet_preprocess, get_likelihood, 
                                load_json_to_dict_list, wildcard_match)
import matplotlib
from tempfile import NamedTemporaryFile
import warnings
import cv2
import pandas as pd
import scipy.ndimage as nd
import copy
from matplotlib.patheffects import withStroke
from skimage.measure import find_contours
from datasets import load_raw_image_label

def collect_gen_table(gen_id_patterns="all_ade20k[ts_sweep]*",
                   model_id_patterns="*",
                   save=False,
                   return_table=True,
                   save_name="",
                   verbose=True,
                   sort_by_save_path=True,
                   make_pretty_table=True,
                   pretty_digit_limit=5,
                   search_gen_setups_instead=False,
                   include_mode="last",
                   record_from_sample_opts=[]):
    if isinstance(record_from_sample_opts,str):
        record_from_sample_opts = [record_from_sample_opts]
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
                if "gen_id" not in column_names:
                    continue
                if search_gen_setups_instead:
                    file_gen_ids = data[:,column_names.index("setup_name")].astype(str)
                else:
                    file_gen_ids = data[:,column_names.index("gen_id")].astype(str)
                match_idx = set()
                
                for idx,fgi in enumerate(file_gen_ids):
                    for gen_id_pattern in gen_id_patterns:
                        if wildcard_match(gen_id_pattern,fgi):
                            match_idx.add(idx)
                            break
                if len(match_idx)==0:
                    continue
                if include_mode=="last":
                    match_idx = [max(match_idx)]
                    if verbose and len(match_idx)>1:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                elif include_mode=="last_per_gen_id":
                    len_before = len(match_idx)
                    match_idx = list(match_idx)
                    match_idx = [max([i for i in match_idx if file_gen_ids[i]==file_gen_ids[j]]) for j in match_idx]
                    if verbose and len(match_idx)<len_before:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                else:
                    match_idx = list(match_idx)
                match_data_s = data[match_idx]
                if len(record_from_sample_opts)>0:
                    column_names += record_from_sample_opts
                    empty_array = np.array(["" for _ in range(match_data_s.shape[0])]).reshape(-1,1)
                    match_data_s = np.concatenate([match_data_s,empty_array],axis=1)
                    gen_id_list = match_data_s[:,column_names.index("gen_id")].tolist()
                    for mds_i,gen_id in enumerate(gen_id_list):
                        sample_opts = gen_id_dict[gen_id]
                        for rfso in record_from_sample_opts:
                            match_data_s[mds_i,column_names.index(rfso)] = sample_opts[rfso]
                table = pd.concat([table,pd.DataFrame(match_data_s,columns=column_names)],axis=0)
                save_paths.extend([v["save_path"] for _ in range(len(match_idx))])
    if table.shape[0]==0:
        if return_table:
            return table
        else:
            return
    table["save_path"] = save_paths
    if sort_by_save_path:
        table = table.sort_values(by=["save_path"])
    table = table.loc[:, (table != "").any(axis=0)]
    table_pd = table.copy()
    table = {k: table[k].tolist() for k in table.keys()}
    if make_pretty_table:
        buffer = 2
        pretty_table = ["" for _ in range(len(table["save_path"])+2)] 
        for k in table.keys():
            pretty_col = ["" for _ in range(len(table["save_path"])+2)]
            
            if table[k][0].replace(".","").isdigit() and table[k][0].find(".")>=0:
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

def make_loss_plot(save_path,step,save=True,show=False,fontsize=14,figsize_per_subplot=(8,2),remove_old=True):
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
            all_logging[["gen_","step_",""][i]+k] = data[:,j].astype(get_dtype(data[:,j]))
    if len(all_logging.keys())==0:
        return
    plot_columns = [["loss","vali_loss"],
                    ["mse_x","vali_mse_x"],
                    ["mse_eps","vali_mse_eps"],
                    ["iou","vali_iou"],
                    ["gen_hiou","gen_max_hiou"],#both vali and train
                    ["gen_ari","gen_max_ari"],#both vali and train
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

def get_mask(mask_vol,idx,onehot=False,onehot_dim=-1):
    if onehot:
        slice_idx = [slice(None) for _ in range(len(mask_vol.shape))]
        slice_idx[onehot_dim] = idx
        return np.expand_dims(mask_vol[tuple(slice_idx)],onehot_dim)
    else:
        return (mask_vol==idx).astype(float)

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]

def mask_overlay_smooth(image,
                        mask,
                        num_spatial_dims=2,
                        pallete=None,
                        pixel_mult=1,
                        class_names=None,
                        show_border=False,
                        border_color="darker",
                        alpha_mask=0.4,
                        dont_show_idx=[255],
                        fontsize=12,
                        text_color="class",
                        text_alpha=1.0,
                        text_border_instead_of_background=True,
                        set_lims=True):
    assert isinstance(image,np.ndarray)
    assert isinstance(mask,np.ndarray)
    assert len(image.shape)>=num_spatial_dims, "image must have at least num_spatial_dims dimensions"
    assert len(mask.shape)>=num_spatial_dims, "mask must have at least num_spatial_dims dimensions"
    assert image.shape[:num_spatial_dims]==mask.shape[:num_spatial_dims], "image and mask must have the same shape"
    if pallete is None:
        pallete = np.concatenate([np.array([[0,0,0]]),jlc.nc.largest_colors],axis=0)
    if image.dtype==np.uint8:
        was_uint8 = True
        image = image.astype(float)/255
    else:
        was_uint8 = False
    if len(mask.shape)==num_spatial_dims:
        onehot = False
        n = mask.max()+1
        uq = np.unique(mask).tolist()
        mask = np.expand_dims(mask,-1)
    else:
        assert len(mask.shape)==num_spatial_dims+1, "mask must have num_spatial_dims (with integers as classes) or num_spatial_dims+1 dimensions (with onehot encoding)"
        if mask.shape[num_spatial_dims]==1:
            onehot = False
            n = mask.max()+1
            uq = np.unique(mask).tolist()
        else:
            onehot = True
            n = mask.shape[num_spatial_dims]
            uq = np.arange(n).tolist()
    image_colored = image.copy()
    if len(image_colored.shape)==num_spatial_dims:
        image_colored = np.expand_dims(image_colored,-1)
    #make rgb
    if image_colored.shape[-1]==1:
        image_colored = np.repeat(image_colored,3,axis=-1)
    color_shape = tuple([1 for _ in range(num_spatial_dims)])+(3,)
    show_idx = [i for i in uq if (not i in dont_show_idx)]
    for i in show_idx:
        reshaped_color = pallete[i].reshape(color_shape)/255
        mask_coef = alpha_mask*get_mask(mask,i,onehot=onehot)
        image_coef = 1-mask_coef
        image_colored = image_colored*image_coef+reshaped_color*mask_coef
    if class_names is not None:
        assert isinstance(class_names,dict), "class_names must be a dictionary that maps class indices to class names"
        for i in uq:
            assert i in class_names.keys(), f"class_names must have a key for each class index, found i={i} not in class_names.keys()"
    assert isinstance(pixel_mult,int), "pixel_mult must be an integer"
    
    if pixel_mult>1:
        image_colored = cv2.resize(image_colored,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_NEAREST)
    
    image_colored = np.clip(image_colored,0,1)
    if show_border or (class_names is not None):
        image_colored = (image_colored*255).astype(np.uint8)
        h,w = image_colored.shape[:2]
        with RenderMatplotlibAxis(h,w,set_lims=set_lims) as ax:
            plt.imshow(image_colored)
            for i in show_idx:
                mask_coef = get_mask(mask,i,onehot=onehot)
                if pixel_mult>1:
                    mask_coef = cv2.resize(mask_coef,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_LANCZOS4)
                else:
                    mask_coef = mask_coef.reshape(h,w)
                if show_border:                    
                    curves = find_contours(mask_coef, 0.5)
                    if border_color=="darker":
                        border_color_i = darker_color(pallete[i]/255)
                    else:
                        border_color_i = border_color
                    k = 0
                    for curve in curves:
                        plt.plot(curve[:, 1], curve[:, 0], linewidth=1, color=border_color_i)
                        k += 1

                if class_names is not None:
                    t = class_names[i]
                    if len(t)>0:
                        dist = distance_transform_edt_border(mask_coef)
                        y,x = np.unravel_index(np.argmax(dist),dist.shape)
                        if text_color=="class":
                            text_color_i = pallete[i]/255
                        else:
                            text_color_i = text_color
                        text_kwargs = {"fontsize": int(fontsize*pixel_mult),
                                       "color": text_color_i,
                                       "alpha": text_alpha}
                        col_bg = "black" if np.mean(text_color_i)>0.5 else "white"             
                        t = plt.text(x,y,t,**text_kwargs)
                        if text_border_instead_of_background:
                            t.set_path_effects([withStroke(linewidth=3, foreground=col_bg)])
                        else:
                            t.set_bbox(dict(facecolor=col_bg, alpha=text_alpha, linewidth=0))
        image_colored = ax.image
    else:
        if was_uint8: 
            image_colored = (image_colored*255).astype(np.uint8)
    return image_colored

def analog_bits_on_image(x_bits,im,ab):
    assert isinstance(x_bits,torch.Tensor), "analog_bits_on_image expects a torch.Tensor"
    x_int = ab.bit2int(x_bits.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
    magnitude = np.minimum(torch.min(x_bits.abs(),0)[0].cpu().detach().numpy(),1)
    mask = np.zeros((im.shape[0],im.shape[1],2**ab.num_bits))
    for i in range(2**ab.num_bits):
        mask[:,:,i] = (x_int==i)*magnitude
    return mask_overlay_smooth(im,mask,alpha_mask=1.0)

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

def contains_key(key,dictionary,ignore_none=True):
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
    left = ["x","final pred_x","image"]*batch_size
    right = ["x_t","pred_x","pred_eps"]*batch_size
    t_vec = np.array(range(num_timesteps, 0, -1))/num_timesteps
    top = bottom = ["","t="]+[f"{t_vec[j]:.2f}" for j in range(num_timesteps)]
    top[1] = "points"
    add_text_axis_to_image(concat_filename,left=left,top=top,right=right,bottom=bottom,xtick_kwargs={"fontsize":20})
    if remove_children:
        for filename in filenames:
            os.remove(filename)
        if len(os.listdir(foldername))==0:
            os.rmdir(foldername)
    if remove_old:
        clean_up(concat_filename)

def normal_image(x,i,imagenet_stats=True): 
    if imagenet_stats:
        x2 = imagenet_preprocess(x.unsqueeze(0),inv=True)
        return x2.squeeze(0).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    else:
        return (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()

def plot_inter(foldername,sample_output,model_kwargs,ab,save_i_idx=None,plot_text=False,imagenet_stats=True):
    t = sample_output["inter"]["t"]
    num_timesteps = len(t)
    
    if save_i_idx is None:
        batch_size = sample_output["pred"].shape[0]
        save_i_idx = np.arange(batch_size)
    else:
        assert isinstance(save_i_idx,list), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx)}"
        assert len(save_i_idx)>0, f"expected save_i_idx to be a list of ints or bools, found {save_i_idx}"
        assert isinstance(save_i_idx[0],(bool,int)), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx[0])}"
        batch_size = len(save_i_idx)
        if isinstance(save_i_idx[0],bool):
            save_i_idx = np.arange(batch_size)[save_i_idx]
        batch_size = len(save_i_idx)
    image_size = sample_output["pred"].shape[-1]
    
    im = np.zeros((batch_size,image_size,image_size,3))+0.5
    aboi = lambda x,i: mask_overlay_smooth(im[i],ab.bit2prob(x.unsqueeze(0))[0].permute(1,2,0).cpu().numpy(),alpha_mask=1.0)
    points_aboi = lambda x,i: aboi(pretty_point(x),i)
    normal_image2 = lambda x,i: normal_image(x,i,imagenet_stats=imagenet_stats)
    map_dict = {"x_t": aboi,
                "pred": aboi,
                "pred_x": aboi,
                "x": aboi,
                "pred_eps": aboi,
                "image": normal_image2,
                "points": points_aboi}
    zero_image = np.zeros((image_size,image_size,3))
    has_classes = contains_key("classes",model_kwargs)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filenames = []
    num_inter_exists = len(glob.glob(bracket_glob_fix(os.path.join(foldername,"intermediate_*.png"))))
    for i in range(batch_size):
        ii = save_i_idx[i]
        images = [[map_dict["x"](sample_output["x"][ii],i)],
                  [map_dict["pred"](sample_output["pred"][ii],i)],
                  [map_dict["image"](model_kwargs["image"][ii],i)] if contains_key("image",model_kwargs) else [zero_image]]
        images[0].append(map_dict["points"](model_kwargs["points"][ii],i) if "points" in model_kwargs.keys() else zero_image)
        images[1].append(zero_image)
        images[2].append(zero_image)
        text = [["x"],["final pred_x"],["image"]]
        for k_i,k in enumerate(["x_t","pred_x","pred_eps"]):
            for j in range(num_timesteps):
                if k in sample_output["inter"].keys():
                    images[k_i].append(map_dict[k](sample_output["inter"][k][j][i],i))
                    text_j = ("    t=" if j==0 else "")+f"{t[j]:.2f}" if k_i==0 else ""
                    text[k_i].append(text_j)
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

def plot_grid(filename,output,ab,max_images=32,remove_old=False,measure='ari',text_inside=False,sample_names=None,imagenet_stats=True):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    show_keys = ["x_init","image","points","target_bit","pred_bit"]
    show_keys_new = []
    for k in show_keys:
        if k in output.keys():
            if output[k] is not None:
                show_keys_new.append(k)
    show_keys = show_keys_new
    k0 = show_keys[0]
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    for k in show_keys:
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
    
    im = np.zeros((bs,image_size,image_size,3))+0.5
    aboi = lambda x,i: mask_overlay_smooth(im[i],ab.bit2prob(x.unsqueeze(0))[0].permute(1,2,0).cpu().numpy(),alpha_mask=1.0)
    points_aboi = lambda x,i: aboi(pretty_point(x),i)
    normal_image2 = lambda x,i: normal_image(x,i,imagenet_stats=imagenet_stats)
    map_dict = {"target_bit": aboi,
                "pred_bit": aboi,
                "x_init": aboi,
                "image": normal_image2,
                "points": points_aboi}
    has_classes = False
    if "classes" in output.keys():
        if output["classes"] is not None:
            has_classes = True
        
    num_votes = output["pred_bit"].shape[1]
    images = []
    text = []
    for k in show_keys:
        if k in output.keys():
            if k=="pred_bit":
                for j in range(num_votes):
                    images.extend([map_dict[k](output[k][i][j],i) for i in range(bs)])
                    text1 = [k if text_inside else ""]+[""]*(bs-1)
                    text2 = ([f"\n{output[measure][i][j]*100:0.1f}" for i in range(bs)]) if measure in output.keys() else (["" for i in range(bs)])
                    if j==0:
                        text1[0] = f"{measure}="+text1[0]
                    text.extend([t1+t2 for t1,t2 in zip(text1,text2)])
            else:
                text.extend([k if text_inside else ""]+[""]*(bs-1))
                images.extend([map_dict[k](output[k][i],i) for i in range(bs)])
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
                               top=sample_names,
                               bottom=bottom_names,
                               left=show_keys2,
                               right=show_keys2)
    if remove_old:
        clean_up(filename)

def likelihood_image(x):
    return cm.inferno(replace_nan_inf(255*mean_dim0(x*2-1)).astype(np.uint8))[:,:,:3]

def plot_forward_pass(filename,output,metrics,ab,max_images=32,remove_old=True,text_inside=False,sort_samples_by_t=True,sample_names=None,imagenet_stats=True):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    show_keys = ["image","points","x_t","pred_x","x","err_x","pred_eps","eps","likelihood"]
    k0 = "x_t"
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    mask = (output["loss_mask"].to(output["x"].device) if "loss_mask" in output.keys() else 1.0)
    output["err_x"] = (output["pred_x"]-output["x"])*mask
    output["likelihood"] = get_likelihood(output["pred_x"],output["x"],mask,ab)[0]
    if "mse_x" not in metrics.keys():
        metrics["mse_x"] = torch.mean(output["err_x"]**2,dim=[1,2,3]).tolist()
    if "self_cond" in output.keys():
        if output["self_cond"] is not None:
            show_keys.append("self_cond")

    for k in show_keys:
        assert k in output.keys(), f"key {k} not in output.keys()"
        if output[k] is None:
            show_keys.remove(k)
            continue
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
        
    im = np.zeros((bs,image_size,image_size,3))+0.5
    aboi = lambda x,i: mask_overlay_smooth(im[i],ab.bit2prob(x.unsqueeze(0))[0].permute(1,2,0).cpu().numpy(),alpha_mask=1.0)
    points_aboi = lambda x,i: aboi(pretty_point(x),i)
    err_im = lambda x,i: error_image(x)
    lik_im = lambda x,i: likelihood_image(x)
    normal_image2 = lambda x,i: normal_image(x,i,imagenet_stats=imagenet_stats)
    map_dict = {"image": normal_image2,
                "x_t": aboi,
                "pred_x": aboi,
                "x": aboi,
                "err_x": err_im,
                "pred_eps": aboi,
                "eps": aboi,
                "self_cond": aboi,
                "points": points_aboi,
                "likelihood": lik_im}
    if sort_samples_by_t:
        perm = torch.argsort(output["t"]).tolist()
    else:
        perm = torch.arange(bs).tolist()
    images = []
    for k in show_keys:
        if k in output.keys():
            images.append([map_dict[k](output[k][i],i) for i in perm])
    text = sum([[k if text_inside else ""]+[""]*(bs-1) for k in show_keys],[])

    has_classes = False
    if "classes" in output.keys():
        if output["classes"] is not None:
            has_classes = True

    if text_inside:
        err_idx = show_keys.index("err_x")*bs
        for i in perm:
            text[i+err_idx] += f"\nmse={metrics['mse_x'][i]:.3f}"
        x_t_idx = show_keys.index("x_t")*bs
        for i in perm:
            text[i+x_t_idx] += f"\nt={output['t'][i].item():.3f}"
        if has_classes:
            points_idx = show_keys.index("points")*bs
            for i in perm:
                text[i+points_idx] += f"\nclass={output['classes'][i].item()}"
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
        if has_classes:
            t_and_mse = [f"class={output['classes'][i].item()}\n"+t for i,t in zip(perm,t_and_mse)]
        add_text_axis_to_image(filename,
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
    old_filenames = list(safe_filename.parent.glob(glob_str))
    for old_filename in old_filenames:
        if old_filename!=safe_filename:
            if verbose:
                print("\nRemoving old file:",old_filename,", based on from safe file: ",safe_filename.parent)
            os.remove(old_filename)

def get_matplotlib_color(color,num_channels=3):
    return render_axis_ticks(23,bg_color=color,xtick_kwargs={"labels": [" "]}, tick_params={"bottom": False})[12,12,:num_channels]

def darker_color(x,power=2,mult=0.5):
    assert isinstance(x,np.ndarray), "darker_color expects an np.ndarray"
    is_int_type = x.dtype in [np.uint8,np.uint16,np.int8,np.int16,np.int32,np.int64]
    if is_int_type:
        return np.round(255*darker_color(x/255,power=power,mult=mult)).astype(np.uint8)
    else:
        return np.clip(x**power*mult,0,1)

def render_axis_ticks(image_width=1000,
                      num_uniform_spaced=None,
                      bg_color="white",
                      xtick_kwargs={"labels": np.arange(5)},
                      tick_params={}):
    old_backend = matplotlib.rcParams['backend']
    old_dpi = matplotlib.rcParams['figure.dpi']
    dpi = 100
    if num_uniform_spaced is None:
        num_uniform_spaced = len(xtick_kwargs["labels"])
    n = num_uniform_spaced
     
    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.use('Agg')
    try:        
        fig = plt.figure(figsize=(image_width/dpi, 1e-15), facecolor=bg_color)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_facecolor(bg_color)
        ax.set_frame_on(False)
        ax.tick_params(**tick_params)
        fig.add_axes(ax)
        
        plt.yticks([])
        plt.xlim(0, n)
        x_pos = np.linspace(0.5,n-0.5,n)
        if not "ticks" in xtick_kwargs:
            xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        else:
            if xtick_kwargs["ticks"] is None:
                xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        plt.xticks(**xtick_kwargs)
        
        with warnings.catch_warnings(record=True) as caught_warnings:
            fig.show()

        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filename = temp_file.name
            fig.savefig(temp_filename, format='png', bbox_inches='tight', pad_inches=0)
        im = np.array(Image.open(temp_filename))
        if not im.shape[1]==image_width:
            #reshape with cv2 linear interpolation
            #warnings.warn("Image width is not as expected, likely due to too large text labels. Reshaping with cv2 linear interpolation.")
            im = cv2.resize(im, (image_width, im.shape[0]), interpolation=cv2.INTER_LINEAR)

        matplotlib.use(old_backend)
        matplotlib.rcParams['figure.dpi'] = old_dpi
    except:
        matplotlib.use(old_backend)
        matplotlib.rcParams['figure.dpi'] = old_dpi
        raise
    return im

def add_text_axis_to_image(filename,
                           new_filename = None,
                           n_horz=None,n_vert=None,
                           top=[],bottom=[],left=[],right=[],
                           bg_color="white",
                           xtick_kwargs={},
                           new_file=False,
                           buffer_pixels=4,
                           add_spaces=True,
                           save=True):
    """
    Function to take an image filename and add text to the top, 
    bottom, left, and right of the image. The text is rendered
    using matplotlib and up to 4 temporary files are created to
    render the text. The temporary files are removed after the
    original file has been modified.

    Parameters
    ----------
    filename : str
        The filename of the image to modify.
    n_horz : int, optional
        The number of horizontal text labels to add. The default
        is None (max(len(top),len(bottom))).
    n_vert : int, optional
        The number of vertical text labels to add. The default
        is None (max(len(left),len(right))).
    top : list, optional
        The list of strings to add to the top of the image. The
        default is [].
    bottom : list, optional
        The list of strings to add to the bottom of the image. The
        default is [].
    left : list, optional
        The list of strings to add to the left of the image. The
        default is [].
    right : list, optional
        The list of strings to add to the right of the image. The
        default is [].
    bg_color : list, optional
        The background color of the text. The default is [1,1,1]
        (white).
    xtick_kwargs : dict, optional
        The keyword arguments to pass to matplotlib.pyplot.xticks.
        The default is {}.        
    new_file : bool, optional
        If True, then a new file is created with the text axis
        added. If False, then the original file is modified. The
        default is False.
    buffer_pixels : int, optional
        The number of pixels to add as a buffer between the image
        and the text. The default is 4.
    add_spaces : bool, optional
        If True, then a space is added to the beginning and end of
        each label. The default is True.
    save : bool, optional
        If True, then the new file is saved. The default is True.
        
    Returns
    -------
    im2 : np.ndarray
        The modified image with the text axis added.
    """
    if n_horz is None:
        n_horz = max(len(top),len(bottom))
    if n_vert is None:
        n_vert = max(len(left),len(right))
    if isinstance(filename,np.ndarray):
        im = filename
    else:
        assert os.path.exists(filename), f"filename {filename} does not exist"
        im = np.array(Image.open(filename))
    h,w,c = im.shape
    xtick_kwargs_per_pos = {"top":    {"rotation": 0,  "labels": top},
                            "bottom": {"rotation": 0,  "labels": bottom},
                            "left":   {"rotation": 90, "labels": left},
                            "right":  {"rotation": 90, "labels": right}}
    tick_params_per_pos = {"top":    {"top":True, "labeltop":True, "bottom":False, "labelbottom":False},
                           "bottom": {},
                           "left":   {},
                           "right":  {"top":True, "labeltop":True, "bottom":False, "labelbottom":False}}
    pos_renders = {}
    pos_sizes = {}
    for pos in ["top","bottom","left","right"]:
        if len(xtick_kwargs_per_pos[pos]["labels"])==0:
            pos_renders[pos] = np.zeros((0,0,c),dtype=np.uint8)
            pos_sizes[pos] = 0
            continue
        xk = dict(**xtick_kwargs_per_pos[pos],**xtick_kwargs)
        if add_spaces:
            xk["labels"] = [" "+l+" " for l in xk["labels"]]
        if not "ticks" in xk.keys():
            n = n_horz if pos in ["top","bottom"] else n_vert

            if len(xk["labels"])<n:
                xk["labels"] += [""]*(n-len(xk["labels"]))
            elif len(xk["labels"])>n:
                xk["labels"] = xk["labels"][:n]
            else:
                assert len(xk["labels"])==n
        pos_renders[pos] = render_axis_ticks(image_width=w if pos in ["top","bottom"] else h,
                                             num_uniform_spaced=n,
                                             bg_color=bg_color,
                                             xtick_kwargs=xk,
                                             tick_params=tick_params_per_pos[pos])[:,:,:c]
        pos_sizes[pos] = pos_renders[pos].shape[0]
    bg_color_3d = get_matplotlib_color(bg_color,c)
    bp = buffer_pixels
    im2 = np.zeros((h+pos_sizes["top"]+pos_sizes["bottom"]+bp*2,
                    w+pos_sizes["left"]+pos_sizes["right"]+bp*2,
                    c),dtype=np.uint8)
    im2 += bg_color_3d
    im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,
        bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = im
    #make sure we have uint8
    pos_renders = {k: np.clip(v,0,255) for k,v in pos_renders.items()}
    for pos in ["top","bottom","left","right"]:
        if pos_renders[pos].size==0:
            continue
        if pos=="top":
            im2[bp:bp+pos_sizes["top"],bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["top"]
        elif pos=="bottom":
            im2[bp+pos_sizes["top"]+h:-bp,bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["bottom"]
        elif pos=="left":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp:bp+pos_sizes["left"]] = np.rot90(pos_renders["left"],k=3)
        elif pos=="right":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp+pos_sizes["left"]+w:-bp] = np.rot90(pos_renders["right"],k=3)
    if new_file:
        if new_filename is None:
            suffix = filename.split(".")[-1]
            new_filename = filename[:-len(suffix)-1]+"_w_text."+suffix
            for i in range(1000):
                if not os.path.exists(new_filename):
                    break
                new_filename = filename[:-len(suffix)-1]+"_w_text("+str(i)+")."+suffix
        filename = new_filename
    if save:
        Image.fromarray(im2).save(filename)
    return im2

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

class RenderMatplotlibAxis:
    def __init__(self, height, width=None, with_axis=False, set_lims=False, with_alpha=False, dpi=100):
        if (width is None) and isinstance(height, (tuple, list)):
            #height is a shape
            height,width = height[:2]
        elif (width is None) and isinstance(height, np.ndarray):
            #height is an image
            height,width = height.shape[:2]
        elif width is None:
            width = height
        self.with_alpha = with_alpha
        self.width = width
        self.height = height
        self.dpi = dpi
        self.old_backend = matplotlib.rcParams['backend']
        self.old_dpi = matplotlib.rcParams['figure.dpi']
        self.fig = None
        self.ax = None
        self._image = None
        self.with_axis = with_axis
        self.set_lims = set_lims

    @property
    def image(self):
        return self._image[:,:,:(3+int(self.with_alpha))]

    def __enter__(self):
        matplotlib.rcParams['figure.dpi'] = self.dpi
        matplotlib.use('Agg')
        figsize = (self.width/self.dpi, self.height/self.dpi)
        self.fig = plt.figure(figsize=figsize,dpi=self.dpi)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        if not self.with_axis:
            self.ax.set_frame_on(False)
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
        self.fig.add_axes(self.ax)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If no exception occurred, save the image to the _image property
            if self.set_lims:
                self.ax.set_xlim(-0.5, self.width-0.5)
                self.ax.set_ylim(self.height-0.5, -0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0, dpi=self.dpi)
            buf.seek(0)
            self._image = np.array(Image.open(buf))

        plt.close(self.fig)
        matplotlib.use(self.old_backend)
        matplotlib.rcParams['figure.dpi'] = self.old_dpi

def to_xy_anchor(anchor):
    anchor_equiv = [["NW","top left","north west","upper left","upper left corner"],
                    ["N","top","north","upper","top center","north center","upper center"],
                    ["NE","top right","north east","upper right","upper right corner"],
                    ["W","left","west","left center","west center","left middle","west middle","mid left"],
                    ["C","CC","center","middle","center middle", "middle center","center center","middle middle","mid"],
                    ["E","right","east","right center","east center","right middle","east middle","mid right"],
                    ["SW","bottom left","south west","lower left","lower left corner"],
                    ["S","bottom","south","lower","bottom center","south center","lower center"],
                    ["SE","bottom right","south east","lower right","lower right corner"]]
    anchor_to_coords = {"NW":(0,0),"N":(0.5,0),"NE":(1,0),
                        "W":(0,0.5),"C":(0.5,0.5),"E":(1,0.5),
                        "SW":(0,1),"S":(0.5,1),"SE":(1,1)}
    if isinstance(anchor,str):
        if anchor in sum(anchor_equiv,[]):
            for i,ae in enumerate(anchor_equiv):
                if anchor in ae:
                    out = anchor_to_coords[anchor_equiv[i][0]]
                    break
        else:
            raise ValueError(f"Unknown anchor string: {anchor}, Use on of {[x[0] for x in anchor_equiv]}")
    else:
        assert len(anchor)==2, f"If anchor is not a str then len(anchor) must be 2, found {len(anchor)}"
        out = anchor
    out = tuple([float(x) for x in out])
    return out

def item_to_rect_lists(item,n1,n2,fill_with_previous=True, fill_val=None):
    fill_val0 = copy.copy(fill_val)
    if fill_with_previous:
        assert fill_val is None, "expected fill_val to be None if fill_with_previous is True" 
    if not isinstance(item,list):
        out = [[item]]
    else:
        if len(item)==0:
            out = [[[] for _ in range(n2)] for _ in range(n1)]
        else:
            if not isinstance(item[0],list):
                out = [item]
            else:
                out = item
    assert len(out)<=n1, f"expected len(out) to be <= {n1}, found {len(out)}"
    if len(out)<n1:
        out.extend([[] for _ in range(n1-len(out))])
    for i in range(len(out)):
        assert len(out[i])<=n2, f"expected len(out[{i}]) to be <= {n2}, found {len(out[i])}"
        if len(out[i])==0 and fill_with_previous:
            all_until_i = sum(out[:i],[])
            all = sum(out,[])
            if len(all_until_i)>0:
                out_i = all_until_i[-1]
            elif len(all)>0:
                out_i = all[-1]
            else:
                out_i = fill_val0
            out[i].append(out_i)
        if len(out[i])<n2:
            if fill_with_previous:
                fill_val = out[i][-1]
            out[i].extend([fill_val for _ in range(n2-len(out[i]))])
    return out

def render_text_gridlike(image, x_sizes, y_sizes, 
                        text_inside=[],
                        transpose_text_inside=False,
                        text_pos_kwargs={},
                        pixel_mult=1, 
                        text_kwargs={"color":"red","fontsize": 20,"verticalalignment":"bottom","horizontalalignment":"left"},
                        anchor_image="NW",
                        border_width_inside=0):
    nx = len(x_sizes)
    ny = len(y_sizes)
    anchor_image = item_to_rect_lists(copy.deepcopy(anchor_image),nx,ny)
    anchor_image = [[to_xy_anchor(a) for a in row] for row in anchor_image]

    if pixel_mult>1:
        h,w = image.shape[:2]
        h,w = (np.round(w*pixel_mult).astype(int),
               np.round(h*pixel_mult).astype(int))
        image = cv2.resize(copy.copy(image),(w,h))
    
    #make sure text_inside is a list of lists, with correct lengths
    text_inside = copy.deepcopy(text_inside)
    assert len(text_inside)<=nx, f"expected len(text_inside) to be <= len(x_sizes), found {len(text_inside)}>{nx}"
    for i in range(len(text_inside)):
        assert len(text_inside[i])<=ny, f"expected len(text_inside[{i}]) to be <= len(y_sizes), found {len(text_inside[i])}>{ny}"
    if len(text_inside)<nx:
        text_inside.extend([[] for _ in range(nx-len(text_inside))])
    for i in range(len(text_inside)):
        if len(text_inside[i])<ny:
            text_inside[i].extend(["" for _ in range(ny-len(text_inside[i]))])

    if transpose_text_inside:
        text_inside = list(zip(*text_inside))
    h,w = image.shape[:2]
    x_sum = sum(x_sizes)
    y_sum = sum(y_sizes)
    if not x_sum==1.0:
        x_sizes = [x/x_sum*w for x in x_sizes]
    if not y_sum==1.0:
        y_sizes = [y/y_sum*h for y in y_sizes]
    with RenderMatplotlibAxis(w,h,set_lims=1) as renderer: #TODO (is this an error? w,h should be switched)
        plt.imshow(image/255)
        for xi in range(len(x_sizes)):
            for yi in range(len(y_sizes)):
                anc_x,anc_y = anchor_image[xi][yi]
                x = sum(x_sizes[:xi])+anc_x*x_sizes[xi]
                y = sum(y_sizes[:yi])+anc_y*y_sizes[yi]
                if len(text_inside[xi][yi])>0:
                    txt = plt.text(x,y,text_inside[xi][yi],**text_kwargs)
                    if border_width_inside>0:
                        txt.set_path_effects([withStroke(linewidth=border_width_inside, foreground='black')])
    rendered = renderer.image
    valid_pos = ["top","bottom","left","right"]
    if any([k in text_pos_kwargs for k in valid_pos]):
        text_pos_kwargs2 = {"n_horz": len(x_sizes), "n_vert": len(y_sizes),"save": False, "buffer_pixels": 0, "add_spaces": 0}
        text_pos_kwargs2.update(text_pos_kwargs)
        rendered = add_text_axis_to_image(rendered,**text_pos_kwargs2)
    return rendered

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


def visualize_dataset_with_labels(dataset_name="totseg",num_images=12,overlay_kwargs = {            
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
    jlc.montage(image_overlays)
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
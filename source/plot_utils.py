
import torch
import numpy as np
import matplotlib.pyplot as plt
import jlc
import nice_colors as nc
import matplotlib.cm as cm
import os
import glob
from pathlib import Path
from PIL import Image
import copy
import matplotlib
from tempfile import NamedTemporaryFile
import warnings
import cv2

def make_loss_plot(save_path,save=True,show=False,fontsize=14,figsize=(10,8),remove_old=True):
    filename = os.path.join(save_path,"progress.csv")
    with open(filename,"r") as f:
        column_names = f.readline()[:-1].split(",")
    #filename_steps = os.path.join(folder_name,"progress_steps.csv")
    data = np.genfromtxt(filename, delimiter=",")[1:]
    if len(data.shape)==1:
        data = np.expand_dims(data,0)
    data = data[~np.any(np.isinf(data),axis=1)]
    
    plot_columns = [["loss","vali_loss"],["mse_x","vali_mse_x"],["mse_eps","vali_mse_eps"]]
    n = len(plot_columns)
    
    x = data[:,column_names.index("step")]
    if np.any(np.isnan(x)):
        print("WARNING: nans found in steps in progress.csv, skipping plotting")
        return
    fig = plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(n,1,i+1)
        Y = []
        for name in plot_columns[i]:
            y = data[:,column_names.index(name)]
            nan_mask = np.isnan(y)
            y = y[~nan_mask]
            x_not_nan = x[~nan_mask]
            Y.append(y)
            fmt = "o-" if len(y)<25 else "-"
            plt.plot(x_not_nan,y,fmt,label=name)
        plt.legend()
        plt.grid()
        plt.xlim(0,x.max())
        Y = np.array(Y).flatten()
        plt.ylim(Y.min(),Y.max())
        plt.xlabel("steps")
    if show:
        plt.show()
    save_name = os.path.join(save_path, f"loss_plot_{int(x[-1]):06d}.png")
    if save:
        fig.savefig(save_name)
    if remove_old:
        clean_up(save_name)    
    plt.close(fig)

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

def mask_overlay_smooth(image,mask,
                 pallete=nc.largest_colors,
                 alpha_mask=0.4):
    assert isinstance(image,np.ndarray)
    assert isinstance(mask,np.ndarray)
    mask = np.atleast_3d(mask)
    if mask.dtype==np.uint8:
        mask = mask.astype(float)/255
    if isinstance(image,np.ndarray):
        if image.dtype==np.uint8:
            image = image.astype(float)/255
            input_is_uint8 = True
        else:
            input_is_uint8 = False
    else:
        image = np.array(image).astype(float)/255
        input_is_uint8 = True
    num_colors = mask.shape[2]
    image_colored = image.copy()
    for i in range(num_colors):
        reshaped_color = pallete[i].reshape(1,1,3)/255
        mask_coef = mask[:,:,i,None]*alpha_mask
        image_coef = 1-mask_coef
        image_colored = image_colored*image_coef+reshaped_color*mask_coef
    image_colored = np.clip(image_colored,0,1)
    if input_is_uint8: 
        image_colored = (image_colored*255).astype(np.uint8)
    return image_colored
    

def analog_bits_on_image(x,im,ab):
    assert isinstance(x,torch.Tensor), "analog_bits_to_image expects a torch.Tensor"
    x = ab.bit2int(x.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
    return mask_overlay_smooth(im,x)

def mean_dim0(x):
    assert isinstance(x,torch.Tensor), "mean_dim2 expects a torch.Tensor"
    return (x*0.5+0.5).clamp(0,1).mean(0).cpu().detach().numpy()

def error_image(x):
    return cm.RdBu((mean_dim0(x)*255).astype(np.uint8))[:,:,:3]

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
    top = bottom = ["t="]+[f"{t_vec[j]:.2f}" for j in range(num_timesteps)]
    add_text_axis_to_image(concat_filename,left=left,top=top,right=right,bottom=bottom,xtick_kwargs={"fontsize":20})
    if remove_children:
        for filename in filenames:
            os.remove(filename)
    if remove_old:
        clean_up(concat_filename)

def plot_inter(foldername,sample_output,model_kwargs,ab,save_i_idx=None,plot_text=False):
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
    nb_3 = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    aboi = lambda x,i: analog_bits_on_image(x,im[i],ab)
    not_nb3 = not ab.num_bits==3
    
    normal_image = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    
    map_dict = {"x_t": aboi if not_nb3 else nb_3,
                "pred": aboi if not_nb3 else nb_3,
                "pred_x": aboi if not_nb3 else nb_3,
                "x": aboi if not_nb3 else nb_3,
                "pred_eps": (lambda x,i: mean_dim0(x)) if not_nb3 else nb_3,
                "image": normal_image}
    zero_image = np.zeros((image_size,image_size,3))
    
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filenames = []
    num_inter_exists = len(glob.glob(os.path.join(foldername,"intermediate_*.png")))
    for i in range(batch_size):
        ii = save_i_idx[i]
        images = [[map_dict["x"](sample_output["x"][ii],i)],
                  [map_dict["pred"](sample_output["pred"][ii],i)],
                  [map_dict["image"](model_kwargs["image"][ii],i)] if contains_key("image",model_kwargs) else [zero_image]]

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
        text = sum(text,[]) if plot_text else []
        jlc.montage_save(save_name=filename,
                        show_fig=False,
                        arr=images,
                        padding=1,
                        n_col=num_timesteps+1,
                        text=text,
                        text_color="red",
                        pixel_mult=max(1,128//image_size),
                        text_size=12)
    

def plot_grid(filename,output,ab,max_images=32,remove_old=False,measure='iou',text_inside=False):
    show_keys = ["x_init","target_bit","pred_bit"]
    if "self_cond" in output.keys():
        if output["self_cond"] is not None:
            show_keys.append("self_cond")
    k0 = show_keys[0]
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    #output["err_x"] = output["pred_bit"]-output["target_bit"]
    
    for k in show_keys:
        assert k in output.keys(), f"key {k} not in output.keys()"
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
        
    im = np.zeros((bs,image_size,image_size,3))+0.5
    nb_3 = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    aboi = lambda x,i: analog_bits_on_image(x,im[i],ab)
    aboi = aboi if (not ab.num_bits==3) else nb_3
    
    map_dict = {"target_bit": aboi,
                "pred_bit": aboi,
                "x_init": aboi,
                "self_cond": aboi}
    
    num_votes = output["pred_bit"].shape[1]
    images = []
    text = []
    for k in show_keys:
        if k in output.keys():
            if k=="pred_bit":
                for j in range(num_votes):
                    images.extend([map_dict[k](output[k][i][j],i) for i in range(bs)])
                    text1 = [k if text_inside else ""]+[""]*(bs-1)
                    text2 = [f"\n{measure}={output[measure][i][j]:.4f}" for i in range(bs)]
                    text.extend([t1+t2 for t1,t2 in zip(text1,text2)])
            else:
                text.extend([k if text_inside else ""]+[""]*(bs-1))
                images.extend([map_dict[k](output[k][i],i) for i in range(bs)])
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
    if not text_inside:
        sample_names = ["s#"+str(i) for i in range(bs)]
        idx = show_keys.index("pred_bit")
        show_keys2 = show_keys[:idx]+["pred_bit\n#"+str(i) for i in range(num_votes)]+show_keys[idx+1:]
        add_text_axis_to_image(filename,
                               top=sample_names,
                               bottom=sample_names,
                               left=show_keys2,
                               right=show_keys2)
    if remove_old:
        clean_up(filename)

def plot_forward_pass(filename,output,metrics,ab,max_images=32,remove_old=True,text_inside=False):
    show_keys = ["image","x_t","pred_x","x","err_x","pred_eps","eps"]
    k0 = show_keys[0]
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    output["err_x"] = output["pred_x"]-output["x"]
    if "mse_x" not in metrics.keys():
        metrics["mse_x"] = torch.mean(output["err_x"]**2,dim=[1,2,3]).tolist()
    if "self_cond" in output.keys():
        if output["self_cond"] is not None:
            show_keys.append("self_cond")

    for k in show_keys:
        assert k in output.keys(), f"key {k} not in output.keys()"
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
        
    im = np.zeros((bs,image_size,image_size,3))+0.5
    normal_image = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    nb_3 = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    aboi = lambda x,i: analog_bits_on_image(x,im[i],ab)
    aboi = aboi if (not ab.num_bits==3) else nb_3
    md0 = (lambda x,i: mean_dim0(x)) if (not ab.num_bits==3) else nb_3
    map_dict = {"image": normal_image,
                "x_t": aboi,
                "pred_x": aboi,
                "x": aboi,
                "err_x": lambda x,i: error_image(x),
                "pred_eps": md0,
                "eps": md0,
                "self_cond": aboi}
    
    images = []
    for k in show_keys:
        if k in output.keys():
            images.append([map_dict[k](output[k][i],i) for i in range(bs)])
    text = sum([[k if text_inside else ""]+[""]*(bs-1) for k in show_keys],[])
    err_idx = show_keys.index("err_x")*bs
    for i in range(bs):
        text[i+err_idx] += f"\nmse={metrics['mse_x'][i]:.4f}"
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
    if not text_inside:
        sample_names = ["s#"+str(i) for i in range(bs)]
        add_text_axis_to_image(filename,
                               top=sample_names,
                               bottom=sample_names,
                               left=show_keys,
                               right=show_keys)
    if remove_old:
        clean_up(filename)
        
def clean_up(filename):
    """
    Removes all files in the same folder as filename that have 
    the same name and format except for the last part of the name
    seperated by an underscore. For example, if filename is
    "folder_name/loss_plot_000000.png", then this function will
    remove all files in folder_name that have the same name and
    format except for the last part of the name seperated by an
    underscore. For example, "folder_name/loss_plot_000001.png"
    """
    safe_filename = Path(filename)
    glob_str = "_".join(safe_filename.name.split("_")[:-1])+"_*"+safe_filename.suffix
    old_filenames = list(safe_filename.parent.glob(glob_str))
    for old_filename in old_filenames:
        if old_filename!=safe_filename:
            os.remove(old_filename)

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
            warnings.warn("Image width is not as expected, likely due to too large text labels. Reshaping with cv2 linear interpolation.")
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
                           add_spaces=True):
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
    """
    if n_horz is None:
        n_horz = max(len(top),len(bottom))
    if n_vert is None:
        n_vert = max(len(left),len(right))
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
                                             tick_params=tick_params_per_pos[pos])
        pos_sizes[pos] = pos_renders[pos].shape[0]
    empty_render_to_get_bg_color = render_axis_ticks(23,bg_color=bg_color,xtick_kwargs={"labels": [" "]}, tick_params={"bottom": False})
    bg_color_3d = empty_render_to_get_bg_color[12,12,:c]
    bp = buffer_pixels
    im2 = np.zeros((h+pos_sizes["top"]+pos_sizes["bottom"]+bp*2,
                    w+pos_sizes["left"]+pos_sizes["right"]+bp*2,
                    c),dtype=np.uint8)
    im2 += bg_color_3d
    im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,
        bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = im
    #make sure we have uint8
    pos_renders = {k: np.clip(v,0,255) for k,v in pos_renders.items()}
    im2[bp:bp+pos_sizes["top"],bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["top"]
    im2[bp+pos_sizes["top"]+h:-bp,bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["bottom"]
    im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp:bp+pos_sizes["left"]] = np.rot90(pos_renders["left"],k=3)
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
    Image.fromarray(im2).save(filename)
    return im2

    

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: add_text_axis_to_image")
        test_filename = "./saves/test2/forward_pass_000010.png"
        add_text_axis_to_image(test_filename,
                               n_horz=4,
                               n_vert=7,
                               top=["abc","","abc"],
                               bottom=["bottom1","bottom2"],
                               left=["left1","left2"],
                               right=["right1","right2"],
                               new_file=True)
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()
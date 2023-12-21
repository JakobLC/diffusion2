
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_batch_metrics
import jlc
import nice_colors as nc
import matplotlib.cm as cm
import os
import glob

def make_loss_plot(save_path,save=True,show=False,fontsize = 14,figsize=(10,8),remove_old=False):
    filename = os.path.join(save_path,"progress.csv")
    #filename_steps = os.path.join(folder_name,"progress_steps.csv")
    data = np.genfromtxt(filename, delimiter=",")[1:]
    if len(data.shape)==1:
        data = np.expand_dims(data,0)
    data = data[~np.any(np.isinf(data),axis=1)]
    column_names = open(filename).readline().strip().split(",")
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
    if save:
        save_name = os.path.join(save_path, f"loss_plot_{int(x[-1]):06d}.png")
        fig.savefig(save_name)
    if remove_old:
        old_plot_files = glob.glob(os.path.join(save_path, f"loss_plot_*.png"))
        if len(old_plot_files)>0:
            for old_filename in old_plot_files:
                if old_filename!=save_name:
                    os.remove(old_filename)
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
    """x = (mean_dim0(x)*255).astype(np.uint8)
    print(x.mean(),x.min(),x.max())
    x = cm.RdBu(x)[:,:,:3]/255
    print(x.mean(),x.min(),x.max())
    assert 1<0"""
    return cm.RdBu((mean_dim0(x)*255).astype(np.uint8))[:,:,:3]/255

def plot_forward_pass(filename,output,metrics,ab,max_images=32,remove_old=False):
    bs = len(output["x"])
    image_size = output["x"].shape[2]
    im = np.zeros((image_size,image_size,3))+0.5
    if bs>max_images:
        bs = max_images
        for k,v in output.items():
            if isinstance(v,torch.Tensor):
                output[k] = v[:max_images]
    nb_3 = lambda x: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    aboi = lambda x: analog_bits_on_image(x,im,ab)
    not_nb3 = not ab.num_bits==3
    
    show_keys = ["x_t","pred_x","x","err_x","pred_eps","eps"]
    map_dict = {"x_t": aboi if not_nb3 else nb_3,
                "pred_x": aboi if not_nb3 else nb_3,
                "x": aboi if not_nb3 else nb_3,
                "err_x": error_image,
                "pred_eps": mean_dim0 if not_nb3 else nb_3,
                "eps": mean_dim0 if not_nb3 else nb_3}
    output["err_x"] = output["pred_x"]-output["x"]
    images = []
    for k in show_keys:
        if k in output.keys():
            images.append([map_dict[k](output[k][i]) for i in range(bs)])
    text = sum([[k]+[""]*(bs-1) for k in show_keys],[])
    err_idx = text.index("err_x")
    for i in range(bs):
        text[i+err_idx] += f"\nmse={metrics['mse_x'][i]:.4f}"
    jlc.montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=bs,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,128//image_size),
                    text_size=12)
    if remove_old:
        old_plot_files = glob.glob(filename.replace("progress.csv", f"forward_pass_*.png"))
        if len(old_plot_files)>0:
            for old_filename in old_plot_files:
                os.remove(old_filename)
    
    
"""    image_size = image.shape[2]
    batch_size = image.shape[0]
    if batch_size>max_images:
        batch_size = max_images
        image = image[:max_images]
        mask = mask[:max_images]
        for k,v in losses.items() or torch.is_tensor(v):
            if isinstance(v,list):
                losses[k] = v[:max_images]
                
    text = ["image","gt","eps","x_t","pred_xstart","pred_eps","err"]
    if "bbox" in losses.keys():
        text.append("bbox")
    if "points" in losses.keys():
        text.append("points")
    
    images = [image,mask]+[losses[k] for k in text[2:]]
    postprocess = lambda x: (x*0.5+0.5).clamp(0,1)
    images = [np_map(postprocess(im)) for im in images]
    images[6] = [cm.RdBu(err[:,:,0])[:,:,:3] for err in images[6]]
    images = sum([[im[i] for i in range(len(im))] for im in images],[])
    text = sum([[t]+[""]*(batch_size-1) for t in text],[])
    if "labels" in losses.keys() and label_to_dataset is not None:
        for i in range(batch_size):
            text[i] += "\n"+label_to_dataset[int(losses['labels'][i])]
    for i in range(batch_size):
        text[i+batch_size*3] += f"\nt={int(losses['t'][i]):d}"
    if losses["model_loss_domain"]=="eps":
        for i in range(batch_size):
            text[i+batch_size*5] += f"\nmse={losses['mse'][i].item():.4f}"
    elif losses["model_loss_domain"]=="xstart":
        for i in range(batch_size):
            text[i+batch_size*4] += f"\nmse={losses['mse'][i].item():.4f}"
    jlc.montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=batch_size,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,128//image_size),
                    text_size=12)"""
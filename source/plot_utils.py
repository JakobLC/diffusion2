
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_batch_metrics
import jlc
import nice_colors as nc
import matplotlib.cm as cm
import os
import glob
from pathlib import Path


def make_loss_plot(save_path,save=True,show=False,fontsize=14,figsize=(10,8),remove_old=True):
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
        clean_up(save_path)
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

def mean_dim0(x,*args):
    assert isinstance(x,torch.Tensor), "mean_dim2 expects a torch.Tensor"
    return (x*0.5+0.5).clamp(0,1).mean(0).cpu().detach().numpy()

def error_image(x,*args):
    return cm.RdBu((mean_dim0(x)*255).astype(np.uint8))[:,:,:3]



def plot_inter(filename,sample_output,show_idx,ab,remove_old=False):
    t = sample_output["inter"]["t"]    
    num_timesteps = len(t)
    image_size = sample_output["pred"].shape[-1]
    batch_size = sample_output["pred"].shape[0]
    if not os.path.exists(filename):
        os.makedirs(filename)
    im = np.zeros((batch_size,image_size,image_size,3))+0.5
    nb_3 = lambda x,i: (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    aboi = lambda x,i: analog_bits_on_image(x,im[i],ab)
    not_nb3 = not ab.num_bits==3
    
    show_keys = ["x","x_t","pred_x","pred_eps"]
    map_dict = {"x_t": aboi if not_nb3 else nb_3,
                "pred_x": aboi if not_nb3 else nb_3,
                "x": aboi if not_nb3 else nb_3,
                "pred_eps": mean_dim0 if not_nb3 else nb_3}
    zero_image = np.zeros((image_size,image_size,3))
    
    for i in range(batch_size):
        images = []
        text = []
        for k in show_keys[1:]:
            if k=="x_t":
                images.append(map_dict[k](sample_output["inter"]["x"][show_idx][i],i))
            else:
                images.append(zero_image)
            for j in range(num_timesteps):
                if k in sample_output["inter"].keys():
                    images.append(map_dict[k](sample_output["inter"][k][j][i],i))
                    if k=="x_t":
                        if j==0:
                            text.append(f"t={t[j]:.2f}")
                        else:
                            text.append(f"{t[j]:.2f}")
                    else:
                        text.append("")
        filename = os.path.join(filename,f"intermediate_{i:03d}.png")
        jlc.montage_save(save_name=filename,
                        show_fig=False,
                        arr=images,
                        padding=1,
                        n_col=num_timesteps+1,
                        text=text,
                        text_color="red",
                        pixel_mult=max(1,128//image_size),
                        text_size=12)
                        
    if remove_old:
        #clean_up(filename)
        raise NotImplementedError("clean_up not implemented for intermediate steps")    

def plot_grid(filename,output,ab,max_images=32,remove_old=False,measure='iou'):
    show_keys = ["x_init","target_bit","pred_bit"]
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
                "x_init": aboi}
    
    images = []
    text = []
    for k in show_keys:
        if k in output.keys():
            if k=="pred_bit":
                for j in range(output[k].shape[1]):
                    images.extend([map_dict[k](output[k][i][j],i) for i in range(bs)])
                    text.extend([f"\n{measure}={output[measure][i][j]:.4f}" for i in range(bs)])
            else:
                text.extend([k]+[""]*(bs-1))
                images.extend([map_dict[k](output[k][i],i) for i in range(bs)])
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
        clean_up(filename)

def plot_forward_pass(filename,output,metrics,ab,max_images=32,remove_old=True):
    show_keys = ["image","x_t","pred_x","x","err_x","pred_eps","eps"]
    k0 = show_keys[0]
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    output["err_x"] = output["pred_x"]-output["x"]
    
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
    md0 = mean_dim0 if (not ab.num_bits==3) else nb_3
    map_dict = {"image": normal_image,
                "x_t": aboi,
                "pred_x": aboi,
                "x": aboi,
                "err_x": error_image,
                "pred_eps": md0,
                "eps": md0}
    
    images = []
    for k in show_keys:
        if k in output.keys():
            images.append([map_dict[k](output[k][i],i) for i in range(bs)])
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
    
import torch
import numpy as np
import random
from pathlib import Path
import csv
import os
import json
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, pair_confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib
import jsonlines
import shutil
import datetime
from functools import partial
import re

def check_keys_are_same(list_of_dicts,verbose=True):
    assert isinstance(list_of_dicts,list), "list_of_dicts must be a list"
    keys = [sorted(list(d.keys())) for d in list_of_dicts]
    if len(keys)==0:
        return True
    else:
        uq_keys = set(sum(keys,[]))
        for k in uq_keys:
            keys_found_k = [int(k in d) for d in keys]
            if not all(keys_found_k):
                if verbose:
                    print(f"Key {k} not found in all dictionaries keys_found_k={keys_found_k}")
                return False
        return True
            
def format_relative_path(path):
    if path is None:
        return path
    return str(Path(path).resolve().relative_to(Path(".").resolve()))

def imagenet_preprocess(x,inv=False,dim=1,maxval=1.0):
    """Normalizes a torch tensor or numpy array with 
    the imagenet mean and std. Can also be used to
    invert the normalization. Assumes the input is
    in the range [0,1]."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    assert x.shape[dim]==3, f"x must have 3 channels in the specified dim={dim}, x.shape: {str(x.shape)}"
    shape = [1 for _ in range(len(x.shape))]
    shape[dim] = 3
    assert torch.is_tensor(x) or isinstance(x,np.ndarray), "x must be a torch tensor or numpy array"
    if torch.is_tensor(x):
        mean = torch.tensor(mean).to(x.device).reshape(shape)
        std = torch.tensor(std).to(x.device).reshape(shape)
    else:
        mean = np.array(mean).reshape(shape)
        std = np.array(std).reshape(shape)
    if inv:
        #y = (x-mean)/std <=> x = y*std + mean
        m = std*maxval
        b = mean*maxval
    else:
        #y = (x-mean)/std = x*1/std - mean/std
        m = 1/std/maxval
        b = -mean/std
    out = x*m+b
    if abs(255-maxval)<1e-6:
        if isinstance(x,np.ndarray):
            out = np.clip(out,0,255).astype(np.uint8)
        else:
            out = torch.clamp(out,0,255).to(torch.uint8)
    return out

class AlwaysReturnsFirstItemOnNext():
    def __init__(self,iterable):
        self.first_item = next(iterable)
        self.iterable = iterable
    def __iter__(self):
        return self
    def __next__(self):
        return self.first_item

def is_infinite_and_not_none(x):
    if x is None:
        return False
    else:
        return torch.isinf(x).any()

def save_dict_list_to_json(data_list, file_path, append=False):
    assert isinstance(file_path,str), "file_path must be a string"
    assert len(file_path)>=5, "File path must end with .json or .jsonl"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if not isinstance(data_list,list):
        data_list = [data_list]
    if file_path[-5:] == ".json":
        if append:
            try:
                existing_data = load_json_to_dict_list(file_path)
                combined_data = existing_data + data_list
            except FileNotFoundError:
                combined_data = data_list
        else:
            combined_data = data_list
        
        with open(file_path, 'w') as json_file:
            json.dump(combined_data, json_file, indent=4)
    elif file_path[-6:] == ".jsonl":
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)
    else:
        raise ValueError("File path must end with .json or .jsonl")

def load_json_to_dict_list(file_path):
    assert len(file_path)>=5, "File path must end with .json"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if file_path[-5:] == ".json":
        with open(file_path, 'r') as json_file:
            data_list = json.load(json_file)
    elif file_path[-6:] == ".jsonl":
        data_list = []
        with jsonlines.open(file_path) as reader:
            for line in reader:
                data_list.append(line)
    return data_list

def longest_common_substring(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    max_length = 0
    end_position = 0
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_position = i
            else:
                dp[i][j] = 0
    return str1[end_position - max_length: end_position]

def fancy_print_kvs(kvs, atmost_digits=5, s="#"):
        """prints kvs in a nice format like
         |#########|########|
         | key1    | value1 |
              ...
         | keyN    | valueN |
         |#########|########|
        """
        values_print = []
        keys_print = []
        for k,v in kvs.items():
            if isinstance(v,float):
                v = f"{v:.{atmost_digits}g}"
            else:
                v = str(v)
            values_print.append(v)
            keys_print.append(k) 
        max_key_len = max([len(k) for k in keys_print])
        max_value_len = max([len(v) for v in values_print])
        print_str = "\n"
        print_str += "|" + s*(max_key_len+2) + "|" + s*(max_value_len+2) + "|\n"
        for k,v in zip(keys_print,values_print):
            print_str += "| " + k + " "*(max_key_len-len(k)+1) + "| " + v + " "*(max_value_len-len(v)+1) + "|\n"
        print_str += "|" + s*(max_key_len+2) + "|" + s*(max_value_len+2) + "|\n"
        return print_str

class MatplotlibTempBackend():
    def __init__(self,backend):
        self.backend = backend
    def __enter__(self):
        self.old_backend = matplotlib.get_backend()
        matplotlib.use(self.backend)
    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self.old_backend)

def bracket_glob_fix(x):
    return "[[]".join([a.replace("]","[]]") for a in x.split("[")])

def get_all_metrics(output,ignore_idx=0,ab=None):
    assert isinstance(output,dict), "output must be an output dict"
    assert "pred_x" in output.keys(), "output must have a pred_x key"
    assert "x" in output.keys(), "output must have an x key"
    mask = output["loss_mask"] if "loss_mask" in output.keys() else None
    metrics = {**get_segment_metrics(output["pred_x"],output["x"],mask=mask,ignore_idx=ignore_idx,ab=ab),
               **get_mse_metrics(output)}
    metrics["likelihood"] = get_likelihood(output["pred_x"],output["x"],output["loss_mask"],ab)[1]
    return metrics

def get_mse_metrics(output):
    metrics = {}
    if ("pred_x" in output.keys()) and ("x" in output.keys()):
        metrics["mse_x"] = mse_loss(output["pred_x"],output["x"],output["loss_mask"]).tolist()
    if ("pred_eps" in output.keys()) and ("eps" in output.keys()):
        metrics["mse_eps"] = mse_loss(output["pred_eps"],output["eps"],output["loss_mask"]).tolist()
    return metrics

def get_likelihood(pred,target,mask,ab,outside_mask_fill_value=0.0,clamp=True):
    assert isinstance(pred,torch.Tensor), "pred must be a torch tensor"
    assert isinstance(target,torch.Tensor), "target must be a torch tensor"
    assert len(pred.shape)==len(target.shape), "pred and target must be 3D or 4D torch tensors. got pred.shape: "+str(pred.shape)+", target.shape: "+str(target.shape)
    if len(pred.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    else:
        was_single = False
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask = mask.to(pred.device)
    bs = pred.shape[0]
    likelihood_images = ab.likelihood(pred,target)
    if clamp:
        likelihood_images = likelihood_images.clamp(min=0.0,max=1.0)
    likelihood_images = likelihood_images*mask + outside_mask_fill_value*(1-mask)
    likelihood = []
    for i in range(bs):
        lh = likelihood_images[i][mask[i]>0].mean().item()
        likelihood.append(lh)
    if was_single:
        likelihood_images = likelihood_images[0]
    return likelihood_images, likelihood

def get_segment_metrics_np(pred,target,**kwargs):
    #simple wrapper for get_segment_metrics
    assert isinstance(pred,np.ndarray), "pred must be a numpy array"
    assert isinstance(target,np.ndarray), "target must be a numpy array"
    if len(pred.shape)==len(target.shape)==2:
        pred = pred.copy()[None]
        target = target.copy()[None]
    if "mask" in kwargs:
        if kwargs["mask"] is not None:
            assert isinstance(kwargs["mask"],np.ndarray), "mask must be a numpy array"
            kwargs["mask"] = torch.tensor(kwargs["mask"].copy())
    return get_segment_metrics(torch.tensor(pred),torch.tensor(target),**kwargs)

def get_segment_metrics(pred,target,mask=None,metrics=["iou","hiou","ari","mi"],ignore_idx=0,ab=None,reduce_to_mean=True):
    assert isinstance(pred,torch.Tensor), "pred must be a torch tensor"
    assert isinstance(target,torch.Tensor), "target must be a torch tensor"
    if len(pred.shape)==len(target.shape)==3:
        was_single = True
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    else:
        was_single = False
    if not pred.shape[1]==target.shape[1]==1:
        if ab is None:
            raise ValueError("ab must be specified if pred and target are not 1-channel. Use Analog bits to get a 1-channel output.")
        pred = ab.bit2int(pred)
        target = ab.bit2int(target)
    assert len(pred.shape)==len(target.shape)==4, "batched_metrics expects 3D or 4D torch tensors"
    bs = pred.shape[0]
    if not isinstance(metrics,list):
        metrics = [metrics]
    metric_dict = {"iou": standard_iou,
                   "hiou": hungarian_iou,
                   "ari": adjusted_rand_score_stable,
                   "mi": adjusted_mutual_info_score,}
    #has to be defined inline for ab to be implicitly passed


    #metric_dict = {k: handle_empty(v) for k,v in metric_dict.items()}
    out = {metric: [] for metric in metrics}
    for i in range(bs):
        pred_i,target_i = metric_preprocess(pred[i],target[i],mask=mask[i] if mask is not None else None)
        for metric in metrics:
            out[metric].append(metric_dict[metric](pred_i,target_i))
    if was_single:
        for metric in metrics:
            out[metric] = out[metric][0]
    if reduce_to_mean:
        for metric in metrics:
            out[metric] = np.mean(out[metric])
    return out


def adjusted_rand_score_stable(target,pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(target.astype(np.uint64),pred.astype(np.uint64))
    tn,fp,fn,tp = np.float64(tn),np.float64(fp),np.float64(fn),np.float64(tp)
    if fp==0 and fn==0:
        return 1.0
    else:
        return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

def handle_empty(metric_func):
    def wrapped(target,pred,*args,**kwargs):
        if len(target)==0 and len(pred)==0:
            return 1.0
        elif len(target)==0 or len(pred)==0:
            return 0.0
        else:
            return metric_func(target,pred,*args,**kwargs)
    return wrapped

def metric_preprocess(target,pred,mask=None,dtype=np.int64):
    assert isinstance(target,np.ndarray) or isinstance(target,torch.Tensor), "target must be a torch tensor or numpy array"
    assert isinstance(pred,np.ndarray) or isinstance(pred,torch.Tensor), "pred must be a torch tensor or numpy array"
    if isinstance(target,torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if mask is None:
        target = target.flatten()
        pred = pred.flatten()
    else:
        if isinstance(mask,torch.Tensor):
            mask = mask.cpu().detach().numpy()>0.5
        target = target[mask]
        pred = pred[mask]
    return target,pred

def extend_shorter_vector(vec1,vec2,fill_value=0):
    if len(vec1)<len(vec2):
        vec1 = np.concatenate([vec1,(fill_value*np.ones(len(vec2)-len(vec1))).astype(vec1.dtype)])
    elif len(vec2)<len(vec1):
        vec2 = np.concatenate([vec2,(fill_value*np.ones(len(vec1)-len(vec2))).astype(vec2.dtype)])
    return vec1,vec2

def hungarian_iou(target,pred,ignore_idx=0,return_assignment=False):
    if ignore_idx is None:
        ignore_idx = []
    if isinstance(ignore_idx,list):
        assert all([isinstance(idx,int) for idx in ignore_idx]), "ignore_idx must be None, int or list[int]"
    else:
        assert isinstance(ignore_idx,int), "ignore_idx must be None, int or list[int]"
        ignore_idx = [ignore_idx]
    
    uq_target,target,conf_rowsum = np.unique(target,return_counts=True,return_inverse=True)
    uq_pred,pred,conf_colsum = np.unique(pred,return_counts=True,return_inverse=True)
    conf_rowsum,conf_colsum = extend_shorter_vector(conf_rowsum,conf_colsum)
    uq_target,uq_pred = extend_shorter_vector(uq_target,uq_pred,fill_value=-1)
    
    conf_rowsum,conf_colsum = conf_rowsum[:,None],conf_colsum[None,:]
    intersection = confusion_matrix(target, pred)

    union = conf_rowsum + conf_colsum - intersection
    iou_hungarian_mat = intersection / union

    mask_pred = np.isin(uq_pred,ignore_idx)
    mask_target = np.isin(uq_target,ignore_idx)
    #handle edge cases
    if all(mask_pred) and all(mask_target):
        val = 1.0
        assign_pred = np.array([],dtype=int)
        assign_target = np.array([],dtype=int)
        iou_per_assignment = np.array([],dtype=float)
    elif all(mask_pred) or all(mask_target):
        val = 0.0
        assign_pred = np.array([],dtype=int)
        assign_target = np.array([],dtype=int)
        iou_per_assignment = np.array([],dtype=float)
    else:
        #force optimal assignment to match ignore_idx with ignore_idx
        iou_hungarian_mat[mask_target,:] = 0
        iou_hungarian_mat[:,mask_pred] = 0
        iou_hungarian_mat += mask_target[:,None]*mask_pred[None,:]

        assignment = linear_sum_assignment(iou_hungarian_mat, maximize=True)

        assign_target = uq_target[assignment[0]]
        assign_pred = uq_pred[assignment[1]]
        iou_per_assignment = iou_hungarian_mat[assignment[0],assignment[1]]
        
        #remove matches which have ignore_idx or dummy (-1) as both target and pred
        ignore_idx.append(-1)
        mask = np.logical_or(~np.isin(assign_pred,ignore_idx),~np.isin(assign_target,ignore_idx))
        assign_target,assign_pred,iou_per_assignment = assign_target[mask],assign_pred[mask], iou_per_assignment[mask]
        
        val = np.mean(iou_per_assignment)

    if return_assignment:
        return val, assign_target, assign_pred, iou_per_assignment
    else:
        return val
    
def standard_iou(target,pred,ignore_idx=0,reduce_classes=True):
    num_classes = max(target.max(),pred.max())+1
    if num_classes==1:
        return 1.0
    intersection = np.histogram(target[pred==target], bins=np.arange(num_classes + 1))[0]
    area_pred = np.histogram(pred, bins=np.arange(num_classes + 1))[0]
    area_target = np.histogram(target, bins=np.arange(num_classes + 1))[0]
    union = area_pred + area_target - intersection
    if ignore_idx is not None:
        union[ignore_idx] = 0
    iou = intersection[union>0] / union[union>0]
    if reduce_classes:
        iou = np.mean(iou)
    return iou

def get_save_name_str(setup_name,gen_id,step):
    if gen_id=="":
        return f"{setup_name}_{step:06d}"
    else:
        return f"{setup_name}_{gen_id}_{step:06d}"

def mse_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    return torch.sum(loss_mask*(pred_x-x)**2, dim=non_batch_dims)

def ce1_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    likelihood = torch.prod(1-0.5*torch.abs(pred_x-x),axis=1,keepdims=True)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_loss(pred_x, x, loss_mask=None, batch_dim=0):
    """crossentropy loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(pred_x.device)
    likelihood = 1-0.5*torch.abs(pred_x-x)
    return -torch.sum(loss_mask*torch.log(likelihood), dim=non_batch_dims)

def ce2_logits_loss(logits, x, loss_mask=None, batch_dim=0):
    """BCEWithLogits loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    if loss_mask is None:
        loss_mask = torch.ones_like(x)*(1/torch.numel(x[0])).to(pred_x.device)
    else:
        div = torch.sum(loss_mask,dim=non_batch_dims,keepdim=True)+1e-14
        loss_mask = (loss_mask/div).to(logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    return torch.mean(bce(logits, (x.clone()>0.0).float(), reduction="none")*loss_mask, dim=non_batch_dims)

def load_state_dict_loose(model_arch,state_dict,allow_diff_size=True,verbose=False):
    arch_state_dict = model_arch.state_dict()
    load_info = {"arch_not_sd": [],"sd_not_arch": [],"match_same_size": [], "match_diff_size": []}
    sd_keys = list(state_dict.keys())
    for name, W in arch_state_dict.items():
        if name in sd_keys:
            sd_keys.remove(name)
            s1 = np.array(state_dict[name].shape)
            s2 = np.array(W.shape)
            l1 = len(s1)
            l2 = len(s2)
            l_max = max(l1,l2)
            if l1<l_max:
                s1 = np.concatenate((s1,np.ones(l_max-l1,dtype=int)))
            if l2<l_max:
                s2 = np.concatenate((s2,np.ones(l_max-l2,dtype=int)))
                
            if all(s1==s2):
                load_info["match_same_size"].append(name)
                arch_state_dict[name] = state_dict[name]
            else:
                if verbose:
                    m = ". Matching." if allow_diff_size else ". Ignoring."
                    print("Param. "+name+" found with sizes: "+str(list(s1[0:l1]))
                                                      +" and "+str(list(s2[0:l2]))+m)
                if allow_diff_size:
                    s = [min(i_s1,i_s2) for i_s1,i_s2 in zip(list(s1),list(s2))]
                    idx1 = [slice(None,s[i],None) for i in range(l2)]
                    idx2 = tuple([slice(None,s[i],None) for i in range(l2)])
                    
                    if l1>l2:
                        idx1 += [0 for _ in range(l1-l2)]
                    idx1 = tuple(idx1)
                    tmp = state_dict[name][idx1]
                    arch_state_dict[name][idx2] = tmp
                load_info["match_diff_size"].append(name)
        else:
            load_info["arch_not_sd"].append(name)
    for name in sd_keys:
        load_info["sd_not_arch"].append(name)
    model_arch.load_state_dict(arch_state_dict)
    return model_arch, load_info




def dump_kvs(filename, kvs, sep=","):
    file_exists = os.path.isfile(filename)
    if file_exists:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=sep)
            old_headers = next(reader, [])
        new_headers = set(kvs.keys()) - set(old_headers)
        if new_headers:
            header_write = old_headers + sorted(new_headers)
            
            with open(filename, 'r', newline='') as file:
                reader = csv.reader(file, delimiter=sep)
                data = list(reader)
            
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=sep)
                writer.writerow(header_write)  # Write sorted headers
                #remove the old header
                data.pop(0)
                # Modify old lines to have empty values for the new columns
                for line in data:
                    line_dict = dict(zip(old_headers, line))
                    line_dict.update({col: "" for col in new_headers})
                    writer.writerow([line_dict[col] for col in header_write])
            
        else:
            header_write = old_headers
    else:
        # create a file with headers
        header_write = sorted(kvs.keys())
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=sep)
            writer.writerow(header_write)  # Write sorted headers
    
    # Write the key-value pairs to the file, taking into account that only some columns might be present
    kvs_write = {col: kvs[col] if col in kvs else "" for col in header_write}
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_write, delimiter=sep)
        writer.writerow(kvs_write)

def load_kvs(filename):
    #loads the key-value pairs in a file with formatting and returns them as a numpy array of objects
    column_names = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1)
    values = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=1)
    values = values.astype(object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            values[i,j] = formatter(values[i,j])
    return column_names, values

def formatter(s,order=[int,float,str]):
    if t=="":
        return float("nan")
    for t in order:
        try:
            return t(s)
        except ValueError:
            pass
    return s

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

class TemporarilyDeterministic:
    def __init__(self,seed=0,torch=True,numpy=True):
        self.seed = seed
        self.torch = torch
        self.numpy = numpy
    def __enter__(self):
        if self.seed is not None:
            if self.numpy:
                self.previous_seed = np.random.get_state()[1][0]
                np.random.seed(self.seed)
            if self.torch:
                self.previous_torch_seed = torch.get_rng_state()
                torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            if self.numpy:
                np.random.seed(self.previous_seed)
            if self.torch:
                torch.set_rng_state(self.previous_torch_seed)

def load_state_dict_loose(model_arch,state_dict,allow_diff_size=True,verbose=False):
    arch_state_dict = model_arch.state_dict()
    load_info = {"arch_not_sd": [],"sd_not_arch": [],"match_same_size": [], "match_diff_size": []}
    sd_keys = list(state_dict.keys())
    """print(sd_keys)
    print(state_dict["state"][0].keys())
    print([type(v) for v in state_dict["state"][0].values()])
    print(len(state_dict["param_groups"]))
    assert 0"""
    for name, W in arch_state_dict.items():
        if name in sd_keys:
            sd_keys.remove(name)
            s1 = np.array(state_dict[name].shape)
            s2 = np.array(W.shape)
            l1 = len(s1)
            l2 = len(s2)
            l_max = max(l1,l2)
            if l1<l_max:
                s1 = np.concatenate((s1,np.ones(l_max-l1,dtype=int)))
            if l2<l_max:
                s2 = np.concatenate((s2,np.ones(l_max-l2,dtype=int)))
                
            if all(s1==s2):
                load_info["match_same_size"].append(name)
                arch_state_dict[name] = state_dict[name]
            else:
                if verbose:
                    m = ". Matching." if allow_diff_size else ". Ignoring."
                    print("Param. "+name+" found with sizes: "+str(list(s1[0:l1]))
                                                      +" and "+str(list(s2[0:l2]))+m)
                if allow_diff_size:
                    s = [min(i_s1,i_s2) for i_s1,i_s2 in zip(list(s1),list(s2))]
                    idx1 = [slice(None,s[i],None) for i in range(l2)]
                    idx2 = tuple([slice(None,s[i],None) for i in range(l2)])
                    
                    if l1>l2:
                        idx1 += [0 for _ in range(l1-l2)]
                    idx1 = tuple(idx1)
                    tmp = state_dict[name][idx1]
                    arch_state_dict[name][idx2] = tmp
                load_info["match_diff_size"].append(name)
        else:
            load_info["arch_not_sd"].append(name)
    for name in sd_keys:
        load_info["sd_not_arch"].append(name)
    model_arch.load_state_dict(arch_state_dict)
    return model_arch, load_info

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    if seed < 0:
        seed = None
    if seed is None:
        np.random.seed()
        seed = np.random.randint(0, 2**16-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def mean_iou(results, gt_seg_maps, num_classes, ignore_index,
            label_map=dict(), reduce_zero_label=False):
    total_intersect, total_union, _, _ = np.zeros((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,)), np.zeros((num_classes,))
    
    for i in range(len(results)):
        pred_label, label = results[i], gt_seg_maps[i]

        if label_map:
            label[label == label_map[0]] = label_map[1]

        if reduce_zero_label:
            label[label == 0] = 255
            label -= 1
            label[label == 254] = 255

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
        area_union = area_pred_label + area_label - area_intersect

        total_intersect += area_intersect
        total_union += area_union

    iou = total_intersect / total_union
    all_acc = total_intersect.sum() / total_union.sum()

    return all_acc, iou

def wildcard_match(pattern, text):
    """
    Perform wildcard pattern matching.

    Parameters:
        pattern (str): The wildcard pattern to match against. '*' matches any character
                      zero or more times.
        text (str): The text to check for a match against the specified pattern.

    Returns:
        bool: True if the text matches the pattern, False otherwise."""
    pattern = re.escape(pattern)
    pattern = pattern.replace(r'\*', '.*')
    regex = re.compile(pattern)
    return bool(regex.search(text))

def get_time(verbosity=4,sep="-"):
    if verbosity==0:
        s = datetime.datetime.now().strftime('%m-%d')
    elif verbosity==1:
        s = datetime.datetime.now().strftime('%m-%d-%H-%M')
    elif verbosity==2:
        s = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
    elif verbosity==3:
        s = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    elif verbosity==4:
        s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    else:
        raise ValueError("Unknown verbosity level: "+str(verbosity))
    if sep!="-":
        s = s.replace("-",sep)
    return s

def nuke_saves_folder(dry_run=False, 
         ask_for_permission=True,
         minimum_save_iteration=1):
    """
    Removes folders for all training runs, under some conditions.
    
    Inputs:
    dry_run (bool): If True, does not remove anything.
    ask_for_permission (bool): If True, asks for permission before removing anything.
    minimum_save_iterations (int): Minimum number of saves for a run to be kept.
    """
    rm_str = "Removing (dry)" if dry_run else "Removing"
    saves_folder = Path("./saves")
    folders = saves_folder.glob("*/*/")
    folders_for_removal = []
    for folder_path in folders:
        if folder_path.is_dir():
            save_files = [x for x in os.listdir(str(folder_path)) if x.endswith(".pt")]
            max_ite = 0
            for save_file in save_files:
                ite_str = save_file.split("_")[-1].split(".")[0]
                if ite_str.isdigit():
                    ite = int(ite_str)
                    if ite>max_ite:
                        max_ite = ite
            save_iteration_is_good = max_ite>=minimum_save_iteration
            if not save_iteration_is_good:
                folders_for_removal.append(folder_path)
    if len(folders_for_removal)==0:
        print("No folders to remove.")
        return
    if ask_for_permission:
        print("The following folders will be removed:")
        for folder in folders_for_removal:
            print(folder)
        if dry_run:
            q_str = "Do you want to proceed (dry, does nothing) ? (y/n)"
        else:
            q_str = "Do you want to proceed? (y/n)"
        if input(q_str)!="y":
            return
    for folder in folders_for_removal:
        print(f"{rm_str} {folder}")
        if not dry_run:
            shutil.rmtree(folder)

def format_save_path(args):
    save_path = str(Path("./saves/") / f"ver-{args.model_version}" / f"{get_time(1)}_{args.model_id}")
    for k,v in args.origin.items():
        if v=="modified_args" and (k not in ["model_id","origin","model_name","save_path"]):
            save_path += f"_({k}={getattr(args,k)})"
    return save_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: nuke test")
        nuke_saves_folder()
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":nuke_saves_folder
import torch
import numpy as np
import random
from pathlib import Path
import csv
import os
import copy
import sys
import argparse
import json
from collections import OrderedDict
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, pair_confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib
from functools import partial
import jsonlines
import shutil

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

def get_model_name_from_written_args(filename):
    loaded = json.loads(Path(filename).read_text())
    return loaded["model_name"]

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
    metrics = {**get_segment_metrics(output["pred_x"],output["x"],ignore_idx=ignore_idx,ab=ab),
               **get_mse_metrics(output)}
    return metrics

def get_mse_metrics(output):
    metrics = {}
    if ("pred_x" in output.keys()) and ("x" in output.keys()):
        metrics["mse_x"] = mse_loss(output["pred_x"],output["x"]).tolist()
    if ("pred_eps" in output.keys()) and ("eps" in output.keys()):
        metrics["mse_eps"] = mse_loss(output["pred_eps"],output["eps"]).tolist()
    return metrics

def get_segment_metrics(pred,target,metrics=["iou","hiou","ari","mi"],ignore_idx=0,ab=None,reduce_to_mean=True):
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
                   "mi": adjusted_mutual_info_score}
    #metric_dict = {k: handle_empty(v) for k,v in metric_dict.items()}
    out = {metric: [] for metric in metrics}
    for i in range(bs):
        pred_i,target_i = metric_preprocess(pred[i],target[i])
        for metric in metrics:
            out[metric].append(metric_dict[metric](pred_i,target_i))
    if was_single:
        for metric in metrics:
            out[metric] = out[metric][0]
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

def metric_preprocess(target,pred,dtype=np.int64):
    assert isinstance(target,np.ndarray) or isinstance(target,torch.Tensor), "target must be a torch tensor or numpy array"
    assert isinstance(pred,np.ndarray) or isinstance(pred,torch.Tensor), "pred must be a torch tensor or numpy array"
    if isinstance(target,torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().detach().numpy()
    target = target.flatten()
    pred = pred.flatten()
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

def mse_loss(pred_x, x, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    return torch.mean((pred_x-x)**2, dim=non_batch_dims)

def load_defaults(idx=0,ordered_dict=False,return_deprecated_keys=False, filename="jsons/args_default.json"):
    default_path = Path(__file__).parent.parent/filename
    if ordered_dict:
        args_dicts = json.loads(default_path.read_text(), object_pairs_hook=OrderedDict)    
    else:
        args_dicts = json.loads(default_path.read_text())
    if return_deprecated_keys:
        return args_dicts["deprecated"].keys()
    args_dict = {}
    for k,v in args_dicts.items():
        if isinstance(v,dict):
            if k!="deprecated":
                for k2,v2 in v.items():
                    args_dict[k2] = v2[idx]
        else:
            args_dict[k] = v[idx]
    return args_dict


class SmartParser():
    def __init__(self,name="args",modify_name_str=None,key_to_type={}):
        if modify_name_str is None:
            modify_name_str = {"args": True,"sample_opts": False}[name]
        self.modify_name_str = modify_name_str
        self.name_str = {"args": "model_name","sample_opts": "gen_setup"}[name]
        self.filename_def   = "jsons/"+name+"_default.json"
        self.filename_model = "jsons/"+name+"_configs.json"
        self.defaults_func = partial(load_defaults,filename=self.filename_def)
        self.descriptions = self.defaults_func(idx=1)
        defaults = self.defaults_func()
        self.type_dict = {}
        self.parser = argparse.ArgumentParser()
        for k, v in defaults.items():
            v_hat = v
            if k in key_to_type.keys():
                t = key_to_type[k]
            else:
                t = self.get_type_from_default(v)
            if isinstance(v, str):
                if v.endswith(","):
                    v_hat = v[:-1]
            self.parser.add_argument(f"--{k}", 
                                     default=v_hat, 
                                     type=t, 
                                     help=self.get_description_from_key(k))
            self.type_dict[k] = t
            
    def parse_types(self, args):
        args_dict = {k: v if isinstance(v,list) else self.type_dict[k](v) for k,v in args.__dict__.items()}
        args = argparse.Namespace(**args_dict)
        return args
        
    def get_args(self,alt_parse_args=None,modified_args={}):
        if alt_parse_args is None:
            args = self.parser.parse_args()
            postprocess_args = sys.argv[1:]
        else:
            assert isinstance(alt_parse_args,list), f"alt_parse_args must be a list or None. alt_parse_args={alt_parse_args}"
            args = self.parser.parse_args(alt_parse_args)
            postprocess_args = alt_parse_args
        model_dicts = json.loads((Path(__file__).parent.parent/self.filename_model).read_text())
        args = model_specific_args(args,model_dicts,self.name_str)
        deprecated_keys = self.defaults_func(return_deprecated_keys=True)
        for k in args.__dict__.keys():
            if k in deprecated_keys:
                raise ValueError(f"key {k} is deprecated.")
        if len(postprocess_args)>0:
            for k,v in zip(postprocess_args[:-1],postprocess_args[1:]):
                if k.startswith("--") and not v.startswith("--"):
                    if k[2:] in args.__dict__.keys():
                        args.__dict__[k[2:]] = self.type_dict[k[2:]](v)
        args = self.parse_types(args)
        for k,v in modified_args.items():
            assert k in args.__dict__.keys(), f"key {k} not found in args.__dict__.keys()={args.__dict__.keys()}"
            assert not isinstance(v,list), f"list not supported in modified_args to avoid recursion."
            if isinstance(v,str):
                assert v.find(";")<0, f"semicolon not supported in modified_args to avoid recursion."
            args.__dict__[k] = v
        if any([isinstance(v,list) for v in args.__dict__.values()]):
            modified_args_list = []
            num_modified_args = 1
            for k,v in args.__dict__.items():
                if isinstance(v,list):
                    if len(v)>1:
                        num_modified_args *= len(v)
                        if num_modified_args>100:
                            raise ValueError(f"Too many modified args. num_modified_args={num_modified_args}")
                        if len(modified_args_list)==0:
                            modified_args_list.extend([{k: v2} for v2 in v])
                        else:
                            modified_args_list = [{**d, k: v2} for d in modified_args_list for v2 in v]
            
            if num_modified_args>1:
                for i in range(num_modified_args):
                    model_name_new = getattr(args,self.name_str)
                    for k,v in modified_args_list[i].items():
                        model_name_new += f"_({k}={v})"
                    if self.modify_name_str:
                        modified_args_list[i][self.name_str] = model_name_new
                args = modified_args_list
        return args
    
    def get_type_from_default(self, default_v):
        assert isinstance(default_v,(float,int,str,bool)), f"default_v={default_v} is not a valid type."
        if isinstance(default_v, str):
            assert default_v.find(";")<0, f"semicolon not supported in default arguments"
        t = list_wrap_type(str2bool if isinstance(default_v, bool) else type(default_v))
        return t
    
    def get_description_from_key(self, k):
        if k in self.descriptions.keys():
            return self.descriptions[k]
        else:
            return ""

def list_wrap_type(t):
    def list_wrap(x):
        if isinstance(x,str):
            if x.find(";")>=0:
                return [t(y) for y in x.split(";")]
            else:
                return t(x)
        else:
            return t(x)
    return list_wrap

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


def load_old_args(args_path,defaults=load_defaults()):
    if isinstance(args_path,str):
        args_path = Path(args_path)
    args = argparse.Namespace(**defaults)
    args_loaded = json.loads(args_path.read_text())
    for k,v in args_loaded.items():
        try:
            args.__dict__[k] = v
        except AttributeError:
            print(f"key {k} not found in defaults. Ignoring.")
    return args

def model_specific_args(args,model_dicts,name_str):
    model_name = getattr(args,name_str)

    if "+" in model_name:
        plus_names = model_name.split("+")[1:]
        model_name = model_name.split("+")[0]
    else:
        plus_names = []
    ver_names = []
    if ("[" in model_name) and ("]" in model_name):
        for _ in range(model_name.count("[")):
            idx0 = model_name.find("[")
            idx1 = model_name.find("]")
            assert idx0<idx1, f"{name_str}={model_name} has mismatched brackets."
            ver_names.append(model_name[idx0+1:idx1])
            model_name = model_name[:idx0] + model_name[idx1+1:]
        
    if not model_name in model_dicts.keys():
        raise ValueError(f"{name_str}={model_name} not found in model_dicts")
    for k,v in model_dicts[model_name].items():
        if k!="versions":
            args.__dict__[k] = v
    if len(ver_names)>0:
        assert "versions" in model_dicts[model_name].keys(), f"{name_str}={model_name} does not have versions."
        for k,v in model_dicts[model_name]["versions"].items():
            if k in ver_names:
                for k2,v2 in v.items():
                    args.__dict__[k2] = v2
    for mn in plus_names:
        assert "+"+mn in model_dicts.keys(), f"{name_str}={'+'+mn} not found in model_dicts."
        for k,v in model_dicts["+"+mn].items():
            args.__dict__[k] = v
    return args

def write_args(args, save_path, match_keys=True):
    if isinstance(save_path,str):
        if not save_path.endswith(".json"):
            save_path += ".json"
        save_path = Path(save_path)
    ref_args = load_defaults(idx=0,ordered_dict=True)
    args_dict = args.__dict__
    if match_keys:
        ref_to_save = all([k in args_dict.keys() for k in ref_args.keys()])
        save_to_ref = all([k in ref_args.keys() for k in args_dict.keys()])
        all_keys_are_there = ref_to_save and save_to_ref
        assert all_keys_are_there, f"args and ref_args do not have the same keys. mismatched keys: {[k for k in ref_args.keys() if not k in args_dict.keys()] + [k for k in args_dict.keys() if not k in ref_args.keys()]}"
        
    args_dict = {k:args_dict[k] for k in ref_args.keys()}
    save_path.write_text(json.dumps(args_dict,indent=4))

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, (int,float)):
        return bool(v)
    elif isinstance(v, str):
        if v.lower() in ["yes", "true", "t", "y", "1"]:
            return True
        elif v.lower() in ["no", "false", "f", "n", "0"]:
            return False
        else:
            raise argparse.ArgumentTypeError("Cannot convert string: {} to bool".format(v))
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


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

def num_of_params(model,print_numbers=True):
    """
    Prints and returns the number of paramters for a pytorch model.
    Args:
        model (torch.nn.module): Pytorch model which you want to extract the number of 
        trainable parameters from.
        print_numbers (bool, optional): Prints the number of parameters. Defaults to True.

    Returns:
        n_trainable (int): Number of trainable parameters.
        n_not_trainable (int): Number of not trainable parameters.
        n_total (int): Total parameters.
    """
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_not_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_total = n_trainable+n_not_trainable
    if print_numbers:
        s = ("The model has:"
            +"\n"+str(n_trainable)+" trainable parameters"
            +"\n"+str(n_not_trainable)+" untrainable parameters"
            +"\n"+str(n_total)+" total parameters")
        print(s)
    return n_trainable,n_not_trainable,n_total

import numpy as np

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

def nuke_saves_folder(dry_run=False, 
         ask_for_permission=True,
         minimum_save_iteration=1,
         minimum_date="2024-01-019-15-00-000000"):
    """
    Removes folders for all training runs, under some conditions.
    
    Inputs:
    dry_run (bool): If True, does not remove anything.
    ask_for_permission (bool): If True, asks for permission before removing anything.
    minimum_save_iterations (int): Minimum number of saves for a run to be kept.
    minimum_date (str): Minimum date for a run to be kept.
    """
    rm_str = "Removing (dry)" if dry_run else "Removing"
    minimum_date = [int(x) for x in minimum_date.split("-")]
    saves_folder = Path("./saves")
    folders_for_removal = []
    for folder in sorted(os.listdir(str(saves_folder))):
        folder_path = saves_folder/folder
        if folder_path.is_dir():
            name = Path(folder).name
            date = [int(x) for x in name.split("-")[:6]]
            date_is_good = True
            for min_d,d in zip(minimum_date,date):
                if d==min_d:
                    continue
                elif d>min_d:
                    break
                elif d<min_d:
                    date_is_good = False
                    break
            save_files = [x for x in os.listdir(str(folder_path)) if x.endswith(".pt")]
            max_ite = 0
            for save_file in save_files:
                ite_str = save_file.split("_")[-1].split(".")[0]
                if ite_str.isdigit():
                    ite = int(ite_str)
                    if ite>max_ite:
                        max_ite = ite
            save_iteration_is_good = max_ite>=minimum_save_iteration
            is_good = date_is_good and save_iteration_is_good
            if not is_good:
                folders_for_removal.append(folder_path)
    if len(folders_for_removal)==0:
        print("No folders to remove.")
        return
    if ask_for_permission:
        print("The following folders will be removed:")
        for folder in folders_for_removal:
            print(folder)
        if input("Do you want to proceed? (y/n)")!="y":
            return
    for folder in folders_for_removal:
        print(f"{rm_str} {folder}")
        if not dry_run:
            shutil.rmtree(folder)

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
        
if __name__=="__main__":
    main()
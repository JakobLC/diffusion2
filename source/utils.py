import torch
import numpy as np
import random
from pathlib import Path
import csv
import os
import copy
import argparse
import json
from collections import OrderedDict
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib

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

def get_segment_metrics(pred,target,metrics=["iou","hiou","ari","mi"],ignore_idx=0,ab=None):
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
                   "ari": adjusted_rand_score,
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

def handle_empty(metric_func):
    def wrapped(target,pred,*args,**kwargs):
        if len(target)==0 and len(pred)==0:
            return 1.0
        elif len(target)==0 or len(pred)==0:
            return 0.0
        else:
            return metric_func(target,pred,*args,**kwargs)
    return wrapped

def metric_preprocess(target,pred):
    assert isinstance(target,np.ndarray) or isinstance(target,torch.Tensor), "target must be a torch tensor or numpy array"
    assert isinstance(pred,np.ndarray) or isinstance(pred,torch.Tensor), "pred must be a torch tensor or numpy array"
    if isinstance(target,torch.Tensor):
        target = target.cpu().detach().numpy()
    if isinstance(pred,torch.Tensor):
        pred = pred.cpu().detach().numpy()
    target = target.flatten()
    pred = pred.flatten()
    return target,pred

def hungarian_iou(target,pred,ignore_idx=0,return_assignment=False):
    uq_target,target = np.unique(target,return_inverse=True)
    uq_pred,pred = np.unique(pred,return_inverse=True)
    intersection = confusion_matrix(target, pred)
    conf_rowsum = np.sum(intersection, axis=1, keepdims=True)
    conf_colsum = np.sum(intersection, axis=0, keepdims=True)
    union = conf_rowsum + conf_colsum - intersection
    iou_hungarian_mat = intersection / union
    assignment = linear_sum_assignment(iou_hungarian_mat, maximize=True)
    val = iou_hungarian_mat[assignment].sum()/min(len(uq_target),len(uq_pred))
    if return_assignment:
        return val, assignment
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

def mse_loss(pred_x, x, batch_dim=0):
    """mean squared error loss reduced over all dimensions except batch"""
    non_batch_dims = [i for i in range(len(x.shape)) if i!=batch_dim]
    return torch.mean((pred_x-x)**2, dim=non_batch_dims)

def model_and_diffusion_defaults(idx=0,ordered_dict=False):
    default_path = Path(__file__).parent/"args_def.json"
    if ordered_dict:
        args_dicts = json.loads(default_path.read_text(), object_pairs_hook=OrderedDict)    
    else:
        args_dicts = json.loads(default_path.read_text())
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
    def __init__(self,defaults_func=model_and_diffusion_defaults):
        self.parser = argparse.ArgumentParser()
        self.descriptions = model_and_diffusion_defaults(idx=1)
        defaults = defaults_func()
        self.type_dict = {}
        for k, v in defaults.items():
            v_hat = v
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
        
    def get_args(self,modified_args={},do_parse_args=True):
        if do_parse_args:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args([])
        args = model_specific_args(args)
        args = self.parse_types(args)
        for k,v in modified_args.items():
            assert k in args.__dict__.keys(), f"key {k} not found in args.__dict__.keys()={args.__dict__.keys()}"
            assert not isinstance(v,list), f"list not supported in modified_args to avoid recursion."
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
                    model_name_new = args.model_name
                    for k,v in modified_args_list[i].items():
                        model_name_new += f"_({k}={v})"
                    modified_args_list[i]["model_name"] = model_name_new
                args = args, modified_args_list
        return args
    
    def list_wrap_type(self,t):
        def list_wrap(x):
            if isinstance(x,str):
                if x.find(";")>=0:
                    return [t(y) for y in x.split(";")]
                else:
                    return t(x)
            else:
                return t(x)
        return list_wrap
    
    def get_type_from_default(self, default_v):
        assert isinstance(default_v,(float,int,str,bool)), f"default_v={default_v} is not a valid type."
        if isinstance(default_v, str):
            assert default_v.find(";")<0, f"semicolon not supported in default arguments"
        t = self.list_wrap_type(str2bool if isinstance(default_v, bool) else type(default_v))
        return t
    
    def get_description_from_key(self, k):
        if k in self.descriptions.keys():
            return self.descriptions[k]
        else:
            return ""
        
def model_specific_args(args):
    model_dicts = json.loads((Path(__file__).parent/"args_model.json").read_text())
    model_name = args.model_name
    if "+" in model_name:
        plus_names = model_name.split("+")[1:]
        model_name = model_name.split("+")[0]
    else:
        plus_names = []
    ver_names = []
    if ("[" in model_name) and ("]" in model_name):
        idx = 0
        for _ in range(model_name.count("[")):
            idx0 = model_name.find("[",idx)
            idx1 = model_name.find("]",idx)
            assert idx0<idx1, f"model_name={model_name} has mismatched brackets."
            ver_names.append(model_name[idx0+1:idx1])
            model_name = model_name[:idx0] + model_name[idx1+1:]
            idx = idx1+1
        
    if not model_name in model_dicts.keys():
        raise ValueError(f"model_name={model_name} not found in model_dicts")
    for k,v in model_dicts[model_name].items():
        if k!="versions":
            args.__dict__[k] = v
    if len(ver_names)>0:
        assert "versions" in model_dicts[model_name].keys(), f"model_name={model_name} does not have versions."
        for k,v in model_dicts[model_name]["versions"].items():
            if k in ver_names:
                for k2,v2 in v.items():
                    args.__dict__[k2] = v2
    for mn in plus_names:
        assert "+"+mn in model_dicts.keys(), f"model_name={'+'+mn} not found in model_dicts."
        for k,v in model_dicts["+"+mn].items():
            args.__dict__[k] = v
    return args

def write_args(args, save_path, match_keys=True):
    if isinstance(save_path,str):
        if not save_path.endswith(".json"):
            save_path += ".json"
        save_path = Path(save_path)
    ref_args = model_and_diffusion_defaults(idx=0,ordered_dict=True)
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




def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: test measures")
        target = torch.randint(0,8,size=(1,1,1,100))
        metrics = ["iou","hiou","ari","mi"]
        metrics_per_t = {k: [] for k in metrics}
        err_ratio = list(range(100))
        for t in err_ratio:
            
            pred = torch.randint_like(target,0,8)
            pred[:,:,:,:t] = target[:,:,:,:t]
            m = batched_metrics(pred.clone(),target.clone(),metrics=metrics,ignore_idx=0)
            for k in metrics:
                metrics_per_t[k].append(m[k])
        import matplotlib.pyplot as plt, jlc
        plt.figure()
        for k,v in metrics_per_t.items():
            plt.plot(err_ratio,v,label=k)
        plt.legend()
        jlc.zoom()
        plt.show()

    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
        
if __name__=="__main__":
    main()
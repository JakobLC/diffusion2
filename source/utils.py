import torch
import numpy as np
import random
from pathlib import Path
import csv
import os

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
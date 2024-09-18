import torch
import numpy as np
import random
from pathlib import Path
import csv
import os
import json
import jsonlines
import shutil
import datetime
import re
import scipy.ndimage as nd
import copy
from jlc import shaprint
import warnings

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
    if not isinstance(data_list,list):
        data_list = [data_list]
    
    if file_path.endswith(".json"):
        loaded_data = []
        if append:
            if Path(file_path).exists():
                loaded_data = load_json_to_dict_list(file_path)
                if not isinstance(loaded_data,list):
                    loaded_data = [loaded_data]
        data_list = loaded_data + data_list
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, indent=4)
    else:
        assert file_path.endswith(".jsonl"), "File path must end with .json or .jsonl"
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)

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


def bracket_glob_fix(x):
    return "[[]".join([a.replace("]","[]]") for a in x.split("[")])

def get_save_name_str(setup_name,gen_id,step):
    if gen_id=="":
        return f"{setup_name}_{step:06d}"
    else:
        return f"{setup_name}_{gen_id}_{step:06d}"

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

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    if seed is not None:
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

def wildcard_match(pattern, text, warning_on_star_in_text=True):
    """
    Perform wildcard pattern matching.

    Parameters:
        pattern (str): The wildcard pattern to match against. '*' matches any character
                      zero or more times.
        text (str): The text to check for a match against the specified pattern.

    Returns:
        bool: True if the text matches the pattern, False otherwise."""
    if '*' in text and warning_on_star_in_text:
        warnings.warn("Wildcard pattern matching with '*' in text is not recommended.")
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
         minimum_save_iteration=1,
         keep_most_recent=True):
    """
    Removes folders for all training runs, under some conditions.
    
    Inputs:
    dry_run (bool): If True, does not remove anything.
    ask_for_permission (bool): If True, asks for permission before removing anything.
    minimum_save_iterations (int): Minimum number of saves for a run to be kept.
    keep_most_recent (bool): If True, keeps the most recent save in the most recent versions folder.
    """
    rm_str = "Removing (dry)" if dry_run else "Removing"
    saves_folder = Path("./saves")
    folders = sorted(list(saves_folder.glob("*/*/")))
    if keep_most_recent:
        folders = folders[:-1]
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

def mask_from_list_of_shapes(imshape,resize):
    assert isinstance(imshape,list), "imshape must be a list or tuple"
    if not isinstance(resize,list):
        assert isinstance(resize,int), "imsize must be an int or a list of ints"
        resize = [resize]*len(imshape)
    return [mask_from_imshape(sh,rs) for sh,rs in zip(imshape,resize)]

def mask_from_imshape(imshape,resize,num_dims=2):
    h,w = imshape[:2]
    new_h,new_w = sam_resize_index(h,w,resize=resize)
    mask = np.zeros((resize,resize),dtype=bool)
    mask[:new_h,:new_w] = True
    for _ in range(num_dims-2):
        mask = mask[None]
    return mask

def sam_resize_index(h,w,resize=64):
    if h>w:
        new_h = resize
        new_w = np.round(w/h*resize).astype(int)
    else:
        new_w = resize
        new_h = np.round(h/w*resize).astype(int)
    return new_h,new_w


def segmentation_gaussian_filter(seg,sigma=1,skip_index=[],skip_spatial=None,padding="constant"):
    assert len(seg.shape)==2 or (len(seg.shape)==3 and seg.shape[-1]==1), f"expected seg to be of shape (H,W) or (H,W,1), found {seg.shape}"
    assert seg.dtype==np.uint8, f"expected seg to be of type np.uint8, found {seg.dtype}"
    if skip_spatial is not None:
        assert skip_spatial.shape==seg.shape, f"skip_spatial.shape={skip_spatial.shape} != seg.shape={seg.shape}"
        ssf = lambda x: np.logical_and(x,np.logical_not(skip_spatial))
    else:
        ssf = lambda x: x
    uq = np.unique(seg)
    best_val = np.zeros_like(seg,dtype=float)
    best_idx = -np.ones_like(seg,dtype=int)
    for i in uq:
        if i in skip_index:
            continue
        val = nd.gaussian_filter(ssf(seg==i).astype(float),sigma=sigma,mode=padding)
        mask = val>best_val
        best_val[mask] = val[mask]
        best_idx[mask] = i
    if any(best_idx.flatten()<0):
        #set all pixels that were not assigned to the closest set pixel class
        _, idx = nd.distance_transform_edt(best_idx<0,return_indices=True)
        best_idx[best_idx<0] = best_idx[tuple(idx[:,best_idx<0])]
        #print("Warning: some pixels were not assigned to any class")
    best_idx = best_idx.astype(np.uint8)
    return best_idx

def segmentation_filter(seg,skip_index=[],skip_spatial=None,padding="constant",kernel=np.ones((3,3))):
    assert len(seg.shape)==2 or (len(seg.shape)==3 and seg.shape[-1]==1)
    assert seg.dtype==np.uint8
    if skip_spatial is not None:
        assert skip_spatial.shape==seg.shape, f"skip_spatial.shape={skip_spatial.shape} != seg.shape={seg.shape}"
        ssf = lambda x: np.logical_and(x,np.logical_not(skip_spatial))
    else:
        ssf = lambda x: x
    uq = np.unique(seg)
    best_val = np.zeros_like(seg,dtype=float)
    best_idx = -np.ones_like(seg,dtype=int)
    for i in uq:
        if i in skip_index:
            continue
        val = nd.convolve(ssf(seg==i).astype(float),kernel,mode=padding)
        mask = val>best_val
        best_val[mask] = val[mask]
        best_idx[mask] = i
    if any(best_idx.flatten()<0):
        #set all pixels that were not assigned to the closest set pixel class
        _, idx = nd.distance_transform_edt(best_idx<0,return_indices=True)
        best_idx[best_idx<0] = best_idx[tuple(idx[:,best_idx<0])]
        print("Warning: some pixels were not assigned to any class")
    best_idx = best_idx.astype(np.uint8)
    return best_idx

def mult_if_float(x,multiplier):
    if isinstance(x,float):
        return np.round(x*multiplier).astype(int)
    else:
        assert isinstance(x,int)
        return x*np.ones_like(multiplier)

def postprocess_batch(seg_tensor,seg_kwargs={},overwrite=False,keep_same_type=True,list_of_imshape=None):
    """
    Wrapper for postprocess_list_of_segs that handles many types of 
    batched inputs and returns the same type for the output.
    """
    expected_seg_tensor_msg = "Expected seg_tensor to be a torch.Tensor or np.ndarray or a list of torch.Tensor or np.ndarray"
    bs = len(seg_tensor)
    if list_of_imshape is not None:
        assert isinstance(list_of_imshape,list), "expected list_of_imshape to be a list"
        assert bs>0, "expected bs>0"
        assert len(list_of_imshape)==len(seg_tensor), f"expected len(list_of_imshape)={len(list_of_imshape)} to be equal to len(seg_tensor)={len(seg_tensor)}"
        assert all([isinstance(imshape,(tuple,list)) for imshape in list_of_imshape]), "expected all elements of list_of_imshape to be a tuple or list"
        assert all([len(imshape)>=2 for imshape in list_of_imshape]), "expected all elements of list_of_imshape to have length >=2"

    if torch.is_tensor(seg_tensor):
        input_mode = "torch"
        dtype = seg_tensor.dtype
        device = seg_tensor.device
        transform = lambda x: x.cpu().numpy()
    elif isinstance(seg_tensor,np.ndarray):
        input_mode = "np"
        transform = lambda x: x
    else:
        assert isinstance(seg_tensor,list), expected_seg_tensor_msg+", found "+str(type(seg_tensor))
        if torch.is_tensor(seg_tensor[0]):
            input_mode = "list_of_torch"
            dtype = seg_tensor[0].dtype
            device = seg_tensor[0].device
            transform = lambda x: x.cpu().numpy()
        else:
            assert isinstance(seg_tensor[0],np.ndarray), expected_seg_tensor_msg+", found "+str(type(seg_tensor))
            input_mode = "list_of_np"
            transform = lambda x: x
    num_dims = [len(im.shape) for im in seg_tensor]
    num_dims0 = num_dims[0]
    assert all([nd==num_dims0 for nd in num_dims]), "expected all images to have the same number of dimensions"
    dim_is_trivial = [all([im.shape[j]==1 for im in seg_tensor]) for j in range(num_dims0)]
    trivial_idx = [i for i in range(num_dims0) if dim_is_trivial[i]]
    dim_is_nontrivial = [not dit for dit in dim_is_trivial]
    assert sum(dim_is_nontrivial)==2, "expected exactly 2 non-trivial dimensions, found dim_is_nontrivial="+str(dim_is_nontrivial)
    d1,d2 = [i for i in range(num_dims0) if dim_is_nontrivial[i]]
    resize = max([max(seg.shape) for seg in seg_tensor])
    list_of_segs = []
    crop_slices = []
    for i in range(bs):
        if list_of_imshape is None:
            new_h,new_w = None,None
        else:
            new_h,new_w = sam_resize_index(*list_of_imshape[i],resize=resize)
        crop_slice = [0 for _ in range(num_dims0)]
        crop_slice[d1] = slice(0,new_h)
        crop_slice[d2] = slice(0,new_w)
        crop_slice = tuple(crop_slice)
        list_of_segs.append(transform(seg_tensor[i])[crop_slice])
        crop_slices.append(crop_slice)
    list_of_segs = postprocess_list_of_segs(list_of_segs,seg_kwargs=seg_kwargs,overwrite=overwrite)
    
    if keep_same_type:
        if input_mode=="torch":
            for i in range(bs):
                seg_tensor[i][crop_slices[i]] = torch.tensor(list_of_segs[i],dtype=dtype,device=device)
        elif input_mode=="np":
            for i in range(bs):
                seg_tensor[i][crop_slices[i]] = list_of_segs[i]
        elif input_mode=="list_of_torch":
            seg_tensor = [torch.tensor(np.expand_dims(seg,trivial_idx),dtype=dtype,device=device) for seg in list_of_segs]
        else:
            seg_tensor = [np.expand_dims(seg,trivial_idx) for seg in list_of_segs]
    else:
        seg_tensor = list_of_segs
    return seg_tensor

def torch_expand_dims(x,list_of_dims):
    # list_of_dims = 3, x.shape = (0,1,2)
    assert max(list_of_dims)-len(list_of_dims)>=len(x.shape), "expected len(list_of_dims)-max(list_of_dims) to be >=len(x.shape)"
    return x

def postprocess_list_of_segs(list_of_segs,seg_kwargs={},overwrite=False):
    out = []
    for seg in list_of_segs:
        out.append(postprocess_seg(seg,**seg_kwargs,overwrite=overwrite))
    return out

def postprocess_seg(seg,
                    mode="gauss_survive",
                    replace_with="nearest",
                    num_objects=8,
                    min_area=0.005,
                    sigma=0.001,
                    overwrite=False):
    """
    Postprocess a segmentation by removing pixels of typically small objects or noise.

    Args:
    seg: np.ndarray, shape (H,W) or (H,W,1), dtype np.uint8
        The segmentation to postprocess.
    mode: str, one of ["num_objects", "min_area", "gauss_raw", "gauss_survive"]
        The mode to use for postprocessing where:
        - "num_objects": remove all but the largest `num_objects` objects
        - "min_area": remove all objects with smaller relative area smaller than 
            `min_area`
        - "gauss_raw": apply a gaussian filter to the onehot of the segmentation
        - "gauss_survive": apply a gaussian filter to the onehot of the segmentation
            and keep the original segmentation for objects that survive the filter
    replace_with: str, one of ["gauss", "new", "nearest"]
        The method to use when replacing pixels of removed objects. Where:
        - "gauss": replace with the result of the gaussian filter
        - "new": replace with a unique label not found in the objects that were kept
        - "nearest": replace with the label of the nearest object from a distance
            transform
    num_objects: int
        The number of objects to keep if `mode` is "num_objects".
    min_area: float
        The minimum relative area of an object to keep if `mode` is "min_area".
    sigma: float
        The sigma of the gaussian filter

    Returns:
    np.ndarray, shape (H,W) or (H,W,1), dtype np.uint8
        The postprocessed segmentation.
    """
    assert mode in ["num_objects", "min_area", "gauss_raw", "gauss_survive"], f"expected mode to be one of ['num_objects', 'min_area', 'gauss_raw', 'gauss_survive'], found {mode}"
    assert replace_with in ["gauss", "new", "nearest"], f"expected replace_with to be one of ['gauss', 'new', 'nearest'], found {replace_with}"
    assert isinstance(seg,np.ndarray), "expected seg to be an np.ndarray"
    assert seg.dtype==np.uint8
    assert len(seg.shape)==2 or (len(seg.shape)==3 and seg.shape[-1]==1)
    if not overwrite:
        seg = seg.copy()
    gauss_seg = None
    sigma_in_pixels = sigma*np.sqrt(np.prod(seg.shape))
    remove_mask = None
    if mode=="num_objects":
        assert num_objects>0 and isinstance(num_objects,int), "num_objects must be a positive integer. found: "+str(num_objects)
        uq, counts = np.unique(seg.flatten(),return_counts=True)
        if len(uq)>num_objects:
            remove_labels = uq[np.argsort(counts)[:-num_objects]]
            remove_mask = np.isin(seg,remove_labels)
    elif mode=="min_area":
        uq, counts = np.unique(seg,return_counts=True)
        area = counts/seg.size
        if any(area<min_area):
            remove_labels = uq[area<min_area]
            remove_mask = np.isin(seg,remove_labels)
    elif mode=="gauss_raw":
        seg = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels)
    elif mode=="gauss_survive":
        gauss_seg = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels)
        uq = np.unique(gauss_seg)
        remove_mask = np.logical_not(np.isin(seg,uq))
    if np.all(remove_mask):
        return np.zeros_like(seg)
    replace_vals = None
    if remove_mask is not None:
        if replace_with=="nearest":
            idx_of_nn = nd.distance_transform_edt(remove_mask,return_indices=True)[1]
            replace_vals = seg[tuple(idx_of_nn)]
        elif replace_with=="new":
            uq = np.unique(seg[np.logical_not(remove_mask)])
            first_idx_not_in_uq = [i for i in range(len(uq)+1) if i not in uq][0]
            replace_vals = np.ones_like(seg)*first_idx_not_in_uq
        elif replace_with=="gauss":
            replace_vals = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels,skip_spatial=remove_mask)
        seg[remove_mask] = replace_vals[remove_mask]
    return seg

def quantile_normalize(x, alpha=0.001, q=None):
    if alpha is not None:
        assert q is None, "expected exactly 1 of alpha or q to be None"
        q = [alpha, 1-alpha]
    assert q is not None, "expected exactly 1 of alpha or q to be None"
    assert len(q)==2, "expected len(q)==2"
    minval,maxval = np.quantile(x,q)
    x = (x-minval)/(maxval-minval)
    x = np.clip(x,0,1)
    return x

def apply_mask(x,mask,is_shape=True):
    assert len(x.shape)>=2, "expected at least 2 dimensions in x"
    assert x.shape[-1]==x.shape[-2], "expected a square image, found "+str(x.shape)
    if is_shape:
        assert len(mask)>=2, "expected len(mask)>=2 when is_shape=True"
        new_h,new_w = sam_resize_index(*mask[:2],resize=x.shape[-1])
    else:
        #use the bbox of nonzero values in mask
        assert len(x.shape)>=2 and len(mask.shape)>=2, "expected at least 2 dimensions in x and mask"
        assert x.shape[-2]==mask.shape[-2] and x.shape[-1]==mask.shape[-1], "expected x.shape[-2:] to be equal to mask.shape[-2:]"
        
        new_h,new_w = max_nonzero_per_dim(mask)[-2:]
    slices = [slice(None) for _ in range(len(x.shape)-2)]+[slice(0,new_h),slice(0,new_w)]
    return x[tuple(slices)]

def torch_any_multiple(x,axis):
    out = x
    for dim in axis:
        out = out.any(dim=dim)
    return out

def max_nonzero_per_dim(x,add_one=True):
    if isinstance(x,np.ndarray):
        f = lambda x,dim: np.nonzero(np.any(x,axis=tuple([i for i in range(x.ndim) if i!=dim])))[0].tolist()
    else:
        f = lambda x,dim: torch.nonzero(torch_any_multiple(x,axis=[i for i in range(x.ndim) if i!=dim])).flatten().tolist()

    nnz = [f(x,i) for i in range(x.ndim)]
    print(nnz)
    nnz = [max([0]+v)+int(add_one) for v in nnz]
    return nnz
    
def to_dev(item,device="cuda"):
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item,list):
        return [to_dev(i,device) for i in item]
    elif item is None:
        return None
    else:
        raise ValueError(f"Unknown type: {type(item)}. Expected list of torch.tensor or None")

def model_arg_is_trivial(model_arg_k):
    out = False
    if model_arg_k is None:
        out = True
    elif isinstance(model_arg_k,list):
        if len(model_arg_k)==0:
            out = True
        elif all([item is None for item in model_arg_k]):
            out = True
    return out
    
def nice_split(s,split_s=",",remove_empty_str=True):
    assert isinstance(s,str), "expected s to be a string"
    assert isinstance(split_s,str), "expected split_s to be a string"
    if len(s)==0:
        out = []
    else:
        out = s.split(split_s)
    if remove_empty_str:
        out = [item for item in out if len(item)>0]
    return out

def str_to_seed(s):
    return int("".join([str(ord(l)) for l in s]))%2**32

def is_nan_float(x):
    out = False
    if isinstance(x,float):
        out = np.isnan(x)
    return out

def get_named_datasets(datasets,datasets_info=None):
    if datasets_info is None:
        datasets_info = load_json_to_dict_list(str(Path(__file__).parent.parent.parent / "data" / "datasets_info_live.json"))
    
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "expected datasets to be a string or a list of strings"
        datasets = datasets.split(",")
    named_datasets_criterion = {"non-medical": lambda d: d["type"]=="pictures",
                                    "medical": lambda d: d["type"]=="medical",
                                    "all": lambda d: True,
                                    "high-qual": lambda d: d["quality"]=="high",
                                    "non-low-qual": lambda d: d["quality"]!="low",
                                    "binary": lambda d: d["num_classes"]==2,}
    dataset_list = copy.deepcopy(datasets)
    if len(datasets)==1:
        if datasets[0] in named_datasets_criterion:
            crit = named_datasets_criterion[datasets[0]]
            dataset_list = [d["dataset_name"] for d in datasets_info if d["live"] and crit(d)]

    available_datasets = [d["dataset_name"] for d in datasets_info if d["live"]]
    assert all([d in available_datasets for d in dataset_list]), "Unrecognized dataset. Available datasets are: "+str(available_datasets)+". Named groups of datasets are: "+str(list(named_datasets_criterion.keys()))+" got "+str(dataset_list)
    
    return dataset_list

def prettify_classname(classname,dataset_name):
    foreground_with_number = ["sa1b","hrsod","dram"]
    no_map_required = ["visor","pascal","msra","fss","ecssd","duts","dis","coift","cityscapes","ade20k"]
    has_underscores = ["totseg","to5k","monu4","monu","lvis"]
    coco_like = ["coco"]
    if dataset_name in foreground_with_number:
        assert classname.find("foreground")>=0 or classname.find("background")>=0, "classname must contain foreground or background"
        return "foreground" if classname.find("foreground")>=0 else "background"
    elif dataset_name in no_map_required:
        return classname
    elif dataset_name in has_underscores:
        return classname.replace("_"," ")
    elif dataset_name in coco_like:
        return classname.split("/")[-1].replace("-other","").replace("-"," ")
    else:
        raise NotImplementedError(f"dataset_name {dataset_name} not implemented")

def fix_clip_matrix_in_state_dict(ckpt_model,model):
    if "vit.class_names_embed.0.weight" in ckpt_model.keys():
        if ckpt_model["vit.class_names_embed.0.weight"].shape[0]!=model.vit.class_names_embed[0].weight.shape[0]:
            print("WARNING: class_names_embed weight shape mismatch. Ignoring.")
            ckpt_model["vit.class_names_embed.0.weight"] = model.vit.class_names_embed[0].weight
    return ckpt_model

def format_model_kwargs(model_kwargs,del_none=True,dev="cuda",list_instead=False):
    """Formats a kwarg dictionary with list arguments as 
    a tensor on the specified device"""
    bs = None
    for k in model_kwargs.keys():
        if not model_arg_is_trivial(model_kwargs[k]):
            if bs is None:
                bs = len(model_kwargs[k])
            else:
                assert bs==len(model_kwargs[k]), f"expected same bs. Found {bs} and {len(model_kwargs[k])} for {k}"
            model_kwargs[k] = unet_kwarg_to_tensor(model_kwargs[k],key=k,dev=dev,list_instead=list_instead)
        else:
            model_kwargs[k] = None

    if del_none:
        for k in list(model_kwargs.keys()):
            if model_kwargs[k] is None:
                del model_kwargs[k]
    return model_kwargs

def unet_kwarg_to_tensor(kwarg,key=None,non_tensor_exception_keys=["class_names"],dev=None,list_instead=False):
    key_exception = False
    if key is not None:
        if key in non_tensor_exception_keys:
            key_exception = True
    if kwarg is None:
        pass
    elif torch.is_tensor(kwarg):
        pass
    elif key_exception:
        crit = [isinstance(item,(str,tuple,int,torch.Tensor,list)) or (item is None) for item in kwarg]
        assert all(crit), f"If kwarg for exception keys is a list, then all elements must be str, tuple, int, or torch.Tensor. kwarg={kwarg}"
        if model_arg_is_trivial(kwarg):
            kwarg = None
        else:
            kwarg = [[] if item is None else item for item in kwarg]
    elif isinstance(kwarg, list):
        assert all([(isinstance(kw, torch.Tensor) or kw is None) for kw in kwarg]), f"If kwarg is a list, all elements must be torch.Tensor or None. kwarg={kwarg}"
        if all([kw is None for kw in kwarg]): #also return true for empty list
            kwarg = None
        elif all([isinstance(kw, torch.Tensor) for kw in kwarg]):
            kwarg = torch.stack(kwarg)
        else:
            bs = len(kwarg)
            shapes = [kw.shape for kw in kwarg if kw is not None]
            s0 = [i for i in range(bs) if kwarg[i] is not None][0]
            assert all([s==shapes[0] for s in shapes]), f"If kwarg is a list, all tensors must have the same shape. kwarg={kwarg}"
            if list_instead:
                full_kwarg = [None for _ in range(bs)]
            else:
                full_kwarg = torch.zeros((bs,)+shapes[0],
                                     dtype=kwarg[s0].dtype,
                                     device=kwarg[s0].device)
            for i in range(bs):
                if kwarg[i] is not None:
                    full_kwarg[i] = kwarg[i]
            kwarg = full_kwarg
    else:
        raise ValueError(f"kwarg={kwarg} is not a valid type. must be None, torch.Tensor, or list of torch.Tensor/None")
    if (dev is not None) and torch.is_tensor(kwarg):
        kwarg = kwarg.to(dev)
    elif (dev is not None) and isinstance(kwarg, list):
        kwarg = [(kw.to(dev) if kw is not None else None) for kw in kwarg]
    return kwarg


def construct_points(points,x,as_tensor=False):
    """Generates point images from ground truth
    
    Args:
        points (list): list of points images or None for batch 
            indices not given points. Points images are torch.tensors 
            of shape (H,W) with values in [0.0,1.0] representing
            the presence of a point at that location if the value
            is 1. 
        x (torch.tensor): Ground truth with shape (bs,num_bits,H,W)
        as_tensor (bool): If True, returns the points as torch.tensors.
            Makes the items which are None into torch.zeros.

    Returns:
        list: list of points images with shape (num_bits,H,W) or None
    """
    assert not model_arg_is_trivial(points), "expected points to be non-trivial"
    assert len(points)==len(x), f"len(points)={len(points)} must be equal to len(x)={len(x)}"
    out = []
    for i in range(len(x)):
        if points[i] is None:
            out.append(None)
        else:
            out.append(points[i]*x[i])
    if as_tensor:
        out = unet_kwarg_to_tensor(out)
    return out

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 0: nuke test")
        nuke_saves_folder()
    elif args.unit_test==1:
        print("UNIT TEST 1: various inputs outputs for postprocess_batch test")
        bs = 5
        d1,d2 = 4,4
        f_torch = lambda *dims: torch.randint(0,10,dims).type(torch.uint8)
        f_np = lambda *dims: np.random.randint(0,10,dims).astype(np.uint8)
        batch1 = f_torch(bs,d1,d2)
        batch2 = f_torch(bs,1,1,d1,d2)
        batch3 = [f_torch(1,d1,d2) for _ in range(bs)]
        batch4 = [f_torch(1,d1,1,d2,1) for _ in range(bs)]
        batch5 = [f_np(d1,d2) for _ in range(bs)]
        batch6 = [f_np(1,d1,d2) for _ in range(bs)]
        batch7 = f_np(bs,d1,d2)

        batches = [batch1,batch2,batch3,batch4,batch5,batch6,batch7]
        for k,batch in enumerate(batches):
            print("TEST batch"+str(k+1))
            shaprint(batch)
            print("result:")
            shaprint(postprocess_batch(batch,seg_kwargs={'mode':'gauss_raw'}))
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")

if __name__=="__main__":
    main()
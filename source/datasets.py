import torch
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel
import jsonlines
import json
from PIL import Image
from pathlib import Path
import os
import albumentations as A
import cv2
import copy
from turbojpeg import TurboJPEG,TJPF_RGB
import warnings
from unet import get_sam_image_encoder
import tqdm

turbo_jpeg = TurboJPEG()

def points_image_from_label(label,num_points=None):
    assert torch.is_tensor(label)
    assert len(label.shape)==3
    assert label.shape[0]==1
    if num_points is None:
        num_points = np.random.choice([1,1,1,1,1,1,1,1,8,
                                       2,2,2,2,2,2,2,7,7,
                                       3,3,3,3,3,3,6,6,6,
                                       4,4,4,4,4,5,5,5,5])
    counts = torch.bincount(label.cpu().flatten())
    nonzero_counts_idx = torch.where(counts>0)[0].cpu().numpy()
    label_indices = np.random.choice(nonzero_counts_idx,size=num_points,replace=True)
    D1 = torch.zeros(num_points,dtype=torch.int64)
    D2 = torch.zeros(num_points,dtype=torch.int64)
    for i in np.unique(label_indices):
        mask_i = label_indices==i
        _,d1,d2 = torch.where(label==i)
        d1,d2 = d1.cpu(),d2.cpu()
        index = torch.randint(0,len(d1),(mask_i.sum().item(),))
        D1[mask_i] = d1[index]
        D2[mask_i] = d2[index]
    points_image = torch.zeros_like(label,dtype=torch.float32)
    points_image[:,D1,D2] = 1
    return points_image.to(label.device)

class AnalogBits(object):
    def __init__(self,num_bits=8,
                 shuffle_zero=False,
                 shuffle=False,
                 permanent_seed=None,
                 bit_dim=1,
                 batch_has_different_seed=False,
                 onehot=False,
                 padding_idx=255):
        self.onehot = onehot
        self.bit_dim = bit_dim
        self.num_bits = num_bits
        self.num_classes = 2**num_bits
        self.padding_idx = padding_idx
        if num_bits>8:
            raise NotImplementedError("num_bits>8 not implemented")
        self.shuffle_zero = shuffle_zero
        self.shuffle = shuffle
        self.permanent_seed = permanent_seed
        self.batch_has_different_seed = batch_has_different_seed
        if batch_has_different_seed:
            assert permanent_seed is None
        if permanent_seed is not None:
            np.random.seed(permanent_seed)
            if shuffle_zero:
                self.perm = np.random.permutation(self.num_classes)
            else:
                self.perm = np.concatenate([[0],np.random.permutation(self.num_classes-1)+1])
        else:
            self.perm = None
            
    def get_perm(self,inv=False):
        if self.shuffle:
            if self.perm is None:
                if self.shuffle_zero:
                    perm = np.random.permutation(2**self.num_bits)
                else:
                    perm = np.concatenate([[0],np.random.permutation(2**self.num_bits-1)+1])
            else:
                perm = self.perm
        else:
            perm = np.arange(2**self.num_bits)
        if inv:
            perm = np.argsort(perm)
        return perm
    
    def int2pad(self,x):
        return x==self.padding_idx
    
    def int2bit(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        assert x.shape[self.bit_dim]==1
        x[self.int2pad(x)] = 0
        assert x.max()<=self.num_classes
        if self.shuffle:
            x = self.get_perm(inv=False)[x]
        
        if self.onehot:
            x_new = np.zeros([x.size,self.num_classes])
            x_new[np.arange(x.size),x.flatten()] = 1
            x_new = x_new.reshape(list(x.shape)+[self.num_classes])
            transpose_list = list(range(len(x_new.shape)))
            transpose_list[-1],transpose_list[self.bit_dim] = self.bit_dim,-1
            x = x_new.transpose(transpose_list).squeeze(-1).astype(np.float32)
        else:
            x = np.unpackbits(x.astype(np.uint8),axis=self.bit_dim,count=self.num_bits,bitorder="little").astype(np.float32)*2-1

        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x
        
    def bit2int(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        #convert to ints if necessary
        if x.dtype in [np.float32,np.float64]:
            x = (x>0).astype(np.uint8)
        if self.onehot:
            x = np.argmax(x,axis=self.bit_dim,keepdims=True)
        else:
            x = np.packbits(x,axis=self.bit_dim,bitorder="little")
        if self.shuffle:
            x = self.get_perm(inv=True)[x]
        if was_torch:
            x = torch.from_numpy(x).to(device)
        return x

    def likelihood(self,x,x_gt):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            x_gt = x_gt.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        if self.onehot:
            raise NotImplementedError("onehot likelihood not implemented")
        else:
            if x_gt.shape[self.bit_dim]==1:
                true_bits = self.int2bit(x_gt)
            else:
                true_bits = x_gt
                assert x_gt.shape[self.bit_dim]==self.num_bits
            likelihood = np.prod(1-0.5*np.abs(true_bits-x),axis=self.bit_dim,keepdims=True)
        if was_torch:
            likelihood = torch.from_numpy(likelihood).to(device)
        return likelihood

    def bit2prob(self,x):
        if isinstance(x,torch.Tensor):
            device = x.device
            x = x.cpu().detach().numpy()
            was_torch = True
        else:
            was_torch = False
        assert isinstance(x,np.ndarray)
        assert len(x.shape)>=self.bit_dim+1
        if self.onehot:
            assert x.shape[self.bit_dim]==self.num_classes, "x.shape: "+str(x.shape)+", self.num_classes: "+str(self.num_classes)+", self.bit_dim: "+str(self.bit_dim)
        else:
            assert x.shape[self.bit_dim]==self.num_bits, "x.shape: "+str(x.shape)+", self.num_bits: "+str(self.num_bits)+", self.bit_dim: "+str(self.bit_dim)
        if self.onehot:
            onehot = x
        else:
            onehot_shape = list(x.shape)
            onehot_shape[self.bit_dim] = self.num_classes
            onehot = np.zeros(onehot_shape)
            for i in range(self.num_classes):
                pure_bits = self.int2bit(np.array([i]).reshape(*[1 for _ in range(len(x.shape))]))
                onehot[:,i] = np.prod(1-0.5*np.abs(pure_bits-x),axis=self.bit_dim)
        if was_torch:
            onehot = torch.from_numpy(onehot).to(device)
        return onehot

class CatBallDataset(torch.utils.data.Dataset):
    def __init__(self,
                 size : int = 64,
                 dtype : str ="uint8",
                 dataset_len : int = 1000,
                 background_is_zero : bool = True,
                 num_balls : list = list(range(1,10)),
                 max_num_classes : int = 8,
                 seed_translation : int = 0):
        assert dtype in ["float","double","uint8"]
        self.dtype = dtype
        self.dataset_len = dataset_len
        self.size = size
        self.background_is_zero = background_is_zero
        if isinstance(num_balls,int):
            num_balls = [num_balls]
        self.num_balls = num_balls
        self.max_num_classes = max_num_classes
        self.x,self.y = np.meshgrid(range(self.size),range(self.size))
        self.seed_translation = seed_translation
        
    def get_image(self,item,noise_coef=0.1,std=0.05):
        noise = np.random.randn(*item.shape)
        foreground = (item>0).astype(float)
        image = (1-noise_coef)*foreground+noise_coef*noise
        image = gaussian_filter(image,sigma=std*self.size)
        return image*2-1
    
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        np.random.seed(idx+self.seed_translation)
        R = []
        CX = []
        CY = []
        item = np.zeros((self.size,self.size),dtype=self.dtype)
        nb = np.random.choice(self.num_balls)
        for i in range(nb):
            cx = self.size*np.random.rand()
            cy = self.size*np.random.rand()
            max_r = min([cx,cy,self.size-cx,self.size-cy])
            r = max_r*np.random.rand()
            dist = r-((self.x-cx)**2+(self.y-cy)**2)**0.5
            item[dist>=0] = np.random.randint(1,self.max_num_classes)
            R.append(r)
            CX.append(cx)
            CY.append(cy)
        info = {"r": R,"cx": CX,"cy": CY, "nb": nb}
        info["image"] = torch.from_numpy(self.get_image(item)).unsqueeze(0)
        item = torch.from_numpy(item).unsqueeze(0)
        return item,info

def custom_collate_with_info(original_batch):
    n = len(original_batch[0])
    normal_batch = []
    for i in range(n):
        list_of_items = [item[i] for item in original_batch]
        if i+1==n:
            info = list_of_items
        else:
            normal_batch.append(torch.stack(list_of_items,axis=0))
    return *normal_batch,info

def mode_resize(label,size):
    assert isinstance(label,np.ndarray)
    assert label.dtype==np.uint8
    uq = np.unique(label)
    max_intensity = np.zeros(size,dtype=np.float32)
    new_label = np.zeros(size,dtype=np.uint8)
    for i in uq:
        new_label_i = cv2.resize((label==i).astype(float),size,interpolation=cv2.INTER_AREA)
        new_label[new_label_i>max_intensity] = i
        max_intensity = np.maximum(max_intensity,new_label_i)
    return new_label

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

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,split="train",
                      image_size=128,
                      datasets="non-medical",
                      split_ratio = [0.8,0.1,0.1],
                      min_label_size=0.0,
                      min_crop=0.5,
                      max_num_classes=8,
                      crop_method="multicrop_most_border",
                      num_crops=3,
                      semantic_prob=0.5,
                      label_map_method="all",
                      shuffle_nonzero_labels=True,
                      shuffle_zero=True,
                      shuffle_datasets=True,
                      data_root=None,
                      use_pretty_data=True,
                      geo_aug_p=0.3,
                      label_padding_val=255,
                      split_method="random",
                      sam_features_idx=-1,
                      ignore_sam_idx=-1):
        self.ignore_sam_idx = ignore_sam_idx
        self.sam_features_idx = sam_features_idx
        self.geo_aug_p = geo_aug_p
        self.shuffle_datasets = shuffle_datasets
        self.use_pretty_data = use_pretty_data
        if data_root is None:
            data_root = str(Path(__file__).parent.parent / "data")
        self.data_root = data_root
        self.image_size = image_size
        self.min_label_size = min_label_size
        self.min_crop = min_crop
        self.max_num_classes = max_num_classes
        self.datasets = datasets
        self.semantic_prob = semantic_prob
        self.label_padding_val = label_padding_val
        assert crop_method in ["multicrop_most_border","multicrop_most_classes","full_image","sam_small","sam_big"]
        if crop_method.startswith("sam"):
            self.sam_aug_small = get_sam_aug(image_size,padval=label_padding_val)
            self.sam_aug_big = get_sam_aug(1024,padval=label_padding_val)
        self.crop_method = crop_method
        assert label_map_method in ["all","largest","random"]
        self.label_map_method = label_map_method
        self.shuffle_nonzero_labels = shuffle_nonzero_labels
        self.shuffle_zero = shuffle_zero
        self.num_crops = num_crops
        self.downscale_thresholding_factor = 3
        self.datasets_info = load_json_to_dict_list(str(Path(data_root) / "datasets_info_live.json"))
        available_datasets = [d["dataset_name"] for d in self.datasets_info if d["live"]]
        if self.datasets==["non-medical"] or self.datasets=="non-medical":
            self.dataset_list = []
            for d in self.datasets_info:
                if d["live"]:
                    if d["type"]=="pictures":
                        self.dataset_list.append(d["dataset_name"])
        elif self.datasets==["medical"] or self.datasets=="medical":
            self.dataset_list = []
            for d in self.datasets_info:
                if d["live"]:
                    if d["type"]=="medical":
                        self.dataset_list.append(d["dataset_name"])
        elif self.datasets==["all"] or self.datasets=="all":
            self.dataset_list = available_datasets
        else:
            if isinstance(self.datasets,list):
                self.dataset_list = self.datasets
            else:
                assert isinstance(self.datasets,str), "invalid datasets input. must be a list of strings or a comma separated string"
                self.dataset_list = self.datasets.split(",")
            
            assert all([d in available_datasets for d in self.dataset_list]), "Unrecognized dataset. Available datasets are: "+str(available_datasets)+" got "+str(self.dataset_list)
        if split in ["train","vali","test","all"]:
            split = {"train": 0,"vali": 1, "test": 2, "all": 3}[split]
        assert split in list(range(-1,4)), "invalid split input. must be one of [0,1,2,3] or ['train','vali','test','all'], found "+str(split)
        sr = split_ratio
        sr = np.array(sr)/sum(sr)
        self.split_start_and_stop = [[0,sr[0]],[sr[0],sr[0]+sr[1]],[sr[0]+sr[1],1.0],[0,1]]
        
        self.split = split
        
        self.items = []
        self.length = 0
        self.idx_to_class = {}
        self.augment_per_dataset = {}
        self.datasets_info = {d["dataset_name"]: d for d in self.datasets_info if d["dataset_name"] in self.dataset_list}

        assert split_method in ["random","native","native_train"]
        self.split_method = split_method

        for dataset_name in self.dataset_list:
            print("processing dataset: ",dataset_name)
            info_json = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,"info.jsonl"))
            if self.ignore_sam_idx>=0:
                info_json = [info for info in info_json if not info.get("sam",[0,0,0])[self.ignore_sam_idx]]
            N = len(info_json)
            previous_seed = np.random.get_state()[1][0]
            if self.shuffle_datasets:
                dataset_specific_seed = sum([ord(l) for l in dataset_name])           
                np.random.seed(seed=dataset_specific_seed)
                randperm = np.random.permutation(N)
                np.random.seed(previous_seed)
            else:
                randperm = np.arange(N)
            
            
            if split_method=="native_train":
                use_idx = self.get_use_idx_native_train(randperm,info_json,dataset_name)
            elif split_method=="native":
                use_idx = np.array([i for i in range(N) if info_json[i].get("split_idx",0)==split])
                if len(use_idx)>0:
                    use_idx = randperm[use_idx]
            elif split_method=="random":
                start = max(0,np.floor(self.split_start_and_stop[split][0]*N).astype(int))
                stop = min(N,np.floor(self.split_start_and_stop[split][1]*N).astype(int))
                use_idx = randperm[start:stop]
            items = []
            if len(use_idx)==0:
                warnings.warn("no data in dataset "+dataset_name+" satisfying the criteria")
                continue
            file_format = self.datasets_info[dataset_name]["file_format"]
            for idx in use_idx:
                item = info_json[idx]
                item["image_path"] = os.path.join("f"+str(idx//1000),str(idx)+"_im."+file_format)
                item["label_path"] = os.path.join("f"+str(idx//1000),str(idx)+"_la.png")
                if self.use_pretty_data and item.get("pretty",False):
                    item["image_path"] = item["image_path"].replace("_im."+file_format,"_pim."+file_format)
                    item["label_path"] = item["label_path"].replace("_la.png","_pla.png")
                item["dataset_name"] = dataset_name
                items.append(item)
        
            class_dict = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,"idx_to_class.json"))[0]
            self.idx_to_class[dataset_name] = class_dict
            self.augment_per_dataset[dataset_name] = get_augmentation(self.datasets_info[dataset_name]["aug"],s=self.image_size,train=split==0,geo_aug_p=self.geo_aug_p)

            self.length += len(items)
            self.items.extend(items)
        
        self.len_per_dataset = {dataset_name: len([item for item in self.items if item["dataset_name"]==dataset_name]) for dataset_name in self.dataset_list}
        #assert all([self.len_per_dataset[dataset_name]>0 for dataset_name in self.dataset_list]), "no data in one of the datasets satisfying the criteria"
            
        self.dataset_weights = {}
        for dataset_name in self.dataset_list:
            w = self.datasets_info[dataset_name]["rel_weight"]/self.len_per_dataset[dataset_name] if self.len_per_dataset[dataset_name]>0 else 0
            self.dataset_weights[dataset_name] = w
        self.dataset_to_label = {dataset: i for i, dataset in enumerate(["none"]+self.dataset_list)}

    def __len__(self):
        return self.length
    
    def get_sampler(self,seed=None):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        p = np.array([self.dataset_weights[item["dataset_name"]] for item in self.items])
        return torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=True,generator=generator)

    def get_gen_dataset_sampler(self,datasets,seed=None):
        if isinstance(datasets,str):
            datasets = datasets.split(",")
        assert all([d in self.dataset_list for d in datasets]), "Unrecognized dataset. Available datasets are: "+str(self.dataset_list)+" got "+str(datasets)
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        p = []
        for item in self.items:
            if item["dataset_name"] in datasets:
                p.append(self.dataset_weights[item["dataset_name"]])
            else:
                p.append(0)
        p = np.array(p)
        return torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=False,generator=generator)

    def convert_to_idx(self,list_of_things):
        """
        Converts a list of things to a list of indices. The items in
        the list should either be:
         - a list of integer indices (where we only check that the indices are valid)
         - a list of info dicts with the fields "dataset_name" and "i"
         - a list of strings formatted like '{dataset_name}/{i}'
        Returns a list of integer indices and checks they are valid
        """
        assert isinstance(list_of_things,list)
        if len(list_of_things)==0: return []
            
        item0 = list_of_things[0]
        if isinstance(item0,int):
            list_of_things2 = list_of_things
        elif isinstance(item0,dict):
            assert "dataset_name" in item0 and "i" in item0, "item0 must be a dict with the fields 'dataset_name' and 'i'"
            d_vec = [item["dataset_name"] for item in list_of_things]
            i_vec = [item["i"] for item in list_of_things]
        elif isinstance(item0,str):
            d_vec = [item.split("/")[0] for item in list_of_things]
            i_vec = [int(item.split("/")[1]) for item in list_of_things]
        else:
            raise ValueError(f"Unrecognized type for item0: {type(item0)}, should be int, dict or str")

        if isinstance(item0,(dict,str)):
            list_of_things2 = []
            for d,i in zip(d_vec,i_vec):
                match_idx = None
                for k,item in enumerate(self.items):
                    if item["dataset_name"]==d and item["i"]==i:
                        match_idx = k
                        break   
                assert match_idx is not None, "No match for dataset_name: "+d+", i: "+str(i)
                list_of_things2.append(match_idx)

        assert all([isinstance(item,int) for item in list_of_things2]), "all items in list_of_things must be integers"
        assert all([0<=item<len(self) for item in list_of_things2]), "all items in list_of_things must be valid indices"
        return list_of_things2

    def get_prioritized_sampler(self,pri_didx,seed=None,use_p=True,shuffle=False):
        """
        Returns a sampler which first samples from the dataset 
        with index in pri_idx and then the rest of the dataset
        """
        pri_idx = self.convert_to_idx(pri_didx)
        non_pri_idx = [i for i in range(len(self)) if i not in pri_idx]
        if shuffle:
            if use_p:
                p = np.array([self.dataset_weights[item["dataset_name"]] for item in self.items])
            else:
                p = np.ones(len(self))
            gen = torch.Generator().manual_seed(seed)
            sampler = torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=False,generator=gen)
            new_pri_idx = []
            new_non_pri_idx = []
            for idx in sampler:
                if idx in pri_idx:
                    new_pri_idx.append(idx)
                else:
                    new_non_pri_idx.append(idx)
            pri_idx = new_pri_idx
            non_pri_idx = new_non_pri_idx
        order = pri_idx+non_pri_idx
        return order

    def get_use_idx_native_train(self,randperm,info_json,dataset_name):
        """Returns a training set of indices based on the native 
        splits of the datasets. Samples are only allowed to migrate from
        lower to higher splits. i.e train->vali->test.
        This only works if the target #train>=#vali+#test, allowing us
        to use native training data as vali/test data."""
        N = len(info_json)
        native_split = []
        for i in range(N):
            native_split.append(info_json[i].get("split_idx",0))
        native = []
        target = []
        for split in range(3):
            start = max(0,np.floor(self.split_start_and_stop[split][0]*N).astype(int))
            stop = min(N,np.floor(self.split_start_and_stop[split][1]*N).astype(int))
            target.append(stop-start)
            native.append(sum([x==split for x in native_split]))
        assert N==sum(native)==sum(target), "dataset_name: "+dataset_name+"N: "+str(N)+", native: "+str(native)+", target: "+str(target)
        #native to actual split. upper triangular matrix to avoid migration from higher to lower splits
        n_to_a = np.zeros((3,3),dtype=int)
        n_to_a[2,2] = native[2]
        n_to_a[1,1] = native[1]
        #take from train to fill up vali and test
        missing_vali = max(0, target[1]-native[1])
        n_to_a[0,1] = missing_vali
        missing_test = max(0,target[2]-native[2])
        n_to_a[0,2] = missing_test
        #remainder goes to train
        assert native[0]>=missing_vali+missing_test, "dataset_name: "+dataset_name+"N: "+str(N)+", native: "+str(native)+", target: "+str(target)
        n_to_a[0,0] = native[0]-missing_test-missing_vali
        assert np.all(n_to_a>=0)
        assert n_to_a.sum()==N, str(n_to_a)+", "+str(n_to_a.sum())+", "+str(N)
        assert np.all(n_to_a.sum(1)==native), str(n_to_a)+", "+str(n_to_a.sum())+", "+str(N)
        #now select indices for the current split of interest based on their ordering in randperm
        target_split = -np.ones(N,dtype=int)
        for i in randperm:
            native_split_i = native_split[i]
            for j in range(3):
                if n_to_a[native_split_i,j]>0:
                    n_to_a[native_split_i,j] -= 1
                    target_split[i] = j
                    break
        assert np.all(target_split>=0), "num fails: "+str(np.sum(target_split==-1))+", dataset_name: "+dataset_name
        if self.split==3:
            use_idx = np.where(target_split>=0)[0]
        else:
            use_idx = np.where(target_split==self.split)[0]
        return use_idx

    def get_crop_params(self,label):
        min_image_sidelegth = min(label.shape[:2])
        max_crop = min_image_sidelegth
        min_crop = min(max(self.image_size,min_image_sidelegth*self.min_crop),max_crop)
        crop_measure_best = 0
        for i in range(self.num_crops):
            if self.crop_method=="multicrop_most_border":
                crop_size = np.random.randint(min_crop,max_crop+1)
                crop_x = np.random.randint(0,label.shape[1]-crop_size+1)+np.array([0,crop_size])
                crop_y = np.random.randint(0,label.shape[0]-crop_size+1)+np.array([0,crop_size])
                crop_measure = total_boundary_pixels(label[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]])
            elif self.crop_method=="multicrop_most_classes":
                crop_size = np.random.randint(min_crop,max_crop+1)
                crop_x = np.random.randint(0,label.shape[1]-crop_size+1)+np.array([0,crop_size])
                crop_y = np.random.randint(0,label.shape[0]-crop_size+1)+np.array([0,crop_size])
                crop_measure = len(np.unique(label[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]))
            elif self.crop_method=="full_image":
                crop_x_best = np.array([0,label.shape[1]])
                crop_y_best = np.array([0,label.shape[0]])
                break
            if crop_measure_best<crop_measure or i==0:
                crop_measure_best = crop_measure
                crop_x_best = crop_x
                crop_y_best = crop_y
        return crop_x_best,crop_y_best
    
    def map_label_to_valid_bits(self,label,info):
        """takes a uint8 label map and depending on the method, maps
        each label to a number between 0 and max_num_classes-1. also
        keeps track of which classes correspond to idx and adds this 
        dict to info.""" 
        idx_to_class_name = {k: self.idx_to_class[info["dataset_name"]][str(c)] for k,c in enumerate(info["classes"])}
        is_semantic = self.semantic_prob>=np.random.rand() and self.semantic_prob>0
        info["is_semantic"] = is_semantic
        if is_semantic:
            for i in np.unique(info["classes"]):
                if sum([x==i for x in info["classes"]])>1:
                    idx_of_uq_i = np.where(info["classes"]==i)[0]
                    for j in idx_of_uq_i[1:]:
                        del idx_to_class_name[j]
                        label[label==j] = idx_of_uq_i[0]
        counts = np.bincount(label.flatten()) #for 1024x1024 np.uint8 image, takes ~1ms on my machine (AMD Ryzen 7 7800X3d 8-Core Processor x 16)
        nnz = len(counts)-1
        mnc = self.max_num_classes

        if self.label_map_method=="largest":
            old_to_new = np.array([0]+list(np.argsort(np.argsort(-counts[1:]))+1),dtype=int)
            old_to_new[old_to_new>=mnc] = 0
        elif self.label_map_method=="random":
            old_to_new = np.array([0]+list(np.random.permutation(nnz)+1),dtype=int)
            old_to_new[old_to_new>=mnc] = 0
        elif self.label_map_method=="all":
            old_to_new = np.array([0]+list(np.argsort(np.argsort(-counts[1:]))+1),dtype=int)
            old_to_new[old_to_new>=mnc] = ((old_to_new[old_to_new>=mnc])%(mnc-1))+1
        #map small classes to zero
        if self.min_label_size>0:
            for i in range(1,nnz+1):
                if counts[i]<self.min_label_size*label.size:
                    old_to_new[i] = 0
        #shuffle nonzero
        if self.shuffle_nonzero_labels:
            if self.shuffle_zero:
                perm = np.random.permutation(mnc)
            else:
                perm = np.array([0]+list(np.random.permutation(mnc-1)+1))
            old_to_new = perm[old_to_new]
        list_of_keys = list(idx_to_class_name.keys())
        print("old_to_new: ",old_to_new)
        print("list_of_keys: ",list_of_keys)
        print("idx_to_class_name: ",idx_to_class_name)
        for k in list_of_keys:
            k_new = old_to_new[k]
            v_new = idx_to_class_name[k]
            del idx_to_class_name[k]
            idx_to_class_name[k_new] = v_new
        info["idx_to_class_name"] = idx_to_class_name
        if self.label_padding_val==255:
            pad_mask = label==255
        label = old_to_new[label]
        if self.label_padding_val==255:
            label[pad_mask] = 255
        return label,info

    def preprocess(self,image,label,info):
        #if image is smaller than image_size, pad it
        if self.crop_method.startswith("sam"):
            if self.crop_method=="sam_big":
                augmented = self.sam_aug_big(image=image,mask=label)
            else:
                assert self.crop_method=="sam_small"
                augmented = self.sam_aug_small(image=image,mask=label)
            image,label = augmented["image"],augmented["mask"]
        else:
            if any([image.shape[i]<self.image_size for i in range(2)]):
                pad_y = max(0,self.image_size-image.shape[0])
                pad_y = (pad_y//2,pad_y-pad_y//2)
                pad_x = max(0,self.image_size-image.shape[1])
                pad_x = (pad_x//2,pad_x-pad_x//2)
                image = np.pad(image,(pad_y,pad_x,(0,0)))
                label = np.pad(label,(pad_y,pad_x))
            if not image.shape[2]==3:
                image = np.repeat(image,3,axis=-1)
            crop_x,crop_y = self.get_crop_params(label)
            image = image[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]
            label = label[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]
            image = cv2.resize(image,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
            label = cv2.resize(label,(self.image_size,self.image_size),interpolation=cv2.INTER_NEAREST)
        return image,label

    def augment(self,image,label,item):
        augmented = self.augment_per_dataset[item["dataset_name"]](image=image,mask=label)
        return augmented["image"],augmented["mask"]
    
    def load_raw_image_label(self,x,longest_side_resize,data_root=None):
        if data_root is None:
            data_root = self.data_root
        if isinstance(x,int):
            image_path = os.path.join(data_root,self.items[x]["dataset_name"],self.items[x]["image_path"])
            label_path = os.path.join(data_root,self.items[x]["dataset_name"],self.items[x]["label_path"])
        elif isinstance(x,dict):
            x = copy.deepcopy(x)
            if not hasattr(x,"image_path"):
                x["image_path"] = load_from_dataset_and_idx(x["dataset_name"],x["i"],im=True)
            if not hasattr(x,"label_path"):
                x["label_path"] = load_from_dataset_and_idx(x["dataset_name"],x["i"],im=False)
            image_path = os.path.join(data_root,x["dataset_name"],x["image_path"])
            label_path = os.path.join(data_root,x["dataset_name"],x["label_path"])
        else:
            assert isinstance(x,list)
            assert len(x)==2
            image_path,label_path = x
        image = np.atleast_3d(open_image_fast(image_path))
        label = open_image_fast(label_path)
        if longest_side_resize>0:
            image = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_AREA, always_apply=True, p=1)(image=image)["image"]
            label = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_NEAREST, always_apply=True, p=1)(image=label)["image"]
        return image,label
    
    def __getitem__(self, idx):
        info = copy.deepcopy(self.items[idx])
        dataset_name = info["dataset_name"]
        image_path = os.path.join(self.data_root,dataset_name,info["image_path"])
        label_path = os.path.join(self.data_root,dataset_name,info["label_path"])
        image = np.atleast_3d(open_image_fast(image_path))
        if image.shape[2]==1:
            image = np.repeat(image,3,axis=-1)
        label = open_image_fast(label_path)
        image,label = self.preprocess(image,label,info)
        label,info = self.map_label_to_valid_bits(label,info)
        if not self.crop_method.startswith("sam"): #TODO
            image,label = self.augment(image,label,info)
            image = image.astype(np.float32)*(2/255)-1
        image = torch.tensor(image).permute(2,0,1)
        label = torch.tensor(label).unsqueeze(0)
        info["image"] = image
        info["num_classes"] = len([uq for uq in torch.unique(label) if uq!=255])
        j = self.sam_features_idx
        if j>=0:
            if info["sam"][j]:
                i = info["i"]
                info["image_features"] = torch.load(os.path.join(self.data_root,dataset_name,f"f{i//1000}",f"{i}_sam{j}.pt"))
            else:
                info["image_features"] = None
        return label,info

def longest_side_resize_func(image,is_label=True,max_size=256):
    return A.LongestMaxSize(max_size=max_size, 
                            interpolation=cv2.INTER_NEAREST if is_label else cv2.INTER_AREA, 
                            always_apply=True, 
                            p=1)(image=image)["image"]

def load_from_dataset_and_idx(dataset_name,i,im=True):
    if im:
        possible_filesnames = ["_im.jpg","_im.png","_pim.jpg","_pim.png"]
    else:
        possible_filesnames = ["_la.png","_pla.png"]
    root_name = f"f{str(i//1000)}{os.sep}{str(i)}"
    root_dir = os.path.join(str(Path(__file__).parent.parent / "data"),dataset_name)
    for pf in possible_filesnames:
        if os.path.exists(os.path.join(root_dir,root_name)+pf):
            return root_name+pf
    raise ValueError("No image file found for dataset_name: "+dataset_name+", i: "+str(i))

def load_raw_image_label(x,longest_side_resize=0,data_root=None):
    assert isinstance(x,dict), "x must be a info dictionary, got: "+str(x)
    if data_root is None:
        data_root = str(Path(__file__).parent.parent / "data")
    return SegmentationDataset.load_raw_image_label(None,x,longest_side_resize,data_root)

def load_raw_image_label_from_didx(didx,longest_side_resize=0,data_root=None):
    assert isinstance(didx,list), "didx must be a list of didx strings"
    assert all([isinstance(didx_i,str) for didx_i in didx]), "didx must be a list of didx strings"
    assert all([len(didx_i.split("/"))==2 for didx_i in didx]), "didx must be a list of didx strings formatted as '{dataset_name}/{i}'"
    x = [{"dataset_name": didx_i.split("/")[0],"i": int(didx_i.split("/")[1])} for didx_i in didx]
    tuples = [load_raw_image_label(x2,longest_side_resize,data_root) for x2 in x]
    ims = [t[0] for t in tuples]
    gts = [t[1] for t in tuples]
    return ims,gts

def get_sam_aug(size,padval=255):
    sam_aug = A.Compose([A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
                     A.Normalize(always_apply=True, p=1), #SAM uses the default imagenet mean and std, which is also default in Albumentations.Normalize
                     A.PadIfNeeded(min_height=size, 
                                   min_width=size, 
                                   border_mode=cv2.BORDER_CONSTANT, 
                                   value=[0, 0, 0], 
                                   mask_value=padval, 
                                   always_apply=True, 
                                   p=1, 
                                   position=A.PadIfNeeded.PositionType.TOP_LEFT)])
    return sam_aug

def open_image_fast(image_path):
    assert image_path.find(".")>=0, "image_path must contain a file extension"
    extension = image_path.split(".")[-1]
    if extension in ["jpg","jpeg"]:
        with open(image_path, "rb") as f:
            image = turbo_jpeg.decode(f.read(),pixel_format=TJPF_RGB)
    else:
        image = np.array(Image.open(image_path))
    return image

def sobel1d(image,axis=0,mode="nearest"):
    from scipy.ndimage import convolve1d
    filter = np.array([1,0,-1])
    return convolve1d(image,filter,axis=axis,mode=mode)

def label_boundaries(image,dims=[-1,-2]):
    if torch.is_tensor(image):
        device = image.device
        image = image.cpu().detach().numpy()
        was_tensor = True
    else:
        was_tensor = False
    if image.dtype==np.uint8:
        image = image.astype(np.float32)
    if not isinstance(dims,list):
        dims = [dims]
    edge = np.zeros(image.shape,dtype=bool)
    for dim in dims:
        edge = np.logical_or(edge,np.abs(sobel1d(image,axis=dim))>0)
    if was_tensor:
        edge = torch.from_numpy(edge).to(device)
    return edge

def total_boundary_pixels(image,dims=[-1,-2]):
    tot = 0
    for dim in dims:
        idx1 = [slice(None) for _ in range(len(image.shape))]
        idx1[dim] = slice(1,-1)
        idx2 = [slice(None) for _ in range(len(image.shape))]
        idx2[dim] = slice(0,-2)
        tot += (image[tuple(idx1)]!=image[tuple(idx2)]).sum()
    return tot

def get_augmentation(augment_name="none",s=128,train=True,global_p=1.0,geo_aug_p=0.3):

    list_of_augs = []
    if geo_aug_p>0:
        geo_augs =  [A.ShiftScaleRotate(rotate_limit=20,p=global_p*geo_aug_p,border_mode=0)]
    else:
        geo_augs = []
    horz_sym_aug = [A.HorizontalFlip(p=global_p*0.5)]
    all_sym_aug = [A.RandomRotate90(p=global_p*0.5),
                   A.HorizontalFlip(p=global_p*0.5),
                   A.VerticalFlip(p=global_p*0.5)]

    common_color_augs = [A.HueSaturationValue(p=global_p*0.3),
                        A.RGBShift(p=global_p*0.1)]

    common_augs = [A.RandomGamma(p=global_p*0.3),
                A.RandomBrightnessContrast(p=global_p*0.3)]
    
    if augment_name == "none":
        pass
    elif augment_name == "pictures":
        if train:
            list_of_augs.extend(horz_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(common_color_augs)
            list_of_augs.extend(geo_augs)
    elif augment_name == "medical_color":
        if train:
            list_of_augs.extend(all_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(common_color_augs)
            list_of_augs.extend(geo_augs)
    elif augment_name == "medical_gray":
        if train:
            list_of_augs.extend(all_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(geo_augs)
    else:
        raise ValueError("invalid augment_name. Expected one of ['none','pictures','medical_color','medical_gray'] got "+str(augment_name))
    return A.Compose(list_of_augs)

def save_sam_features(datasets="ade20k",
                 sam_idx_or_name=0,
                 split="vali",
                 split_method="native",
                 batch_size=4,
                 ratio_of_dataset=1.0,
                 device="cuda",
                 dry=False,
                 progress_bar=True,
                 verbose=False,
                 dtype=torch.float16,
                 skip_existing=True):
    dataset = SegmentationDataset(split=split,
                            image_size=64,
                            datasets=datasets,
                            shuffle_zero=0,
                            geo_aug_p=0,
                            crop_method="sam_big",
                            split_method=split_method,
                            shuffle_datasets=False,
                            ignore_sam_idx=0)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            drop_last=False,
                                            collate_fn=custom_collate_with_info)
    n_batches = np.ceil(len(dataloader)*ratio_of_dataset).astype(int)
    data_iter = iter(dataloader)
    sam = get_sam_image_encoder(sam_idx_or_name)
    wrap = tqdm.tqdm if progress_bar else lambda x: x
    for ii in wrap(range(n_batches)):
        infos = next(data_iter)[-1]
        images = torch.stack([info["image"] for info in infos]).to(device)
        with torch.no_grad():
            image_features = sam(images)
        image_features = [f.cpu().type(dtype) for f in image_features]
        for i in range(len(infos)):
            f_i = infos[i]['i']
            filename = Path("./data") / infos[i]["dataset_name"] / f"f{f_i//1000}"/ f"{f_i}_sam{sam_idx_or_name}.pt"
            if verbose:
                print(filename)
            if not dry:
                torch.save(image_features[i],filename)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=-1)
    parser.add_argument("--process", type=int, default=-1)
    args = parser.parse_args()
    if args.process==0:
        print("PROCESSING sam features for native validation sets")
        save_sam_features(sam_idx_or_name=0,
                        split="vali",
                        split_method="native",
                        datasets="cityscapes,coco,ade20k",
                        batch_size=4,
                        progress_bar=True)
    elif args.process==1:
        print("PROCESSING sam features for all datasets")
        save_sam_features(sam_idx_or_name=0,
                        split="all",
                        split_method="random",
                        datasets="all",
                        batch_size=4,
                        progress_bar=True)
    if args.unit_test==0:
        print("UNIT TEST 0: display batch of categorical ball data")
        import jlc
        import matplotlib.pyplot as plt
        dataloader = torch.utils.data.DataLoader(CatBallDataset(size=64,dataset_len=10),batch_size=10,shuffle=False,collate_fn=custom_collate_with_info)
        x,info = next(iter(dataloader))
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        images = torch.cat((x,im),dim=0).clamp(0,1)
        jlc.montage(images)
        plt.show()
    elif args.unit_test==1:
        print("UNIT TEST 1: display batch of segmentation data")
        import jlc
        import matplotlib.pyplot as plt
        datasets = "non-medical"
        dataloader = torch.utils.data.DataLoader(SegmentationDataset(split="train",
                                                                     datasets=datasets,
                                                                     image_size=128,
                                                                     label_map_method="all",
                                                                     semantic_prob=0.5,
                                                                     shuffle_nonzero_labels=True,
                                                                     ),batch_size=20,shuffle=True,collate_fn=custom_collate_with_info)
        x,info = next(iter(dataloader))
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        images = torch.cat((x.repeat(1,3,1,1)/7,im*0.5+0.5),dim=0).clamp(0,1)
        text = [str(info_i["dataset_name"])+","+str(info_i["label_path"]) for info_i in info]*2
        jlc.montage(images,n_col=10,text=text,text_color="red")
        plt.show()
    elif args.unit_test==2:
        print("UNIT TEST 2: display edges with cv2 of the label")
        import cv2
        import jlc
        import matplotlib.pyplot as plt
        datasets = "non-medical"
        dataloader = torch.utils.data.DataLoader(SegmentationDataset(split="train",datasets=datasets,image_size=128),batch_size=20,shuffle=True,collate_fn=custom_collate_with_info)
        x,info = next(iter(dataloader))
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        x = label_boundaries(x)
        dataset = [info_i["dataset_name"] for info_i in info]
        text = dataset*2
        images = torch.cat((x.repeat(1,3,1,1),im*0.5+0.5),dim=0).clamp(0,1)
        jlc.montage(images,n_col=10,text=text,text_color="red")
        plt.show()
    elif args.unit_test==3:
        print("UNIT TEST 3: display n random crops of the label")
        import cv2
        import jlc
        import matplotlib.pyplot as plt
        num_images_vis = 10
        num_crops_vis = 10
        dataloader = torch.utils.data.DataLoader(SegmentationDataset(split="train",
                                                                     datasets="non-medical",#"coift,hrsod".split(","),#
                                                                     image_size=128,
                                                                     crop_method="full_image"),
                                                 batch_size=num_images_vis,shuffle=True,collate_fn=custom_collate_with_info)
        dataloader.dataset.num_crops = 3
        dataloader.dataset.crop_method = "most_border"
        gcp = dataloader.dataset.get_crop_params
        _,info = next(iter(dataloader))
        crops = []

        fig,ax = plt.subplots(2,num_images_vis)
        print(ax.shape)
        for i in range(num_images_vis):
            label = np.array(Image.open(os.path.join(dataloader.dataset.data_root,info[i]["dataset_name"],info[i]["label_path"])))
            image = np.array(Image.open(os.path.join(dataloader.dataset.data_root,info[i]["dataset_name"],info[i]["image_path"])))
            crops.append([])
            for j in range(num_crops_vis):
                crop_x,crop_y = gcp(label)
                crops[i].append((crop_x,crop_y))
            ax[0,i].imshow(image)
            ax[1,i].imshow(label)
            for x,y in crops[i]:
                ax[1,i].plot([x[0],x[0],x[1],x[1],x[0]],[y[0],y[1],y[1],y[0],y[0]],color="red")

        plt.show()
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        images = torch.cat((x.repeat(1,3,1,1),im*0.5+0.5),dim=0).clamp(0,1)
        jlc.montage(images,n_col=10,text=text,text_color="red")
        plt.show()
    elif args.unit_test==4:
        print("UNIT TEST 4: cityscapes semantic segmentation")
        import jlc
        import matplotlib.pyplot as plt
        datasets = "cityscapes"
        dataset = SegmentationDataset(split="train",
                                    datasets=datasets,
                                    image_size=128,
                                    label_map_method="all",
                                    semantic_prob=1.0,
                                    shuffle_nonzero_labels=False,
                                    shuffle_datasets=False,
                                    )
        x,info = dataset[16]
        print(info["i"])
        im = info["image"]
        label = x
        plt.subplot(1,2,1)
        plt.imshow(im.permute(1,2,0)*0.5+0.5)
        plt.subplot(1,2,2)
        plt.imshow(label[0])
        plt.show()
    elif args.unit_test==5:
        print("UNIT TEST 5: total_boundary_pixels")
        import jlc
        import matplotlib.pyplot as plt
        datasets = "non-medical"
        dataloader = torch.utils.data.DataLoader(SegmentationDataset(split="train",datasets=datasets,image_size=128),batch_size=20,shuffle=True,collate_fn=custom_collate_with_info)
        x,info = next(iter(dataloader))
        im = torch.stack([info_i["image"] for info_i in info],dim=0)
        tot = [total_boundary_pixels(label) for label in x]
        dataset = [info_i["dataset_name"] for info_i in info]
        text = dataset+tot
        maxvals_dim123 = torch.amax(x,dim=(1,2,3),keepdim=True)
        images = torch.cat((x.repeat(1,3,1,1)/maxvals_dim123,im*0.5+0.5),dim=0).clamp(0,1)
        jlc.montage(images,n_col=10,text=text,text_color="red")
        plt.show()
        
if __name__=="__main__":
    main()
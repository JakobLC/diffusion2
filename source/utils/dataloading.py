import torch
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
from pathlib import Path
import os
import albumentations as A
import cv2
import copy
from argparse import Namespace
import jlc
from turbojpeg import TurboJPEG,TJPF_RGB
import warnings
#add source folder if it is not already in PATH
import sys
if not str(Path(__file__).parent.parent) in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from source.models.unet import get_sam_image_encoder, cond_image_prob_keys, dynamic_image_keys
import tqdm
#from source.models.cond_vit import ModelInputKwargs, cond_image_keys, cond_image_prob_keys, dynamic_image_keys
from source.utils.mixed import (load_json_to_dict_list, save_dict_list_to_json,
                                sam_resize_index,is_nan_float,get_named_datasets, 
                                nice_split,str_to_seed,ambiguous_info_from_fn)
from source.utils.argparsing import get_current_default_version,load_existing_args,TieredParser
import shutil
import pandas as pd

turbo_jpeg = TurboJPEG()

required_class_table_info_keys = ["i","imshape","classes","class_counts","dataset_name"]

def bbox_image_from_label(label,num_bbox=None,padding_idx=255,fill_val=-1):
    return points_image_from_label(label,num_points=num_bbox,padding_idx=padding_idx,bbox_instead=True,fill_val=fill_val)

def points_image_from_label(label,num_points=None,padding_idx=255,bbox_instead=False,fill_val=0):
    assert torch.is_tensor(label)
    assert len(label.shape)==3
    assert label.shape[0]==1
    if num_points == 0:
        return torch.zeros_like(label,dtype=torch.float32)
    if num_points is None:
        num_points = np.random.choice([1,1,1,1,1,1,1,1,8,
                                       2,2,2,2,2,2,2,7,7,
                                       3,3,3,3,3,3,6,6,6,
                                       4,4,4,4,4,5,5,5,5])
    counts = torch.bincount(label.cpu().flatten())
    if len(counts)>padding_idx:
        counts[padding_idx] = 0
    nonzero_counts_idx = torch.where(counts>0)[0].cpu().numpy()
    label_indices = np.random.choice(nonzero_counts_idx,size=num_points,replace=True)
    
    if bbox_instead:
        #constructs bounding boxes as squares around masks with the same index.
        #non bbox pixels are set to padding
        bbox_image = torch.ones_like(label,dtype=int)*fill_val
        for i in np.unique(label_indices):
            mask_i = label_indices==i
            mask = label==i
            if mask.sum()==0:
                continue
            _,d1,d2 = torch.nonzero(mask,as_tuple=True)
            d1,d2 = d1.cpu(),d2.cpu()
            d1_min = d1.min()
            d1_max = d1.max()
            d2_min = d2.min()
            d2_max = d2.max()
            bbox_image[:,d1_min:d1_max+1,d2_min] = i
            bbox_image[:,d1_min:d1_max+1,d2_max] = i
            bbox_image[:,d1_min,d2_min:d2_max+1] = i
            bbox_image[:,d1_max,d2_min:d2_max+1] = i
        return bbox_image.to(label.device)
    else:
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

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,split="train",
                      image_size=128,
                      datasets="non-medical",
                      split_ratio = [0.8,0.1,0.1],
                      min_rel_class_area=0.0,
                      min_crop=0.5,
                      max_num_classes=8,
                      crop_method="multicrop_most_border",
                      num_crops=3,
                      semantic_prob=0.5,
                      map_excess_classes_to="largest",
                      shuffle_labels=True,
                      shuffle_zero=True,
                      shuffle_datasets=True,
                      data_root=None,
                      use_pretty_data=True,
                      geo_aug_p=0.3,
                      padding_idx=255,
                      split_method="random",
                      sam_features_idx=-1,
                      ignore_sam_idx=-1,
                      delete_info_keys=[],
                      conditioning=False,
                      load_cond_probs=None,
                      load_matched_items=True,
                      save_matched_items=False,
                      imagenet_norm=True,
                      ambiguous_mode=False,):
        self.ambiguous_mode = ambiguous_mode
        self.delete_info_keys = delete_info_keys
        self.conditioning = conditioning
        self.load_cond_auto = load_cond_probs is not None
        self.load_cond_probs = load_cond_probs
        if self.load_cond_probs is not None:
            assert isinstance(self.load_cond_probs,dict), "load_cond_probs must be a dict or None"
            assert all([k in cond_image_prob_keys for k in self.load_cond_probs.keys()]), "unexpected key, got "+str(self.load_cond_probs.keys())+" expected "+str(cond_image_prob_keys)
        self.ignore_sam_idx = ignore_sam_idx
        if isinstance(sam_features_idx,str):
            sam_strs = ['none','sam_vit_b','sam_vit_l','sam_vit_h']
            assert sam_features_idx in sam_strs, "sam_features_idx must be one of "+str(sam_strs)+", got "+sam_features_idx
            sam_features_idx = sam_features_idx.index(sam_features_idx)-1
        self.sfi = sam_features_idx
        self.geo_aug_p = geo_aug_p
        self.shuffle_datasets = shuffle_datasets
        self.use_pretty_data = use_pretty_data
        if data_root is None:
            data_root = str(Path(__file__).parent.parent.parent / "data")
        self.data_root = data_root
        self.image_size = image_size
        self.min_rel_class_area = min_rel_class_area
        self.min_crop = min_crop
        self.max_num_classes = max_num_classes
        if max_num_classes>255:
            raise NotImplementedError("max_num_classes>255 not implemented since the implementation being based on uint8. Note the value 255 is reserved for padding") #TODO: enable max_num_classes>255
        self.datasets = datasets
        self.semantic_prob = semantic_prob
        self.padding_idx = padding_idx
        self.save_matched_items = save_matched_items
        self.load_matched_items = load_matched_items
        self.gen_mode = False
        legal_crops = ["multicrop_most_border","multicrop_most_classes","full_image","sam_small","sam_big","sam_lidc64"]
        assert crop_method in legal_crops, f"crop_method must be one of {legal_crops}, got {crop_method}"
        if crop_method.startswith("sam"):
            self.sam_aug_small = get_sam_aug(image_size,padval=padding_idx, imagenet_norm_p=float(imagenet_norm))
            self.sam_aug_big = get_sam_aug(1024,padval=padding_idx, imagenet_norm_p=float(imagenet_norm))
            self.lidc64_aug = get_lidc64_aug()
        else:
            raise NotImplementedError("crop_method not implemented for non-sam")
        self.crop_method = crop_method
        assert map_excess_classes_to in ["largest","random_different","random_same","zero","same","nearest_expensive"]
        self.map_excess_classes_to = map_excess_classes_to
        self.shuffle_labels = shuffle_labels
        self.shuffle_zero = shuffle_zero
        self.num_crops = num_crops
        self.downscale_thresholding_factor = 3
        self.datasets_info = load_json_to_dict_list(str(Path(data_root) / "datasets_info_live.json"))
        self.dataset_list = get_named_datasets(self.datasets,datasets_info=self.datasets_info)
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
        self.didx_to_item_idx = []
        self.datasets_info = {d["dataset_name"]: d for d in self.datasets_info if d["dataset_name"] in self.dataset_list}

        assert split_method in ["random","native","native_train"]
        self.split_method = split_method
        for dataset_name in self.dataset_list:
            print("processing dataset: ",dataset_name)
            info_json = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,"info.jsonl"))
            if self.ignore_sam_idx>=0:
                info_json = [info for info in info_json if not info.get("sam",[0,0,0])[self.ignore_sam_idx]]
            N = len(info_json)
            if self.shuffle_datasets:
                previous_seed = np.random.get_state()[1][0]
                dataset_specific_seed = str_to_seed(dataset_name)
                self.datasets_info[dataset_name]["dataset_specific_seed"] = dataset_specific_seed           
                np.random.seed(seed=dataset_specific_seed)
                randperm = np.random.permutation(N)
                np.random.seed(previous_seed)
            else:
                randperm = np.arange(N)
            if split_method=="native_train":
                use_idx = self.get_use_idx_native_train(randperm,info_json,dataset_name)
            elif split_method=="native":
                use_idx = np.array([i for i in range(N) if info_json[i].get("split_idx",0)==split])
            elif split_method=="random":
                start = max(0,np.floor(self.split_start_and_stop[split][0]*N).astype(int))
                stop = min(N,np.floor(self.split_start_and_stop[split][1]*N).astype(int))
                use_idx = randperm[start:stop]
            
            if len(use_idx)==0:
                warnings.warn("no data in dataset "+dataset_name+" satisfying the criteria")
                continue
            
            items = []
            if self.load_matched_items:
                items,match_dict = self.get_matched_items(dataset_name,use_idx)
            if len(items)==0:
                file_format = self.datasets_info[dataset_name]["file_format"]
                for idx in use_idx:
                    item = info_json[idx]
                    item["image_path"] = os.path.join("f"+str(idx//1000),str(idx)+"_im."+file_format)
                    item["label_path"] = os.path.join("f"+str(idx//1000),str(idx)+"_la.png")
                    if self.use_pretty_data and item.get("pretty",False):
                        item["image_path"] = item["image_path"].replace("_im."+file_format,"_pim."+file_format)
                        item["label_path"] = item["label_path"].replace("_la.png","_pla.png")
                    if self.conditioning:
                        item = self.process_dyn_cond(item,use_idx)                 
                    else:
                        if "conditioning" in item:
                            del item["conditioning"]
                    for k in delete_info_keys:
                        del item[k]
                    item["dataset_name"] = dataset_name
                    items.append(item)
                if self.save_matched_items and len(items)>0:
                    self.save_matched_items_new(items,match_dict,dataset_name)
                    print("Saved new items for dataset: ",dataset_name)

            class_dict = load_json_to_dict_list(os.path.join(self.data_root,dataset_name,"idx_to_class.json"))[0]
            self.idx_to_class[dataset_name] = class_dict
            assert len(class_dict)==self.datasets_info[dataset_name]["num_classes"], ("num_classes in idx_to_class.json does not match num_classes in info.json. found "+
                                                                                      str(len(class_dict))+" and "+str(self.datasets_info[dataset_name]["num_classes"])+
                                                                                      " for dataset "+dataset_name)
            self.augment_per_dataset[dataset_name] = get_augmentation(self.datasets_info[dataset_name]["aug"],s=self.image_size,train=split==0,geo_aug_p=self.geo_aug_p)
            self.didx_to_item_idx.extend([f"{dataset_name}/{i}" for i in use_idx])
            self.length += len(items)
            self.items.extend(items)

        self.didx_to_item_idx = {k: i for i,k in enumerate(self.didx_to_item_idx)}
        self.len_per_dataset = {dataset_name: len([item for item in self.items if item["dataset_name"]==dataset_name]) for dataset_name in self.dataset_list}
        #assert all([self.len_per_dataset[dataset_name]>0 for dataset_name in self.dataset_list]), "no data in one of the datasets satisfying the criteria"
            
        self.dataset_weights = {}
        for dataset_name in self.dataset_list:
            w = self.datasets_info[dataset_name]["rel_weight"]/self.len_per_dataset[dataset_name] if self.len_per_dataset[dataset_name]>0 else 0
            self.dataset_weights[dataset_name] = w
        self.dataset_to_label = {dataset: i for i, dataset in enumerate(["none"]+self.dataset_list)}

    def get_matched_items(self,dataset_name,use_idx,
                          match_other=["len_use_idx","current_args_version","dataset_specific_seed"],
                          match_keys=["split_method",
                                      "split",
                                      "delete_info_keys",
                                      "conditioning",
                                      "use_pretty_data",
                                      "shuffle_datasets",
                                      "ignore_sam_idx"]):
        """
        loads a saved file of processed items to save time. The items are
        only loaded if a dict that fully matches the previous saved setup
        """
        match_dict = {key: getattr(self,key) for key in match_keys}
        if "len_use_idx" in match_other:
            match_dict["len_use_idx"] = len(use_idx)
        if "current_args_version" in match_other:
            match_dict["current_args_version"] = get_current_default_version()
        if "dataset_specific_seed" in match_other:
            match_dict["dataset_specific_seed"] = self.datasets_info[dataset_name].get("dataset_specific_seed",-1)
        match_dict_filepath = f"./data/{dataset_name}/match_items/match_dict.json" 
        if Path(match_dict_filepath).exists():
            loaded_match_dicts = load_json_to_dict_list(match_dict_filepath)
        else:
            loaded_match_dicts = []
        matched = False
        for k,match_dict_compare in enumerate(loaded_match_dicts):
            if match_dict_compare==match_dict:
                matched = True
                break
        if matched:
            items = load_json_to_dict_list(f"./data/{dataset_name}/match_items/items_{k:05d}.json")
        else:
            items = []
        return items,match_dict

    def save_matched_items_new(self,items,match_dict,dataset_name):
        filepath = f"./data/{dataset_name}/match_items"
        match_dict_filepath = f"./data/{dataset_name}/match_items/match_dict.json" 
        if not Path(filepath).exists():
            os.makedirs(filepath)
        if Path(match_dict_filepath).exists():
            loaded_match_dicts = load_json_to_dict_list(match_dict_filepath)
        else:
            loaded_match_dicts = []
        matched = False
        for match_dict_compare in loaded_match_dicts:
            if match_dict_compare==match_dict:
                matched = True
                break
        if not matched:
            #The items are new, and we therefore save
            if not Path(match_dict_filepath).exists():
                save_dict_list_to_json(match_dict,match_dict_filepath,append=False)
            else:
                save_dict_list_to_json(match_dict,match_dict_filepath,append=True)
            savename = os.path.join(filepath,f"items_{len(loaded_match_dicts):05d}.json")
            save_dict_list_to_json(items,savename)
        return
        
    def __len__(self):
        return self.length
    
    def process_dyn_cond(self,item,use_idx,keep_atmost=8):
        if "conditioning" not in item.keys():
            item["conditioning"] = {}
        item["conditioning"]["same_dataset"] = np.random.choice(use_idx,min(keep_atmost,len(use_idx)),replace=False).tolist()
        for key in dynamic_image_keys:
            if key in item["conditioning"] and key!="same_dataset":
                good_idx = np.flatnonzero(np.isin(item["conditioning"][key],use_idx))
                if len(good_idx)>keep_atmost:
                    good_idx = good_idx[:keep_atmost]
                item["conditioning"][key] = np.array(item["conditioning"][key])[good_idx].tolist()
        return item
    
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

    def class_table_from_info(self,info,label=None,as_dict=False,origin="image"):
        #["i","imshape","classes","class_counts","dataset_name"]
        rqtik = required_class_table_info_keys
        assert all([k in info for k in rqtik]), "info must contain the keys: "+str(rqtik)+", got "+str(info.keys())
        if label is not None:
            counts = np.bincount(label[label!=self.padding_idx].flatten()) #for 1024x1024 np.uint8 image, takes ~1ms on my machine (AMD Ryzen 7 7800X3d 8-Core Processor x 16)
        else:
            #we use the precomputed vals in info, and rescale to number of nonpadded pixels
            h,w = sam_resize_index(*info["imshape"][:2],self.image_size)
            num_pixels_unpadded = h*w
            counts = np.array(info["class_counts"],dtype=float)
            counts = counts/sum(counts)
            counts = np.round(num_pixels_unpadded*counts).astype(int).tolist()

        class_table = pd.DataFrame(columns=["idx_dataset","idx_old","idx_new","name","count","key_origin"],
                                   index=range(len(counts)))
        #set the counts
        class_table["count"] = counts
        class_table["idx_old"] = range(len(counts))
        class_table["idx_dataset"] = info["classes"]
        class_table["name"] = [self.idx_to_class[info["dataset_name"]][str(c)] for c in info["classes"]]
        class_table["key_origin"] = [origin for _ in range(len(counts))]
        assert origin in ["image"]+dynamic_image_keys, f"origin must be 'image' or in dynamic_image_keys={dynamic_image_keys}, got {origin}"
        origin_int = len(dynamic_image_keys) if origin=="image" else dynamic_image_keys.index(origin)
        class_table["key_int"] = [origin_int for _ in range(len(counts))]
        class_table["rel_area"] = class_table["count"]/sum(class_table["count"])
        class_table["rel_area_big_enough"] = (class_table["rel_area"]>=self.min_rel_class_area).astype(int)
        class_table["is_main_image_zero"] = np.logical_and(class_table["key_origin"]=="image",class_table["idx_dataset"]==0).astype(int)
        if as_dict:
            class_table = class_table.to_dict()
        return class_table

    def process_class_table(self,class_table,semantic,delete_empty_from_class_table=True,info=None):
        assert self.map_excess_classes_to in ["largest","random_different","random_same","zero","same","nearest_expensive"], f"expected map_excess_classes_to to be one of ['largest','random_different','random_same','zero','same','nearest_expensive'], got {self.map_excess_classes_to}"
        mnc = self.max_num_classes
        sort_keys = ["is_main_image_zero","rel_area_big_enough","key_int","count"]
        sorted_indices = np.argsort(class_table.copy().sort_values(sort_keys,ascending=False).index)
        class_table["priority"] = sorted_indices
        if semantic:
            #each unique class has its lowest priority index assigned to it
            minimum_class_priority = {}
            for row in class_table.iterrows():
                j = row[1]["idx_dataset"]
                p = row[1]["priority"]
                minimum_class_priority[j] = minimum_class_priority.get(j,[])+[p]

            for k,v in minimum_class_priority.items():
                minimum_class_priority[k] = min(v)
            
            keys = list(minimum_class_priority.keys())
            vals = list(minimum_class_priority.values())
            #remove gaps in priority, i.e [0,4,3] -> [0,2,1]
            vals = np.argsort(np.argsort(vals))
            minimum_class_priority = {k: v for k,v in zip(keys,vals)}
            for i in range(len(class_table)):
                j = class_table.loc[i]["idx_dataset"]
                class_table.loc[i,"idx_new"] = minimum_class_priority[j]
        else:
            class_table["idx_new"] = class_table["priority"]
        
        class_table["is_excess"] = np.logical_or(class_table["idx_new"]>=mnc, 
                                                 class_table["rel_area_big_enough"]==False
                                                )
        excess_idx = class_table["is_excess"].to_numpy().nonzero()[0].astype(int).tolist()
        if len(excess_idx)>0:
            #what should be done with the excess classes ?
            if self.map_excess_classes_to=="largest": #smallest classes with less area in the image are mapped to the largest label
                image_origin_mask = np.array(class_table["key_origin"])=="image"
                i_largest_class = np.argmax(class_table["count"].to_numpy()*image_origin_mask).item()
                try:
                    class_table.loc[excess_idx,"idx_new"] = class_table.loc[i_largest_class]["idx_new"]
                except KeyError:
                    print("i_largest_class: ",i_largest_class)
                    print("excess_idx: ",excess_idx)
                    print("class_table: ",class_table)
                    raise
            elif self.map_excess_classes_to=="random_different": #all labels are randomly associated with a different label
                if semantic:
                    random_classes = np.random.choice(mnc,size=len(excess_idx))
                    dataset_idx_of_excess = class_table.loc[excess_idx,"idx_dataset"].to_numpy()
                    for uq in np.unique(dataset_idx_of_excess):
                        random_classes[dataset_idx_of_excess==uq] = np.random.choice(random_classes[dataset_idx_of_excess==uq])
                else:
                    random_classes = np.random.choice(mnc,size=len(excess_idx))
                class_table.loc[excess_idx,"idx_new"] = random_classes
            elif self.map_excess_classes_to=="random_same": #all labels are randomly associated with the same label
                random_class = np.random.choice(mnc)
                class_table.loc[excess_idx,"idx_new"] = random_class
            elif self.map_excess_classes_to=="zero":
                class_table.loc[excess_idx,"idx_new"] = 0
            elif self.map_excess_classes_to=="same":
                pass
            elif self.map_excess_classes_to=="nearest_expensive": #not implemented yet, supposed to find maximum bordering class and assign to that
                raise NotImplementedError("nearest_expensive not implemented yet")
        if self.shuffle_labels:
            if self.shuffle_zero:
                perm = np.random.permutation(mnc)
            else:
                perm = np.array([0]+list(np.random.permutation(mnc-1)+1))
            perm_f = lambda x: np.vectorize(lambda y: perm[y])(x)
        else:
            perm_f = lambda x: x
        class_table["idx_new"] = perm_f(class_table["idx_new"].to_numpy())
        old_to_new = {k: v for k,v in zip(class_table["idx_old"],class_table["idx_new"])}
        old_to_new[-1] = self.padding_idx
        
        new_class_table_columns = ["idx_new","idx_old","idx_dataset","name","count","key_origin"]
        new_class_table = pd.DataFrame(columns=new_class_table_columns,index=range(mnc))
        
        new_class_table["idx_new"] = range(mnc)
        new_class_table = new_class_table.map(lambda x: [] if is_nan_float(x) else x)
        for row in class_table.iterrows():
            i = row[1]["idx_new"]
            for k in new_class_table_columns[1:]:
                new_class_table.loc[i,k].append(row[1][k])

        idx_to_dataset_idx = {self.padding_idx: -1}
        idx_to_class_name = {self.padding_idx: "padding"}
        num_main_image_classes = 0
        for i in range(mnc):
            if len(new_class_table.loc[i]["name"])>0:
                if "image" in new_class_table.loc[i]["key_origin"]:
                    #j is the best (largest area) index of "image" origin classes
                    image_origin_mask = np.array(new_class_table.loc[i]["key_origin"])=="image"
                    c = np.array(new_class_table.loc[i]["count"])
                    j = np.argmax(c*image_origin_mask)
                    num_main_image_classes += 1
                else:
                    #j is the best (largest area) index of any origin classes
                    j = np.argmax(new_class_table.loc[i]["count"])
                idx_to_dataset_idx[i] = new_class_table.loc[i]["idx_dataset"][j]
                idx_to_class_name[i] = new_class_table.loc[i]["name"][j]
        new_class_table["new_count"] = new_class_table["count"].apply(sum)
        if delete_empty_from_class_table:
            new_class_table = new_class_table[new_class_table["idx_dataset"].apply(len)>0]
        return (class_table, 
                new_class_table, 
                old_to_new, 
                idx_to_dataset_idx, 
                idx_to_class_name, 
                num_main_image_classes)

    def map_label_to_valid_bits(self,label,info):
        """takes a uint8 label map and depending on the method, maps
        each label to a number between 0 and max_num_classes-1. also
        keeps track of which classes correspond to idx and adds this 
        dict to info."""
        has_cond = len(info.get("cond",{}))>0
        semantic = self.semantic_prob>=np.random.rand() and self.semantic_prob>0
        info["semantic"] = semantic
        class_table = self.class_table_from_info(info)
        if has_cond:
            for k in list(info["cond"].keys()):
                delta = len(class_table)
                class_table_v = self.class_table_from_info(info["cond"][k][-1],origin=k)
                class_table_v["idx_old"] += delta
                info["cond"][k][0] = info["cond"][k][0].astype(int)
                info["cond"][k][0][info["cond"][k][0]==self.padding_idx] = -1
                info["cond"][k][0][info["cond"][k][0]>0] += delta
                class_table = pd.concat([class_table,class_table_v],axis=0,ignore_index=True)
        class_table,new_class_table,old_to_new,idx_to_dataset_idx,idx_to_class_name,num_classes = self.process_class_table(class_table,semantic,info=info)
        label = label.astype(int)
        label[label==self.padding_idx] = -1
        if not self.ambiguous_mode:
            label = np.vectorize(old_to_new.get)(label)

        info["idx_to_class_name"] = idx_to_class_name
        info["idx_to_dataset_idx"] = idx_to_dataset_idx
        info["old_to_new"] = old_to_new
        info["num_labels"] = num_classes
        if has_cond:
            for k in info["cond"].keys():
                info["cond"][k][0] = np.vectorize(old_to_new.get)(info["cond"][k][0])
        return label,info

    def load_cond_image_label(self,info,probs):
        if "conditioning" not in info:
            return info
        dataset_name = info["dataset_name"]
        type_of_load = []
        didx_to_load = []
        illegal_idx = [info["i"]]
        for key in dynamic_image_keys:
            if self.gen_mode:
                p = 1.0
            else:
                p = probs.get("p_"+key,0.0)
            if p>0:
                if np.random.rand()<=p:
                    idx = sample_from_list_in_dict(dict_=info["conditioning"], key=key, num_samples=1, illegal_idx=illegal_idx)
                    type_of_load.extend([key for _ in range(len(idx))])
                    didx_to_load.extend([f"{dataset_name}/{i}" for i in idx])
        if len(didx_to_load)>0:
            info["cond"] = {}
            for t,didx in zip(type_of_load,didx_to_load):
                item = self.__getitem__({"didx": didx, "load_cond": False, "is_cond_call": True})
                info_subset = {k: v for k,v in item[-1].items() if k in required_class_table_info_keys}
                info["cond"][t] = [item[0],item[1],info_subset]
        
    def preprocess(self,image,label,info):
        #if image is smaller than image_size, pad it
        if self.crop_method.startswith("sam"):
            image,label = self.augment(image,label,info)
            if self.crop_method=="sam_big":
                augmented = self.sam_aug_big(image=image,mask=label)
            elif self.crop_method=="sam_lidc64":
                augmented = self.lidc64_aug(image=image,mask=label)
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
    
    def load_raw_image_label(self,x,longest_side_resize=0,data_root=None):
        if data_root is None:
            data_root = self.data_root
        if isinstance(x,str):
            x = {"dataset_name": x.split("/")[0], "i": int(x.split("/")[1])}
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
        image = open_image_fast(image_path,num_channels=3)
        label = open_image_fast(label_path,num_channels=1)
        if longest_side_resize>0:
            image = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_AREA, always_apply=True, p=1)(image=image)["image"]
            label = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_NEAREST, always_apply=True, p=1)(image=label)["image"]
        return image,label

    def process_input(self,idx):
        if isinstance(idx,int):
            idx_d = {"idx": idx}
        elif isinstance(idx,str):
            idx_d = {"idx": self.didx_to_item_idx[idx]}
        else:
            assert isinstance(idx,dict), "idx must be an integer or a dictionary or a str, got: "+str(type(idx))
            idx_d = idx
            if "idx" in idx_d:
                pass
            elif "didx" in idx_d:
                idx_d["idx"] = self.didx_to_item_idx[idx["didx"]]
            else:
                assert "i" in idx and "dataset_name" in idx, "idx must contain 'i' and 'dataset_name' keys, or be a didx str or have the 'didx' field"
                idx["idx"] = self.didx_to_item_idx[f"{idx['dataset_name']}/{idx['i']}"]
        load_cond = idx_d.get("load_cond",self.load_cond_auto)
        load_cond_probs = idx_d.get("load_cond_probs",self.load_cond_probs)
        is_cond_call = idx_d.get("is_cond_call",False)
        if load_cond and is_cond_call:
            raise ValueError("Cannot load cond inputs in a cond function call to avoid recursions")
        return idx_d["idx"],load_cond,load_cond_probs,is_cond_call

    def images_to_torch(self,image,label,info):
        image = torch.tensor(image).permute(2,0,1)
        label = torch.tensor(label).unsqueeze(0)
        if "cond" in info.keys():
            info["cond"] = {k: [torch.tensor(v[0]).unsqueeze(0),
                                torch.tensor(v[1]).permute(2,0,1),
                                v[2]] for k,v in info["cond"].items()}
        return image,label,info
    
    def load_all_ambiguous_labels(self,info):
        i = info["i"]
        fn_info = ambiguous_info_from_fn(info["fn"])
        labels = []
        for j in range(-fn_info["m_i"],fn_info["m_tot"]-fn_info["m_i"]):
            idx = i+j
            label_path = os.path.join(self.data_root,
                                      info["dataset_name"],
                                      os.path.join("f"+str(idx//1000),str(idx)+"_la.png"))
            labels.append(open_image_fast(label_path,num_channels=0))
        return np.stack(labels,axis=-1),fn_info["m_i"]

    def __getitem__(self, idx):
        idx,load_cond,load_cond_probs,is_cond_call = self.process_input(idx)
        info = copy.deepcopy(self.items[idx])
        if load_cond:
            self.load_cond_image_label(info,load_cond_probs)
        dataset_name = info["dataset_name"]
        image_path = os.path.join(self.data_root,dataset_name,info["image_path"])
        label_path = os.path.join(self.data_root,dataset_name,info["label_path"])
        image = open_image_fast(image_path,num_channels=3)
        if self.ambiguous_mode:
            label,m_i = self.load_all_ambiguous_labels(info)
        else:
            label = open_image_fast(label_path,num_channels=0) #num_channels=0 means 2D

        image,label = self.preprocess(image,label,info)
        if is_cond_call:
            if not self.crop_method.startswith("sam"):
                raise NotImplementedError("Conditional calls are only implemented for sam copping method")
            return label,image,info
        label,info = self.map_label_to_valid_bits(label,info)
        if not self.crop_method.startswith("sam"): #TODO, not implemented
            image,label = self.augment(image,label,info)
            image = image.astype(np.float32)*(2/255)-1
        if self.ambiguous_mode:
            info["amb_label"] = label
            label = label[:,:,m_i]
        image,label,info = self.images_to_torch(image,label,info)
        info["image"] = image
        if self.sfi>=0:
            if info["sam"][self.sfi]:
                i = info["i"]
                info["image_features"] = torch.load(os.path.join(self.data_root,dataset_name,f"f{i//1000}",f"{i}_sam{self.sfi}.pt"))
            else:
                info["image_features"] = None
        return label,info

def delete_all_matched_items(datasets=None,dry=False,verbose=True):
    #if datasets is none, we delete all
    if datasets is None:
        datasets = get_all_valid_datasets()
    for dataset_name in datasets:
        match_items_dir = os.path.join(str(Path(__file__).parent.parent / "data"),dataset_name,"match_items")
        if os.path.exists(match_items_dir):
            if not dry:
                shutil.rmtree(match_items_dir)
            if verbose:
                print(f"Deleted matched items for dataset {('(dry)' if dry else '')}: ",dataset_name)

def sample_from_list_in_dict(dict_,key,illegal_idx=[],num_samples=1):
    out = []
    if key in dict_.keys():
        v = dict_[key]
        lenv = len(v)
        n = num_samples
        if lenv>0 and n>0:
            out = np.random.choice(v,size=min(n,lenv),replace=False)
            if len(illegal_idx)>0:
                out = [o for o in out if o not in illegal_idx]
    return out

def longest_side_resize_func(image,is_label=True,max_size=256):
    return A.LongestMaxSize(max_size=max_size, 
                            interpolation=cv2.INTER_NEAREST if is_label else cv2.INTER_AREA, 
                            always_apply=True, 
                            p=1)(image=image)["image"]

def load_from_dataset_and_idx(dataset_name,i,im=True):
    if im:
        possible_filesnames = ["_pim.jpg","_pim.png","_im.jpg","_im.png"]
    else:
        possible_filesnames = ["_pla.png","_la.png"]
    root_name = f"f{str(i//1000)}{os.sep}{str(i)}"
    root_dir = os.path.join(str(Path(__file__).parent.parent.parent / "data"),dataset_name)
    for pf in possible_filesnames:
        if os.path.exists(os.path.join(root_dir,root_name)+pf):
            return root_name+pf
    raise ValueError("No image file found for dataset_name: "+dataset_name+", i: "+str(i))

def load_raw_image_label(x,longest_side_resize=0,data_root=None):
    if data_root is None:
        data_root = str(Path(__file__).parent.parent.parent / "data")
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

def get_sam_aug(size,padval=255,imagenet_norm_p=1):
    #SAM uses the default imagenet mean and std, also default in A.Normalize
    sam_aug = A.Compose([A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, always_apply=True, p=1),
                     A.Normalize(always_apply=imagenet_norm_p>=1.0, p=imagenet_norm_p),
                     A.PadIfNeeded(min_height=size, 
                                   min_width=size, 
                                   border_mode=cv2.BORDER_CONSTANT, 
                                   value=[0, 0, 0], 
                                   mask_value=padval, 
                                   always_apply=True, 
                                   p=1, 
                                   position=A.PadIfNeeded.PositionType.TOP_LEFT)])
    return sam_aug

def get_lidc64_aug():
    lidc64_aug = A.Compose([A.RandomCrop(width=64, height=64, always_apply=True, p=1),
                     A.Normalize(always_apply=True, p=1),
                     ])
    return lidc64_aug

def open_image_fast(image_path,
                    num_channels=None):
    assert image_path.find(".")>=0, "image_path must contain a file extension"
    extension = image_path.split(".")[-1]
    if extension in ["jpg","jpeg"]:
        with open(image_path, "rb") as f:
            image = turbo_jpeg.decode(f.read(),pixel_format=TJPF_RGB)
    else:
        image = np.array(Image.open(image_path))
    if num_channels is not None:
        assert num_channels in [0,1,3,4], f"Expected num_channels to be in [0,1,3,4], got {num_channels}"
        if num_channels==0: #means only 2 dims
            if (len(image.shape)==3 and image.shape[2]==1):
                image = image[:,:,0]
            else:
                assert len(image.shape)==2, f"loaded image must either be 2D or have 1 channel when num_channels=0. got shape: {image.shape}"
        else:
            if len(image.shape)==2:
                image = image[:,:,None]
            if num_channels==1:
                assert image.shape[2]==1, f"loaded image must have at most 1 channel if num_channels==1, found {image.shape[2]}"
            elif num_channels==3:
                if image.shape[2]==1:
                    image = np.repeat(image,num_channels,axis=-1)
                elif image.shape[2]==4:
                    image = image[:,:,:3]
                else:
                    assert image.shape[2]==3, f"loaded image must have 1,3 or 4 channels if num_channels==3, found {image.shape[2]}"
            elif num_channels==4:
                if image.shape[2]==1:
                    image = np.concatenate([image,image,image,np.ones_like(image)*255],axis=-1)
                elif image.shape[2]==3:
                    image = np.concatenate([image,np.ones_like(image[:,:,0:1])*255],axis=-1)
                else:
                    assert image.shape[2]==4, f"loaded image must have 1,3 or 4 channels if num_channels==4, found {image.shape[2]}"
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
    elif augment_name in ["medical_gray","medical_grey"]:
        if train:
            list_of_augs.extend(all_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(geo_augs)
    else:
        raise ValueError("invalid augment_name. Expected one of ['none','pictures','medical_color','medical_gray'] got "+str(augment_name))
    return A.Compose(list_of_augs)

def get_all_valid_datasets():
    datasets_info = load_json_to_dict_list("./data/datasets_info_live.json")
    valid_datasets = [d["dataset_name"] for d in datasets_info if d["live"]]
    return valid_datasets

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

def get_dataset_from_args(args_or_model_id=None,
                          split="vali",
                          prioritized_didx=None,
                          mode="training",
                          return_type="dli",
                          load_cond_probs_override=None,
                          ambiguous_mode=False,
                          ):
    if args_or_model_id is None:
        #use default args with data
        args_or_model_id = TieredParser().get_args(alt_parse_args=["--model_name","default"])
        args_or_model_id.datasets = "all"
    assert split in ["train","vali","test","all",0,1,2,3], f"split must be in ['train','vali','test','all'] or its index, got {split}"
    split = {0:"train",1:"vali",2:"test",3:"all"}.get(split,split)
    assert return_type in ["dli","dl","ds"], f"return_type must be in ['dli','dl','ds'], got {return_type}"
    assert mode in ["training","pure_gen","pri_didx",None], f"sampler_mode must be in ['training','pure_gen','pri_didx',None], got {mode}"
    if isinstance(args_or_model_id,dict):
        args = Namespace(**args_or_model_id)
    elif isinstance(args_or_model_id,str):
        args = load_existing_args(args_or_model_id)
    else:
        assert isinstance(args_or_model_id,Namespace)
        args = args_or_model_id

    if load_cond_probs_override is not None:
        assert isinstance(load_cond_probs_override,float), f"load_cond_probs_override must be a float, got {load_cond_probs_override}"
        load_cond_probs = {k: load_cond_probs_override for k in dynamic_image_keys}
    else:
        load_cond_probs = {"p_"+k: args.__dict__["p_"+k] for k in dynamic_image_keys}
        
    if max(load_cond_probs.values())==0:
        conditioning = False
        load_cond_probs = None
    else:
        conditioning = True
    ds = SegmentationDataset(split=split,
                            split_ratio=[float(item) for item in args.split_ratio.split(",")],
                            image_size=args.image_size,
                            datasets=args.datasets,
                            min_rel_class_area=args.min_label_size,
                            max_num_classes=2**args.diff_channels,
                            shuffle_zero=args.shuffle_zero,
                            crop_method=args.crop_method,
                            padding_idx=255 if args.ignore_padded else 0,
                            split_method=args.split_method,
                            sam_features_idx=args.image_encoder,
                            semantic_prob=args.semantic_dl_prob,
                            conditioning=conditioning,
                            load_cond_probs=load_cond_probs,
                            save_matched_items=args.dataloader_save_processing,
                            shuffle_labels=args.agnostic,
                            imagenet_norm=args.imagenet_norm,
                            ambiguous_mode=ambiguous_mode,
                            )
    if return_type=="ds":
        return ds
    tbs = args.train_batch_size
    vbs = args.vali_batch_size if args.vali_batch_size>0 else tbs
    bs = {"train": tbs,
          "vali": vbs,
          "test": vbs,
          "all": vbs}[split]
    if mode=="training":
        sampler = ds.get_sampler(args.seed)
    elif mode=="pure_gen":
        sampler = ds.get_gen_dataset_sampler(args.datasets,args.seed)
    elif mode=="pri_didx":
        assert prioritized_didx is not None, "prioritized_didx must be provided if mode is pri_didx"
        sampler = ds.get_prioritized_sampler(prioritized_didx,seed=args.seed)
    dl = torch.utils.data.DataLoader(ds,
                                    batch_size=bs,
                                    sampler=sampler,
                                    shuffle=(sampler is None),
                                    drop_last=mode!="pure_gen",
                                    collate_fn=custom_collate_with_info,
                                    num_workers=args.dl_num_workers)
    if return_type=="dl":
        return dl
    elif return_type=="dli":
        return jlc.DataloaderIterator(dl)
    
def dummy_label(n=128,num_classes=10,as_torch=True):
    """
    Constructs a segmentation mask by choosing a set of center points
    for the classes and then labeling each pixel with the class of the
    nearest center point. 
    """
    points = np.random.rand(num_classes,2)*n
    X,Y = np.meshgrid(np.arange(n),np.arange(n))
    dist = []
    for i in range(num_classes):
        dist.append(np.sqrt((X-points[i,0])**2+(Y-points[i,1])**2))
    label = np.argmin(np.stack(dist,axis=2),axis=2)
    if as_torch:
        return torch.tensor(label).unsqueeze(0)
    else:
        return label
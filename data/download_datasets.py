import os,sys

sys.path.append(os.path.abspath('./'))

from urllib.parse import urlparse
from scipy.io import loadmat
import numpy as np
import re
import shutil
from PIL import Image
import tqdm
from data.data_utils import (unpack_files, save_dict_list_to_json, load_json_to_dict_list, rle_to_mask)
import glob
import json
import jlc.nc as nc 
from pathlib import Path
import scipy.ndimage as nd
from source.utils.mixed import quantile_normalize, prettify_classname, str_to_seed
import cv2
from source.utils.dataloading import SegmentationDataset, save_sam_features, get_all_valid_datasets
import pandas as pd
import zipfile
import skimage
import nibabel as nib
import clip
import torch
import pydicom
import pickle
import pycocotools.mask as mask_util

def default_do_step():
    return {"unpack": 0,
                "delete_f_before": 0,
                "make_f0": 1,
                "save_images": 1,
                "reset_info": 1,
                "save_info": 1,
                "save_global_info": 1,
                "save_class_dict": 1,
                "delete_unused_files": 0,
                "prettify": 0,
                "add_prettify_to_info": 1,
                "sam_features": 0,
                "add_sam_to_info": 0}

class DatasetDownloader:
    def __init__(self, mainfolder = None):
        if mainfolder is None:
            mainfolder = str(Path(os.getcwd()).parent / "diffusion2" /"data")
        self.mainfolder = mainfolder
        self.files_per_folder = 1000
        self.allowed_failure_rate = 0.01

    def create_subfolder(self, folder_name):
        subfolder_path = os.path.join(self.mainfolder, folder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        return subfolder_path

    def download_files(self, dataset_dict):
        subfolder_name = dataset_dict["dataset_name"]
        urls = dataset_dict["urls"]
        for url in urls:
            filename = os.path.basename(urlparse(url).path)
            subfolder_path = self.create_subfolder(subfolder_name)
            file_path = os.path.join(subfolder_path, filename)
            if not os.path.exists(file_path):
                print(f"Downloading {filename}...")
                wget.download(url, file_path)
                print("Download complete.")
            else:
                print(f"{filename} already exists. Skipping download.")
    
    def unpack_all_in_folder(self, folder_name):
        subfolder_path = os.path.join(self.mainfolder, folder_name)
        unpack_list = [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path)]
        for file in unpack_list:
            print(f"Unpacking {file}...")
            unpack_files(file)

    def process_files(self, 
                      name, 
                      do_step = default_do_step()):
        if isinstance(name,list):
            for n in name:
                self.process_files(n, do_step=do_step)
            return
        folder_path = os.path.join(self.mainfolder, name)
        folder_i = 0
        num_attempts = 0
        num_saved_images = 0
        file_i = 0
        split_idx = 0
        show_error_flag = False
        label_suffix = ".png"
        image_suffix = ".jpg"
        pallete=nc.largest_pallete
        save_binary_overlaps = False
        
        if do_step["unpack"]:
            self.unpack_all_in_folder(name)

        if name=="pascal":
            with open(os.path.join(folder_path,"labels.txt")) as f:
                lines = f.readlines()
            with open(os.path.join(folder_path,"train.txt")) as f:
                train = f.readlines()
            train = [(t[:-1] if t.endswith("\n") else t) for t in train]
            with open(os.path.join(folder_path,"val.txt")) as f:
                val = f.readlines()
            val = [(v[:-1] if v.endswith("\n") else v) for v in val]
            class_dict = {l[:l.find(":")]: l[l.find(":")+2:(-1 if l[-1]=="\n" else None)] for l in lines}
            class_dict[0] = "background"
            unpack_list = [os.path.join(folder_path, file_name) for file_name in ["trainval.tar.gz","trainval.tar","VOCtrainval_11-May-2012.tar"]]
            file_name_list = [file_name[:-4] for file_name in os.listdir(os.path.join(folder_path,"trainval"))]
            def load_image_label_info(file_name):
                if file_name in train:
                    split_idx = 0
                elif file_name in val:
                    split_idx = 1
                else:
                    raise ValueError("Image not found."+file_name)
                label_path = os.path.join(folder_path,"trainval",f"{file_name}.mat")
                image_path = os.path.join(folder_path,"VOCdevkit/VOC2010/JPEGImages",f"{file_name}.jpg")
                label = loadmat(label_path)["LabelMap"]
                image = Image.open(image_path)
                uq = np.unique(label).tolist()
                label2 = np.zeros_like(label)
                info = {"classes": [0], "split_idx": split_idx}
                for i,u in enumerate(uq):
                    label2[label==u] = i+1
                    info["classes"].append(u)
                    if i==255:
                        break
                label = Image.fromarray(label2)
                return image,label,info
        elif name=="coco":
            unpack_list = [os.path.join(folder_path, file_name) for file_name in [l+"2017.zip" for l in ["train", "val", "stuff_annotations_trainval"]]]
            file_name_list = (
                [file_name[:-4] for file_name in os.listdir(os.path.join(folder_path, "val2017"))]+
                [file_name[:-4] for file_name in os.listdir(os.path.join(folder_path, "train2017"))])
            cats_folder = os.path.join(folder_path,"stuff_annotations_trainval2017","annotations")
            cats_name = os.path.join(cats_folder,"stuff_val2017.json")
            cats = load_json_to_dict_list(cats_name)["categories"]
            class_dict = {0: "background"}
            for cat in cats:
                class_dict[cat["id"]] = cat["supercategory"]+"/"+cat["name"]
            def load_image_label_info(file_name):
                if os.path.isfile(os.path.join(folder_path,"train2017",file_name+".jpg")):
                    split_idx = 0
                elif os.path.isfile(os.path.join(folder_path,"val2017",file_name+".jpg")):
                    split_idx = 1
                else:
                    raise ValueError("Image not found."+file_name)
                tv = ["train","val"][split_idx]
                label_path = os.path.join(cats_folder,"stuff_"+tv+"2017_pixelmaps",file_name+".png")
                image_path = os.path.join(folder_path,tv+"2017",file_name+".jpg")
                
                label = np.array(Image.open(label_path))
                image = Image.open(image_path)
                uq = np.unique(label).tolist()
                assert len(uq)<=256, "uint8 format fails if more than 256 classes are present."
                label2 = np.zeros_like(label)
                info = {"classes": [0],
                        "split_idx": split_idx}
                for i,u in enumerate(uq):
                    if u>0:
                        if do_step["save_images"]:
                            label2[label==u] = i
                        info["classes"].append(u)
                label = Image.fromarray(label2)
                return image,label,info
        elif name=="ade20k":
            unpack_list = ["jakoblc_73c65c3c.zip"]
            root_ade20k_dir = os.path.join(folder_path,"ADE20K_2016_07_26","images","ADE",)
            search_path = os.path.join(root_ade20k_dir,"*","*","*","*.jpg")
            file_name_list = glob.glob(search_path)
            category_filename = os.path.join(folder_path,"ADE20K_2016_07_26","index_ade20k.mat")
            cats = loadmat(category_filename)["index"]
            cats = [str(c) for c in cats[0][0][6]]
            class_dict = {i+1: cats[i] for i in range(len(cats))}
            class_dict[0] = "background"
            def load_image_label_info(file_name2):
                file_name = os.path.relpath(file_name2,root_ade20k_dir)
                image_path = os.path.join(root_ade20k_dir,file_name)
                json_path = image_path.replace(".jpg",".json")
                info_json = json.load(open(json_path))

                image = Image.open(image_path)
                label = np.zeros((image.size[1],image.size[0]),dtype=np.uint8)

                info = {"classes": [0],
                        "split_idx": ["training","validation"].index(file_name.split("/")[0])}
                assert len(info_json["annotation"]["object"])<=256, "uint8 format fails if more than 256 classes are present."
                for i,d in enumerate(info_json["annotation"]["object"]):
                    if do_step["save_images"]:
                        label_i = np.array(Image.open(os.path.join(root_ade20k_dir,*file_name.split("/")[:-1],d["instance_mask"])))
                        label[label_i==255] = i+1
                    info["classes"].append(d["name_ndx"])
                label = Image.fromarray(label)
                return image,label,info
        elif name=="ade20k_v2":
            save_binary_overlaps = True
            folder_path1 = folder_path.replace("_v2","")
            unpack_list = ["jakoblc_73c65c3c.zip"]
            root_ade20k_dir = os.path.join(folder_path1,"ADE20K_2016_07_26","images","ADE",)
            search_path = os.path.join(root_ade20k_dir,"*","*","*","*.jpg")
            file_name_list = glob.glob(search_path)
            category_filename = os.path.join(folder_path1,"ADE20K_2016_07_26","index_ade20k.mat")
            category_keyname = "objectnames"
            cats = loadmat(category_filename)[category_keyname]
            class_dict = {i+1: cats[i] for i in range(len(cats))}
            class_dict[0] = "background"
            def load_image_label_info(file_name2):
                file_name = os.path.relpath(file_name2,root_ade20k_dir)
                image_path = os.path.join(root_ade20k_dir,file_name)
                json_path = image_path.replace(".jpg",".json")
                info_json = json.load(open(json_path))

                image = Image.open(image_path)
                label = np.zeros((image.size[1],image.size[0]),dtype=np.uint8)
                label_overlapped = np.zeros((image.size[1],image.size[0],3),dtype=np.uint8)
                bbox_areas = []
                for i,d in enumerate(info_json["annotation"]["object"]):
                    x_range = max(d["polygon"]["x"])-min(d["polygon"]["x"])
                    y_range = max(d["polygon"]["y"])-min(d["polygon"]["y"])
                    bbox_areas.append(x_range*y_range)
                top_24_area_idx = np.argsort(bbox_areas)[-min(24,len(bbox_areas)):][::-1]
                
                info = {"classes": [0],
                        "split_idx": ["training","validation"].index(file_name.split("\\")[0])}
                info["class_counts_overlap"] = []
                for i in range(min(24,len(bbox_areas))):
                    ii = top_24_area_idx[i]
                    d = info_json["annotation"]["object"][ii]
                    label_i = np.array(Image.open(os.path.join(root_ade20k_dir,*file_name.split("\\")[:-1],d["instance_mask"])))
                    if do_step["save_images"]:
                        mask = label_i>0
                        channel_idx = i//8
                        sub_i = i%8
                        label_overlapped[:,:,channel_idx][mask] += 2**sub_i
                        info["class_counts_overlap"].append(int(np.sum(mask)))
                        label[label_i==255] = i+1
                    info["classes"].append(d["name_ndx"])
                num_zeros = int(np.all(label_overlapped==0,axis=2).sum())
                info["class_counts_overlap"] = [num_zeros]+info["class_counts_overlap"]
                label = Image.fromarray(label)
                label_overlapped = Image.fromarray(label_overlapped)
                return image,(label,label_overlapped),info
        elif name=="cityscapes":
            image_suffix = ".png"
            labels = ['background', 'ego vehicle', 'rectification border', 'out of roi', 
            'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 
            'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 
            'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
            background_classes = list(range(7))
            class_dict = {i: labels[i] for i in range(len(labels))}
            unpack_list = ["gtFine_trainvaltest.zip","leftImg8bit_trainvaltest.zip"]
            data_folder_train = Path(self.mainfolder)/ "cityscapes" / "gtFine" / "train"
            data_folder_val = Path(self.mainfolder) / "cityscapes" / "gtFine" / "val"
            file_name_list = list(data_folder_train.glob("*/*_instanceIds.png"))+list(data_folder_val.glob("*/*_instanceIds.png"))
            file_name_list = [str(f)[:-4] for f in file_name_list]
            crop = 12
            def load_image_label_info(file_name):
                label_path = str(file_name)+".png"
                image_path = label_path.replace("_gtFine_instanceIds","_leftImg8bit").replace("gtFine","leftImg8bit")[:-4]+image_suffix

                image = np.array(Image.open(image_path))[crop:-crop,crop*2:-crop*2,:]
                image = Image.fromarray(image)
                label_instance = np.array(Image.open(label_path))[crop:-crop,crop*2:-crop*2]

                info = {"classes": [0],
                        "split_idx": ["train","val"].index(file_name.replace("/","\\").split("\\")[-3])}
                
                label = np.zeros_like(label_instance,dtype=np.uint8)
                k = 0
                for u in np.unique(label_instance):
                    if u not in background_classes:
                        k += 1
                        label[label_instance==u] = k
                        u_hat = u if u<1000 else u//1000
                        info["classes"].append(u_hat.item())
                    if k==255:
                        break
                label = Image.fromarray(label)
                return image,label,info
        elif name=="coift":
            unpack_list = ["COIFT"]
            all_names = os.listdir(os.path.join(folder_path,"COIFT","images"))
            file_name_list = [l[:l.find(".jpg")] for l in all_names]
            class_dict = {0: "background",
                          1: "foreground"}
            def load_image_label_info(file_name):
                image_path = os.path.join(folder_path,"COIFT","images",file_name+".jpg")
                label_path = os.path.join(folder_path,"COIFT","masks",file_name+".png")
                image = Image.open(image_path)
                label = Image.fromarray((np.array(Image.open(label_path))==255).astype(np.uint8))
                info = {"classes": [0,1],
                        "split_idx": split_idx}
                return image,label,info
        elif name=="hrsod":
            all_names = os.listdir(os.path.join(folder_path,"HRSOD","images"))
            file_name_list = [l[:l.find(".jpg")] for l in all_names]
            class_dict = {0: "background"}
            for i in range(5):
                class_dict[i+1] = "foreground"+str(i+1)
            def load_image_label_info(file_name):
                image_path = os.path.join(folder_path,"HRSOD","images",file_name+".jpg")
                label_path = os.path.join(folder_path,"HRSOD","masks",file_name+".png")
                image = Image.open(image_path)
                label = Image.open(label_path)
                info = {"classes": list(range(np.array(label).max()+1)),
                        "split_idx": split_idx}
                return image,label,info
        elif name=="to5k":
            with open(os.path.join(folder_path,"ThinObject5K","list","train.txt")) as f:
                train_names = f.readlines()
            train_names = [l[:l.find(".png")] for l in train_names]
            with open(os.path.join(folder_path,"ThinObject5K","list","test.txt")) as f:
                test_names = f.readlines()
            test_names = [l[:l.find(".png")] for l in test_names]
            with open(os.path.join(folder_path,"ThinObject5K","list","trainval.txt")) as f:
                trainval_names = f.readlines()
            trainval_names = [l[:l.find(".png")] for l in trainval_names]
            file_name_list = trainval_names+test_names
            dataset_idx_dict = {}
            for file_name in file_name_list:
                dataset_idx_dict[file_name] = 0
                #code below: not used since all are trainval
                """if name in train_names:
                    dataset_idx_dict[file_name] = 0
                elif name in test_names:
                    dataset_idx_dict[file_name] = 2
                else:
                    dataset_idx_dict[file_name] = 1"""
            class_dict = {0: "background"}
            class_dict_inv = {"background": 0}
            i = 1
            for l in file_name_list:
                k = l[:l.find("_PNG")]
                if not k in class_dict.values():
                    class_dict[i] = k
                    class_dict_inv[k] = i
                    i += 1
            def load_image_label_info(file_name):
                image_path = os.path.join(folder_path,"ThinObject5K","images",file_name+".jpg")
                label_path = os.path.join(folder_path,"ThinObject5K","masks",file_name+".png")
                split_idx = dataset_idx_dict[file_name]
                image = Image.open(image_path)
                label = np.array(Image.open(label_path))
                label = (label>=128).astype(np.uint8)
                label = Image.fromarray(label)
                class_idx = class_dict_inv[file_name[:file_name.find("_PNG")]]
                info = {"classes": [0,class_idx],
                        "split_idx": split_idx}
                return image,label,info
        elif name=="sa1b":
            unpack_list = [os.path.join(folder_path, "An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0.gz")]
            file_name_list = [l for l in os.listdir(os.path.join(folder_path,"SA-1B-Part-000004")) if l.find(".jpg")>=0]
            class_dict = {i: "foreground"+str(i) for i in range(1,256)}
            class_dict["0"] = "background"
            
            def load_image_label_info(file_name):
                image_path = os.path.join(folder_path,"SA-1B-Part-000004",file_name)
                json_path = image_path.replace(".jpg",".json")
                info_json = json.load(open(json_path))
                assert len(info_json['annotations'])<=255, "uint8 format fails if more than 255 classes are present."
                rle = info_json['annotations']
                image = Image.open(image_path)
                if do_step["save_images"]:
                    label = rle_to_mask(rle)
                    label = Image.fromarray(label)
                else:
                    label = None
                info = {"classes": list(range(len(info_json['annotations'])+1)),"split_idx": 0}
                
                return image,label,info
        elif name=="monuseg":
            unpack_list = [os.path.join(folder_path,"MoNuSeg" ,"MoNuSeg 2018 Training Data.zip"),
                            os.path.join(folder_path,"MoNuSeg" ,"MoNuSegTestData.zip")]
            train_files = os.listdir(os.path.join(folder_path,"MoNuSeg 2018 Training Data"))
            test_files = os.listdir(os.path.join(folder_path,"MoNuSegTestData"))
            file_name_list = [l[:-4] for l in train_files+test_files if l.find(".tif")>=0]
            image_suffix = ".tif"
            class_dict = {1: "cell_nuclei"}
            class_dict["0"] = "background"
            
            def load_image_label_info(file_name):
                if file_name+".tif" in train_files:
                    split_idx = 0
                    image_path = os.path.join(folder_path,"MoNuSeg 2018 Training Data",file_name+".tif")
                    label_path = os.path.join(folder_path,"MoNuSeg 2018 Training Data",file_name+".png")
                else:
                    assert file_name+".tif" in test_files
                    split_idx = 1
                    image_path = os.path.join(folder_path,"MoNuSegTestData",file_name+".tif")
                    label_path = os.path.join(folder_path,"MoNuSegTestData",file_name+".png")
                image = Image.open(image_path)
                label = np.array(Image.open(label_path))
                label = (label==255).astype(np.uint8)
                label = Image.fromarray(label)
                info = {"classes": [0,1],"split_idx": split_idx}
                return image,label,info
        elif name=="dram":
            labels_folder = Path(self.mainfolder) / "dram" / "DRAM_processed" / "labels"
            file_name_list = list(labels_folder.glob("*/*/*.png"))
            file_name_list = [str(f)[:-4] for f in file_name_list]
            class_dict = {str(i+1): "foreground"+str(i+1) for i in range(5)}
            class_dict["0"] = "background"
            def load_image_label_info(file_name):
                image_path = file_name.replace("labels","test")+".jpg"
                label_path = file_name+".png"
                image = Image.open(image_path)
                label = np.array(Image.open(label_path))
                info = {"classes": list(range(len(np.unique(label)))),"split_idx": 0}
                label2 = np.zeros_like(label)
                k = 0
                for u in np.unique(label):
                    if u>0:
                        k += 1
                        label2[label==u] = k
                label = Image.fromarray(label2)
                return image,label,info
        elif name=="monu4":
            file_name_list = (Path(self.mainfolder)/"monu/").glob("*/*.tif")
            file_name_list = [str(f) for f in file_name_list]
            #repeat each entry 4 times
            file_name_list = sum([[f for _ in range(4)]for f in file_name_list],[])
            patch_idx = sum([list(range(4)) for _ in range(len(file_name_list))],[])
            file_name_list = list(zip(file_name_list,patch_idx))
            image_suffix = ".png"
            class_dict = {1: "cell_nuclei"}
            class_dict["0"] = "background"
            
            def load_image_label_info(file_name):
                image_path,patch_num = file_name
                label_path = image_path.replace("_im.tif","_la.png")
                image = np.array(Image.open(image_path))
                label = np.array(Image.open(label_path))
                half = label.shape[0]//2
                if patch_num==0:
                    label = label[:half,:half]
                    image = image[:half,:half]
                elif patch_num==1:
                    label = label[:half,half:]
                    image = image[:half,half:]
                elif patch_num==2:
                    label = label[half:,:half]
                    image = image[half:,:half]
                elif patch_num==3:
                    label = label[half:,half:]
                    image = image[half:,half:]
                label = nd.label(label)[0]
                label[label>255] = (label[label>255]%254)+1
                n_cc = label.max()
                label = Image.fromarray(label.astype(np.uint8))
                image = Image.fromarray(image)
                info = {"classes": [0]+[1 for _ in range(n_cc)],"split_idx": split_idx}
                return image,label,info
        elif name=="uvo":
            labels_folder = Path(self.mainfolder) / "dram" / "DRAM_processed" / "labels"
            file_name_list = list(labels_folder.glob("*/*/*.png"))
            file_name_list = [str(f)[:-4] for f in file_name_list]
            class_dict = {str(i+1): "foreground"+str(i+1) for i in range(5)}
            class_dict["0"] = "background"
            def load_image_label_info(file_name):
                image_path = file_name.replace("labels","test")+".jpg"
                label_path = file_name+".png"
                image = Image.open(image_path)
                label = np.array(Image.open(label_path))
                info = {"classes": list(range(len(np.unique(label)))),"split_idx": 0}
                label2 = np.zeros_like(label)
                k = 0
                for u in np.unique(label):
                    if u>0:
                        k += 1
                        label2[label==u] = k
                label = Image.fromarray(label2)
                return image,label,info
        elif name=="totseg":
            file_name_list = [str(f) for f in (Path(folder_path)/"Totalsegmentator_dataset_v201").glob("*/samples/dim*_gt.png")]
            """
            #filter out uninteresting images
            mean_image_values = []
            for f in file_name_list:
                mean_image_values.append((np.array(Image.open(f.replace("_gt","_im")))<30).mean())
            #at most 80% of the image in the very low intensity interval of [0,29]
            file_name_list = [f for f,m in zip(file_name_list,mean_image_values) if m<0.8]"""

            class_dict_path = "/home/jloch/Desktop/diff/diffusion2/data/totseg/Totalsegmentator_dataset_v201/idx_to_class_alphabetical.json"
            class_dict = load_json_to_dict_list(class_dict_path)[0]
            image_suffix = ".png"
            meta = "/home/jloch/Desktop/diff/diffusion2/data/totseg/Totalsegmentator_dataset_v201/meta.csv"
            meta_loaded = pd.read_csv(meta)
            def load_image_label_info(file_name):
                image_path = file_name.replace("_gt.","_im.")
                label_path = file_name
                image = Image.open(image_path)
                label = np.array(Image.open(label_path))
                image_id = file_name.split("/")[-3]
                iloc_i = np.flatnonzero(meta_loaded["image_id"]==image_id)
                split_str = meta_loaded.iloc[iloc_i]["split"].item()
                split_idx = ["train","val","test"].index(split_str)
                uq_classes, label = np.unique(label,return_inverse=True)
                info = {"classes": uq_classes.tolist(),"split_idx": split_idx}
                w,h = image.size
                label = Image.fromarray(label.reshape((h,w)).astype(np.uint8))
                return image,label,info
        elif name=="visor":
            zipname = "2v6cgv1x04ol22qp9rm9x2j6a7"
            classes_filename = str(Path(folder_path)/zipname/"EPIC_100_noun_classes_v2.csv")
            classes = np.loadtxt(classes_filename, dtype=str, delimiter=',', skiprows=1, usecols=(1)).tolist()
            classes = ["background"]+[c[1:-1].replace(":", " ") for c in classes]
            idx_to_class = dict(enumerate(classes))
            class_to_idx = {v:k for k,v in idx_to_class.items()}
            class_dict = idx_to_class
            assert len(idx_to_class) == len(class_to_idx)
            image_folder = Path(folder_path)/zipname/"GroundTruth-SparseAnnotations"/"rgb_frames/"
            label_folder =  Path(folder_path)/zipname/"GroundTruth-SparseAnnotations"/"annotations/"
            file_name_list = []
            image_path_to_split = {}
            for split in ["train","val"]:
                list_of_files = [str(f) for f in Path(label_folder).glob(split+"/*.json")]
                for file in tqdm.tqdm(list_of_files):
                    gt = json.load(open(file))
                    for v in gt["video_annotations"]:
                        image_path = v["image"]["name"]
                        file_name_list.append(image_path)
                        image_path_to_split[image_path] = ["train","val"].index(split)
            def load_image_label_info(image_path):
                split = image_path_to_split[image_path]
                split_name = ["train","val"][split]
                idx1 = image_path.split("_")[0]
                idx12 = "_".join(image_path.split("_")[:2])
                label_filename = label_folder/split_name/f"{idx12}.json"
                zip_filename = Path(image_folder)/split_name/f"{idx1}"/f"{idx12}.zip"
                zip_sub_filename = image_path
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    with zip_ref.open(zip_sub_filename) as f:
                        image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
                h,w = image.shape[:2]
                label = np.zeros((h,w),dtype=np.uint8)
                annotations = json.load(open(label_filename))["video_annotations"]
                idx3 = [idx3 for idx3,v in enumerate(annotations) if v["image"]["name"]==image_path][0]
                info = {"split_idx": split,
                        "classes": [0]}
                k = 0
                for annotation_dict in annotations[idx3]["annotations"]:
                    k += 1
                    if k>255:
                        break
                    idx = annotation_dict["class_id"]+1
                    info["classes"].append(idx)
                    for poly in annotation_dict["segments"]:
                        poly = np.array(poly)[:,::-1]
                        mask = skimage.draw.polygon2mask((h,w), poly)
                        label[mask] = k
                image = Image.fromarray(image)
                label = Image.fromarray(label)
                return image,label,info
        elif name=="duts":
            cascade_folder = "/home/jloch/Desktop/diff/sam-hq-training/data/cascade_psp/"
            class_dict = {"0": "background", "1": "foreground"}
            filenames_train =  [str(f) for f in (Path(cascade_folder)/"DUTS-TR").glob("*.png")]
            filenames_test =  [str(f) for f in (Path(cascade_folder)/"DUTS-TE").glob("*.png")]
            filename_to_split = {f:i for i,f in zip([0]*len(filenames_train)+[1]*len(filenames_test),filenames_train+filenames_test)}
            file_name_list = filenames_train+filenames_test
            def load_image_label_info(filename):
                label = np.array(Image.open(filename))
                label = (label>=128)
                if len(label.shape)==3:
                    label = label.mean(2)
                assert 0 in np.unique(label)
                assert 1 in np.unique(label)
                label = Image.fromarray(label.astype(np.uint8))
                image = Image.open(filename.replace(".png",".jpg"))
                info = {"split_idx": filename_to_split[filename],
                        "classes": [0,1]}
                return image,label,info
        elif name=="ecssd":
            cascade_folder = "/home/jloch/Desktop/diff/sam-hq-training/data/cascade_psp/"
            file_name_list =  [str(f) for f in (Path(cascade_folder)/"ecssd").glob("*.png")]
            class_dict = {"0": "background", "1": "foreground"}
            def load_image_label_info(filename):
                label = np.array(Image.open(filename))
                label = (label>=128)
                if len(label.shape)==3:
                    label = label.mean(2)
                assert 0 in np.unique(label)
                assert 1 in np.unique(label)
                label = Image.fromarray(label.astype(np.uint8))
                image = Image.open(filename.replace(".png",".jpg"))
                info = {"split_idx": 0,
                        "classes": [0,1]}
                return image,label,info
        elif name=="fss":
            cascade_folder = "/home/jloch/Desktop/diff/sam-hq-training/data/cascade_psp/"
            file_name_list =  [str(f) for f in (Path(cascade_folder)/"fss_all").glob("*.png")]
            foreground_classes = []
            for f in file_name_list:
                f = " ".join(Path(f).name.split("_")[:-1])
                if f not in foreground_classes:
                    foreground_classes.append(f)

            class_dict = {"0": "background"}
            class_dict.update({str(i+1): c for i,c in enumerate(foreground_classes)})
            class_dict_inv = {v:k for k,v in class_dict.items()}

            def load_image_label_info(filename):
                label = np.array(Image.open(filename))
                label = (label>=128).mean(2).astype(np.uint8)
                assert 0 in np.unique(label)
                assert 1 in np.unique(label)
                label = Image.fromarray(label)
                image = Image.open(filename.replace(".png",".jpg"))
                class_name = " ".join(Path(filename).name.split("_")[:-1])
                info = {"split_idx": 0,
                        "classes": [0,int(class_dict_inv[class_name])]}
                return image,label,info
        elif name=="msra":
            cascade_folder = "/home/jloch/Desktop/diff/sam-hq-training/data/cascade_psp/"
            file_name_list =  [str(f) for f in (Path(cascade_folder)/"MSRA_10K").glob("*.png")]
            class_dict = {"0": "background", "1": "foreground"}
            def load_image_label_info(filename):
                label = np.array(Image.open(filename))
                label = (label>=128)
                if len(label.shape)==3:
                    label = label.mean(2)
                assert 0 in np.unique(label)
                assert 1 in np.unique(label)
                label = Image.fromarray(label.astype(np.uint8))
                image = Image.open(filename.replace(".png",".jpg"))
                info = {"split_idx": 0,
                        "classes": [0,1]}
                return image,label,info
        elif name=="dis":
            dis_folder = '/home/jloch/Desktop/diff/sam-hq-training/data/DIS5K/'
            filenames_train =  [str(f) for f in (Path(dis_folder)/"DIS-TR"/"gt").glob("*.png")]
            filenames_vali =  [str(f) for f in (Path(dis_folder)/"DIS-VD"/"gt").glob("*.png")]
            filename_to_split = {f:i for i,f in zip([0]*len(filenames_train)+[1]*len(filenames_vali),filenames_train+filenames_vali)}
            file_name_list = filenames_train+filenames_vali
            def replace_upper_with_lower_and_space(s):
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                for a in alphabet:
                    s = s.replace(a," "+a.lower())
                if s[0] == " ":
                    s = s[1:]
                return s
            foreground_classes = []
            for f in file_name_list:
                f = Path(f).name.split("#")[3]
                f = replace_upper_with_lower_and_space(f)
                if f not in foreground_classes:
                    foreground_classes.append(f)
            class_dict = {"0": "background"}
            class_dict.update({str(i+1): c for i,c in enumerate(foreground_classes)})
            class_dict_inv = {v:k for k,v in class_dict.items()}

            def load_image_label_info(filename):
                label = np.array(Image.open(filename))
                label = (label>=128)
                if len(label.shape)==3:
                    label = label.mean(2)
                assert 0 in np.unique(label)
                assert 1 in np.unique(label)
                label = Image.fromarray(label.astype(np.uint8))
                image_filename = filename.replace(".png",".jpg").replace("/gt/","/im/")
                image = Image.open(image_filename)
                class_name = Path(filename).name.split("#")[3]
                class_name = replace_upper_with_lower_and_space(class_name)
                info = {"split_idx": filename_to_split[filename],
                        "classes": [0,int(class_dict_inv[class_name])]}
                return image,label,info
        elif name=="lvis":
            val_data = load_json_to_dict_list("./data/lvis/lvis_v1_val.json")
            train_data = load_json_to_dict_list("./data/lvis/lvis_v1_train.json")

            train_ids = [item["id"] for item in train_data["images"]]
            val_ids = [item["id"] for item in val_data["images"]]

            assert len(set(train_ids).intersection(set(val_ids)))==0

            idx_to_class = {d["id"]: d["name"] for d in train_data["categories"]}
            idx_to_class[0] = "background"
            class_dict = idx_to_class

            coco_info = load_json_to_dict_list(f"./data/coco/info.jsonl")
            coco_i_to_id = {item["i"]: item["fn"] for item in coco_info}
            coco_id_to_i = {int(v): k for k,v in coco_i_to_id.items()}
            def loadpath_from_coco_id(image_id):
                return f"./data/coco/f{coco_id_to_i[image_id]//1000}/{coco_id_to_i[image_id]}_im.jpg"

            annotation_dict = {k: [] for k in train_ids+val_ids}
            for annotation in train_data["annotations"]+val_data["annotations"]:
                annotation_dict[annotation["image_id"]].append(annotation)
            #sort all annotations by size (biggest first)
            for k in list(annotation_dict.keys()):
                v = annotation_dict[k]
                v.sort(key=lambda x: -x["area"])
                annotation_dict[k] = v

            file_name_list = [loadpath_from_coco_id(k) for k in annotation_dict.keys()]

            def load_image_label_info(image_path):
                dl_id = int(image_path.split("/")[-1].split("_")[0])
                image_id = int(coco_i_to_id[dl_id])
                if image_id in train_ids:
                    split = 0
                else:
                    assert image_id in val_ids, f"Could not match image_id {image_id}"
                    split = 1
                image = np.array(Image.open(image_path))
                h,w = image.shape[:2]
                label = np.zeros((h,w),dtype=np.uint8)
                info = {"split_idx": split,
                        "classes": [0]}
                k = 0
                for annotation in annotation_dict[image_id]:
                    k += 1
                    if k>255:
                        break
                    idx = annotation["category_id"]
                    info["classes"].append(idx)
                    x = annotation["segmentation"][0][::2]
                    y = annotation["segmentation"][0][1::2]
                    mask = skimage.draw.polygon2mask((h,w), np.array([y,x]).T)
                    label[mask] = k
                label = Image.fromarray(label)
                image = Image.fromarray(image)
                return image,label,info
        elif name=="lidc":
            always_4_labels = False 
            lidc_path = "/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare3/"
            data = load_json_to_dict_list(lidc_path+"data.jsonl")
            class_dict = {0: "background",1: "nodule"}
            vali_idx_path = Path(lidc_path).parent/"vali_patient_ids.txt"
            test_idx_path = Path(lidc_path).parent/"test_patient_ids.txt"
            vali_idx = [l.replace("\n","") for l in open(vali_idx_path).readlines()]
            test_idx = [l.replace("\n","") for l in open(test_idx_path).readlines()]
            split_from_patient_id = lambda id: 1 if id in vali_idx else 2 if id in test_idx else 0
            image_suffix = ".png"
            file_data = []
            for scan_id,scan in enumerate(data):
                for n_i,nodule in enumerate(scan["nodules"]):
                    for s_i,image_dict in enumerate(nodule):
                        if always_4_labels:
                            if len(image_dict["masks_id"])>4:
                                image_dict["masks_id"] = image_dict["masks_id"][:4]
                            elif len(image_dict["masks_id"])<4:
                                image_dict["masks_id"] += [-1]*(4-len(image_dict["masks_id"]))
                        for m_i,mask_id in enumerate(image_dict["masks_id"]):
                            file_data.append({"split_idx": split_from_patient_id(scan["patient_id"]),
                                            "bbox"       : image_dict["bbox"],
                                            "image_id"   : image_dict["image_id"],

                                            "m_i"        : m_i,
                                            "s_i"        : s_i,
                                            "n_i"        : n_i,

                                            "tot_m"      : len(image_dict["masks_id"]),
                                            "tot_s"      : len(nodule),
                                            "tot_n"      : len(scan["nodules"]),

                                            "mask_id"    : mask_id,
                                            "patient_id" : scan["patient_id"],
                                            "scan_id"    : scan_id})
            def fmt(f):
                return ",".join([
                    f"mask-i_{f['m_i']}/{f['tot_m']}",
                    f"slice-i_{f['s_i']}/{f['tot_s']}",
                    f"nodule-i_{f['n_i']}/{f['tot_n']}",
                    f"scan_{f['scan_id']}",
                    f"img_{f['image_id']:06d}.dcm",
                    f"mask_{f['mask_id']:06d}.png"
                ])
            file_name_list = [fmt(f) for f in file_data]
            if not always_4_labels:
                assert len(file_name_list)==mask_id+1, f"Expected {mask_id+1} files, got {len(file_name_list)}." 
            def load_image_label_info(file_name):
                file_data_idx = file_name_list.index(file_name)
                file_data_dict = file_data[file_data_idx]
                image_path = lidc_path+f"images/img_{file_data_dict['image_id']:06d}.dcm"
                image = pydicom.dcmread(image_path).pixel_array
                label = np.zeros(image.shape[:2],dtype=np.uint8)
                if file_data_dict['mask_id']>0:
                    label_path = lidc_path+f"masks/mask_{file_data_dict['mask_id']:06d}.png"
                    b1,b2,b3,b4 = file_data_dict["bbox"]
                    label[b1:b2,b3:b4] = (np.array(Image.open(label_path))>0).astype(np.uint8)
                image,label = bbox_crop(image,label,file_data_dict["bbox"],size=128)
                image = (quantile_normalize(image,alpha=0.001)*255).astype(np.uint8)
                info = {"classes": [0]+([1] if np.sum(label)>0 else []),
                        "split_idx": file_data_dict["split_idx"]}
                image = Image.fromarray(image)
                label = Image.fromarray(label)
                return image,label,info
        elif name=="lidc15096":
            data = pickle.load(open("/home/jloch/Desktop/diff/diffusion2/data/lidc15096/data_lidc.pickle","rb"))
            data = pickle.load(open("/home/jloch/Desktop/diff/diffusion2/data/lidc15096/data_lidc.pickle","rb"))
            patient_ids = [k.split("_")[0] for k in data.keys()]
            patient_ids = np.unique(patient_ids).tolist()

            test_ids = np.loadtxt("/home/jloch/Desktop/diff/diffusion2/data/lidc15096/test_ids.txt", dtype=str)
            vali_ids = np.loadtxt("/home/jloch/Desktop/diff/diffusion2/data/lidc15096/vali_ids.txt", dtype=str)
            file_data = []
            image_id = 0
            for k,v in data.items():
                patient_id = patient_ids.index(k.split("_")[0])
                if "lesion" in k:
                    n_i = int(k.split("lesion")[-1])-1
                    tot_n = 2
                else:
                    n_i = 0
                    tot_n = 1
                rawslice = int(k.split("_")[-1].split("lesion")[0].replace("slice",""))
                for m_i in range(len(v["masks"])):
                    file_data.append({"split_idx": 2 if k.split("_")[0] in test_ids else (1 if k.split("_")[0] in vali_ids else 0),
                                    "m_i"        : m_i,
                                    "s_i"        : None,
                                    "n_i"        : n_i,

                                    "tot_m"      : len(v["masks"]),
                                    "tot_s"      : None,
                                    "tot_n"      : tot_n,

                                    "rawslice": rawslice,
                                    "image_id"   : image_id,
                                    "mask_id"    : m_i,
                                    "patient_id" : patient_id})
                image_id += 1

            # go through all the data and add slice indices for each unique patient,nodule pair
            pair_to_slices = {} # (patient_id,n_i) -> {rawslice: [i1,i2,...]}
            for i,f in enumerate(file_data):
                key = (f["patient_id"],f["n_i"])
                if not key in pair_to_slices:
                    pair_to_slices[key] = {}
                pair_to_slices[key][f["rawslice"]] = [i] if not f["rawslice"] in pair_to_slices[key] else pair_to_slices[key][f["rawslice"]]+[i]
            for k,v in pair_to_slices.items():
                tot_s = len(v)
                keys = list(v.keys())
                values = list(v.values())
                for s_i,indices in enumerate([values[i] for i in np.argsort(keys)]):
                    for i in indices:
                        file_data[i]["s_i"] = s_i
                        file_data[i]["tot_s"] = tot_s
            
            def fmt(f):
                return ",".join([
                    f"mask-i_{f['m_i']}/{f['tot_m']}",
                    f"slice-i_{f['s_i']}/{f['tot_s']}",
                    f"nodule-i_{f['n_i']}/{f['tot_n']}",
                    f"rawslice_{f['rawslice']}",
                    f"img_{f['image_id']}",
                    f"mask_{f['mask_id']}"
                ])
            file_name_list = [fmt(f) for f in file_data]
            class_dict = {0: "background",1: "nodule"}
            keylist = list(data.keys())
            image_suffix = ".png"
            def load_image_label_info(file_name):
                file_data_idx = file_name_list.index(file_name)
                file_data_dict = file_data[file_data_idx]
                data_i = data[keylist[file_data_dict["image_id"]]]
                image = data_i["image"]
                label = data_i["masks"][file_data_dict["m_i"]].astype(np.uint8)
                image = (quantile_normalize(image,alpha=0.001)*255).astype(np.uint8)
                info = {"classes": [0]+([1] if np.sum(label)>0 else []),
                        "split_idx": file_data_dict["split_idx"]}
                image = Image.fromarray(image)
                label = Image.fromarray(label)
                return image,label,info
        elif name=="entityseg":
            train_path = "/home/jloch/Desktop/diff/BIG_dataset/entityseg_train_lr.json"
            test_path = "/home/jloch/Desktop/diff/BIG_dataset/entityseg_val_lr.json"
            with open(train_path) as f:
                train_loaded = json.load(f)
            with open(test_path) as f:
                test_loaded = json.load(f)
            file_name_dict = {}
            id_to_filename = {}
            for d in train_loaded["images"]:
                id_to_filename[d["id"],0] = d["file_name"]
                file_name_dict[d["file_name"]] = {"ann_ids": [], 
                                                "split_idx": 0}
            for d in test_loaded["images"]:
                id_to_filename[d["id"],2] = d["file_name"]
                file_name_dict[d["file_name"]] = {"ann_ids": [],
                                                "split_idx": 2}
                
            file_name_list = list(file_name_dict.keys())
            for ann_id,ann in enumerate(train_loaded["annotations"]+test_loaded["annotations"]):
                s = (0 if ann_id<len(train_loaded["annotations"]) else 2)
                fn = id_to_filename[ann["image_id"],s]
                file_name_dict[fn]["ann_ids"].append(ann_id)
            class_dict = {0: "background"}
            for cat in train_loaded["categories"]:
                class_dict[cat["id"]+1] = cat["name"]

            def load_image_label_info(file_name):
                d = file_name_dict[file_name]
                image = np.array(Image.open(os.path.join("/home/jloch/Desktop/diff/BIG_dataset",file_name)))
                h,w,c = np.array(image).shape
                if c==1:
                    image = np.repeat(image,3,axis=-1)
                elif c==4:
                    image = image[:,:,:3]
                else:
                    assert c==3, f"Image has an unexpected number of {c} channels. Expected 1,3 or 4."
                seg = np.zeros((h,w),dtype=np.uint8)
                info = {"classes": [0],
                        "split_idx": d["split_idx"]}
                for k,ann_id in enumerate(d["ann_ids"]):
                    ann = (train_loaded["annotations"]+test_loaded["annotations"])[ann_id]
                    seg += mask_util.decode(ann["segmentation"])*(k+1)
                    info["classes"].append(ann["category_id"]+1)
                return Image.fromarray(image),Image.fromarray(seg),info
        else:
            raise ValueError(f"Dataset {name} not supported.")
        if do_step["delete_f_before"]:
            f_files = [f for f in os.listdir(folder_path) if re.match(r'f\d', f)]
            for f in f_files:
                print(f"Deleting {os.path.join(folder_path, f)}...")
                shutil.rmtree(os.path.join(folder_path, f))
        #unpack files:
        if do_step["make_f0"]:
            os.makedirs(os.path.join(folder_path,f"f{folder_i}"),exist_ok=True)
        if do_step["save_class_dict"]:
            save_dict_list_to_json([class_dict], os.path.join(folder_path,"idx_to_class.json"),append=False)
        jsonl_save_path = os.path.join(folder_path,"info.jsonl")
        if do_step["reset_info"]:
            save_dict_list_to_json([], jsonl_save_path, append=False)
        #loop over label-image pairs
        samples_per_split = [0,0,0]
        for file_name in tqdm.tqdm(file_name_list):
            if not any([do_step[k] for k in ["save_images","save_info"]]): break
            num_failures = num_attempts-file_i
            failure_rate = num_failures/num_attempts if num_attempts>0 else 0
            if failure_rate>self.allowed_failure_rate:
                show_error_flag = True
            num_attempts += 1
            try:
                image,label,info = load_image_label_info(file_name)
                if save_binary_overlaps:
                        label,label_overlapped = label
                if not image.size==label.size:
                    print(f"Image and label size mismatch. Skipping at file_i={file_i}.")
                    raise ValueError(f"Image and label size mismatch: {image.size} vs {label.size}")
                if do_step["save_images"]:
                    image.save(os.path.join(folder_path,f"f{folder_i}",str(file_i)+"_im"+image_suffix))
                    if save_binary_overlaps:
                        label = label.convert("P")
                        label.putpalette(pallete)
                        label.save(os.path.join(folder_path,f"f{folder_i}",str(file_i)+"_la"+label_suffix))
                        label_overlapped.save(os.path.join(folder_path,f"f{folder_i}",str(file_i)+"_ola"+label_suffix))
                    else:
                        label = label.convert("P")
                        label.putpalette(pallete)
                        label.save(os.path.join(folder_path,f"f{folder_i}",str(file_i)+"_la"+label_suffix))
                if "split_idx" in info.keys():
                    samples_per_split[info["split_idx"]] += 1
                else:
                    samples_per_split[0] += 1
                num_saved_images += 1
                
                if do_step["save_info"]:
                    info["i"] = file_i
                    info["fn"] = str(file_name)
                    info["imshape"] = np.array(image).shape
                    info["class_counts"] = [int(np.sum(np.array(label)==i)) for i in range(len(info["classes"]))]
                    save_dict_list_to_json(info, jsonl_save_path, append=True)
                
                file_i += 1
                if num_saved_images==self.files_per_folder and file_i<len(file_name_list):
                    folder_i += 1
                    num_saved_images = 0
                    os.makedirs(os.path.join(folder_path,f"f{folder_i}"),exist_ok=True)
            except Exception:
                if show_error_flag:
                    raise Exception(f"Too many failed attempts. {num_attempts} attempts, {file_i} saved images. Last error shown above.")
        if do_step["save_global_info"]:
            global_info = {"dataset_name": name,
                            "urls": [],
                            "num_samples": file_i,
                            "num_classes": len(class_dict),
                            "file_format": image_suffix,
                            "samples_per_split": samples_per_split}
            save_dict_list_to_json([global_info], "./data/datasets_info.json", append=True)
        #delete unused files
        if do_step["delete_unused_files"]:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if re.match(r'f\d', file_name):
                    continue
                elif file_name in ["info.jsonl","idx_to_class.json"]:
                    continue
                else:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.rm(file_path)
        if do_step["prettify"]:
            prettify_data(name)
        if do_step["add_prettify_to_info"]:
            add_existence_of_prettify_to_info_jsonl(name)
        if do_step["sam_features"]:
            print("Saving SAM features...")
            save_sam_features(name)
        if do_step["add_sam_to_info"]:
            add_existence_of_sam_features_to_info_jsonl(name)
        print(f"Finished processing dataset: {name}. Num failures: {num_attempts-file_i}/{num_attempts}.")


def delete_all_sam_features(sam_index=0,dry=False):
    #delete all sam features to save memory
    list_of_sam_features = list(Path("./data").glob(f"*/f*/*sam{sam_index}.pt"))
    if dry:
        n = len(list_of_sam_features)
        idx = np.random.randint(n,size=5)
        some_random_files = [list_of_sam_features[i] for i in idx]
        print(f"Found {n} files for deletion. Some random files:")
        for f in some_random_files:
            print(f)
    else:
        for f in list_of_sam_features:
            os.remove(f)

def process_bg(label,
                always_override=False,
                background_area_thresh=0.005,
                opening_width=0.005,
                minimum_affected_ratio=0.1,
                conn_comp_r_balls=10,
                conn_comp_r_balls_all=0):
    """
    Processes a label with bad boundaries for the background class
    and overrides the background with the nearest class.

    Inputs:
    - label: 2d array
        Image of integer labels to be processed
    - always_override: bool
        If True, always override the background with the nearest class
    - background_area_thresh: float
        Threshold of the background area relative to the total area, if
        the area of the background is less than this, it is overridden
        the sme way as always_override=True
    - opening_width: float
        Width of the opening operation (relative to the average side length 
        of the image)
    - minimum_affected_ratio: float
        Minimum relative area any operation can affect. If less than this,
        the operation is not performed.
    - conn_comp_area_thresh: float
        Connected components which represent less than this relative area of 
        the total image area are removed.

    Returns:
    - result_array: 2d array
        Processed label image
    """
    background_mask_orig = label==0
    if np.sum(background_mask_orig) < 1:
        return label
    distances,indices = nd.distance_transform_edt(label == 0, return_indices=True)
    if always_override or np.mean(background_mask_orig) < background_area_thresh:
        result_array = np.where(label == 0, label[tuple(indices)], label)
    else:
        #use distance transform to do opening, since it can consider non-integer distances
        mean_sidelength = 0.5*(label.shape[0]+label.shape[1])
        opening_width *= mean_sidelength
        #erode the background mask
        background_mask = distances > opening_width
        #dilate the background mask, dilate with twice the original radius but confined to the original mask (keeps high-fidelity)
        distances_mask = nd.distance_transform_edt(background_mask==0)
        background_mask = np.logical_and(distances_mask < opening_width*2,background_mask_orig)
        if (1-np.mean(background_mask))/np.mean(background_mask_orig) < minimum_affected_ratio:
            #dont accept the opening if it affects too few pixels
            return label
        else:
            #accept the opening
            #remove small connected components
            conn_comp = nd.label(background_mask)[0]
            conn_comp_area = np.bincount(conn_comp.flatten())
            conn_comp_thresh = conn_comp_r_balls*np.pi*opening_width**2
            background_mask = np.logical_and(conn_comp>0,(conn_comp_area>conn_comp_thresh)[conn_comp])
        modify_mask = np.logical_and(background_mask_orig,np.logical_not(background_mask))
        distances,indices = nd.distance_transform_edt(modify_mask, return_indices=True)
        result_array = np.where(modify_mask, label[tuple(indices)], label)
    if conn_comp_r_balls_all>0:
        modify_mask = np.zeros(result_array.shape,dtype=bool)
        conn_comp_thresh_all = conn_comp_r_balls_all*np.pi*opening_width**2
        for i in range(1,np.max(result_array)+1):
            conn_comp_i = nd.label(result_array==i)[0]
            conn_comp_area = np.bincount(conn_comp_i.flatten())
            modify_mask = np.logical_or(modify_mask,(conn_comp_area<conn_comp_thresh_all)[conn_comp_i])
        distances,indices = nd.distance_transform_edt(modify_mask, return_indices=True)
        result_array = np.where(modify_mask, result_array[tuple(indices)], result_array)
    return result_array

def prettify_data(dataset,suffix="p",max_save_sidelength=1024,max_process_sidelength=2048):
    if isinstance(dataset,str):
        dataset = SegmentationDataset(datasets=dataset,split="all",shuffle_datasets=False,use_pretty_data=False)
    dataset_counter = 0
    for item in tqdm.tqdm(dataset):
        info = item[-1]
        label_path = Path("./data/")/ info["dataset_name"] / info["label_path"]
        image_path = Path("./data/")/ info["dataset_name"] / info["image_path"]
        label = np.array(Image.open(label_path))
        image = np.array(Image.open(image_path))
        if len(image.shape)==2:
            image = image[:,:,None]
        if image.shape[2]==1:
            image = np.concatenate([image for _ in range(3)],axis=2)
        shape = label.shape
        if shape[0]>shape[1]:
            #makes sure that the smallest size is height
            image = image.transpose(1,0,2)
            label = label.transpose(1,0)
            transposed=True
        else:
            transposed=False
        h,w = label.shape
        h1 = min(h,max_process_sidelength)
        w1 = int(w*h1/h)
        h2 = min(h1,max_save_sidelength)
        w2 = int(w1*h2/h1)
        image = cv2.resize(image,(w2,h2),interpolation=cv2.INTER_AREA)
        
        thin_bg = dataset.datasets_info[info["dataset_name"]]["thin_bg"]
        if thin_bg=="always":
            label = cv2.resize(label,(w1,h1),interpolation=cv2.INTER_NEAREST)
            label_after = process_bg(label,always_override=True)
        elif thin_bg=="never":
            label_after = label
            #dont save the files which are not processed and are the same size
            if h2==h and w2==w:
                continue
        elif thin_bg=="sometimes":
            label = cv2.resize(label,(w1,h1),interpolation=cv2.INTER_NEAREST)
            label_after = process_bg(label)
        elif thin_bg=="sa1b":
            label = cv2.resize(label,(w1,h1),interpolation=cv2.INTER_NEAREST)
            label_after = process_bg(label,conn_comp_r_balls_all=5)
        else:
            raise ValueError("thin_bg must be in ['always','never','sometimes','sa1b']")

        label_after = cv2.resize(label_after,(w2,h2),interpolation=cv2.INTER_NEAREST)
        if transposed:
            image = image.transpose(1,0,2)
            label_after = label_after.transpose(1,0)
        label_after = Image.fromarray(label_after)
        label_after = label_after.convert("P")
        label_after.putpalette(nc.largest_pallete)
        filename_label = str(label_path).replace("_la.",f"_{suffix}la.")
        filename_image = str(image_path).replace("_im.",f"_{suffix}im.")
        label_after.save(filename_label)
        Image.fromarray(image).save(filename_image)
        dataset_counter += 1
    return dataset_counter/len(dataset)

def add_existence_of_prettify_to_info_jsonl(dataset_names=None):
    if dataset_names is None:
        list_of_info_jsons = Path("./data/").glob("*/info.jsonl")
    elif isinstance(dataset_names,str):
        list_of_info_jsons = [Path(f"./data/{dataset_names}/info.jsonl")]
        assert list_of_info_jsons[0].exists(), f"File {list_of_info_jsons[0]} does not exist."
    else:
        list_of_info_jsons = []
        for dataset_name in dataset_names:
            list_of_info_jsons.append(Path(f"./data/{dataset_name}/info.jsonl"))
            assert list_of_info_jsons[-1].exists(), f"File {list_of_info_jsons[-1]} does not exist."
    for infopath in tqdm.tqdm(list_of_info_jsons):
        infopath = str(infopath)
        info_list = load_json_to_dict_list(infopath)
        for j in range(len(info_list)):
            i = info_list[j]["i"]
            folder_i = np.floor(i/1000).astype(int)
            filename = Path(infopath).parent / f"f{folder_i}" / f"{i}_pla.png"
            if filename.exists():
                info_list[j]["pretty"] = True
            else:
                info_list[j]["pretty"] = False
        save_dict_list_to_json(info_list,infopath,append=False)

def add_existence_of_sam_features_to_info_jsonl(dataset_names=None):
    if dataset_names is None:
        list_of_info_jsons = Path("./data/").glob("*/info.jsonl")
    elif isinstance(dataset_names,str):
        list_of_info_jsons = [Path(f"./data/{dataset_names}/info.jsonl")]
        assert list_of_info_jsons[0].exists(), f"File {list_of_info_jsons[0]} does not exist."
    else:
        list_of_info_jsons = []
        for dataset_name in dataset_names:
            list_of_info_jsons.append(Path(f"./data/{dataset_name}/info.jsonl"))
            assert list_of_info_jsons[-1].exists(), f"File {list_of_info_jsons[-1]} does not exist."
    for infopath in tqdm.tqdm(list_of_info_jsons):
        infopath = str(infopath)
        info_list = load_json_to_dict_list(infopath)
        for j in range(len(info_list)):
            i = info_list[j]["i"]
            folder_i = np.floor(i/1000).astype(int)
            sam_features = []
            for sam_idx in range(7):
                filename = Path(infopath).parent / f"f{folder_i}" / f"{i}_sam{sam_idx}.pt"
                if filename.exists():
                    sam_features.append(sam_idx)
            info_list[j]["sam"] = sam_features
        save_dict_list_to_json(info_list,infopath,append=False)

def class_dict_from_info(info,ignore_zero=True):
    class_dict = {}
    for class_idx,class_count in zip(info["classes"],info["class_counts"]):
        if ignore_zero:
            if class_idx!=0:
                class_dict[class_idx] = class_count
        else:
            class_dict[class_idx] = class_count
    L2_norm = np.linalg.norm(list(class_dict.values()))+1e-12
    class_dict = {k: v/L2_norm for k,v in class_dict.items()}
    return class_dict

def class_balance_similarity(info1,info2,ignore_zero=True):
    class_dict1 = class_dict_from_info(info1,ignore_zero)
    class_dict2 = class_dict_from_info(info2,ignore_zero)
    intersection_keys = set(class_dict1.keys()).intersection(set(class_dict2.keys()))
    sim = 0
    for key in intersection_keys:
        c1,c2 = class_dict1[key],class_dict2[key]
        sim += c1*c2
    return sim
    

def add_same_class_reference(datasets=get_all_valid_datasets(),
                             max_neighbours=32,
                             save=True,
                             num_search_neighbours=1000,
                             num_per_dataset=-1,
                             min_sim=0.5,
                             dry=True,
                             add_dataset_name_key=False):
    if not isinstance(datasets,list):
        assert isinstance(datasets,str), "datasets must be a string or a list of strings"
        datasets = [datasets]

    live_info_path = "./data/datasets_info_live.json"
    live_info = load_json_to_dict_list(live_info_path)
    #pprint([(item["dataset_name"],item["num_classes"]) for item in live_info])
    for global_info in live_info:
        dataset_name = global_info["dataset_name"]
        if not global_info["live"]:
            continue
        if not dataset_name in datasets:
            continue
        if global_info["num_classes"]<=2:
            continue
        print(f"Processing: {dataset_name}")
        info_jsonl_path = f"./data/{dataset_name}/info.jsonl"
        info_list = load_json_to_dict_list(info_jsonl_path)
        if add_dataset_name_key:
            info_list = [{"dataset_name": dataset_name, **info} for info in info_list]
        info_add = []
        irange = tqdm.tqdm(range(num_per_dataset) if num_per_dataset>0 else range(len(info_list)))
        for i in irange:
            info = info_list[i]
            class_sim = {-i:0 for i in range(1,1+max_neighbours)}
            for info2_idx in np.random.choice(range(len(info_list)),min(num_search_neighbours,len(info_list)),replace=False):
                info2 = info_list[info2_idx]
                if info2["i"]!=info["i"]:
                    sim = class_balance_similarity(info,info2)
                    if sim>min(class_sim.values()):
                        min_key = min(class_sim,key=class_sim.get)
                        del class_sim[min_key]
                        class_sim[info2["i"]] = sim
            class_sim = {"sim": [int(round(v,3)*1000) for v in class_sim.values()], "idx": [k for k in class_sim.keys()]}
            #remove sim too low or negative index
            mask = np.flatnonzero(np.logical_and(np.array(class_sim["sim"])>=min_sim*1000,np.array(class_sim["idx"])>=0))
            class_sim = {k: [class_sim[k][i] for i in mask] for k in class_sim.keys()}
            order_by_sim = np.argsort(class_sim["sim"])[::-1]
            same_classes =[class_sim["idx"][i] for i in order_by_sim]
            class_sim = {"same_classes": same_classes}
            info_add.append(class_sim)
        for i in range(len(info_list)):
            if "conditioning" in info_list[i].keys():
                info_list[i]["conditioning"].update(info_add[i])
            else:
                info_list[i]["conditioning"] = info_add[i]
        if save:
            if dry:
                save_dict_list_to_json(info_list,info_jsonl_path.replace(".jsonl","_dry.jsonl"))
            else:
                save_dict_list_to_json(info_list,info_jsonl_path)
    return info_list, info_add


def add_conditioning_adjacent_slices_and_same_vol(dataset_name="totseg",max_neighbours=32,save=True):
    info_jsonl_path = f"./data/{dataset_name}/info.jsonl"
    info_list = load_json_to_dict_list(info_jsonl_path)
    info_add = [{"adjacent": [], "same_vol": []} for _ in info_list]
    if dataset_name=="totseg":
        samples_dict = {}
        for i,info in enumerate(info_list):
            sample_name = info["fn"].split('/')[-3]
            if not sample_name in samples_dict.keys():
                samples_dict[sample_name] = []
            samples_dict[sample_name].append(i)
        col_names = ["info_idx","name","volname","dim","slice_idx"]
        for k,v in tqdm.tqdm(samples_dict.items()):
            df = pd.DataFrame(columns=col_names)
            for idx in v:
                sample = {"info_idx": idx, "adjacent": []}
                sample["name"] = Path(info_list[idx]["fn"]).name
                sample["volname"] = k
                sample["dim"] = int(sample["name"].split('_')[0][3:])
                sample["slice_idx"] = int(sample["name"].split('_')[1][5:])
                # add the sample
                new_row_df = pd.DataFrame([sample])
                df = pd.concat([df,new_row_df], ignore_index=True)
            #sort by first dim, otherwise slices:
            df = df.sort_values(by=["dim","slice_idx"])
            for i in range(len(df)):
                dim_now = df.iloc[i]["dim"]
                if i>0:
                    dim_prev = df.iloc[i-1]["dim"]
                    if dim_now == dim_prev:
                        df.iloc[i]["adjacent"].append(df.iloc[i-1]["info_idx"])
                if i<len(df)-1:
                    dim_next = df.iloc[i+1]["dim"]
                    if dim_now == dim_next:
                        df.iloc[i]["adjacent"].append(df.iloc[i+1]["info_idx"])
            for row in df.iterrows():
                idx = row[1]["info_idx"]
                info_add[idx]["adjacent"] = row[1]["adjacent"]
                n = min(max_neighbours,len(df))
                info_add[idx]["same_vol"] = np.random.choice(df["info_idx"].tolist(),n,replace=False).tolist()
    
    elif dataset_name=="visor":
        samples_dict = {}
        for i,info in enumerate(info_list):
            sample_name = "_".join(info["fn"].split('_')[:2])
            if not sample_name in samples_dict.keys():
                samples_dict[sample_name] = []
            samples_dict[sample_name].append(i)
        col_names = ["info_idx","name","volname","framenum"]
        for k,v in tqdm.tqdm(samples_dict.items()):
            df = pd.DataFrame(columns=col_names)
            for idx in v:
                sample = {"info_idx": idx, "adjacent": []}
                sample["name"] = Path(info_list[idx]["fn"]).name
                sample["volname"] = k
                sample["framenum"] = int(sample["name"].split('_')[3][:-4])
                # add the sample
                new_row_df = pd.DataFrame([sample])
                df = pd.concat([df,new_row_df], ignore_index=True)
            #sort by first dim, otherwise slices:
            df = df.sort_values(by=["framenum"])
            for i in range(len(df)):
                volname_now = df.iloc[i]["volname"]
                if i>0:
                    volname_prev = df.iloc[i-1]["volname"]
                    if volname_now == volname_prev:
                        df.iloc[i]["adjacent"].append(df.iloc[i-1]["info_idx"])
                if i<len(df)-1:
                    volname_next = df.iloc[i+1]["volname"]
                    if volname_now == volname_next:
                        df.iloc[i]["adjacent"].append(df.iloc[i+1]["info_idx"])
            for row in df.iterrows():
                idx = row[1]["info_idx"]
                info_add[idx]["adjacent"] = row[1]["adjacent"]
                n = min(max_neighbours,len(df))
                info_add[idx]["same_vol"] = np.random.choice(df["info_idx"].tolist(),n,replace=False).tolist()
    elif dataset_name=="lidc":
        lidc_path = "/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare/"
        data = load_json_to_dict_list(lidc_path+"data.jsonl")
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    for i in range(len(info_list)):
        if "conditioning" in info_list[i].keys():
            info_list[i]["conditioning"].update(info_add[i])
        else:
            info_list[i]["conditioning"] = info_add[i]
    
    if save:
        save_dict_list_to_json(info_list,info_jsonl_path)
    return info_list

def create_totseg_label(f):
    #saves a segmentation as a combined file in the same folder as the volume 
    f2 = f.replace("ct.nii.gz","seg.nii.gz")
    vol, seg = load_totseg(f)
    nib.save(nib.Nifti1Image(seg, np.eye(4)), f2)

def load_totseg(f,class_to_idx=None):
    #loads a volume and its segmentation
    vol = nib.load(f).get_fdata()
    if os.path.exists(f.replace("ct.nii.gz","seg.nii.gz")):
        seg = nib.load(f.replace("ct.nii.gz","seg.nii.gz")).get_fdata().astype(np.uint8)
    else:
        assert class_to_idx is not None 
        seg_files = Path(f).parent.glob("segmentations/*.nii.gz")
        seg = np.zeros(vol.shape,dtype=np.uint8)
        for s in seg_files:
            class_name = Path(s).name.replace(".nii.gz","")
            mask = nib.load(s).get_fdata()>0
            seg[mask] = class_to_idx[class_name]
    return vol, seg

def totseg_vol_to_slices(f,delta=10):
    vol, seg = load_totseg(f)
    vol = quantile_normalize(vol)
    s = vol.shape
    slice_idx = [list(range(0,s[dim_i],delta)) for dim_i in range(3)]
    slices = []
    slice_segs = []
    for dim_i in range(3):
        idx = [slice(None) for _ in range(3)]
        slices.append([])
        slice_segs.append([])
        for i in slice_idx[dim_i]:
            idx[dim_i] = i
            slices[-1].append(vol[tuple(idx)])
            slice_segs[-1].append(seg[tuple(idx)])
    return slices, slice_segs

def axis_metrics(vol,seg,vol_alpha=0.01):
    """sums values along xy, xz, yz, also returns 
    bounding boxes for the segmentation and volume (approximate)
    bbox returned as [min_d1,max_d1,min_d2,max_d2,min_d3,max_d3]
    Note that max_d1 is inclusive, so the slice should be [min_d1:max_d1+1]
    """
    sum_dims = [[1,2],[0,2],[0,1]]
    summed_vals_vol = []
    summed_vals_seg = []
    for d in range(3):
        summed_vals_vol.append(np.sum(vol,axis=tuple(sum_dims[d])))
        summed_vals_seg.append(np.sum(seg,axis=tuple(sum_dims[d])))
    summed_vals_vol = [v/np.sum(v) for v in summed_vals_vol]
    bbox_vol = []
    bbox_seg = []
    assert seg.sum()>0, "no segmentation found"
    cumsum_vals_vol = [np.cumsum(v) for v in summed_vals_vol]
    for d in range(3):
        first_vol, last_vol = np.logical_and(cumsum_vals_vol[d]>vol_alpha,cumsum_vals_vol[d]<1-vol_alpha).nonzero()[0][[0,-1]]

        bbox_vol.extend([first_vol,last_vol])
        seg_bool = np.flatnonzero(summed_vals_seg[d])
        first_seg, last_seg = seg_bool[0], seg_bool[-1]
        bbox_seg.extend([first_seg,last_seg])
    max_bbox = sum([[s-1,s-1] for s in seg.shape[:3]],[])
    min_bbox = [0 for _ in range(6)]
    bbox_vol = [max(min_bbox[i],bbox_vol[i]) for i in range(6)]
    bbox_vol = [min(max_bbox[i],bbox_vol[i]) for i in range(6)]
    bbox_seg = [max(min_bbox[i],bbox_seg[i]) for i in range(6)]
    bbox_seg = [min(max_bbox[i],bbox_seg[i]) for i in range(6)]
    return summed_vals_vol, summed_vals_seg, bbox_vol, bbox_seg

def create_totseg_samples(f,
                        save=False,
                        delta_abs=0,
                        delta_rel=0.2,
                        max_images=20,
                        min_dist=10,
                        delete_before=False):
    vol,seg = load_totseg(f)
    vol = quantile_normalize(vol)
    bbox_seg = axis_metrics(vol,seg)[-1]

    vol = np.clip(np.round(vol*255),0,255).astype(np.uint8)
    idx_sample = axis_bbox_to_idx(vol.shape,bbox_seg,delta_abs=delta_abs,delta_rel=delta_rel,
                                            max_images=max_images,min_dist=min_dist)
    samples = {"name": [], "image": [], "gt": []}
    for dim_i in range(len(idx_sample)):
        for i in idx_sample[dim_i]:
            samples["name"].append(f"dim{dim_i}_slice{i}")
            s = [slice(None) for _ in range(3)]
            s[dim_i] = i
            t = vol[tuple(s)]
            samples["image"].append(vol[tuple(s)])
            samples["gt"].append(seg[tuple(s)])
    if delete_before:
        f = Path(f)
        f2 = f.parent / "samples"
        if f2.exists():
            shutil.rmtree(f2)
    t = samples["image"][0]
    if save:
        save_folder = Path(f).parent / "samples"
        os.makedirs(save_folder,exist_ok=True)
        for i in range(len(samples["name"])):
            name = samples["name"][i]
            img1 = Image.fromarray(samples["image"][i])
            img1.save(f"{save_folder}/{name}_im.png")
            img2 = Image.fromarray(samples["gt"][i])
            img2.putpalette(nc.largest_pallete)
            img2.save(f"{save_folder}/{name}_gt.png")
    return samples

def axis_bbox_to_idx(volshape,bbox_seg,delta_abs=0,delta_rel=0.05,
                     max_images=20,
                     min_dist=10):
    min_bbox = [0 for _ in range(6)]
    max_bbox = sum([[s-1,s-1] for s in volshape],[])
    width = [bbox_seg[i+1]-bbox_seg[i] for i in [0,2,4]]
    delta = [delta_abs+delta_rel*w for w in width]
    bbox_delta = sum([[-d,d] for d in delta],[])
    bbox = [bbox_seg[i]+bbox_delta[i] for i in range(6)]
    bbox = [max(min_bbox[i],bbox[i]) for i in range(6)]
    bbox = [min(max_bbox[i],bbox[i]) for i in range(6)]
    
    idx_sample = []
    for d in range(3):
        width = bbox[d*2+1]-bbox[d*2]
        n = np.round(min(max_images,width//min_dist)).astype(int)
        idx_sample.append(np.round(np.linspace(bbox[d*2],bbox[d*2+1],n)).astype(int).tolist())
    return idx_sample
    
def save_clip_vectors(dataset_name=None,batch_size=16,save_path = f"./data/CLIP_emb.pth", clip_model="ViT-B/32"):
    idx_to_class = load_json_to_dict_list(f"./data/{dataset_name}/idx_to_class.json")[0]
    idx_to_class_pretty = {k: prettify_classname(v,dataset_name) for k,v in idx_to_class.items()}
    crit = [int(i) for i in list(idx_to_class_pretty.keys())]
    sort_order = np.argsort(crit)
    sorted_crit = [str(crit[i]) for i in sort_order] 
    list_of_classnames_pretty = [idx_to_class_pretty[i] for i in sorted_crit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device)
    n_batches = len(idx_to_class_pretty)//batch_size
    clip_emb_matrix = np.zeros((len(idx_to_class_pretty),512))
    with torch.no_grad():
        for i in tqdm.tqdm(range(n_batches)):
            batch_idx = range(i*batch_size,min((i+1)*batch_size,len(idx_to_class_pretty)))
            list_of_classnames_batch = [list_of_classnames_pretty[i] for i in batch_idx]
            text = clip.tokenize(list_of_classnames_batch).to(device)
            text_features = model.encode_text(text)
            clip_emb_matrix[batch_idx] = text_features.cpu().numpy()
    if Path(save_path).exists():
        loaded = torch.load(save_path)
    else:
        loaded = {}
    loaded[dataset_name] = {"class_names": list_of_classnames_pretty,
                            "class_names_pretty": list_of_classnames_pretty, 
                            "embeddings": clip_emb_matrix,
                            "class_idx": [int(i) for i in sorted_crit]}
    torch.save(loaded,save_path)

def show_clip_vectors(save_path = f"./data/CLIP_emb.pth"):
    check_keys=["class_names","class_names_pretty","embeddings","class_idx"]
    loaded = torch.load(save_path)
    dataset_names = list(loaded.keys())
    for dataset_name in dataset_names:
        idx_to_class = load_json_to_dict_list(f"./data/{dataset_name}/idx_to_class.json")[0]
        n = len(idx_to_class)
        list_of_pretty_classnames = [prettify_classname(v,dataset_name) for v in idx_to_class.values()]
        dataset_was_saved_correctly = True
        #check all keys are there and have length n
        for key in check_keys:
            if not key in loaded[dataset_name].keys():
                dataset_was_saved_correctly = False
            elif len(loaded[dataset_name][key])!=n:
                dataset_was_saved_correctly = False
        #check all pretty classnames are present
        if not all([cn in loaded[dataset_name].get("class_names_pretty",[]) for cn in list_of_pretty_classnames]):
            dataset_was_saved_correctly = False
        if dataset_was_saved_correctly:
            print(f"Dataset {dataset_name} was saved correctly.")
        else:
            print(f"Dataset {dataset_name} was not saved correctly.")

def extract_data_subsets(dataset_names=None,subset_size=200,save_name="info_subset.jsonl"):
    """Saves a random subset of the dataset info jsonl file to a new file"""
    
    if dataset_names is None:
        #do it fo all live datasets
        live_info = load_json_to_dict_list("./data/datasets_info_live.json")
        dataset_names = [d["dataset_name"] for d in live_info if d["live"]]
    if isinstance(dataset_names,str):
        dataset_names = [dataset_names]
    for dataset_name in dataset_names:
        info_jsonl_path = f"./data/{dataset_name}/info.jsonl"
        info_list = load_json_to_dict_list(info_jsonl_path)
        n = len(info_list)
        if isinstance(subset_size,float):
            n_use = np.ceil(subset_size*n).astype(int)
        else:
            n_use = min(subset_size,n)
        dataset_specific_seed = str_to_seed(dataset_name)
        previous_seed = np.random.get_state()[1][0]
        np.random.seed(seed=dataset_specific_seed)
        indices = np.random.permutation(n)[:n_use]
        np.random.seed(previous_seed)
        subset_info = [info_list[i] for i in indices]

        save_filename = f"./data/{dataset_name}/{save_name}"
        save_dict_list_to_json(subset_info,save_filename)
        print(f"Saved {n_use} sample infos from {dataset_name} to {save_filename}")
        

def zip_data_subsets(dataset_names=None,
                     save_path="./data/data_subset_[info_name].zip",
                     info_name="info_subset.jsonl",
                     same_folder=True,
                     include_info=True,
                     include_idx_to_class=True,
                     include_datasets_info_live=True):
    """
    Zips all data in in specified info file into a zip file
    while retaining structure of folders
    """
    if dataset_names is None:
        #do it fo all live datasets
        live_info = load_json_to_dict_list("./data/datasets_info_live.json")
        dataset_names = [d["dataset_name"] for d in live_info if d["live"]]
    if isinstance(dataset_names,str):
        dataset_names = [dataset_names]
    datasets_info_path = "./data/datasets_info_live.json"
    datasets_info = load_json_to_dict_list(datasets_info_path)
    dataset_name_to_fmt = {d["dataset_name"]: d["file_format"] for d in datasets_info if d["live"]}
    save_path = save_path.replace("[info_name]",info_name.split(".")[0])
    with zipfile.ZipFile(save_path, 'w') as zipf:
        for dataset_name in dataset_names:
            dataset_fmt = dataset_name_to_fmt[dataset_name]
            info_jsonl_path = f"./data/{dataset_name}/{info_name}"
            if not Path(info_jsonl_path).exists():
                print(f"File {info_jsonl_path} does not exist. Skipping.")
                continue
            info_list = load_json_to_dict_list(info_jsonl_path)
            for info in info_list:
                folder_i = np.floor(info["i"]/1000).astype(int)
                filename_im = f"./data/{dataset_name}/f{folder_i}/{info['i']}_im."+dataset_fmt
                filename_la = f"./data/{dataset_name}/f{folder_i}/{info['i']}_la.png"
                assert Path(filename_im).exists(), f"File {filename_im} does not exist."
                assert Path(filename_la).exists(), f"File {filename_la} does not exist."
                if same_folder:
                    savename_im = f"./data/{dataset_name}/files/{info['i']}_im."+dataset_fmt
                    savename_la = f"./data/{dataset_name}/files/{info['i']}_la.png"
                else:
                    savename_im = f"./data/{dataset_name}/f{folder_i}/{info['i']}_im."+dataset_fmt
                    savename_la = f"./data/{dataset_name}/f{folder_i}/{info['i']}_la.png"
                zipf.write(filename_im,savename_im)
                zipf.write(filename_la,savename_la)
            if include_info:
                zipf.write(info_jsonl_path)
            if include_idx_to_class:
                idx_to_class_path = f"./data/{dataset_name}/idx_to_class.json"
                if Path(idx_to_class_path).exists():
                    zipf.write(idx_to_class_path)
            print(f"Saved {len(info_list)} samples from {dataset_name} to {save_path}")
        if include_datasets_info_live:
            zipf.write(datasets_info_path)
    save_path = save_path.replace("[info_name]",info_name)

def get_slice_bbox(mask):
    assert np.any(mask), "mask is empty"
    if len(mask.shape)==3:
        mask = mask.any(axis=-1)
    b1,b2 = np.where(mask.any(axis=1))[0][[0,-1]]
    b3,b4 = np.where(mask.any(axis=0))[0][[0,-1]]
    return b1,b2+1,b3,b4+1

def bbox_crop(image,mask,bbox,size,mask_pad=0,image_pad=0):
    """crops centrally around the bbox, adding enough pixels to reach
    the desired size. If the larger size exceeds the image size, the
    image is padded with zeros. if the number of needed pixels is odd,
    one more pixel is added to the top left corner (smaller index)"""
    if isinstance(size,int):
        d1,d2 = size,size
    else:
        d1,d2 = size
    if bbox is None:
        bbox = get_slice_bbox(mask)
    b1,b2,b3,b4 = bbox # image[b1:b2,b3:b4] is the bbox
    h,w = image.shape
    assert mask.shape[:2]==(h,w), f"mask shape {mask.shape} does not match image shape {image.shape}"
    #number of available pixels on each side
    avail = b1,h-b2,b3,w-b4
    assert all([val>=0 for val in avail]), f"bbox is invalid since it exceeds bounds: {bbox}. Image shape: {image.shape}"
    assert b1<b2 and b3<b4, f"bbox is invalid since we dont have (stop > start): {bbox}"
    #number of additional pixels needed on each side
    a = (d1-(b2-b1))/2, (d2-(b4-b3))/2
    c = lambda x: int(np.ceil(x))
    f = lambda x: int(np.floor(x))
    a = [c(a[0]),f(a[0]),c(a[1]),f(a[1])]
    #number of additional pixels needed on each side, inside the image
    a_i = [min(a[i],avail[i]) for i in range(4)]
    #number of additional pixels needed on each side, which must be padded since they are outside the image
    a_o = [val1-val2 for val1,val2 in zip(a,a_i)]
    #old_slice
    old_slice = (slice(b1-a_i[0],b2+a_i[1]),slice(b3-a_i[2],b4+a_i[3]))
    #new_slice
    new_slice = (slice(a_o[0],d1-a_o[1]),slice(a_o[2],d2-a_o[3]))
    image_new = np.zeros((d1,d2),dtype=image.dtype)+image_pad
    image_new[new_slice] = image[old_slice]
    mask_shape_new = list(mask.shape)
    mask_shape_new[0] = d1
    mask_shape_new[1] = d2
    mask_new = np.zeros(mask_shape_new,dtype=mask.dtype)+mask_pad
    mask_new[new_slice] = mask[old_slice]
    return image_new,mask_new


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="pascal")
    args = parser.parse_args()
    if args.process==0:
        print("PROCESS 0: variable dataset")
        downloader = DatasetDownloader()        
        downloader.process_files(args.dataset)
    elif args.process==1:
        print("PROCESS 1: sa1b")
        downloader = DatasetDownloader()
        downloader.process_files("sa1b")
    elif args.process==2:
        print("PROCESS 2: coco")
        downloader = DatasetDownloader()
        downloader.process_files("coco")
    elif args.process==3:
        print("PROCESS 3: ade20k")
        downloader = DatasetDownloader()
        downloader.process_files("ade20k")
    elif args.process==4:
        print("PROCESS 4: monu4")
        downloader = DatasetDownloader()
        downloader.allowed_failure_rate = 0
        downloader.process_files("monu4")
    elif args.process==5:
        print("PROCESS 5: prettify_data only ugly datasets")
        for dataset in ["sa1b","coco","ade20k"]:
            prop = prettify_data(dataset)
            print(f"Finished {dataset}. Saved images for {prop*100:.2f}% of the dataset")
    elif args.process==6:
        print("PROCESS 6: add_existence_of_prettify_to_info_jsonl")
        add_existence_of_prettify_to_info_jsonl()
    elif args.process==7:
        print("PROCESS 7: prettify ade20k")
        for dataset in ["ade20k"]:
            prop = prettify_data(dataset)
            print(f"Finished {dataset}. Saved images for {prop*100:.2f}% of the dataset")
    elif args.process==8:
        print("PROCESS 8: add_existence_of_sam_features_to_info_jsonl")
        add_existence_of_sam_features_to_info_jsonl(dataset_names=args.dataset)
    elif args.process==9:
        print("PROCESS 9: delete_all_sam_features dry")
        delete_all_sam_features(dry=True)
    elif args.process==10:
        print("PROCESS 10: delete_all_sam_features")
        delete_all_sam_features()
    elif args.process==11:
        print("PROCESS 11: totseg")
        downloader = DatasetDownloader()
        downloader.allowed_failure_rate = 0
        do_step = default_do_step()
        do_step["delete_f_before"] = 1
        downloader.process_files("totseg",do_step=do_step)
    elif args.process==12:
        print("PROCESS 12: duts, ecssd, fss, msra, dis")
        names = ["duts","ecssd","fss","msra","dis"]
        downloader = DatasetDownloader()        
        downloader.process_files(names)
    elif args.process==13:
        print("PROCESS 13: prettify_data all datasets, loaded from live info")
        live_info = load_json_to_dict_list("./data/datasets_info_live.json")
        #live_datasets = [d["dataset_name"] for d in live_info if d["live"]]
        live_datasets = ['duts', 'ecssd', 'fss', 'msra', 'dis']
        for dataset in live_datasets:
            print(f"Processing {dataset}")
            info_jsonl = load_json_to_dict_list(f"./data/{dataset}/info.jsonl")
            if any([inf["pretty"] for inf in info_jsonl if "pretty" in inf.keys()]):
                print(f"{dataset} already prettified. Skipping.")
                continue            
            prop = prettify_data(dataset)
            print(f"Finished {dataset}. Saved images for {prop*100:.2f}% of the dataset")
        #add existence of prettify to info jsonl
        add_existence_of_prettify_to_info_jsonl()
    elif args.process==14:
        print("PROCESS 14: embedding CLIP vectors")
        live_info = load_json_to_dict_list("./data/datasets_info_live.json")
        all_dataset_names = [d["dataset_name"] for d in live_info if d["live"]]
        for dataset_name in all_dataset_names:
            print(f"Processing {dataset_name}")
            save_clip_vectors(dataset_name)
    elif args.process==15:
        print("PROCESS 15: show CLIP vectors")
        show_clip_vectors()
    elif args.process==16:
        print("PROCESS 16: embed single dataset CLIP vectors")
        save_clip_vectors(args.dataset)
    elif args.process==17:
        print("PROCESS 17: extract data subsets")
        extract_data_subsets()
    elif args.process==18:
        print("PROCESS 18: zip data subsets")
        zip_data_subsets()
    elif args.process==19:
        print("PROCESS 19: create test_patient_ids.txt and vali_patient_ids.txt for lidc")
        if Path("/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare/vali_patient_ids.txt").exists():
            raise ValueError("Files already exist. Delete them first to create new ones.")
        data = load_json_to_dict_list("/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare/data.jsonl")
        patient_ids = list(set([d["patient_id"] for d in data]))
        perm = np.random.permutation(len(patient_ids))
        n = len(patient_ids)
        vali_idx = perm[int(n*0.8):int(n*0.9)]
        test_idx = perm[int(n*0.9):]
        vali_patient_ids = [patient_ids[i] for i in vali_idx]
        test_patient_ids = [patient_ids[i] for i in test_idx]
        with open("/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare/vali_patient_ids.txt","w") as f:
            for pid in vali_patient_ids:
                f.write(f"{pid}\n")
        with open("/home/jloch/Desktop/diff/diffusion2/data/lidc/lidcshare/test_patient_ids.txt","w") as f:
            for pid in test_patient_ids:
                f.write(f"{pid}\n")
    elif args.process==20:
        print("PROCESS 20: save sam features")
        save_sam_features(args.dataset, sam_idx_or_name=0)
    else:
        raise ValueError(f"Unknown process: {args.process}")
    
if __name__=="__main__":
    main()
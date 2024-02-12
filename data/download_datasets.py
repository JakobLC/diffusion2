import os,sys

sys.path.append(os.path.abspath('./source/'))
sys.path.append(os.path.abspath('./data/'))

from urllib.parse import urlparse
from scipy.io import loadmat
import numpy as np
import re
import shutil
from PIL import Image
import tqdm
from data_utils import (unpack_files, save_dict_list_to_json, load_json_to_dict_list, rle_to_mask)
import glob
import json
import pickle
import jlc.nc as nc 
from pathlib import Path
import scipy.ndimage as nd
import cv2
from datasets import SegmentationDataset

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
                      do_step = {"unpack": 0,
                                 "make_f0": 1,
                                 "save_images": 1,
                                 "reset_info": 1,
                                 "save_info": 1,
                                 "save_global_info": 1,
                                 "save_class_dict": 1,
                                 "delete_unused_files": 0}):
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
            class_dict = {l[:l.find(":")]: l[l.find(":")+2:(-1 if l[-1]=="\n" else None)] for l in lines}
            class_dict[0] = "background"
            unpack_list = [os.path.join(folder_path, file_name) for file_name in ["trainval.tar.gz","trainval.tar","VOCtrainval_11-May-2012.tar"]]
            file_name_list = [file_name[:-4] for file_name in os.listdir(os.path.join(folder_path,"trainval"))]
            def load_image_label_info(file_name):
                label_path = os.path.join(folder_path,"trainval",f"{file_name}.mat")
                image_path = os.path.join(folder_path,"VOCdevkit/VOC2012/JPEGImages",f"{file_name}.jpg")
                label = loadmat(label_path)["LabelMap"]
                image = Image.open(image_path)
                uq = np.unique(label).tolist()
                assert len(uq)<=256, "uint8 format fails if more than 256 classes are present."
                label2 = np.zeros_like(label)
                info = {"classes": [0]}
                for i,u in enumerate(uq):
                    label2[label==u] = i
                    info["classes"].append(u)
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
                if name in train_names:
                    dataset_idx_dict[file_name] = 0
                elif name in test_names:
                    dataset_idx_dict[file_name] = 2
                else:
                    dataset_idx_dict[file_name] = 1
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
                info = {"classes": list(range(len(np.unique(label)))),"split_idx": 2}
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
        else:
            raise ValueError(f"Dataset {name} not supported.")
        #unpack files:
        if do_step["make_f0"]:
            os.makedirs(os.path.join(folder_path,f"f{folder_i}"),exist_ok=True)
        if do_step["save_class_dict"]:
            save_dict_list_to_json(class_dict, os.path.join(folder_path,"idx_to_class.json"),append=False)

        jsonl_save_path = os.path.join(folder_path,"info.jsonl")
        if do_step["reset_info"]:
            save_dict_list_to_json([], jsonl_save_path, append=False)
            
        if not any([do_step[k] for k in ["save_images","save_info","save_global_info","delete_unused_files"]]): return

        #loop over label-image pairs
        samples_per_split = [0,0,0]
        for file_name in tqdm.tqdm(file_name_list):
            has_run_long_enough = num_attempts>np.ceil(2/(self.allowed_failure_rate+0.01)) or self.allowed_failure_rate<0.01
            if (self.allowed_failure_rate*num_attempts >= file_i) and has_run_long_enough:
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
                if num_saved_images==self.files_per_folder:
                    folder_i += 1
                    num_saved_images = 0
                    os.makedirs(os.path.join(folder_path,f"f{folder_i}"),exist_ok=True)
            except Exception:
                if show_error_flag:
                    raise Exception(f"Too many failed attempts. {num_attempts} attempts, {file_i} saved images. Last error shown above.")
        if do_step["save_global_info"]:
            global_info = {"dataset_name": name,
                            "urls": [],
                            "num_samples": len(file_name_list),
                            "num_classes": len(class_dict),
                            "file_format": image_suffix,
                            "samples_per_split": samples_per_split}
            save_dict_list_to_json(global_info, "./datasets_info.json", append=True)
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
        
        print(f"Finished processing dataset: {name}. Num failures: {num_attempts-file_i}/{num_attempts}.")

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
        dataset = SegmentationDataset(datasets=dataset,split="all",shuffle_datasets=False,prettify_data=False)
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
            image = image.transpose(1,0,2)
            label = label.transpose(1,0)
            transposed=True
        else:
            transposed=False
        #now we smallest size is height
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
            raise ValueError("thin_bg must be in ['always','never','sometimes']")

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

def add_existence_of_prettify_to_info_jsonl():
    list_of_info_jsons = Path("./data/").glob("*/info.jsonl")
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
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=int, default=0)
    args = parser.parse_args()
    if args.process==0:
        print("PROCESS 0: pascal")
        downloader = DatasetDownloader()
        downloader.process_files("pascal")
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
        print("PROCESS 5: prettify_data")
        for dataset in ["coco","hrsod","to5k","dram","coift","cityscapes","pascal","sa1b","ade20k","monu4"]:
            prop = prettify_data(dataset)
            print(f"Finished {dataset}. Saved images for {prop*100:.2f}% of the dataset")
    elif args.process==6:
        print("PROCESS 6: add_existence_of_prettify_to_info_jsonl")
        add_existence_of_prettify_to_info_jsonl()
    elif args.process==7:
        print("PROCESS 7: prettify dram")
        for dataset in ["dram"]:
            prop = prettify_data(dataset)
            print(f"Finished {dataset}. Saved images for {prop*100:.2f}% of the dataset")

    else:
        raise ValueError(f"Unknown process: {args.process}")
if __name__=="__main__":
    main()
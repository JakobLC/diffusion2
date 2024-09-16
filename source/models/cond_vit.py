# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os

if __name__=="__main__":
    sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type
import copy
import math
import warnings
import pandas as pd
from source.utils.mixed_utils import get_named_datasets,nice_split
from argparse import Namespace
from source.utils.argparse_utils import TieredParser

tp = TieredParser()
arg_items = tp.get_args(alt_parse_args=["--model_name","default"]).__dict__.items()
cond_image_prob_keys = [k for k,v in arg_items if k.startswith("p_")]
cond_image_keys =[k[2:] for k in cond_image_prob_keys]

#dynamic_image_keys = ["same_classes","same_dataset","same_vol","adjacent"]
#all_input_keys = cond_image_keys+dynamic_image_keys

def assert_one_to_one_list_of_str(list1,list2):
    assert isinstance(list1,list) and isinstance(list2,list), "Expected list, found: "+str(type(list1))+" and "+str(type(list2))
    for k in list1+list2:
        assert isinstance(k,str), "Expected str, found: "+str(type(k))
        assert k in list1, "Expected "+k+" from list2 to be in list1="+str(list1)
        assert k in list2, "Expected "+k+" from list1 to be in list2="+str(list2)

class ModelInputKwargs:
    """
    Class to manage kwargs for both model (ViT and UNet),
    as well as the dataset
    """
    def __init__(self,args,construct_args=False,assert_valid=True):
        if args is None:
            #get default args
            args = tp.get_args(alt_parse_args=["--model_name","default"]).__dict__
        if isinstance(args,Namespace):
            args = copy.deepcopy(args.__dict__)
        else:
            args = copy.deepcopy(args)
        self.args = args
        self.columns = ["name","type","spatialness","unet","vit","support"]
        if construct_args:
            self.construct_kwarg_table()
            if assert_valid:
                self.assert_inputs_are_valid()

    def compute_hyper_params(self):
        self.hyper_params = {}
        self.hyper_params["diff_channels"] = int(torch.log2(torch.tensor(self.args["max_num_classes"])).ceil().item())
        self.hyper_params["image_channels"] = 3
        self.hyper_params["image_encoder"] = 256
        self.hyper_params["class_names_datasets"] = get_named_datasets(self.args["datasets"])

    def use_input_criteria(self):
        #a=self.args
        inputs = {
            "sample":         lambda a: True,
            "time":           lambda a: True,
            "image":          lambda a: a["p_image"]>0,
            "image_features": lambda a: a["p_image"]>0 and a["image_encoder"]!="none",
            "points":         lambda a: a["p_points"]>0,
            "bbox":           lambda a: a["p_bbox"]>0,
            "self_cond":      lambda a: a["p_self_cond"]>0,
            "num_classes":    lambda a: a["class_type"]=="num_classes" and a["p_classes"]>0,
            "same_vol":       lambda a: a["p_same_vol"]>0,
            "same_classes":   lambda a: a["p_same_classes"]>0,
            "same_dataset":   lambda a: a["p_same_dataset"]>0,
            "adjacent":       lambda a: a["p_adjacent"]>0,
            "class_names":    lambda a: a["p_class_names"]>0,
            "semantic":       lambda a: (0<a["p_semantic"]) and (0<a["semantic_dl_prob"]<1)
            }
        return inputs
    
    def supported_inputs(self):
        unet_support = ["time","sample","image","image_features",
                        "points","bbox","self_cond","num_classes",
                        "semantic","adjacent","same_vol","same_classes",
                        "same_dataset"]
        vit_support = unet_support+["class_names"]
        return unet_support, vit_support

    def construct_kwarg_table(self,return_df=False):
        if not hasattr(self,"hyper_params"):
            self.compute_hyper_params()
        
        im_d = {"img_size": self.args["image_size"], 
                "patch_size": self.args["cond_patch_size"],
                "type": "image"}
        c_diff = self.hyper_params["diff_channels"]
        c_im = self.hyper_params["image_channels"]
        c_im_d = c_diff+c_im
        c_im_enc = self.hyper_params["image_encoder"]
        cnd = self.hyper_params["class_names_datasets"]

        inputs = {
            "sample":         {**im_d, "in_chans": c_diff},#6X
            "image":          {**im_d, "in_chans": c_im  },#3X
            "image_features": {**im_d, "in_chans": c_im_enc},
            "points":         {**im_d, "in_chans": c_diff},#6X
            "bbox":           {**im_d, "in_chans": c_diff},
            "self_cond":      {**im_d, "in_chans": c_diff},#6X
            "same_vol":       {**im_d, "in_chans": c_im_d},
            "same_classes":   {**im_d, "in_chans": c_im_d},
            "same_dataset":   {**im_d, "in_chans": c_im_d},
            "adjacent":       {**im_d, "in_chans": c_im_d},
            "time":           {"type": "scalar_continuous", "min": 0.0, "max": 1.0},
            "num_classes":    {"type": "scalar_discrete", "size": 64},
            "class_names":    {"type": "vocabulary", "size": -1, "class_names_datasets": cnd},
            "semantic":       {"type": "scalar_discrete", "size": 2}
            }
        #add what is needed to load each input
        load_type =  {"dynamic": ["adjacent","same_vol","same_classes","same_dataset"], #dynamic loading inside dataloader
                      "unique": ["image_features","time","sample","num_classes","semantic","self_cond","points","bbox"], #unique processing required
                      "info": ["class_names","semantic","num_classes","image"]#ready as-is: simply take from info
                      }
        #check that load_type is defined for all inputs
        assert_one_to_one_list_of_str(list(inputs.keys()),sum([v for v in load_type.values()],[]))
        for k,v in load_type.items():
            for k2 in v:
                inputs[k2]["load_type"] = k
        
        need_int2bit = ["same_classes","same_dataset","same_vol","adjacent","bbox","points","sample"]
        #               (im,gt)         (im,gt)        (im,gt)    (im,gt)    (gt)   (gt)     (gt)
        for k in need_int2bit:
            inputs[k]["int2bit"] = True
        #add spatialness
        spatialness = {0: ["num_classes","class_names","semantic"], #non-spatial
                       1: ["same_vol","same_classes","same_dataset"], #style-like spatial inputs (images)
                       2: ["image","image_features","bbox","points","self_cond","adjacent"], #pixelwise spatial inputs (images)
                       3: ["sample","time"]} #minimum required diffusion args
        #check that spatialness is defined for all inputs
        assert_one_to_one_list_of_str(list(inputs.keys()),sum([v for v in spatialness.values()],[]))
        for k,v in spatialness.items():
            for k2 in v:
                inputs[k2]["spatialness"] = k
        #add supports
        unet_support, vit_support = self.supported_inputs()
        for k in inputs.keys():
            inputs[k]["support"] = []
            if k in unet_support:
                inputs[k]["support"].append("unet")
            if k in vit_support:
                inputs[k]["support"].append("vit")

        input_criteria = self.use_input_criteria()

        self.kwarg_table = pd.DataFrame(columns=self.columns+["etc"])
        for name,v in inputs.items():
            append_dict = {"name": name,"etc": {}}
            for k2,v2 in v.items():
                if k2 in self.columns:
                    append_dict[k2] = v2
                else:
                    append_dict["etc"][k2] = v2
            use_input = input_criteria[name](self.args)
            unet_spatial_input = str(v["spatialness"]) in self.args["unet_spatialness"].split(",")
            vit_spatial_input = str(v["spatialness"]) in self.args["vit_spatialness"].split(",")
            append_dict["unet"] = use_input and unet_spatial_input
            append_dict["vit"] = use_input and vit_spatial_input

            self.kwarg_table.loc[len(self.kwarg_table)] = append_dict

        if return_df:
            return self.kwarg_table

    def get_input_probs(self,only_nonzero=False,only_used_inputs=False,only_dynamic=True):
        probs = {k: v for k,v in self.args.items() if k.startswith("p_")}
        if only_nonzero:
            probs = {k: v for k,v in probs.items() if v>0}
        if only_used_inputs:
            unet = self.kwarg_table["unet"]
            vit = self.kwarg_table["vit"]
            used_inputs = self.kwarg_table[unet|vit]["name"]
            probs = {k: v for k,v in probs.items() if k[2:] in used_inputs}
        if only_dynamic:
            probs = {k: v for k,v in probs.items() if k[2:] in dynamic_image_keys}
        return probs

    def get_input_dict(self,model_type="vit"):
        input_dict = {}
        for row in self.kwarg_table.iterrows():
            if row[1][model_type.lower()]:
                input_dict[row[1]["name"]] = {**row[1]["etc"],"input_type": row[1]["type"]}
        return input_dict

    def assert_inputs_are_valid(self,raise_error=True):
        try:
            assert len(self.kwarg_table)>0, "Need to construct kwarg table first, before checking validity"
            name_to_row_idx = {k: i for i,k in enumerate(self.kwarg_table["name"])}
            #assert image is an actual input
            assert len(self.kwarg_table.loc[name_to_row_idx["image"]]["support"])>=0, "Image has to be supported by either unet or vit"
            #check availability of all inputs
            for row in self.kwarg_table.iterrows():
                if row[1]["unet"]:
                    assert "unet" in row[1]["support"], "The input "+row[1]["name"]+" is not supported by the unet"
                if row[1]["vit"]:
                    assert "vit" in row[1]["support"], "The input "+row[1]["name"]+" is not supported by the vit"
                if row[1]["type"].startswith("image"):
                    im_s = row[1]["etc"]["img_size"]
                    pa_s = row[1]["etc"].get("patch_size",1)
                    #assert img_size is divisible by patch_size
                    assert im_s%pa_s==0, f"Image size must be divisible by patch size. Found image_size={im_s} and patch_size={pa_s}"
            return True
        except AssertionError as e:
            if raise_error:
                raise e
            else:
                return False

mik = ModelInputKwargs(args=None,construct_args=True,assert_valid=True)
all_input_keys = mik.kwarg_table["name"].tolist()
all_load_types = [item["load_type"] for item in mik.kwarg_table["etc"]]
dynamic_image_keys = [k for k,lt in zip(all_input_keys,all_load_types) if lt=="dynamic"]

vit_seperate_params = { 'num_params':          [5113088, 28510720, 89670912, 308278272, 637026048],
                        'model_name':          ['vit_tiny', 'vit_small', 'vit_b', 'vit_l', 'vit_h'],
                        'idx':                 [-2, -1, 0, 1, 2],
                        'embed_dim':           [256, 512, 768, 1024, 1280],
                        'depth':               [4, 8, 12, 24, 32],
                        'num_heads':           [4, 8, 12, 16, 16],
                        'global_attn_indexes': [[1, 2, 3], [1, 3, 5, 7], [2, 5, 8, 11], [5, 11, 17, 23], [7, 15, 23, 31]]}

additional_fields = ["num_params","model_name","idx"]

vit_shared_params = {'out_chans':   256,
                'img_size':         1024,
                'patch_size':       16,
                'qkv_bias':         True,
                'use_rel_pos':      True,
                'window_size':      14,
                'mlp_ratio':        4}
non_fancy_vit_shared_keys = "input_dict,img_size,patch_size,global_attn_indexes,block_types".split(",")
fancy_vit_shared_keys = ""
default_input_dict = {"image": {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3}}

def vit_args_from_idx(idx):
    args = copy.deepcopy(vit_shared_params)
    assert idx in list(range(-2,3)), f"Invalid index {idx} for ViT. Must be in range -2 to 2"
    dict_idx = vit_seperate_params['idx'].index(idx)
    for k in vit_seperate_params.keys():
        if k not in additional_fields:
            args[k] = vit_seperate_params[k][dict_idx]
    return args

def fancy_vit_from_idx(idx,del_keys=["global_attn_indexes","img_size","patch_size"]):
    args = vit_args_from_idx(idx)
    args["block_types"] = ["global" if i in args["global_attn_indexes"] else "window" for i in range(args["depth"])]
    args["input_dict"] = {"image": {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3}}
    for k in del_keys:
        del args[k]
    return args

def fancy_vit_from_args(mik_or_args):
    if isinstance(mik_or_args,(Namespace,dict)):
        if isinstance(mik_or_args,Namespace):
            args = copy.deepcopy(mik_or_args.__dict__)
        else:
            args = copy.deepcopy(mik_or_args)
        mik = ModelInputKwargs(args)
        mik.construct_kwarg_table()
        mik.assert_inputs_are_valid()
    else:
        assert isinstance(mik_or_args,ModelInputKwargs), "Expected ModelInputKwargs (mik) instance or dict/Namespace (args), found: "+str(type(mik))
        mik = copy.deepcopy(mik_or_args)
    args = mik.args
    fancy_vit_args = fancy_vit_from_idx(args["cond_sam_idx"])
    fancy_vit_args["input_dict"] = mik.get_input_dict("vit")
    index_to_opt = is_valid_cond_vit_setup(args["cond_vit_setup"])
    fancy_vit_args["injection_type"] = index_to_opt[4]
    fancy_vit_args["pre_reduction"] = {"a": "spatial", "b": "none"}[index_to_opt[1]]
    fancy_vit_args["post_reduction"] = {"a": "cls_token", "b": "mean_token", "c": "spatial", "d": "none"}[index_to_opt[3]]
    block_types = get_appropriate_block_types(depth=fancy_vit_args["depth"],block_types=index_to_opt[2])
    fancy_vit_args["block_types"] = block_types
    no_unet = len(nice_split(args["unet_spatialness"]))==0
    if no_unet:
        assert index_to_opt[3]=="c", "Only option 3c is allowed for models with no inputs for the unet, found: 3"+index_to_opt[3]+". cond_vit_setup="+args["cond_vit_setup"]
        fancy_vit_args["post_reduction"] = "diffusion_sample"
    return fancy_vit_args

def sam_vit_from_idx(idx):
    args = vit_args_from_idx(idx)
    return ImageEncoderViT(**args)

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def timestep_embedding(timesteps, dim, max_period=10):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ContinuousVariable(nn.Module):
    def __init__(
        self,
        dim: int,
        min_val: float,
        max_val: float,
        max_period: int = 10,):
        super().__init__()
        assert dim%2==0, "dim must be even"
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val
        self.map_to_0_1 = lambda x: (x-self.min_val)/(self.max_val-self.min_val)
        self.max_period = max_period
        self.layers = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, x):
        x = self.map_to_0_1(x)
        freqs = torch.exp(
                -math.log(self.max_period) * torch.arange(start=0, end=self.dim//2, dtype=torch.float32) / self.dim
            ).to(device=x.device)
        args = x[:, None].float() * freqs[None]
        return self.layers(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


cond_vit_setup_long = {1: {"option_name": "preprocess" ,
                           "a": "Stack channels spatially",
                           "b": "No reduction"},
                       2: {"option_name": "attention", "multiple": True,
                           "a": "global",
                           "b": "image",
                           "c": "cross",
                           "d": "grouped",
                           "e": "window",
                           "f": "window_shifted"},
                       3: {"option_name": "postprocess",
                           "a": "Only cls token",
                           "b": "Mean token",
                           "c": "Mean without spatial reduction",
                           "d": "No reduction"},
                       4: {"option_name": "injection"  , "requires": {"a": {3: ["a","b"]}, "c": {3: ["c"]}, "d": {3: ["c"]}},
                           "a": "Embed vec (single token)",
                           "b": "UNet cross-attention",
                           "c": "Spatial Embed (once)",
                           "d": "Spatial Embed (many times)",
                           "e": "Spatial Embed (after Downsample/Upsample)"}
                           }

def get_appropriate_block_types(depth=12,block_types="ae",transform_to_names=True):
    for letter in block_types:
        assert letter in cond_vit_setup_long[2].keys(), "did not recognize letter, found: "+letter+" in "+block_types+" should be in "+str(cond_vit_setup_long[2].keys())
    block_types = block_types[::-1]
    if block_types.find("f")<0:
        block_types = block_types.replace("e","ef")
    block_types = (block_types*depth)[:depth] #repeat pattern and crop
    if transform_to_names:
        block_types = [cond_vit_setup_long[2][b] for b in block_types]
    return block_types

def get_opt4_from_cond_vit_setup(cond_vit_setup):
    index_to_opt = is_valid_cond_vit_setup(cond_vit_setup)
    return index_to_opt[4]

def is_valid_cond_vit_setup(cond_vit_setup,long_names_instead=False):
    assert isinstance(cond_vit_setup,str), "expected str, found: "+str(cond_vit_setup)
    opt_numbers = list(range(1,5))
    opt_numbers_str = [str(opt) for opt in opt_numbers]
    assert all([cond_vit_setup.count(str(opt))==1 for opt in opt_numbers]), "Must have at exactly one of each option, found: "+str(cond_vit_setup)
    index_of_on = [j for j in range(len(cond_vit_setup)) if cond_vit_setup[j] in opt_numbers_str]
    index_of_on.append(len(cond_vit_setup))
    assert len(index_of_on)==5
    index_to_opt = {k: cond_vit_setup[index_of_on[k-1]+1:index_of_on[k]] for k in opt_numbers}
    index_to_opt2 = {}
    for k,v in cond_vit_setup_long.items():
        letters = index_to_opt[k]
        mult = v.get("multiple",False)
        if len(letters)>1:
            assert mult, "Option "+str(k)+" does not allow multiple values"
        for letter in letters:
            assert letter in v.keys(), "Invalid option for "+str(k)+": "+letter+" not in "+str([k for k in v.keys() if len(k)==1])
            if long_names_instead:
                val = cond_vit_setup_long[k]["option_name"]+": "
                val += ",".join([cond_vit_setup_long[k][v] for v in letters])
            else:
                val = letters
            index_to_opt2[k] = val
        if "requires" in v.keys():
            if val in v["requires"].keys():
                for k_req,v_req in v["requires"][val].items():
                    assert index_to_opt2[k_req] in v_req, f"Invalid requirement. When having option {str(k)+val}, you need one of: {[str(k_req)+item for item in v_req]}, found: {str(k_req)+index_to_opt2[k_req]}"
    return index_to_opt2

class FancyViT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        block_types: list = [],
        share_image_rel_pos: bool = True,
        share_image_patch_embed: bool = True,
        input_dict: dict = default_input_dict,
        max_seq_len: int = None,
        pre_reduction: str = "none",
        post_reduction: str = "spatial",
        injection_type: str = "a",
        diff_channels: int = 6,
    ) -> None:
        super().__init__()
        assert pre_reduction in ["spatial","none"], "Invalid pre_reduction: "+pre_reduction
        assert post_reduction in ["cls_token","mean_token","spatial","diffusion_sample","none"], "Invalid post_reduction: "+post_reduction
        self.pre_reduction = pre_reduction
        self.post_reduction = post_reduction
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.share_image_rel_pos = share_image_rel_pos
        self.share_image_patch_embed = share_image_patch_embed
        valid_input_types = ["image","scalar_continuous","scalar_discrete","vocabulary"]
        assert len(block_types)==depth, "Fancy block type must be specified for each block"
        self.img_sizes = []
        self.patch_sizes = []
        self.in_chans = []
        self.vocab_keys = []
        input_dict["cls"] = {"input_type": "scalar_discrete", "size": 1}
        has_image_input = False
        for input_name, input_info in input_dict.items():
            t = input_info["input_type"]
            assert t in valid_input_types, f"Invalid input type specified. Found {t}"
            if t=="image":
                for k in ["img_size","patch_size","in_chans"]:
                    assert isinstance(input_info[k],int), f"Expected {k} to be int, found {input_info[k]}"
                self.img_sizes.append(input_info["img_size"])
                self.patch_sizes.append(input_info["patch_size"])
                self.in_chans.append(input_info["in_chans"])
                has_image_input = True
            elif t=="scalar_continuous":
                setattr(self, input_name+"_embed", ContinuousVariable(dim=embed_dim, min_val=input_info["min"], max_val=input_info["max"]))
            elif t=="scalar_discrete":
                setattr(self, input_name+"_embed", nn.Embedding(input_info["size"], embed_dim))
            elif t=="vocabulary":
                if len(input_info.get("class_names_datasets","")):
                    assert "class_names"==input_name, "Must have class_names key if class_names_datasets is specified"
                    assert input_info["size"]<=0, "Size must be non-positive (since it isn't used) for vocabulary based on class names. Found: "+str(input_info[input_name]["size"])
                    clip_matrix, dataset_idx_to_clip_idx, pretty_name_to_clip_idx = get_clip_matrix(input_info["class_names_datasets"])
                    input_dict[input_name]["size"] = clip_matrix.shape[0]
                    clip_feat_dim = clip_matrix.shape[1]
                    self.class_names_embed = nn.Sequential(nn.Embedding(input_info["size"], 
                                                                        clip_feat_dim,
                                                                        _weight=clip_matrix.float(),
                                                                        _freeze=True),
                                                           nn.Linear(clip_feat_dim,embed_dim))
                    to_vocab_idx = {j: j for j in range(input_info["size"])}
                    to_vocab_idx.update(pretty_name_to_clip_idx)
                    to_vocab_idx.update(dataset_idx_to_clip_idx)
                    to_vocab_idx = WrapToVocabDict(to_vocab_idx)
                else:
                    assert input_info["size"]>0, "Size must be positive for vocabulary. Found: "+str(input_info["size"])
                    setattr(self, input_name+"_embed", nn.Embedding(input_info["size"], embed_dim))
                    to_vocab_idx = DummyIdentityDict()
                setattr(self, input_name+"_to_vocab_idx", to_vocab_idx)
                self.vocab_keys.append(input_name)
        self.input_dict = input_dict
        self.img_size = self.img_sizes[0] if len(set(self.img_sizes))==1 else None
        self.patch_size = self.patch_sizes[0] if len(set(self.patch_sizes))==1 else None
        if self.pre_reduction=="spatial":
            assert self.share_image_patch_embed, "Must share image patch embed for spatial reduction"
            irsum = lambda x: int(round(sum(x)))
            image_input_names = [k for k,v in input_dict.items() if v["input_type"]=="image"]
            self.cls_image_channel_slice = {n: slice(irsum(self.in_chans[:i]),
                                                     irsum(self.in_chans[:i+1]
                                                           )) for i,n in enumerate(image_input_names)}
            self.in_chans = [irsum(self.in_chans)]
            self.cls_image_size = lambda bs: (bs,self.in_chans[0],self.img_size,self.img_size)
            self.input_dict["cls_image"] = {"input_type": "image", 
                                            "img_size": self.img_size, 
                                            "patch_size": self.patch_size, 
                                            "in_chans": self.in_chans[0]}

        self.token_img_size = None
        if self.share_image_patch_embed:
            if "image_features" in input_dict.keys():
                warnings.warn("share_image_patch_embed is pretty ineffecient when using image_features since it has many channels")
            if len(self.img_sizes)>0:
                #assert we have same patch_size and image_size
                assert len(set(self.img_sizes))==1, "Image sizes must be the same for shared patch embed. found: "+str(self.img_size)
                assert len(set(self.patch_sizes))==1, "Patch sizes must be the same for shared patch embed. found: "+str(self.patch_size)
                self.shared_patch_embed = PatchEmbed(
                    kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size),
                    in_chans=max(self.in_chans),
                    embed_dim=self.embed_dim,
                )
                self.token_img_size = self.img_size // self.patch_size
                if use_abs_pos:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, self.token_img_size, self.token_img_size, self.embed_dim)
                    )
                else:
                    self.pos_embed: Optional[nn.Parameter] = None
        else:
            raise NotImplementedError
            for input_name, input_info in input_dict.items():
                if input_info["input_type"]=="image":
                    patch_embed = PatchEmbed(
                        kernel_size=(input_info["patch_size"], input_info["patch_size"]),
                        stride=(input_info["patch_size"], input_info["patch_size"]),
                        in_chans=input_info["in_chans"],
                        embed_dim=self.embed_dim,
                    )
                    setattr(self, input_name+"_embed", patch_embed)


        if not has_image_input:
            for bt in ["grouped","window","window_shifted","image"]:
                assert bt not in block_types, "No image input, so cannot have "+bt+" attention which requires image-like inputs."
        attn_size_dict = {"global": None,
                        "image": self.token_img_size,
                        "window": window_size,
                        "window_shifted": window_size,
                        "grouped": self.token_img_size,
                        "cross": None}
        self.blocks = nn.ModuleList()
        use_rel_pos_attention_types = ["window","window_shifted","image"]
        for i in range(depth):
            attention_type = block_types[i]
            attention_size = attn_size_dict[attention_type]
            block = FancyBlock(
                attention_size=attention_size,
                attention_type=attention_type,
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos and (attention_type in use_rel_pos_attention_types),
                rel_pos_zero_init=rel_pos_zero_init,
                )
            self.blocks.append(block)
        if self.post_reduction=="spatial":
            self.neck = nn.Sequential(
                nn.Conv2d(self.embed_dim,out_chans,kernel_size=1,bias=False,),
                LayerNorm2d(out_chans),
                nn.Conv2d(out_chans,out_chans,kernel_size=3,padding=1,bias=False),
                LayerNorm2d(out_chans),
            )
        elif self.post_reduction=="diffusion_sample":
            #self.inverse_patch_embed = PatchEmbed(kernel_size=(self.patch_size, self.patch_size),
            #                                    stride=(self.patch_size, self.patch_size),
            #                                    in_chans=max(self.in_chans),
            #                                    embed_dim=self.embed_dim,
            #                                    is_inverse=True)
            #upscale features, then add conv block
            self.neck = nn.Sequential(
                nn.Conv2d(self.embed_dim,out_chans,kernel_size=1,bias=False,),
                LayerNorm2d(out_chans),
                nn.Conv2d(out_chans,out_chans,kernel_size=3,padding=1,bias=False),
                LayerNorm2d(out_chans),
                nn.Upsample(scale_factor=self.patch_size, mode='bilinear', align_corners=False),
                nn.Conv2d(out_chans,out_chans,kernel_size=3,padding=1,bias=False),
                act_layer(),
                nn.Conv2d(out_chans,out_chans,kernel_size=3,padding=1,bias=False),
                act_layer(),
                nn.Conv2d(out_chans,out_chans,kernel_size=3,padding=1,bias=False),
            )
        elif self.post_reduction in ["cls_token","mean_token"]:
            self.neck = nn.Sequential(
                nn.Linear(self.embed_dim,out_chans,bias=False,),
                nn.LayerNorm(out_chans),
                nn.Linear(out_chans,out_chans,bias=False)
            )
    def reset_idx(self,bs):
        self.idx_now = [0 for _ in range(bs)]

    def iterate_idx(self,idx,num_iter,max_seq_len):
        assert idx<len(self.idx_now), f"Index {idx} out of range"
        self.idx_now[idx]+=num_iter
        if max_seq_len is None:
            space_enough = True
        else:
            if self.idx_now[idx]>max_seq_len:
                space_enough = False
            else:
                space_enough = True
        return space_enough, slice(self.idx_now[idx]-num_iter,self.idx_now[idx])

    def get_seq_len(self,inputs,bs,assert_expected=True):
        """finds and returns the appropriate max_seq_len for a given input"""
        ae = assert_expected
        seq_len = [0 for _ in range(bs)]
        for k,v in inputs.items():
            for i in range(bs):
                if v is not None:
                    if ae: assert len(v)==bs, f"Expected length of bs={bs}, got length={len(v[i])} for k={k}"
                    if v[i] is None:
                        continue
                    else:
                        assert k in self.input_dict, f"Input name {k} not found in input_dict. Has to be one of: {self.input_dict.keys()}"
                    if torch.is_tensor(v[i]):
                        v_i = torch.atleast_1d(v[i])
                        assert not isinstance(v_i,list), "Expected tensor for input. Found list for k="+k
                        assert len(v_i.shape)>0, "Expected tensor with at least one dimension. Found :" +str(v_i.shape)+" for k="+k
                        if self.input_dict[k]["input_type"]=="image":
                            if ae: assert v_i.shape[1]==v_i.shape[2]==self.input_dict[k]["img_size"], f"Expected image size {self.input_dict[k]['img_size']}, got {v_i.shape[1]}x{v_i.shape[2]} for k={k}, i={i}"
                            seq_len[i] += self.token_img_size**2
                        elif self.input_dict[k]["input_type"]=="scalar_continuous":
                            if ae: assert v_i.shape[0]==1, f"Expected scalar_continuous to have shape (bs,1), got {v_i.shape} for k={k}, i={i}"
                            seq_len[i] += 1
                        elif self.input_dict[k]["input_type"]=="scalar_discrete":
                            if ae: assert v_i.shape[0]==1, f"Expected scalar_discrete to have shape (bs,1), got {v_i.shape} for k={k}, i={i}"
                            seq_len[i] += 1
                        elif self.input_dict[k]["input_type"]=="vocabulary":
                            if ae: assert v_i.shape[0]>=1, f"Expected vocabulary to have shape (bs,n>=1), got {v_i.shape} for k={k}, i={i}"
                            #assert all index are valid
                            if ae: assert (v_i<self.input_dict[k]["size"]).all(), f"Invalid index found for vocabulary. Expected index to be in range 0-{self.input_dict[k]['size']}, found: {v_i} for k={k}"
                            seq_len[i] += len(v_i)
                    else:
                        assert isinstance(v[i],list), "Expected list, tensor or none as inputs, found: "+str(type(v[i]))+" for k="+k
                        assert self.input_dict[k]["input_type"]=="vocabulary", "Only vocabulary input types is supported for list inputs, found: "+self.input_dict[k]["input_type"]+" for k="+k
                        v_i = v[i]
                        assert all([v_ii in getattr(self, k+"_to_vocab_idx").keys() for v_ii in v_i]), f"Invalid index found for vocabulary. Expected index to be in range 0-{self.input_dict[k]['size']} or mappables such as tuples of (dataset_name,dataset_idx) or pretty_name strings, found: {v_i} for k={k}"
                        seq_len[i] += len(v_i)
        return max(seq_len)
    
    def filter_inputs(self,inputs):
        """returns a new dict with only expected keys for the ViT based on self.input_dict"""
        return {k: v for k,v in inputs.items() if k in self.input_dict.keys()}

    def tokenize_inputs(self, inputs, 
                        crop_unused_tokens=True, 
                        raise_error_on_zero_tokens=False):
        #inputs = copy.deepcopy(inputs)
        if "bs_device_for_empty_input" in inputs:
            bs,device = inputs["bs_device_for_empty_input"]
            del inputs["bs_device_for_empty_input"]
        else:
            if torch.is_tensor(inputs):
                inputs = {"image": inputs}
            device,bs = get_device_and_bs_from_input_dict(inputs)
            assert bs is not None, "No inputs found. Manually pass batch_size and device with key 'bs_device_for_empty_input' to avoid errors"
        #always have atleast a cls token to avoid problems with no inputs
        inputs["cls"] = torch.tensor([0 for _ in range(bs)],device=device).unsqueeze(1)
        
        max_seq_len = self.get_seq_len(inputs,bs)
        self.reset_idx(bs)
        tokens = [[] for _ in range(bs)]
        token_info = [[] for _ in range(bs)]
        input_names = list(inputs.keys())
        if self.pre_reduction=="spatial":
            assert self.img_size is not None, "Image size must be specified for spatial reduction. add an image input to the model"
            inputs["cls_image"] = torch.zeros(self.cls_image_size(bs),device=device)
            #make sure "cls_image" is at the end so it contains all the information
            input_names.append("cls_image")

        for input_name in input_names:
            item = inputs[input_name]
            assert input_name in self.input_dict, f"Input name {input_name} not found in input_dict. Has to be one of: {self.input_dict.keys()}"
            t = self.input_dict[input_name]["input_type"]
            if item is None:
                continue
            else:
                assert len(item)==bs, f"expected length of bs={bs}, got length={len(item)} for input_name={input_name}"
            for i in range(bs):
                item_i = item[i]
                if item_i is None:
                    continue
                else:
                    item_i = item_i.unsqueeze(0) if torch.is_tensor(item_i) else item_i
                info_i = None
                tokenized_item = None
                if t=="image":
                    if self.pre_reduction=="spatial" and input_name!="cls_image":
                        #assert len of slice corresponds to channels in the item:
                        slice_channels = self.cls_image_channel_slice[input_name].stop-self.cls_image_channel_slice[input_name].start
                        assert item_i.shape[1]==slice_channels, f"Expected {slice_channels} channels, found {item_i.shape[1]} for input_name={input_name}"
                        inputs["cls_image"][i,self.cls_image_channel_slice[input_name]] = item_i
                        continue
                    if self.share_image_patch_embed:
                        if item_i.shape[1]<max(self.in_chans):
                            item_i = F.pad(item_i,(0,0,0,0,0,max(self.in_chans)-item_i.shape[1]))
                        tokenized_item = self.shared_patch_embed(item_i)
                    else:
                        tokenized_item = getattr(self, input_name+"_embed")(item)
                    info_i = sum([[{"input_name": input_name, 
                                        "input_type": t, 
                                        "h": k, 
                                        "w": j} for j in range(self.token_img_size)] 
                                                for k in range(self.token_img_size)],[])
                    if self.pos_embed is not None:
                        tokenized_item = tokenized_item + self.pos_embed
                    tokenized_item = tokenized_item.reshape(1,-1,self.embed_dim)
                elif t=="scalar_continuous":
                    if item_i>=0:
                        tokenized_item = getattr(self, input_name+"_embed")(item_i.reshape(1,1))
                elif t=="scalar_discrete":
                    if item_i>=0:
                        tokenized_item = getattr(self, input_name+"_embed")(item_i.reshape(1,1))
                elif t=="vocabulary":
                    vocab_indices = torch.tensor(
                        [getattr(self, input_name+"_to_vocab_idx")[item_i[j]] for j in range(len(item_i))],
                        device=device,
                        dtype=torch.long).unsqueeze(0)
                    tokenized_item = getattr(self, input_name+"_embed")(vocab_indices)
                else:
                    raise ValueError(f"Invalid input_name, must be one of {self.input_dict.keys()}")
                if tokenized_item is None:
                    continue
                assert len(tokenized_item.shape)==3, "Expected 3D tensor for tokenized item, found: "+str(tokenized_item.shape)+" for input_name="+input_name
                assert tokenized_item.shape[0]==1, "Expected batch size 1 for tokenized item, found: "+str(tokenized_item.shape[0])
                assert tokenized_item.shape[2]==self.embed_dim, "Expected embed_dim to be "+str(self.embed_dim)+", found: "+str(tokenized_item.shape[2])
                seq_len = tokenized_item.shape[1]
                space_enough,seq_idx = self.iterate_idx(i,seq_len,self.max_seq_len)
                if not space_enough:
                    raise ValueError("Not enough space for tokens, increase max_seq_len")
                else:
                    if info_i is None:
                        info_i = [{"input_name": input_name, "input_type": t, "j": j} for j in range(seq_len)]
                    if space_enough:
                        tokens[i].append(tokenized_item)
                        token_info[i].append(info_i)
        if self.pre_reduction=="spatial":
            pass
        #tokens = [torch.cat(item,dim=1) if len(item)>0 else torch.zeros(1,0,self.embed_dim,device=device) for item in tokens]
        tokens = [torch.cat(item,dim=1) for item in tokens]
        token_info = [sum(item,[]) for item in token_info]
        if self.max_seq_len is None:
            max_actual_len = max([len(item) for item in token_info])
        else:
            if crop_unused_tokens:
                max_actual_len = max([len(item) for item in token_info])
            else:
                max_actual_len = self.max_seq_len
        for i in range(bs):
            if len(token_info[i])<max_actual_len:
                add = max_actual_len-len(token_info[i])
                token_info[i] += [{"input_name": "padding"} for _ in range(add)]
                tokens[i] = F.pad(tokens[i],(0,0,0,add),value=0)
        tokens = torch.cat(tokens,dim=0)
        if raise_error_on_zero_tokens:
            if max_actual_len==0:
                raise ValueError("No tokens found in input")
        return tokens, token_info

    def post_reduction_func(self,x,token_info):
        if self.post_reduction=="cls_token":
            #assert all items in batch have exactly 1 cls token
            cls_idx = [[j for j in range(len(ti)) if ti[j]["input_name"]=="cls"] for ti in token_info]
            assert all([len(idx)==1 for idx in cls_idx]), "Expected 1 cls token per batch, got: "+str(cls_idx)
            cls_idx = torch.tensor([idx[0] for idx in cls_idx])
            x = x[range(len(cls_idx)),cls_idx]
            x = self.neck(x)
        elif self.post_reduction=="mean_token":
            x_mean = []
            for i in range(len(x)):
                nonpad_mask = [j for j in range(len(token_info[i])) if token_info[i][j]["input_name"]!="padding"]
                x_mean.append(x[i,nonpad_mask].mean(dim=0))
            x = torch.stack(x_mean)
            x = self.neck(x)
        elif self.post_reduction in ["spatial","diffusion_sample"]:
            x, _ = group_window_multi(x,
                                    token_info=token_info,
                                    window_size=self.token_img_size,
                                    window_delta=0,
                                    full_image_is_window=True,
                                    reduce_spatially=True)
            # old, only works with 1 image and no non-image tokens.
            #  x = x.reshape(x.shape[0],self.token_img_size,self.token_img_size,self.embed_dim)
            x = self.neck(x.permute(0, 3, 1, 2))
        elif self.post_reduction=="none":
            pass
        else:
            raise ValueError("Invalid post_reduction: "+self.post_reduction)
        return x

    def forward(self, inputs: torch.Tensor, return_token_info=False) -> torch.Tensor:
        x, token_info = self.tokenize_inputs(inputs)
        for blk in self.blocks:
            x = blk(x,token_info)
        x = self.post_reduction_func(x,token_info)
        if return_token_info:
            return x, token_info
        else:
            return x

def vuft(x):
    return very_unique_from_tensor(x)

def very_unique_from_tensor(x):
    """returns a very unique number from a tensor to compare code"""
    if torch.is_tensor(x):
        y = 1/torch.arange(1,x.numel()+1,device=x.device)
        out = (y*(x.flatten())).abs().sum().item()
    elif isinstance(x,nn.Module):
        flat_param_vec = torch.cat([p.flatten() for p in x.parameters()])
        out = very_unique_from_tensor(flat_param_vec)
    return out

class DummyIdentityDict(dict):
    def __init__(self):
        self.super().__init__()
    def __getitem__(self, key):
        return key

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            
            block = Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

def get_device_and_bs_from_input_dict(inputs,assert_consistency=True):
    bs = []
    device = []
    assert_info = {}
    for k,v in inputs.items():
        if v is None:
            continue
        if torch.is_tensor(v):
            bs.append(len(v))
            device.append(v.device)
            assert_info[str(k)+" shape"] = v.shape
        elif isinstance(v,list):
            torch_tensors_in_list = list([item for item in v if torch.is_tensor(item)])
            assert_info[str(k)+" len"] = len(v)
            if len(torch_tensors_in_list)>0:
                bs.append(len(v))
                device.append(torch_tensors_in_list[0].device)
    if assert_consistency:
        if len(set(bs))>1:
            assert_info = "\n".join([str(k)+": "+str(v) for k,v in assert_info.items()])
            raise ValueError("Batch sizes must be consistent across inputs, got: "+str(bs)+" \n assert_info: \n"+assert_info)
        if len(set(device))>1:
            raise ValueError("Devices must be consistent across inputs, got: "+str(device))
    if len(bs)==0:
        device = None
        bs = None
    else:
        device = device[0]
        bs = bs[0]
    return device,bs

class FancyBlock(nn.Module):
    """Transformer blocks with support for various forms of attention:
        - global_attention (basic form for ViT)
        - window_attention (SWIN)
        - grouped_attention (grouped attenation token_info for multiple images)
        - cross (global attention wrt. a small subset of tokens, considered the queries for X-attn, but attending all other tokens (including themselves)
        """
    def __init__(
        self,
        attention_type: str,
        attention_size: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True
    ) -> None:
        super().__init__()
        attn_func_dict = {
            "global": GlobalAttention,
            "image": WindowedAttention,
            "window": WindowedAttention,
            "window_shifted": partial(WindowedAttention, is_shifted=True),
            "grouped": GroupedAttention,
            "cross": CrossAttention,
        }
        assert attention_type in attn_func_dict.keys(), f"Invalid attention type {attention_type}"
        self.attention_type = attention_type
        self.attention_size = attention_size

        self.norm1 = norm_layer(dim)
        
        attn_func = attn_func_dict[self.attention_type]
        
        self.attn = attn_func(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=(attention_size,attention_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        
    def forward(self, x: torch.Tensor, token_info: torch.Tensor = None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn.forward_wrapped(x,token_info)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, f"dim must be divisible by num_heads. got dim={dim} and num_heads={num_heads}"
        self.dim = dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H==W==self.input_size[0], f"Input size mismatch. Expected {self.input_size[0]}x{self.input_size[1]}, got {H}x{W}"
        assert C==self.dim, f"Input dim mismatch. Expected {self.dim}, got {C}"
        C_heads = C // self.num_heads
        
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C_heads).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, C_heads).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj(x)
        return x
    
    def forward_wrapped(self, x: torch.Tensor, token_info = None):
        assert token_info is None, "token_info not supported for basic attention, use a subclass"
        return self.forward(x)
    
def get_all_image_idx(token_info,H=None,W=None):
    token_idx = []
    batch_idx = []
    bs = len(token_info)
    for i in range(bs):
        J = len(token_info[i])
        input_types = [token_info[i][j].get("input_type","padding") for j in range(J)]
        input_names = [token_info[i][j].get("input_name","padding") for j in range(J)]
        unique_image_input_names = list(set([n for n,t in zip(input_names,input_types) if t=="image"]))
        for n_ij in unique_image_input_names:
            idx_x = torch.tensor([j for j in range(J) if input_names[j]==n_ij])
            h_of_image = [token_info[i][j]["h"] for j in idx_x]
            w_of_image = [token_info[i][j]["w"] for j in idx_x]
            H_hat,W_hat = max(h_of_image)+1, max(w_of_image)+1
            if H is None:
                H = H_hat
            else:
                assert H==H_hat
            if W is None:
                W = W_hat
            else:
                assert W==W_hat
            assert len(idx_x)==H*W
            order_criterion = torch.tensor([h+W*w for h,w in zip(h_of_image,w_of_image)])
            idx_x = idx_x[torch.argsort(order_criterion)]
            token_idx.append(idx_x)
            batch_idx.append(i)
    return batch_idx, token_idx

def get_submatrix(x,idx):
    return x[idx,:][:,idx]

def set_submatrix(x,values,idx1,idx2=None):
    if isinstance(idx1,list):
        idx1 = torch.tensor(idx1)
    elif isinstance(idx1,slice):
        idx1 = torch.arange(idx1.start,idx1.stop,idx1.step)
    if idx2 is None:
        idx2 = idx1
    idx_d1,idx_d2 = torch.meshgrid(idx1,idx2,indexing="ij")
    if isinstance(values,(int,float)):
        values = torch.ones(idx_d1.shape)*values
    assert len(values.shape)==2
    if values.shape[0]==1 and idx_d1.shape[0]>1:
        values = values.repeat(idx_d1.shape[0],1)
    if values.shape[1]==1 and idx_d1.shape[1]>1:
        values = values.repeat(1,idx_d1.shape[1])
    assert values.shape[0]==idx_d1.shape[0] and values.shape[1]==idx_d1.shape[1], f"tried to index with shape={idx_d1.shape}, but values had shape={values.shape}"
    print(x.shape,values.shape)
    x[idx_d1.flatten(),idx_d2.flatten()] = values.type(x.dtype).flatten()
    #Very slow and bad D:


class GlobalAttention(Attention):
    def __init__(self,*args,**kwargs):
        if kwargs.get("use_rel_pos",False):
            warnings.warn("GlobalAttention does not support relative positional embeddings. Ignoring use_rel_pos=True")
            kwargs.pop("use_rel_pos")
        super().__init__(*args,**kwargs)

    def add_image_decomposed_rel_pos(self, attn, q, token_info):
        """Extracts images based off of token_info and 
        adds rel_pos embeddings appropriately to the attn matrix"""
        H, W = self.input_size
        nh = self.num_heads
        batch_idx, token_idx  = get_all_image_idx(token_info,H=H,W=W)
        attn_modified = []
        for bs_i,idx in zip(batch_idx, token_idx):
            attn_modified.extend([get_submatrix(attn[i],idx) for i in range(bs_i*nh,(bs_i+1)*nh)])
        attn_modified = torch.stack(attn_modified,dim=0)
        attn_modified = add_decomposed_rel_pos(attn_modified, q, 
                                               self.rel_pos_h, 
                                               self.rel_pos_w, 
                                               (H, W), 
                                               (H, W))
        for bs_i,idx in zip(batch_idx, token_idx):
            for i in range(bs_i*nh,(bs_i+1)*nh):
                set_submatrix(attn[i],values=attn_modified[i],idx1=idx)
        return attn

    def forward_wrapped(self, x, token_info = None):
        assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor"
        B, L, C = x.shape
        assert C==self.dim, f"Input dim mismatch. Expected {self.dim}, got {C}"
        C_heads = C // self.num_heads
        
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C_heads).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, L, C_heads).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:#not used TODO
            attn = self.add_image_decomposed_rel_pos(attn, q, token_info)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, L, C_heads).permute(0, 2, 1, 3).reshape(B, L, C)
        x = self.proj(x)
        return x
    
def group_tokens_for_2d_window(x, window_size, window_delta):
    assert len(x.shape)==4, "Expected tokens to be stored in a (B,H,W,C) tensor, found size: "+str(x.shape)
    s,d = window_size, window_delta
    B, H, W, C = x.shape
    pad_h0,pad_w0 = (s-d)%s,(s-d)%s
    pad_h1,pad_w1 = s-((H+pad_h0)%s), s-((W+pad_w0)%s)
    paddings = [0,0,pad_w0,pad_w1,pad_h0,pad_h1] #F.pad uses reverse dim order, so (C0, C1, W0, W1, H0, H1)
    paddings = [p%s for p in paddings]
    x_padded = torch.nn.functional.pad(x,paddings)
    _, H2, W2, _ = x_padded.shape
    group_inv = {"unpad_slice_H": slice(paddings[4],H2-paddings[5]),
                "unpad_slice_W": slice(paddings[2],W2-paddings[3]),
                "x_padded.shape": x_padded.shape}
    group_slice = []
    groups = []
    nH,nW = x_padded.shape[1]//s, x_padded.shape[2]//s
    for bs_i in range(B):
        for i in range(nH):
            for j in range(nW):
                slice_H = slice(i*s,(i+1)*s)
                slice_W = slice(j*s,(j+1)*s)
                groups.append(x_padded[bs_i,slice_H,slice_W,:])
                group_slice.append({"slice_H":slice_H,
                                "slice_W":slice_W,
                                "bs_i":bs_i})
    group_inv["group_slice"] = group_slice
    groups = torch.stack(groups,dim=0)
    return groups, group_inv

def ungroup_tokens_for_2d_window(groups,group_inv):
    x_padded = torch.zeros(group_inv["x_padded.shape"],dtype=groups.dtype,device=groups.device)
    for i,group in enumerate(groups):
        slice_H = group_inv["group_slice"][i]["slice_H"]
        slice_W = group_inv["group_slice"][i]["slice_W"]
        bs_i = group_inv["group_slice"][i]["bs_i"]
        x_padded[bs_i,slice_H,slice_W,:] = group
    x = x_padded[:,group_inv["unpad_slice_H"],group_inv["unpad_slice_W"]]
    return x

def group_window_multi(x,token_info, window_size, window_delta, 
                       full_image_is_window=False,
                       reduce_spatially=False,
                       reduce_spatially_add_zeros_on_missing=True,
                       ):
    """groups tokens for window attention with multiple images, based on information in token_info"""
    assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor when using the token_info argument, found size: "+str(x.shape)
    assert len(token_info)==x.shape[0], "Expected token_info to have length equal to batch size, got: "+str(len(token_info))
    B, L, C = x.shape
    #raise NotImplementedError
    if full_image_is_window:
        H = W = window_size
        group_inv_for_image_attention = {"unpad_slice_H": slice(None),
                                    "unpad_slice_W": slice(None),
                                    "x_padded.shape": (B,H,W,C),
                                    "group_slice": [{"slice_H": slice(None),
                                                    "slice_W": slice(None),
                                                    "bs_i": i} for i in range(B)]}
        assert window_delta==0, "Expected window_delta to be 0 for when the full image is the window, got: "+str(window_delta)

    group_inv_multi = []
    window_grouped_tokens = []
    for i in range(B):
        input_types = [token_info[i][j].get("input_type","padding") for j in range(len(token_info[i]))]
        input_names = [token_info[i][j].get("input_name","padding") for j in range(len(token_info[i]))]
        unique_image_input_names = list(set([n for n,t in zip(input_names,input_types) if t=="image"]))
        for n_ij in unique_image_input_names:
            y,idx_x = extract_image_from_tokens(x_i=x[i],token_info_i=token_info[i],input_name=n_ij)
            if full_image_is_window:
                assert y.shape[1]==H and y.shape[2]==W, f"Expected image to have shape ({H},{W}), got: {y.shape}"
                z = y
                group_inv = group_inv_for_image_attention
            else:
                z,group_inv = group_tokens_for_2d_window(y, window_size, window_delta)
            lwgt = sum([len(w) for w in window_grouped_tokens])
            idx_z = slice(lwgt,lwgt+len(z))
            group_inv_multi.append({"i": i,
                                    "n": n_ij,
                                    "idx_x": idx_x,
                                    "idx_z": idx_z,
                                    "group_inv": group_inv})
            window_grouped_tokens.append(z)
    if reduce_spatially:
        rsazom = reduce_spatially_add_zeros_on_missing
        batch_i = [g["i"] for g in group_inv_multi]
        num_images_per_batch_i = [sum([bi==i for bi in batch_i]) for i in range(B)]
        window_grouped_tokens_new = []
        for i,num_images in enumerate(num_images_per_batch_i):
            images = []
            if num_images==0:
                if rsazom:
                    images.append(torch.zeros((1,H,W,C),dtype=window_grouped_tokens[0].dtype,device=window_grouped_tokens[0].device))
                else:
                    continue
            else:
                for _ in range(num_images):
                    images.append(window_grouped_tokens.pop(0))
            window_grouped_tokens_new.append(torch.cat(images,dim=0).mean(dim=0,keepdim=True))
        window_grouped_tokens = window_grouped_tokens_new
    window_grouped_tokens = torch.cat(window_grouped_tokens,dim=0) if len(window_grouped_tokens)>0 else None
    return window_grouped_tokens, group_inv_multi

def ungroup_window_multi(x,z,group_inv_multi):
    for g in group_inv_multi:
        i,idx_x,idx_z,group_inv = g["i"],g["idx_x"],g["idx_z"],g["group_inv"]
        y = ungroup_tokens_for_2d_window(z[idx_z],group_inv)
        x[i,idx_x] = y.reshape(-1,y.shape[-1])
    return x

class WindowedAttention(Attention):
    def __init__(self, *args, is_shifted=False, is_cyclical=False, full_image_is_window=False, **kwargs): #TODO: add support for cyclical
        super().__init__(*args, **kwargs)
        self.full_image_is_window = full_image_is_window
        self.window_size = self.input_size[0]
        assert self.window_size>0, "Window size must be greater than 0"
        if is_shifted:
            self.window_delta = self.window_size//2
        else:
            self.window_delta = 0
    
    def forward_wrapped_w_window_partition(self, x, token_info = None):
        assert len(x.shape)==4, "Expected tokens to be stored in a (B,H,W,C) tensor, found size: "+str(x.shape)
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)
        x = self.forward(x)
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        return x
    
    def forward_wrapped_w_old_window(self, x, token_info = None):
        B,HW,C = x.shape
        H = W = int(round(HW**0.5))
        h = w = self.window_size
        x, group_inv = self.group_tokens_for_2d_window(x.reshape((B,H,W,C)))
        b = x.shape[0]
        x = self.forward(x).reshape((b,h,w,C))
        x = self.ungroup_tokens_for_2d_window(x, group_inv).reshape((B,HW,C))
        return x
    
    def forward_wrapped(self, x, token_info = None):
        assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor when using the token_info argument, found size: "+str(x.shape)
        y,group_inv_multi = group_window_multi(x,token_info,self.window_size,self.window_delta,full_image_is_window=self.full_image_is_window)
        if y is None: 
            #y is None when there are no images in x
            #fix for when we have no images in x
            return x 
        else:
            y = self.forward(y)
            x = ungroup_window_multi(x,y,group_inv_multi)
            return x

def extract_image_from_tokens(x_i,token_info_i,input_name,pad_if_missing=False):
    idx_x = [i for i in range(len(token_info_i)) if token_info_i[i]["input_name"]==input_name]
    h_of_image = [token_info_i[k]["h"] for k in idx_x]
    w_of_image = [token_info_i[k]["w"] for k in idx_x]
    H,W = max(h_of_image)+1, max(w_of_image)+1
    if not len(idx_x)==H*W:
        if pad_if_missing:
            raise NotImplementedError
        else:
            nonpresent = [f"h={h}, w={w}" for (h,w) in zip(range(H),range(W)) if (h,w) not in zip(h_of_image,w_of_image)]
            assert len(idx_x)==H*W, "Expected all tokens to be present in the image. did not find: "+str(nonpresent[:min(5,len(nonpresent))])
    order_criterion = torch.tensor([h+W*w for h,w in zip(h_of_image,w_of_image)])
    idx_x = [idx_x[i] for i in torch.argsort(order_criterion)]
    idx_x = convert_to_slice_if_possible(idx_x)
    C = x_i.shape[-1]
    y = x_i[idx_x].reshape(1,H,W,C)
    return y, idx_x

def convert_to_slice_if_possible(idx):
    if len(idx)==0 or len(idx)==1:
        return idx
    steps = [idx[i+1]-idx[i] for i in range(len(idx)-1)]
    if len(set(steps))==1:
        step = steps[0]
        return slice(idx[0],idx[-1]+step,step)
    else:
        return idx

class CrossAttention(Attention):
    """Multi-head Attention block which computes the full attention matrix for a specified subset of tokens.
    essentially cross attention but where the cross tokens originate from the decoder."""
    def __init__(self, *args, nametype_to_query = lambda n,t: t!="image", **kwargs):
        super().__init__(*args, **kwargs)
        self.nametype_to_query = nametype_to_query
        self.nh = self.num_heads
        self.largest_neg_value = -torch.finfo(torch.float32).max
        if self.use_rel_pos:
            warnings.warn("Relative positional embeddings are not supported for cross attention")

    def get_attn_idx(self,token_info):
        bs = len(token_info)
        attn_idx = [[] for _ in range(bs)]
        for i in range(bs):
            input_types = [token_info[i][j].get("input_type","padding") for j in range(len(token_info[i]))]
            input_names = [token_info[i][j].get("input_name","padding") for j in range(len(token_info[i]))]
            for j in range(len(token_info[i])):
                if self.nametype_to_query(input_names[j],input_types[j]):
                    attn_idx[i].append(j)
        lengths = [len(item) for item in attn_idx]
        max_idx_len = max(lengths)
        bs = len(attn_idx)
        #padding is for multi-headed
        padding = torch.zeros((bs*self.nh,max_idx_len),dtype=torch.bool)
        for i in range(bs):
            bs_slice = slice(i*self.nh,(i+1)*self.nh)
            padding[bs_slice,:lengths[i]] = 1
        #mask is after joining heads
        y_mask = torch.zeros((bs,max_idx_len),dtype=torch.bool)
        x_mask = torch.zeros((bs,len(token_info[0])),dtype=torch.bool)
        for i in range(bs):
            y_mask[i,:lengths[i]] = 1
            x_mask[i,attn_idx[i]] = 1

        return attn_idx, padding, max_idx_len, x_mask, y_mask

    
    def forward_wrapped(self, x, token_info = None):
        assert token_info is not None, "token_info must be provided for cross attention"
        assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor"
        B, L, C = x.shape
        assert C==self.dim, f"Input dim mismatch. Expected {self.dim}, got {C}"
        C_heads = C // self.num_heads
        attn_idx, padding, Lx, x_mask, y_mask = self.get_attn_idx(token_info)
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, C_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.clone().reshape(3, B * self.num_heads, L, C_heads).unbind(0)
        #sub_q = torch.zeros((B*self.num_heads,Lx,C_heads),dtype=q.dtype,device=q.device)
        sub_q = []
        for idx in range(len(attn_idx)):
            bs_slice = slice(idx*self.nh,(idx+1)*self.nh)
            #sub_q[bs_slice,:len(attn_idx[idx]),:] = q[bs_slice,attn_idx[idx],:]
            sub_q.append(torch.nn.functional.pad(q[bs_slice,attn_idx[idx],:],(0,0,0,Lx-len(attn_idx[idx]))))
        sub_q = torch.cat(sub_q,dim=0)
        attn = (sub_q * self.scale) @ k.transpose(-2, -1)
        attn[padding] = self.largest_neg_value
        attn = attn.softmax(dim=-1)
        y = (attn @ v).view(B, self.num_heads, Lx, C_heads).permute(0, 2, 1, 3).reshape(B, Lx, C)
        #change only the subset of tokens that were in the query group
        new_x = x.clone()
        new_x[x_mask] = y[y_mask]
        return new_x

def token_info_to_group_default(d,W):
    if ("h" in d) and ("w" in d):
        return d["h"]*W+d["w"]
    else:
        return None

class GroupedAttention(GlobalAttention):
    def __init__(self, *args, 
                 token_info_to_group = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        """
        Wrapper for Attention which does fully connected attention but only
        within groups, and no inter-group attention. The variable 
        token_info_to_group should return the same item for all elements in
        a group, and None for items which are ignored for attention
        """
        
        assert isinstance(self.input_size,tuple), "input_size must be a tuple, found: "+str(self.input_size)
        assert isinstance(self.input_size[0],int), "input_size must be a tuple of integers, found: "+str(self.input_size)
        self.W = self.input_size[0]

        if token_info_to_group is None:
            token_info_to_group = lambda x: token_info_to_group_default(x,self.W)
        self.token_info_to_group = token_info_to_group
        padding_dict = {"input_name": "padding", "input_type": "padding"}
        assert token_info_to_group(padding_dict) is None, f"token_info_to_group should return None for padding tokens, found: {self.token_info_to_group(padding_dict)}"
        #redefine the self.forward_wrapped from globalAttention
        self.forward_wrapped_ga = super().forward_wrapped

    def forward_wrapped(self, x, token_info = None):
        assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor, found size: "+str(x.shape)
        assert token_info is not None, "token_info must be provided for grouped attention"
        B, L, C = x.shape
        y, group_inv = group_tokens(x, token_info, self.token_info_to_group)
        if y is None: return x #fix for when we have no groups in x
        y = self.forward_wrapped_ga(y)
        x = ungroup_tokens(x, y, group_inv)
        return x

def group_tokens(x: torch.Tensor, 
                 token_info: torch.Tensor,
                 group_f) -> torch.Tensor:
    B = x.shape[0]
    groups = []
    group_inv = []
    for i in range(B):
        group_names = [group_f(d) for d in token_info[i]]
        uq_groups = list(set(group_names))
        j = 0
        for uq_name in uq_groups:
            if uq_name is not None:
                group_idx = [j for j in range(len(token_info[i])) if group_names[j]==uq_name]
                groups.append(x[i,group_idx])
                group_inv.append({"i": i,
                                  "uq_name": uq_name,
                                  "group_idx": group_idx,
                                  "j": j})
                j += 1
    
    if len(groups)==0:
        return None, group_inv
    len_per_group = [len(g) for g in groups]
    max_len = max(len_per_group)

    groups = [F.pad(g,(0,0,0,max_len-len(g))) for g in groups]
    groups = torch.stack(groups)
    return groups, group_inv

def ungroup_tokens(x: torch.Tensor, 
                   y: torch.Tensor,
                   group_inv: list) -> torch.Tensor:
    for g in group_inv:
        i,group_idx,j = g["i"],g["group_idx"],g["j"]
        x[i,group_idx] = y[j,:len(group_idx)]
    return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def token_info_overview(token_info,only_len=True,as_df=False):
    bs = len(token_info)
    out = [{} for _ in range(bs)]
    for i in range(bs):
        for j in range(len(token_info[i])):
            input_name = token_info[i][j]["input_name"]
            if not input_name in out[i]:
                out[i][input_name] = []
            out[i][input_name].append(j)
    for i in range(bs):
        for k in out[i].keys():
            if only_len:
                out[i][k] = len(out[i][k])
            else:
                out[i][k] = convert_to_slice_if_possible(out[i][k])
    if as_df:
        out = pd.DataFrame(out).fillna(0).astype(int)
    return out

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): grouped sequence size of query q with (q_h, q_w).
        k_size (Tuple): grouped sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    if False:# TODO q_size[0]!=14:
        print(attn.shape,q.shape,rel_pos_h.shape,rel_pos_w.shape,q_size,k_size)
        assert 0
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    if False:#q_size[0]!=14: TODO
        print("shape1:",(B, q_h, q_w, k_h, k_w))
        print("shape2:",rel_h.shape)
        print("shape3:",rel_w.shape)
        print("shape4:",attn.shape)
        assert 0
    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        is_inverse: bool = False,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.is_inverse = is_inverse
        self.in_chans = in_chans
        if self.is_inverse:
            self.proj = nn.ConvTranspose2d(
                embed_dim, in_chans, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_chans, f"Expected tensor of shape (B,{self.in_chans},H,W). Found: {x.shape}"
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

def num_tokens_from_token_info(token_info,count_padding=False,sum_batch=True):
    toi = token_info_overview(token_info)
    num_tokens = []
    for i in range(len(token_info)):
        nt = 0
        for k,v in toi[i].items():
            if k=="padding" and count_padding:
                nt += v
            else:
                nt += v
        num_tokens.append(nt)
    if sum_batch:
        num_tokens = sum(num_tokens)
    return num_tokens

def get_clip_matrix(datasets,save_path = f"./data/CLIP_emb.pth"):
    loaded = torch.load(save_path)
    dataset_idx_to_clip_idx = {}
    pretty_name_to_clip_idx = {}
    clip_matrix = []
    k = 0
    for dataset_name in datasets:
        assert dataset_name in loaded.keys(), f"Dataset {dataset_name} not found in loaded keys."
        clip_matrix.append(torch.from_numpy(loaded[dataset_name]["embeddings"]))
        for i in range(len(loaded[dataset_name]["class_idx"])):
            idx = loaded[dataset_name]["class_idx"][i]
            pretty_name = loaded[dataset_name]["class_names_pretty"][i]
            dataset_idx_to_clip_idx[(dataset_name,idx)] = k
            pretty_name_to_clip_idx[pretty_name] = k
            k += 1
    clip_matrix = torch.cat(clip_matrix,dim=0)
    return clip_matrix, dataset_idx_to_clip_idx, pretty_name_to_clip_idx

class WrapToVocabDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if torch.is_tensor(key):
            #map to int
            key = key.item() 
        v = super().__getitem__(key)
        if isinstance(v,int):
            v = torch.tensor(v)
        return v

def main():
    import argparse
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
    from source.utils.mixed_utils import set_random_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 1: Testing default ver")
        set_random_seed(123)
        model = sam_vit_from_idx(-1)
        model.to('cuda')
        model.eval()
        print("sum_of_params: ",sum([p.abs().sum().item() for p in model.parameters()]))
        #set seed
        img = torch.randn(2, 3, 1024, 1024).to('cuda')
        pred = model(img)
        print(f"input shape: {img.shape}, output shape: {pred.shape}")
        print(f"input absum val: {img.abs().sum()}")
        print(f"output absum val: {pred.abs().sum()}")
        print(f"vuft: {vuft(pred)}")
        #expected result:
        #output absum val: 1675137.25
    elif args.unit_test==1:
        print("UNIT TEST 1: fancy")
        set_random_seed(123)
        model = FancyViT(**fancy_vit_from_idx(-1))
        model.to('cuda')
        model.eval()
        print("sum_of_params: ",sum([p.abs().sum().item() for p in model.parameters()]))
        img = torch.randn(2, 3, 1024, 1024).to('cuda')
        pred = model(img)
        print(f"input absum val: {img.abs().sum()}")
        print(f"output absum val: {pred.abs().sum()}")
        print(f"output vuft={vuft(pred)}")
    elif args.unit_test==2:
        print("UNIT TEST 2: fancy with lots of args")
        img_size = 128
        inputs = default_input_dict(img_size=img_size,patch_size=16)
        set_random_seed(123)
        dim = 128
        num_heads = 8
        block_types = ["global","window","window_shifted","image","cross","grouped"]*2
        #block_types = ["cross"]*1
        depth = len(block_types)
        model = FancyViT(embed_dim=dim,block_types=block_types,depth=depth,input_dict=inputs,num_heads=num_heads)
        import jlc
        jlc.num_of_params(model)
        bs = 5
        class_names = torch.randint(0,8096,(bs,5))
        padding_mask = 0.4>torch.rand(bs,5)
        class_names[padding_mask] = -1
        probs_per_arg = {"image": 0.5,
                      "same_vol": 0.5,
                    "same_classes": 0.5,
                    "same_dataset": 0.5,
                    "adjacent": 0.5,
                    "time": 0.5,
                    "num_classes": 0.5,
                    "class_names": 0.5}
        input_dict = {"image": torch.randn(bs,3,img_size,img_size),
                      "same_vol": torch.randn(bs,3+6,img_size,img_size),
                    "same_classes": torch.randn(bs,3+6,img_size,img_size),
                    "same_dataset": torch.randn(bs,3+6,img_size,img_size),
                    "adjacent": torch.randn(bs,3+6,img_size,img_size),
                    "time": torch.rand(bs,1),
                    "num_classes": torch.randint(0,64,(bs,1)),
                    "class_names": class_names}
        input_dict = {k: [v[i] if torch.rand(1)>probs_per_arg[k] else None for i in range(bs)] for k,v in input_dict.items()}
        pred = model(input_dict)
        print("input sizes: ",{k: v.shape if torch.is_tensor(v) else len(v) for k,v in input_dict.items() if v is not None})
        print(f"output shape: {pred.shape}")
        if True:
            token_info = model.tokenize_inputs(input_dict)[1]
            print("token_info_overview:")
            print(token_info_overview(token_info))
            print(num_tokens_from_token_info(token_info))
            print(num_tokens_from_token_info(token_info,1,0))
            
    elif args.unit_test==3:
        with torch.no_grad():
            for _ in range(10):
                print("UNIT TEST 3: fancy_vit_from_args")
                device = "cuda"
                seed = set_random_seed(None)
                print("seed: ",seed)
                args = {"cond_vit_setup": "1b2abcde3a4b", #err: ["1b2abcde3c4b"]
                        "max_num_classes": 64,
                        "cond_img_size": 128,
                        "cond_patch_size": 16,
                        "cond_sam_idx": -1}
                fancy_vit_args = fancy_vit_from_args(args)
                model = FancyViT(**fancy_vit_args)
                model = model.to(device)
                import jlc
                jlc.num_of_params(model)
                img_size = model.img_size
                bs = 16
                class_name_length = 7
                class_names = torch.randint(0,8096,(bs,class_name_length))
                padding_mask = 0.4>torch.rand(bs,class_name_length)
                class_names[padding_mask] = -1
                probs_per_arg = {"image": 0.5,
                            "same_vol": 0.5,
                            "same_classes": 0.5,
                            "same_dataset": 0.5,
                            "adjacent": 0.5,
                            "time": 0.5,
                            "num_classes": 0.5,
                            "class_names": 0.5}
                probs_per_arg = {k: 0.5 for k in probs_per_arg.keys()}
                input_dict = {"image": torch.randn(bs,3,img_size,img_size),
                            "same_vol": torch.randn(bs,3+6,img_size,img_size),
                            "same_classes": torch.randn(bs,3+6,img_size,img_size),
                            "same_dataset": torch.randn(bs,3+6,img_size,img_size),
                            "adjacent": torch.randn(bs,3+6,img_size,img_size),
                            "time": torch.rand(bs,1),
                            "num_classes": torch.randint(0,64,(bs,1)),
                            "class_names": class_names}
                input_dict = {k: v.to(device) if torch.is_tensor(v) else v for k,v in input_dict.items()}
                input_dict = {k: [v[i] if torch.rand(1)<probs_per_arg[k] else None for i in range(bs)] for k,v in input_dict.items()}
                input_dict["bs_device_for_empty_input"] = bs,device
                pred = model(input_dict)
                print("input sizes: ",{k: v.shape if torch.is_tensor(v) else len(v) for k,v in input_dict.items() if v is not None})
                print(f"output shape: {pred.shape}")
                if True:
                    token_info = model.tokenize_inputs(input_dict)[1]
                    print("token_info_overview:")
                    print(token_info_overview(token_info,as_df=1))
    elif args.unit_test==4:
        print("UNIT TEST 4: debug transposed image problem")
        from pprint import pprint
        set_random_seed(123)
        args = TieredParser().get_args(alt_parse_args=["--model_name", "vittest[v2]","--cond_vit_setup", "1a2b3c4b"])
        mik = ModelInputKwargs(args)
        t = mik.construct_kwarg_table(True)
        model = FancyViT(**fancy_vit_from_args(args))
        print(t)
        pprint(is_valid_cond_vit_setup(args.cond_vit_setup,long_names_instead=1))
    elif args.unit_test==5:
        print("UNIT TEST 5: model input kwargs")
        set_random_seed(123)
        args = TieredParser().get_args(alt_parse_args=["--model_name", "vit128[T0]"])
        mik = ModelInputKwargs(args)
        t = mik.construct_kwarg_table(True)
        print(t)
        mik.assert_inputs_are_valid()
    elif args.unit_test==6:
        print("UNIT TEST 6: model input kwargs, probs")
        set_random_seed(123)
        args = TieredParser().get_args(alt_parse_args=["--model_name", "vit128[T0]+allcond"])
        mik = ModelInputKwargs(args)
        t = mik.construct_kwarg_table(True)
        print(t)
        #mik.assert_inputs_are_valid()
        probs = mik.get_input_probs()
        print(probs)
    elif args.unit_test==7:
        import jlc
        print("UNIT TEST 7: forward pass with NAN pixel to test information flow")
        set_random_seed(123)
        args = TieredParser().get_args(alt_parse_args=["--model_name", "vit128[T0]+allcond"])
        args.cond_vit_setup = "1a2e3c4b"
        args.vit_spatialness = "0,1,2,3"
        vit_args = fancy_vit_from_args(args)
        print(vit_args)
        vit = FancyViT(**vit_args)
        inputs = {"image": torch.randn(1,3,128,128)}
        outputs = vit(inputs)
        jlc.shaprint(outputs)
    else:
        raise ValueError("Invalid unit test")
    
if __name__=="__main__":
    main()
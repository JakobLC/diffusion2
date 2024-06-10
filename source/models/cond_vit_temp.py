
def default_input_dict(img_size=1024,patch_size=16,image_channels=3,diff_channels=6):
    #input types: ["image","scalar_continuous","scalar_discrete","vocabulary"]
    im_dict = {"input_type": "image",
               "img_size": img_size, 
               "patch_size": patch_size, 
               "in_chans": image_channels+diff_channels}
    inputs = {"sample":       {**im_dict, "in_chans": diff_channels},#X
            "image":          {**im_dict, "in_chans": 3},#X
            "image_features": {**im_dict, "in_chans": diff_channels},#X
            "points":         {**im_dict, "in_chans": diff_channels},#X
            "bbox":           {**im_dict, "in_chans": diff_channels},#X
            "self_cond":      {**im_dict, "in_chans": diff_channels},#(X)
            "same_vol":       im_dict,#X
            "same_classes":   im_dict,#X
            "same_dataset":   im_dict,#X
            "adjacent":       im_dict,#X
            "time":           {"input_type": "scalar_continuous", "min": 0.0, "max": 1.0},#X
            "num_classes":    {"input_type": "scalar_discrete", "size": 64},#(X)
            "class_names":    {"input_type": "vocabulary", "size": -1},#X
            "semantic":       {"input_type": "scalar_discrete", "size": 2},#X
    }
    return inputs
new_prob_keys = "semantic_kwarg_prob,adjacent_prob,same_vol_prob,same_classes_prob,same_dataset_prob,class_names_prob".split(",")
cond_image_keys = ["same_classes","same_dataset","same_vol","adjacent"]
spatial_input_keys = ["image","bbox","points","self_cond"]


def fancy_vit_from_args(args,return_input_dict_instead=False):
    keys = ["cond_vit_setup",
            "max_num_classes",
            "cond_img_size",
            "cond_patch_size",
            "cond_sam_idx",
            "vit_unet_cond_mode",
            "weak_bbox_prob",
            "weak_points_prob",
            "weak_signals",
            "class_names_prob",
            "datasets"]
    keys += new_prob_keys

    keys = []

    if not isinstance(args,dict):
        args = copy.deepcopy(args.__dict__)
    
    assert all([k in args.keys() for k in keys]), "Missing keys in args: "+str([k for k in keys if k not in args.keys()])
    if args["vit_unet_cond_mode"]=="no_vit":
        assert not any([args[k]>0 for k in new_prob_keys]), "No ViT, so new probabilities should be 0. found: "+str({k: args[k] for k in new_prob_keys if args[k]>0})
        return {}
    index_to_opt = is_valid_cond_vit_setup(args["cond_vit_setup"])



    img_size = args["cond_img_size"] if args["cond_img_size"]>0 else args["image_size"]
    
    diff_channels = int(torch.log2(torch.tensor(args["max_num_classes"])).round().item())
    del_keys = "max_seq_len,input_dict,img_size,patch_size,global_attn_indexes,block_types"
    fancy_vit_args = fancy_vit_from_idx(args["cond_sam_idx"],del_keys=del_keys.split(","))
    fancy_vit_args["input_dict"] = default_input_dict(img_size=img_size, 
                                                        patch_size=args["cond_patch_size"],
                                                        image_channels=3,
                                                        diff_channels=diff_channels)
    
    fancy_vit_args["diff_channels"] = diff_channels
    del_keys2 = []
    for k in new_prob_keys:
        if args[k]<=0 or (k=="semantic_prob" and args[k]==1):
            del_keys2.append(k.replace("_prob",""))
    
    #remove keys from the input dict based on args
    if args["weak_points_prob"]<=0 or not args["weak_signals"]:
        del_keys2.append("points")

    if args["weak_bbox_prob"]<=0 or not args["weak_signals"]:
        del_keys2.append("bbox")

    if args["class_names_prob"]<=0:
        del_keys2.append("class_names")
    else:
        fancy_vit_args["input_dict"]["class_names"]["class_names_datasets"] = get_named_datasets(args["datasets"])
        
    if args["vit_unet_cond_mode"]=="both_spatial_unet":
        del_keys2.extend(spatial_input_keys)
    elif not args["vit_unet_cond_mode"]=="no_unet":
        del_keys2.append("sample")
    else:
        assert args["vit_unet_cond_mode"] in ["both_spatial_vit","no_unet"], "Invalid vit_unet_cond_mode: "+args["vit_unet_cond_mode"]

    for k in del_keys2:
        if k in fancy_vit_args["input_dict"].keys():
            del fancy_vit_args["input_dict"][k]

    fancy_vit_args["injection_type"] = index_to_opt[4]
    fancy_vit_args["pre_reduction"] = {"a": "spatial", "b": "none"}[index_to_opt[1]]
    fancy_vit_args["post_reduction"] = {"a": "cls_token", "b": "mean_token", "c": "spatial", "d": "none"}[index_to_opt[3]]
    if args["vit_unet_cond_mode"]=="no_unet":
        assert index_to_opt[3]=="c", "Only option 3c is allowed for no_unet, found: 3"+index_to_opt[3]
        fancy_vit_args["post_reduction"] = "diffusion_sample"
    block_types = get_appropriate_block_types(depth=fancy_vit_args["depth"],block_types=index_to_opt[2])
    fancy_vit_args["block_types"] = block_types
    if return_input_dict_instead:
        return fancy_vit_args["input_dict"]
    else:
        return fancy_vit_args



def unet_vit_input_dicts_from_args(args):
    diff_channels = int(torch.log2(torch.tensor(args["max_num_classes"])).round().item())
    out_channels = diff_channels
    no_diffusion = False
    image_channels = 3
    self_cond = args["p_self_cond"]>0
    
    full_unet_input_dict = {"sample": out_channels if not no_diffusion else 0,
                        "image": image_channels,
                        "bbox": out_channels*int(args["p_bbox"]>0),
                        "points": out_channels*int(args["p_points"]>0),
                        "self_cond": out_channels if self_cond else 0
                        }
    task_name = args["vit_unet_cond_mode"]
    if task_name=="no_vit": #T0: no ViT
        unet_input_keys = list(full_unet_input_dict.keys())
    elif task_name=="both_spatial_unet": #T1: The unet gets the image
        unet_input_keys = ["sample","image","bbox","points","self_cond"]
    elif task_name=="both_spatial_vit": #T2: The unet does not get the image and other spatial-directly conditionable inputs (bbox,points)
        unet_input_keys = ["sample"]
    elif task_name=="no_unet": #T3: full ViT, no unet
        unet_input_keys = []
    else:
        raise ValueError("Invalid task_name: "+task_name)
    unet_input_dict = {k: full_unet_input_dict[k] for k in unet_input_keys}

    vit_input_dict = fancy_vit_from_args(args,return_input_dict_instead=True)
    
    return unet_input_dict,vit_input_dict


def pd_table_of_inputs(args):
    if isinstance(args,Namespace):
        args = copy.deepcopy(args.__dict__)
    elif isinstance(args,str):
        args = TieredParser().get_args(alt_parse_args=["--model_name","hq"]).__dict__
    else:
        assert isinstance(args,dict), "Expected Namespace or dict, str (as model name) or Namespace found: "+str(type(args))
    unet_input_dict,vit_input_dict = unet_vit_input_dicts_from_args(args)
    if args["vit_unet_cond_mode"]!="no_unet":
        unet_input_dict["time"] = 1
        if args["image_encoder"]!="none":
            unet_input_dict["image_features"] = 1
        if args["classes_prob"]>0 and args["class_type"]!="none":
            unet_input_dict["num_classes"] = 1
    df = pd.DataFrame(columns=["name","input_type","is_spatial","where"])
    #types: "image-non-spatial", "image-spatial", ""
    uq_keys = list(set(list(unet_input_dict.keys())+list(vit_input_dict.keys())))
    spatial_image_keys = ["image","bbox","points","self_cond","sample","image_features"]
    d = default_input_dict()
    d["image_features"] = {"input_type": "image"}
    for key in uq_keys:
        assert key in d, "Key "+key+" not found in default input dict. Found only: "+str(d.keys())
        where = str(int(key in unet_input_dict.keys()))+str(int(key in vit_input_dict.keys()))
        where = {"00": " ", "10": "UNet", "01": "ViT", "11": "Unet,ViT"}[where]
        input_type = d[key]["input_type"]
        is_spatial = key in spatial_image_keys
        df.loc[len(df)] = [key,input_type,"X" if is_spatial else " ",where]
    df = df.sort_values(by=["where","is_spatial","input_type","name"])
    return df
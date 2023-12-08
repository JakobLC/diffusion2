import argparse
import inspect

#from . import gaussian_diffusion as gd TODO
#from .unet import UNetModel TODO
from pathlib import Path
import json
from collections import OrderedDict

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
        
    def get_args(self,modified_args={}):
        args = self.parser.parse_args()
        args = model_specific_args(args)
        args = self.parse_types(args)
        for k,v in modified_args.items():
            assert k in args.__dict__.keys(), f"key {k} not found in args.__dict__.keys()={args.__dict__.keys()}"
            args.__dict__[k] = v
            assert not isinstance(v,list), f"list not supported in modified_args to avoid recursion."
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

def create_model_from_args(args):
    key_names = ("image_encoder,dropout,predict,use_scale_shift_norm,"+
                 "num_res_blocks,num_heads,num_heads_upsample,num_channels,"+
                 "attention_resolutions,image_size,deeper_net,weak_signals,"+
                 "num_classes,datasets_as_classes,self_conditioning").split(",")
    (image_encoder,dropout,predict,use_scale_shift_norm,
        num_res_blocks,num_heads,num_heads_upsample,num_channels,
        attention_resolutions,image_size,deeper_net,weak_signals,
        num_classes,datasets_as_classes,self_conditioning) = tuple([args.__dict__.get(k) for k in key_names])

def create_diffusion_from_args(args):
    pass

"""def create_model_and_diffusion(args):
    key_names =("image_size,num_classes,learn_sigma,sigma_small,num_channels,num_res_blocks,"+
                "num_heads,num_heads_upsample,attention_resolutions,dropout,rrdb_blocks,"+
                "deeper_net,diffusion_steps,noise_schedule,timestep_respacing,use_kl,predict_xstart,"+
                "rescale_timesteps,rescale_learned_sigmas,use_checkpoint,use_scale_shift_norm,"+
                "rrdb_channels,image_encoder,weak_signals,eval_timestep_respacing,"+
                "no_diffusion,bce_loss,loss_domain").split(",")
    (image_size,num_classes,learn_sigma,sigma_small,num_channels,num_res_blocks,
    num_heads,num_heads_upsample,attention_resolutions,dropout,rrdb_blocks,
    deeper_net, diffusion_steps, noise_schedule, timestep_respacing, use_kl, predict_xstart,
    rescale_timesteps, rescale_learned_sigmas, use_checkpoint, use_scale_shift_norm,
    rrdb_channels, image_encoder, weak_signals, eval_timestep_respacing,
    no_diffusion, bce_loss, loss_domain) = tuple([args.__dict__.get(k) for k in key_names])

    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=False if no_diffusion else learn_sigma,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        rrdb_blocks=rrdb_blocks,
        deeper_net=deeper_net,
        rrdb_channels=rrdb_channels,
        image_encoder=image_encoder,
        weak_signals=weak_signals,
        no_diffusion=no_diffusion,
    )
    if no_diffusion:
        loss_type = gd.LossType.BCE if bce_loss else gd.LossType.MSE
        diffusion = gd.DummyDiffusion(loss_type=loss_type,num_timesteps=diffusion_steps)
        eval_diffusion = diffusion
    else:
        diffusion = create_gaussian_diffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
            loss_domain=loss_domain,
        )
        eval_diffusion = create_gaussian_diffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=eval_timestep_respacing,
            loss_domain=loss_domain,
        )
    return model, diffusion, eval_diffusion"""


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    num_classes,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    rrdb_blocks,
    deeper_net,
    rrdb_channels,
    image_encoder,
    weak_signals,
    no_diffusion,
):
    if image_size == 256:
        if deeper_net:
            channel_mult = (1, 1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        if deeper_net:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 16:
        channel_mult = (1, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    
    if isinstance(attention_resolutions, int):
        num_att = int(attention_resolutions)
        num_ds = len(channel_mult)-1
        assert num_ds>=num_att, "number of attention resolutions should be less than number of downsampling layers"
        if num_att<0:
            attention_ds = []
            for i in range(-num_att):
                attention_ds.append(2**(num_ds-i))
        elif num_att>0:
            attention_ds = [2**i for i in range(num_att)]
        else:
            attention_ds = []
    elif isinstance(attention_resolutions, list):
        attention_ds = []
        for res in attention_resolutions:
            attention_ds.append(image_size // int(res))
    else:
        assert isinstance(attention_resolutions, str)
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=1,
        model_channels=num_channels,
        out_channels=(1 if not learn_sigma else 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        rrdb_blocks=rrdb_blocks,
        rrdb_channels=rrdb_channels,
        image_encoder=image_encoder,
        weak_signals=weak_signals,
        no_diffusion=no_diffusion,
    )


"""def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    loss_domain="same"
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
        
    if loss_domain=="same":
        if predict_xstart:
            loss_domain = gd.LossDomain.START_X
        else:
            loss_domain = gd.LossDomain.EPSILON
    elif loss_domain.lower() in ["xstart","x0","x_0","image","mask","start_x","x_start","startx"]:
        loss_domain = gd.LossDomain.START_X
    elif loss_domain.lower() in ["eps","noise","epsilon"]:
        loss_domain = gd.LossDomain.EPSILON
    else:
        raise ValueError(f"loss_domain={loss_domain} not recognized.")
    if isinstance(timestep_respacing, int):
        timestep_respacing = str(timestep_respacing)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        loss_domain=loss_domain,
    )"""

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

from abc import abstractmethod

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import os
from segment_anything.modeling import ImageEncoderViT
from source.models.cond_vit import FancyViT, fancy_vit_from_args, unet_vit_input_dicts_from_args
from source.utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from source.utils.utils import model_arg_is_trivial
from source.models.nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
import pandas as pd

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x,y=x_attn)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False, with_xattn=False, xattn_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.with_xattn = with_xattn
        if self.with_xattn:
            if xattn_channels is None:
                xattn_channels = channels
            self.xattn_channels = xattn_channels
            self.qk_x = conv_nd(1, xattn_channels, 2*channels, 1) 
            self.v_x = conv_nd(1, channels, channels, 1)
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x, y=None):
        b, c, *spatial = x.shape
        #assert c==self.channels, f"expected {self.channels} channels, got {c} channels"
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        if y is not None:
            assert self.with_xattn, "y is is only supported as an input for AttentionBlocks with cross attention"
            b, cx, *spatial2 = y.shape
            assert cx==self.xattn_channels, f"expected {self.xattn_channels} channels, got {cx} channels"
            y = y.reshape(b, cx, -1)
            qk = self.qk_x(self.norm(y))
            v = self.v_x(self.norm(h))
            qkv_x = torch.cat([qk,v],dim=-1).reshape(b * self.num_heads, -1, qk.shape[2])
            h = self.attention(qkv_x)+h
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])

def unet_kwarg_to_tensor(kwarg):
    if (kwarg is None) or isinstance(kwarg, torch.Tensor):
        #output is correctly formatted
        pass
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
            full_kwarg = torch.zeros((bs,)+shapes[0],
                                     dtype=kwarg[s0].dtype,
                                     device=kwarg[s0].device)
            for i in range(bs):
                if kwarg[i] is not None:
                    full_kwarg[i] = kwarg[i]
            kwarg = full_kwarg
    else:
        raise ValueError(f"kwarg={kwarg} is not a valid type. must be None, torch.Tensor, or list of torch.Tensor/None")
    return kwarg
            
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        image_size=32,
        out_channels=1,
        image_channels=3,
        model_channels=64,
        num_res_blocks=3,
        attention_resolutions="-2,-1",
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        weak_signals=False,#TODO, remove unused inputs
        no_diffusion=False,
        self_cond=False,
        cond=False,
        is_pred_both=False,
        debug_flag=False,
        final_act="none",
        image_encoder_shape=(256,64,64),
        image_encoder_depth=-1,
        vit_args=None,
        no_unet=False,
        unet_input_dict=None,
        vit_feature_depth=1,
    ):
        super().__init__()
        self.debug_flag = debug_flag
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.vit_args = vit_args
        self.image_size = image_size
        time_embed_dim = model_channels * 4
        self.legal_keys = list(unet_input_dict.keys())
        self.vit_feature_depth = vit_feature_depth
        if vit_args is not None:
            self.vit = FancyViT(**self.vit_args)
            self.vit_injection_type = {"a": "emb", "b": "xattn", "c": "spatial_once", "d": "spatial_all", "e": "spatial_after_resize"}[self.vit_args["injection_type"]]
            self.legal_keys += list(self.vit_args["input_dict"].keys())
        else:
            assert unet_input_dict is not None, "unet_input_dict must be provided if vit_args is None"
            unet_input_dict = unet_input_dict
            self.vit = None
            self.vit_injection_type="none"
        
        
        if no_unet:
            assert self.vit_args is not None, "vit_args must be provided if no_unet is True"
            self.has_unet = False
            ch = self.vit_args["out_chans"]
            self.fp16_attrs = []
        else:
            self.has_unet = True
            self.model_channels = model_channels
            self.num_res_blocks = num_res_blocks
            
            self.attention_resolutions = []
            if attention_resolutions not in ["", ","]:
                for ar in attention_resolutions.split(","):
                    ar = int(ar)
                    if ar < 0:
                        ar = len(channel_mult) + ar
                    self.attention_resolutions.append(ar)
            self.dropout = dropout
            self.channel_mult = channel_mult
            self.conv_resample = conv_resample
            self.num_classes = None if num_classes==0 else num_classes
            self.use_checkpoint = use_checkpoint
            self.num_heads = num_heads
            self.num_heads_upsample = num_heads_upsample
            self.no_diffusion = no_diffusion
            self.image_encoder_depth = image_encoder_depth
            self.use_image_features = image_encoder_depth >= 0
            self.fp16_attrs = ["input_blocks","middle_block","output_blocks"]

            if self.use_image_features:
                self.fp16_attrs.append("preprocess_img_enc")

                s_in = image_encoder_shape
                d = self.image_size//(2**image_encoder_depth)
                s_out = (channel_mult[image_encoder_depth]*model_channels,d,d)
                self.preprocess_img_enc = [conv_nd(2,in_channels=s_in[0],out_channels=s_out[0],kernel_size=3,padding=1)]
                
                if s_in[1] == s_out[1]:
                    pass
                elif s_in[1] > s_out[1]:
                    if np.isclose(s_in[1]/s_out[1],int(s_in[1]/s_out[1])):
                        self.preprocess_img_enc.append(avg_pool_nd(2,s_in[1]//s_out[1]))
                    else:
                        #ugly downsampling due to non-integer ratio
                        self.preprocess_img_enc.append(nn.Upsample(size=(s_out[1],s_out[2]),mode="bilinear"))
                else:
                    #Bilinear upsampling is as good as it gets
                    self.preprocess_img_enc.append(nn.Upsample(size=(s_out[1],s_out[2]),mode="bilinear"))
                self.preprocess_img_enc = nn.Sequential(*self.preprocess_img_enc)

            self.unet_input_dict = unet_input_dict
            self.in_channels = 0
            for k,v in self.unet_input_dict.items():
                self.unet_input_dict[k] = [i+self.in_channels for i in range(v)]
                self.in_channels += v
            
            self.fp16_attrs.append("time_embed")
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            
            if self.num_classes is not None:
                self.fp16_attrs.append("class_emb")
                self.class_emb = nn.Embedding(num_classes, time_embed_dim)
            
            self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, self.in_channels, model_channels, 3, padding=1))])
            input_block_chans = [model_channels]
            ch = model_channels
            res_block_kwargs = {"emb_channels": time_embed_dim,
                                "dropout": dropout,
                                "dims": dims,
                                "use_checkpoint": use_checkpoint,
                                "use_scale_shift_norm": use_scale_shift_norm}
            attn_kwargs = {"use_checkpoint": use_checkpoint,
                           "num_heads": num_heads,
                           "with_xattn": self.vit_injection_type=="xattn",
                           "xattn_channels": vit_args.get("out_chans",None) if vit_args is not None else None}
            resolution = 0
            
            assert channel_mult[0]==1, "channel_mult[0] must be 1"
            for level, mult in enumerate(channel_mult):
                for _ in range(num_res_blocks):
                    layers = [
                        ResBlock(ch,out_channels=mult*model_channels,**res_block_kwargs)
                    ]
                    ch = mult*model_channels
                    if resolution in self.attention_resolutions:
                        layers.append(AttentionBlock(ch,**attn_kwargs))
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                    input_block_chans.append(ch)
                if level != len(channel_mult) - 1:
                    self.input_blocks.append(
                        TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                    )
                    input_block_chans.append(ch)
                    resolution += 1
            middle_layers = ([ResBlock(ch,**res_block_kwargs)] +
                             ([AttentionBlock(ch,**attn_kwargs)] if len(self.attention_resolutions)>0 else []) +
                             [ResBlock(ch,**res_block_kwargs)])
            self.middle_block = TimestepEmbedSequential(*middle_layers)

            attn_kwargs["num_heads"] = num_heads_upsample
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    layers = [ResBlock(ch + input_block_chans.pop(),out_channels=model_channels*mult,**res_block_kwargs)] 
                    ch = model_channels * mult
                    if resolution in self.attention_resolutions:
                        layers.append(AttentionBlock(ch,**attn_kwargs))
                    if level and i == num_res_blocks:
                        layers.append(Upsample(ch, conv_resample, dims=dims))
                        resolution -= 1
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
            self.make_block_info()

            if self.vit_injection_type=="spatial_all":
                self.vit_to_unet_map = nn.ModuleDict()
                self.vit_injection_depths = []
                for row in self.block_info.iterrows():
                    depth = row[1]["depth"]
                    size = (int(row[1]["image_size_out"]),int(row[1]["image_size_out"]))
                    vit_ch = vit_args["out_chans"]
                    unet_ch = row[1]["ch_out"]
                    self.vit_to_unet_map[str(depth)] = nn.Sequential(conv_nd(2,vit_ch,unet_ch,1),nn.Upsample(size=size,mode="bilinear"))
                    self.vit_injection_depths.append(depth)
            elif self.vit_injection_type=="spatial_once":
                self.vit_to_unet_map = nn.ModuleDict()
                assert self.vit_feature_depth in self.block_info["depth"].values, "vit_feature_depth must be in block_info['depth']="+str(self.block_info["depth"].values)
                vit_ch = vit_args["out_chans"]
                unet_ch = self.block_info.iloc[self.vit_feature_depth]["ch_out"]
                size = self.block_info.iloc[self.vit_feature_depth]["image_size_out"]
                size = (int(size),int(size))
                self.vit_to_unet_map[str(self.vit_feature_depth)] = nn.Sequential(conv_nd(2,vit_ch,unet_ch,1),nn.Upsample(size=size,mode="bilinear"))
                self.vit_injection_depths = [self.vit_feature_depth]
            elif self.vit_injection_type=="xattn":
                self.vit_to_unet_map = None
            elif self.vit_injection_type=="emb":
                self.vit_to_unet_map = nn.Linear(self.vit_args["out_chans"],time_embed_dim)
                self.vit_injection_depths = []
            elif self.vit_injection_type=="spatial_after_resize":
                self.vit_to_unet_map = nn.ModuleDict()
                self.vit_injection_depths = []
                for row in self.block_info.iterrows():
                    if ("Upsample" in row[1]["subclasses"]) or ("Downsample" in row[1]["subclasses"]):
                        depth = row[1]["depth"]
                        size = (int(row[1]["image_size_out"]),int(row[1]["image_size_out"]))
                        vit_ch = vit_args["out_chans"]
                        unet_ch = row[1]["ch_out"]
                        self.vit_to_unet_map[str(depth)] = nn.Sequential(conv_nd(2,vit_ch,unet_ch,1),nn.Upsample(size=size,mode="bilinear"))
                        self.vit_injection_depths.append(depth)
            else:
                assert self.vit_injection_type=="none", "vit_injection_type must be one of 'emb','xattn','spatial_once','spatial_all','spatial_after_resize', or 'none'"
                self.vit_injection_depths = []
        both_mult = 2 if is_pred_both else 1
        final_act_dict = {"none": nn.Identity(),
                        "softmax": nn.Softmax(dim=1),
                        "tanh": nn.Tanh()}
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, ch, out_channels*both_mult, 3, padding=1)),
            final_act_dict[final_act.lower()]
        )

    def make_block_info(self,as_df=True):
        block_info_keys = ["depth","resolution",
                           "subclasses","image_size_in","image_size_out",
                           "ch_in","ch_out","backbone_part","has_attention"]
        self.block_info = {k: [] for k in block_info_keys}

        depth = 0
        resolution = 0

        subclasses = []
        image_size_in = self.image_size
        image_size_out = self.image_size

        ch_in = self.model_channels
        ch_out = self.model_channels
        backbone_part = "input"
        has_attention = False
        
        for i in range(len(self.input_blocks)): 
            subclasses = [type(m).__name__ for m in self.input_blocks[i]]
            has_attention = "AttentionBlock" in subclasses
            has_downsample = "Downsample" in subclasses
            if has_downsample:
                resolution += 1
                image_size_out //= 2
            ch_in, ch_out = get_ch_in_out(self.input_blocks[i])
            values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
            self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
            depth += 1
            if has_downsample:
                image_size_in //= 2
        backbone_part = "middle"
        has_downsample = False
        subclasses = [type(m).__name__ for m in self.middle_block]
        has_attention = "AttentionBlock" in subclasses
        ch_in, ch_out = get_ch_in_out(self.middle_block)
        values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
        self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
        depth += 1
        backbone_part = "output"
        for i in range(len(self.output_blocks)):
            subclasses = [type(m).__name__ for m in self.output_blocks[i]]
            has_attention = "AttentionBlock" in subclasses
            has_upsample = "Upsample" in [type(m).__name__ for m in self.output_blocks[i]]
            if has_upsample:
                resolution -= 1
                image_size_out *= 2
            ch_in, ch_out = get_ch_in_out(self.output_blocks[i])
            values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
            self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
            depth += 1
            if has_upsample:
                image_size_in *= 2
        
        if as_df:
            self.block_info = pd.DataFrame(self.block_info)
        #print as df
        #print(self.block_info)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def to_xattn(self, vit_output, depth):
        if self.vit_injection_type!="xattn":
            out = None
        else:
            if self.block_info.iloc[depth]["has_attention"]:
                out = vit_output
            else:
                out = None
        return out

    def forward(self, sample, timesteps, **kwargs):
        """
        Apply the model to an input batch.

        :param sample: an [N x C x ...] Diffusion sample tensor.
        :param timesteps: a 1-D batch of timesteps.
        :param kwargs: additional kwargs for the model. see self.unet_input_dict for available kwargs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        if self.vit is not None:
            vit_output,token_info = self.vit(self.vit.filter_inputs({**kwargs, 
                                                          "sample": sample, 
                                                          "time": timesteps.repeat(len(sample)//timesteps.numel())}),
                                                          return_token_info=True)
            self.last_token_info = token_info
        else:
            vit_output = None

        if self.has_unet:
            h,timesteps,classes,image_features = self.prepare_inputs(sample, timesteps, **kwargs)
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
            if self.no_diffusion:
                emb *= 0
            if self.vit_injection_type=="emb":
                emb = emb + self.vit_to_unet_map(vit_output)
            if self.num_classes is not None:
                emb = emb + self.class_emb(classes)
            hs = []
            depth = 0
            for module in self.input_blocks:
                h = module(h, emb, x_attn = self.to_xattn(vit_output,depth))
                hs.append(h)
                if (depth == self.image_encoder_depth) and self.use_image_features and (image_features is not None):
                    h = h + self.preprocess_img_enc(image_features.type(h.dtype))
                if depth in self.vit_injection_depths:
                    h = h + self.vit_to_unet_map[str(depth)](vit_output).type(h.dtype)
                depth += 1
            h = self.middle_block(h, emb, x_attn = self.to_xattn(vit_output,depth))
            if depth in self.vit_injection_depths:
                h = h + self.vit_to_unet_map[str(depth)](vit_output).type(h.dtype)
            depth += 1
            for module in self.output_blocks:
                cat_in = torch.cat([h, hs.pop()], dim=1)
                h = module(cat_in, emb, x_attn = self.to_xattn(vit_output,depth))
                if depth in self.vit_injection_depths:
                    h = h + self.vit_to_unet_map[str(depth)](vit_output).type(h.dtype)
                depth += 1
            h = h.type(sample.dtype)
        else:
            h = vit_output
        h = self.out(h)
        return h
    
    def prepare_inputs(self, sample, timesteps, **kwargs):
        """
        prepare inputs for the model.
        Each keyword argument must be either
            - a tensor of shape [batch_size, num_channels, image_size, image_size]
            - None
            - a list where each element is either [num_channels, 
              image_size, image_size] or None
        The function will construct the by formatting the tensors into a tensor of 
        shape [batch_size, sum(num_channels_i), image_size, image_size].
        """
        if self.no_diffusion:
            assert sample is None, "sample must be None if no_diffusion is True"
        else:
            kwargs["sample"] = sample
        bs = sample.shape[0]
        shape = [bs,self.in_channels,self.image_size,self.image_size]
        h = torch.zeros(shape,device=sample.device).type(self.inner_dtype)
        for k,v in kwargs.items():
            if k not in ["classes","image_features"]:
                assert (k in self.legal_keys) or (v is None), f"k={k} with type {type(v)} is not a legal input for the model: {str(self.legal_keys)}"
                if k in self.unet_input_dict.keys():
                    if torch.is_tensor(v):
                        assert len(self.unet_input_dict[k])>0, k+" is not an available kwarg for the model (unless None). Inputs which are allowed to be tensors: "+str([k for k in self.unet_input_dict.keys() if len(self.unet_input_dict[k])>0])
                        assert len(v.shape) == 4, "Expected 4 dimensions for input "+k+", got: "+str(len(v.shape))+" instead."
                        assert v.shape[1] == len(self.unet_input_dict[k]), "Expected "+str(len(self.unet_input_dict[k]))+" channels for input "+k+", got: "+str(v.shape[1])+" instead."
                        assert v.shape[2] == v.shape[3] == self.image_size, "Expected last two dimensions to be "+str(self.image_size)+" for input "+k+", got: "+str(v.shape[2:])+" instead."
                        h[:,self.unet_input_dict[k],:,:] = v.type(self.inner_dtype)
                    else:
                        assert v is None, "input "+k+" must be a tensor or None"
        has_nontrivial_classes = kwargs["classes"] is not None if "classes" in kwargs.keys() else False
        if has_nontrivial_classes:
            assert self.num_classes is not None, "num_classes must be specified if classes are provided"
            classes = kwargs["classes"]
            if classes.numel() == 1:
                classes = classes.expand(bs)
            assert 0<=classes.min() and classes.max()<self.num_classes, "classes must be in range [0,"+str(self.num_classes)+"). classes: "+str(classes)
            assert classes.shape == (bs,)
        else:
            #no classes (embedding 0)
            if self.num_classes is not None:
                classes = torch.zeros(bs,dtype=torch.long,device=sample.device)
            else:
                classes = None
        if "image_features" in kwargs.keys():
            assert self.use_image_features, "image_features provided but model has does not use image features"
            image_features = kwargs["image_features"]
        else:
            image_features = None
        
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(bs)
        assert timesteps.shape == (bs,), "timesteps must be a vector of length batch size"
        
        return h, timesteps, classes, image_features


def create_unet_from_args(args):
    if not isinstance(args,dict):
        args = copy.deepcopy(args.__dict__)
    
    if args["channel_multiplier"]=="auto":
        image_size = args["image_size"]
        if image_size == 256:
            if args["deeper_net"]:
                channel_mult = (1, 1, 1, 2, 2, 4, 4)
            else:
                channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            if args["deeper_net"]:
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
    else:
        channel_mult = tuple([int(x) for x in args["channel_multiplier"].split(",")])
    if args["onehot"]:
        out_channels = args["max_num_classes"]
    else:
        out_channels = np.ceil(np.log2(args["max_num_classes"])).astype(int)
    if args["class_type"]=="none":
        num_classes = None
    elif args["class_type"]=="num_classes":
        num_classes = args["max_num_classes"]+1
    else:
        raise ValueError(f"unknown class_type: {args['class_type']}")
    if args["debug_run"]=="dummymodel":
        unet = DummyModel(out_channels)
    else:
        unet_input_dict,_ = unet_vit_input_dicts_from_args(args)
        vit_args = fancy_vit_from_args(args)
        if len(vit_args)==0:
            vit_args = None
        unet = UNetModel(image_size=args["image_size"],
                    is_pred_both=args["predict"]=="both",
                    out_channels=out_channels,
                    image_channels=1 if args["cat_ball_data"] else 3,
                    num_res_blocks=args["num_res_blocks"],
                    model_channels=args["num_channels"],
                    attention_resolutions=args["attention_resolutions"],
                    dropout=args["dropout"],
                    channel_mult=channel_mult,
                    num_classes=num_classes,
                    num_heads=args["num_heads"],
                    num_heads_upsample=args["num_heads_upsample"],
                    weak_signals=args["weak_signals"],
                    self_cond=args["self_cond"],
                    cond=args["cond_type"]!="none",
                    debug_flag=args["debug_run"],
                    final_act=args["final_activation"],
                    image_encoder_depth=args["image_encoder_depth"] if args["image_encoder"]!="none" else -1,
                    vit_args=vit_args,
                    unet_input_dict=unet_input_dict,
                    no_unet=args["vit_unet_cond_mode"]=="no_unet")
    return unet

def get_sam_image_encoder(model_type="vit_b",device="cuda"):
    if isinstance(model_type,int):
        assert model_type in [0,1,2]
        model_type = ["vit_h","vit_l","vit_b"][model_type]
    elif isinstance(model_type,str):
        model_type = model_type.replace("-","_").lower()
        if model_type.startswith("sam_"):
            model_type = model_type[4:]
        elif model_type.startswith("sam"):
            model_type = model_type[3:]
        assert model_type in ["vit_h","vit_l","vit_b"]
    else:
        raise ValueError("model_type must be int or str")
    checkpoint_idx = ["vit_h","vit_l","vit_b"].index(model_type)
    sam_checkpoint = ["sam_vit_h_4b8939.pth","sam_vit_l_0b3195.pth","sam_vit_b_01ec64.pth"][checkpoint_idx]
    sam_checkpoint = os.path.join(os.path.abspath(".."),"segment-anything","segment_anything","checkpoint",sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_image_encoder = sam.image_encoder
    sam_image_encoder.eval()
    for p in sam_image_encoder.parameters():
        p.requires_grad = False
    return sam_image_encoder

class DummyModel(nn.Module):
    def __init__(self, num_bits):
        super().__init__()
        self.conv_layer = conv_nd(2,num_bits,num_bits,1)

    def convert_to_fp32(self):
        self.conv_layer.apply(convert_module_to_f32)

    def convert_to_fp16(self):
        self.conv_layer.apply(convert_module_to_f16)

    @property
    def inner_dtype(self):
        return next(self.conv_layer.parameters()).dtype
    
    def forward(self,x,timesteps,**kwargs):
        y = self.conv_layer(x.type(self.inner_dtype))
        return y.type(x.dtype)

def get_ch_in_out(module):
    first_block_type = type(module[0]).__name__
    last_block_type = type(module[-1]).__name__
    if first_block_type=="ResBlock":
        ch_in = module[0].channels
    elif first_block_type in ["Downsample","Upsample"]:
        ch_in = module[0].channels
    elif first_block_type=="AttentionBlock":
        ch_in = module[0].channels
    elif first_block_type=="Conv2d":
        ch_in = module[0].in_channels
    else:
        raise ValueError("first block type not recognized: "+first_block_type)

    if last_block_type=="ResBlock":
        ch_out = module[-1].out_channels
    elif last_block_type in ["Downsample","Upsample"]:
        ch_out = module[-1].channels
    elif last_block_type=="AttentionBlock":
        ch_out = module[-1].channels
    elif last_block_type=="Conv2d":
        ch_out = module[-1].out_channels
    else:
        raise ValueError("last block type not recognized: "+last_block_type)
    return ch_in, ch_out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST: try forward pass on cuda")
        model = UNetModel(num_classes=5, num_heads=4, num_heads_upsample=-1, weak_signals=1, no_diffusion=False, self_cond=1, cond=1)
        model.to("cuda")
        print(model.input_dict)
        imsize = 32
        bs = 2
        sample = torch.randn(bs,1,imsize,imsize).cuda()
        timesteps = torch.rand(bs).cuda()
        classes = torch.randint(0,5,(bs,)).cuda()
        kwargs = {"image": torch.randn(bs,3,imsize,imsize).cuda(),
                  "bbox": torch.randn(bs,1,imsize,imsize).cuda(),
                  "points": torch.randn(bs,1,imsize,imsize).cuda(),
                  "self_cond": torch.randn(bs,3,imsize,imsize).cuda(),
                  "cond": torch.randn(bs,4,imsize,imsize).cuda(),
                  "classes": classes}
        print("sample.shape:",sample.shape)
        output = model(sample, timesteps, **kwargs)
        print("output.shape:",output.shape)
    elif args.unit_test==1:
        print("UNIT TEST: try forward pass on cuda, with ugly kwargs")
        model = UNetModel(num_classes=5, num_heads=4, num_heads_upsample=-1, weak_signals=1, no_diffusion=False, self_cond=1, cond=1)
        model.to("cuda")
        print(model.input_dict)
        imsize = 32
        bs = 2
        sample = torch.randn(bs,1,imsize,imsize).cuda()
        timesteps = torch.rand(bs).cuda()
        classes = torch.randint(0,5,(bs,)).cuda()
        kwargs = {"image": torch.randn(bs,3,imsize,imsize).cuda(),
                  "bbox": None,
                  "points": [None for _ in range(bs)],
                  "self_cond": [None]+[torch.randn(4,imsize,imsize).cuda() for _ in range(bs-1)],
                  "cond": torch.randn(bs,4,imsize,imsize).cuda(),
                  "classes": classes}
        kwargs = {k:unet_kwarg_to_tensor(v) for k,v in kwargs.items()}
        print("sample.shape:",sample.shape)
        output = model(sample, timesteps, **kwargs)
        print("output.shape:",output.shape)
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
    
if __name__=="__main__":
    main()
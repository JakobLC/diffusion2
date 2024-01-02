from abc import abstractmethod

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fp16_util import convert_module_to_f16, convert_module_to_f32
from nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


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

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
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

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
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
        weak_signals=False,
        no_diffusion=False,
        self_cond=False,
        cond=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = []
        for ar in attention_resolutions.split(","):
            ar = int(ar)
            if ar < 0:
                ar = len(channel_mult) + ar - 1
            self.attention_resolutions.append(ar)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = None if num_classes==0 else num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.no_diffusion = no_diffusion
        
        self.input_dict = {"sample": out_channels if not self.no_diffusion else 0,
                           "image": image_channels,
                           "bbox": int(weak_signals),
                           "points": int(weak_signals),
                           "self_cond": out_channels+image_channels if self_cond else 0,
                           "cond": out_channels+image_channels if cond else 0,
                           }
        self.in_channels = 0
        for k,v in self.input_dict.items():
            self.input_dict[k] = [i+self.in_channels for i in range(v)]
            self.in_channels += v
            
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None and self.num_classes>0:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        resolution = 0
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if resolution in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                resolution += 1

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if resolution in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    resolution -= 1
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, sample, timesteps, labels=None, **kwargs):
        """
        Apply the model to an input batch.

        :param sample: an [N x C x ...] Diffusion sample tensor.
        :param timesteps: a 1-D batch of timesteps.
        :param labels: an [N] Tensor of labels, if class-conditional.
        :param kwargs: additional kwargs for the model. see self.input_dict for available kwargs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h, emb = self.prepare_inputs(sample, timesteps, labels, **kwargs)
        
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(sample.dtype)
        h = self.out(h)
        return h
    
    def prepare_inputs(self, sample, timesteps, labels=None, **kwargs):
        """
        prepare inputs for the model.
        """
        if self.no_diffusion:
            assert sample is None, "sample must be None if no_diffusion is True"
        else:
            kwargs["sample"] = sample
        bs = sample.shape[0]
        shape = [bs,self.in_channels,self.image_size,self.image_size]
        h = torch.zeros(shape,device=sample.device).type(self.inner_dtype)
        for k,v in kwargs.items():
            assert k in self.input_dict.keys(), k+" is not an available kwarg for the model. legal inputs: "+str(self.input_dict.keys())
            if isinstance(v,list):
                if any([isinstance(item,torch.Tensor) for item in v]):
                    v = torch.stack(v,dim=1)
                else:
                    continue #TODO
            assert isinstance(v,torch.Tensor), k+" must be a tensor or list of objects (None or tensor) to be concatenated as a tensor"
            if len(self.input_dict[k])>0:
                assert len(v.shape) == 4, "Expected 4 dimensions for input "+k+", got: "+str(len(v.shape))+" instead."
                assert v.shape[1] == len(self.input_dict[k]), "Expected "+str(len(self.input_dict[k]))+" channels for input "+k+", got: "+str(v.shape[1])+" instead."
                assert v.shape[2] == v.shape[3] == self.image_size, "Expected last two dimensions to be "+str(self.image_size)+" for input "+k+", got: "+str(v.shape[2:])+" instead."
                if v.shape[0]==1:
                    v = v.repeat(bs,1,1,1)
                assert v.shape[0] == bs, "Expected first dimension to be batch size, got: "+str(v.shape[0])+" instead."
                
                h[:,self.input_dict[k],:,:] = v.type(self.inner_dtype)
            else:
                assert v is None, k+" is not an available kwarg for the model. legal inputs: "+str(self.input_dict.keys())
            
        if (labels is not None):
            assert self.num_classes is not None, "num_classes must be specified if labels are provided"
            if labels.numel() == 1:
                labels = labels.expand(bs)
            assert 0<=labels.min() and labels.max()<self.num_classes, "labels must be in range [0,"+str(self.num_classes)+")"
        else:
            if self.num_classes is not None:
                labels = torch.zeros(sample.shape[0],dtype=torch.long,device=sample.device)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(bs)
        assert timesteps.shape == (bs,), "timesteps must be a vector of length batch size"
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.no_diffusion:
            emb *= 0
            
        if self.num_classes is not None:
            assert labels.shape == (sample.shape[0],)
            emb = emb + self.label_emb(labels)
            
        return h, emb


def create_unet_from_args(args):
    if not isinstance(args,dict):
        args = copy.deepcopy(args.__dict__)
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
    out_channels = np.ceil(np.log2(args["max_num_classes"])).astype(int)
    if args["predict"]=="both":
        out_channels *= 2
    unet = UNetModel(image_size=args["image_size"],
                    out_channels=out_channels,
                    image_channels=0 if args["cat_ball_data"] else 3,
                    num_res_blocks=args["num_res_blocks"],
                    attention_resolutions=args["attention_resolutions"],
                    dropout=args["dropout"],
                    channel_mult=channel_mult,
                    num_classes=None,
                    num_heads=args["num_heads"],
                    num_heads_upsample=args["num_heads_upsample"],
                    weak_signals=args["weak_signals"],
                    self_cond=args["self_conditioning"],
                    cond=args["conditioning_type"]!="none")
    return unet

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
        labels = torch.randint(0,5,(bs,)).cuda()
        kwargs = {"image": torch.randn(bs,3,imsize,imsize).cuda(),
                  "bbox": torch.randn(bs,1,imsize,imsize).cuda(),
                  "points": torch.randn(bs,1,imsize,imsize).cuda(),
                  "self_cond": torch.randn(bs,4,imsize,imsize).cuda(),
                  "cond": torch.randn(bs,4,imsize,imsize).cuda(),
                  }
        print("sample.shape:",sample.shape)
        output = model(sample, timesteps, labels, **kwargs)
        print("output.shape:",output.shape)
    else:
        raise ValueError(f"Unknown unit test index: {args.unit_test}")
    
if __name__=="__main__":
    main()
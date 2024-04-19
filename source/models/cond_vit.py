# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type
import copy
import math

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
                'mlp_ratio':        4,
                'norm_layer':       partial(torch.nn.LayerNorm, eps=1e-6)}


def vit_args_from_idx(idx):
    args = copy.deepcopy(vit_shared_params)
    assert idx in list(range(-2,3)), f"Invalid index {idx} for ViT. Must be in range -2 to 2"
    dict_idx = vit_seperate_params['idx'].index(idx)
    for k in vit_seperate_params.keys():
        if k not in additional_fields:
            args[k] = vit_seperate_params[k][dict_idx]
    return args

def fancy_vit_from_idx(idx):
    args = vit_args_from_idx(idx)
    args["block_types"] = ["global" if i in args["global_attn_indexes"] else "window" for i in range(args["depth"])]
    args["input_dict"] = {"image": {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3}}
    args["max_seq_len"] = 64**2
    args["output_image_only"] = True
    for k in ["global_attn_indexes","img_size","patch_size"]:
        del args[k]
    return FancyViT(**args)

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


def default_input_dict():
    #input types: ["image","scalar_continuous","scalar_discrete","vocabulary"]
    #
    inputs = {"image":        {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3},
            "same_vol":       {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3+6},
            "same_classes":   {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3+6},
            "same_dataset":   {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3+6},
            "adjecant":       {"input_type": "image", "img_size": 1024, "patch_size": 16, "in_chans": 3+6},
            "time":           {"input_type": "scalar_continuous", "min": 0.0, "max": 1.0},
            "num_classes":    {"input_type": "scalar_discrete", "size": 64},
            "class_names":    {"input_type": "vocabulary", "size": 8096},}
    return inputs

class FancyViT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        block_types: list = [],
        group_size: int = 0,
        share_image_rel_pos: bool = True,
        share_image_patch_embed: bool = True,
        input_dict: dict = default_input_dict(),
        max_seq_len: int = 1024,
        output_image_only: bool = False,
    ) -> None:
        super().__init__()
        self.output_image_only = output_image_only
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.share_image_rel_pos = share_image_rel_pos
        self.share_image_patch_embed = share_image_patch_embed
        valid_input_names = ["image","scalar_continuous","scalar_discrete","vocabulary"]
        assert len(block_types)==depth, "Fancy block type must be specified for each block"
        self.img_size = []
        self.patch_size = []
        self.in_chans = []
        for input_name, input_info in input_dict.items():
            t = input_info["input_type"]
            assert t in valid_input_names, f"Invalid input type specified. Found {input_name}"
            if t=="image":
                self.img_size.append(input_info["img_size"])
                self.patch_size.append(input_info["patch_size"])
                self.in_chans.append(input_info["in_chans"])
            elif t=="scalar_continuous":
                setattr(self, input_name+"_embed", ContinuousVariable(dim=embed_dim, min_val=input_info["min"], max_val=input_info["max"]))
            elif t=="scalar_discrete":
                setattr(self, input_name+"_embed", nn.Embedding(input_info["size"], embed_dim))
            elif t=="vocabulary":
                setattr(self, input_name+"_embed", nn.Embedding(input_info["size"], embed_dim))
        self.input_dict = input_dict
        if self.share_image_patch_embed:
            #assert we have same patch_size and image_size
            assert len(set(self.img_size))==1, "Image sizes must be the same for shared patch embed. found: "+str(self.img_size)
            assert len(set(self.patch_size))==1, "Patch sizes must be the same for shared patch embed. found: "+str(self.patch_size)
            self.patch_size=self.patch_size[0]
            self.img_size=self.img_size[0]
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


        self.blocks = nn.ModuleList()
        for i in range(depth):
            attention_type = block_types[i]
            attention_size = {"global": self.token_img_size,
                                "window": window_size,
                                "grouped": group_size,
                                }[attention_type]
            block = FancyBlock(
                attention_size=attention_size,
                attention_type=attention_type,
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                self.embed_dim,
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

    def reset_idx(self,bs):
        self.idx_now = [0 for _ in range(bs)]

    def iterate_idx(self,idx,num_iter):
        assert idx<len(self.idx_now), f"Index {idx} out of range"
        self.idx_now[idx]+=num_iter
        if self.idx_now[idx]>self.max_seq_len:
            space_enough = False
        else:
            space_enough = True
        return space_enough, slice(self.idx_now[idx]-num_iter,self.idx_now[idx])

    def tokenize_inputs(self, inputs, 
                        crop_unused_tokens=True, 
                        raise_error_on_zero_tokens=True):
        if torch.is_tensor(inputs):
            inputs = {"image": inputs}
        device,bs = get_device_and_bs_from_input_dict(inputs)
        self.reset_idx(bs)
        tokens = torch.zeros(bs,self.max_seq_len,self.embed_dim, device=device)
        token_info = [[{"input_name": "padding"} for _ in range(self.max_seq_len)] for _ in range(bs)]
        for input_name, item in inputs.items():
            assert input_name in self.input_dict, f"Input name {input_name} not found in input_dict. Has to be one of: {self.input_dict.keys()}"
            t = self.input_dict[input_name]["input_type"]
            if item is None:
                continue
            for i in range(bs):
                item_i = item[i]
                if item_i is None:
                    continue
                else:
                    item_i = item_i.unsqueeze(0)
                tokenitem_info = None
                if t=="image":
                    if self.share_image_patch_embed:
                        if item_i.shape[1]<max(self.in_chans):
                            item_i = F.pad(item_i,(0,0,0,0,0,max(self.in_chans)-item_i.shape[1]))
                        tokenized_item = self.shared_patch_embed(item_i)
                    else:
                        tokenized_item = getattr(self, input_name+"_embed")(item)
                    tokenitem_info = sum([[{"input_name": input_name, 
                                        "input_type": t, 
                                        "h": k, 
                                        "w": j} for j in range(self.token_img_size)] 
                                                for k in range(self.token_img_size)],[])
                    if self.pos_embed is not None:
                        tokenized_item += self.pos_embed
                    tokenized_item = tokenized_item.reshape(1,-1,self.embed_dim)
                elif t=="scalar_continuous":
                    if item_i>=0:
                        tokenized_item = getattr(self, input_name+"_embed")(item_i)
                elif t=="scalar_discrete":
                    if item_i>=0:
                        tokenized_item = getattr(self, input_name+"_embed")(item_i)
                elif t=="vocabulary":
                    vocab_indices = item_i
                    if (vocab_indices>=0).any():
                        vocab_indices = vocab_indices[vocab_indices>=0].reshape(1,-1)
                        tokenized_item = getattr(self, input_name+"_embed")(vocab_indices)
                else:
                    raise ValueError(f"Invalid input_name, must be one of {self.input_dict.keys()}")
                seq_len = tokenized_item.shape[1]
                space_enough,seq_idx = self.iterate_idx(i,seq_len)
                if tokenitem_info is None:
                    tokenitem_info = [{"input_name": input_name, "input_type": t, "j": j} for j in range(seq_len)]
                if space_enough:
                    tokens[i,seq_idx,:] = tokenized_item
                    for j in range(seq_len):
                        token_info[i][j] = tokenitem_info[j]
        max_actual_len = max([len([item2 for item2 in item if item2["input_name"]!="padding"]) for item in token_info])
        if crop_unused_tokens:
            tokens = tokens[:,:max_actual_len]
        if raise_error_on_zero_tokens:
            if max_actual_len==0:
                raise ValueError("No tokens found in input")
        return tokens, token_info

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, token_info = self.tokenize_inputs(inputs)
        
        for blk in self.blocks:
            x = blk(x,token_info)

        if self.output_image_only:
            x = x.reshape(x.shape[0],self.token_img_size,self.token_img_size,self.embed_dim)
        x = self.neck(x.permute(0, 3, 1, 2))

        return x



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
        norm_layer: Type[nn.Module] = nn.LayerNorm,
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

def get_device_and_bs_from_input_dict(inputs):
    first_torch_item = list([item for item in inputs.values() if torch.is_tensor(item)])[0]
    bs = first_torch_item.shape[0]
    device = first_torch_item.device
    return device,bs

class FancyBlock(nn.Module):
    """Transformer blocks with support for various forms of attention:
        - global_attention (basic form for ViT)
        - window_attention (SWIN)
        - grouped_attention (grouped attenation token_info for multiple images)
        - global_sparse_attention (global attention wrt. a small set of tokens, but attending all other tokens)
        """
    def __init__(
        self,
        attention_type: str,
        attention_size: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True
    ) -> None:
        super().__init__()
        assert attention_type in ["global", "window", "grouped", "global_sparse"], f"Invalid attention type {attention_type}"
        self.attention_type = attention_type
        self.attention_size = attention_size

        self.norm1 = norm_layer(dim)
        attn_func_dict = {
            "global": GlobalAttention,
            "window": WindowedAttention,
            "grouped": GroupedAttention,
            "global_sparse": SparseGlobalAttention,
        }
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
        norm_layer: Type[nn.Module] = nn.LayerNorm,
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
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    
    def forward_wrapped(self, x: torch.Tensor, token_info = None):
        assert token_info is None, "token_info not supported for basic attention, use a subclass"
        return self.forward(x)

class GlobalAttention(Attention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward_wrapped(self, x, token_info = None):
        if len(x.shape)==3:
            s1 = x.shape
            s2 = (x.shape[0],self.input_size[0],self.input_size[1],x.shape[-1])
            return self.forward(x.reshape(s2)).reshape(s1)
        else:
            return self.forward(x)
        
class WindowedAttention(Attention):
    def __init__(self, *args, is_shifted=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = self.input_size[0]
        if is_shifted:
            self.window_delta = self.window_size//2
        else:
            self.window_delta = 0
    
    def group_tokens_for_2d_window(self,x):
        assert len(x.shape)==4, "Expected tokens to be stored in a (B,H,W,C) tensor, found size: "+str(x.shape)
        s,d = self.window_size, self.window_delta
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
    
    def ungroup_tokens_for_2d_window(self,groups,group_inv):
        x_padded = torch.zeros(group_inv["x_padded.shape"],dtype=groups.dtype,device=groups.device)
        for i,group in enumerate(groups):
            slice_H = group_inv["group_slice"][i]["slice_H"]
            slice_W = group_inv["group_slice"][i]["slice_W"]
            bs_i = group_inv["group_slice"][i]["bs_i"]
            x_padded[bs_i,slice_H,slice_W,:] = group
        x = x_padded[:,group_inv["unpad_slice_H"],group_inv["unpad_slice_W"]]
        return x
    
    def group_window_multi(self,x,token_info):
        """groups tokens for window attention with multiple images, based on information in token_info"""
        group_inv_multi = {}
        window_grouped_tokens = []
        for i in range(len(token_info)):
            for j in range(len(token_info[i])):
                t_ij = token_info[i][j]["input_type"]
                n_ij = token_info[i][j]["input_name"]
                if t_ij=="image":
                    item = (i,n_ij)
                    if item not in group_inv_multi:
                        idx_x = [k for k in range(len(token_info[i])) if token_info[i][k]["input_name"]==n_ij]
                        h_of_image = [token_info[i][k]["h"] for k in idx_x]
                        w_of_image = [token_info[i][k]["w"] for k in idx_x]
                        H,W = max(h_of_image)+1, max(w_of_image)+1
                        order_criterion = torch.tensor([W*h+w for h,w in zip(h_of_image,w_of_image)])
                        idx_x = [idx_x[k] for k in torch.argsort(order_criterion)]
                        y = x[i,idx_x].reshape(1,H,W,-1)
                        z,group_inv = self.group_tokens_for_2d_window(y)
                        lwgt = len(window_grouped_tokens)
                        group_inv_multi[item] = {"i": i,
                                                 "n": n_ij,
                                                 "idx_x": idx_x,
                                                 "idx_z": list(range(lwgt,lwgt+len(z))),
                                                 "group_inv": group_inv}
                        window_grouped_tokens.append(z)
        
        window_grouped_tokens = torch.cat(window_grouped_tokens,dim=0)
        return window_grouped_tokens, group_inv_multi

    def ungroup_window_multi(self,x,z,group_inv_multi):
        for g in group_inv_multi.values():
            i,idx_x,idx_z,group_inv = g["i"],g["idx_x"],g["idx_z"],g["group_inv"]
            y = self.ungroup_tokens_for_2d_window(z[idx_z],group_inv)
            x[i,idx_x] = y.reshape(-1,y.shape[-1])
        return x
            
    def forward_wrapped_w_window_partition(self, x, token_info = None):
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)
        x = self.forward(x)
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        return x
    
    def forward_wrapped(self, x, token_info = None):
        """if token_info is None:
            x, group_inv = self.group_tokens_for_2d_window(x)
            x = self.forward(x)
            x = self.ungroup_tokens_for_2d_window(x, group_inv)
        else:"""
        assert len(x.shape)==3, "Expected tokens to be stored in a (B,L,C) tensor when using the token_info argument, found size: "+str(x.shape)
        y,group_inv_multi = self.group_window_multi(x,token_info)
        y = self.forward(y)
        x = self.ungroup_window_multi(x,y,group_inv_multi)
        return x

class SparseGlobalAttention(Attention):
    """Multi-head Attention block which computes the full attention matrix for a specified subset of tokens."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SparseGlobalAttention not implemented yet")
    
class GroupedAttention(Attention):
    """Multi-head Attention block which reorders a batch of items to a larger group specific batch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = self.input_size[0]
    
    def forward_wrapped(self, x, token_info = None):
        H, W = x.shape[1], x.shape[2]
        x, y, group_inv = group_tokens(x, token_info, self.group_size)
        x = self.forward(x)
        y = self.forward(y)
        x = ungroup_tokens(x, y, token_info, group_inv)
        return x

def group_tokens(x: torch.Tensor, 
                 token_info: torch.Tensor,
                 group_size: int) -> torch.Tensor:
    assert x.shape[:-1]==token_info.shape, "expected all but channel dimension (last) to match, found: "+str(x.shape[:-1])+" and "+str(token_info.shape)
    B = x.shape[0]
    groups = []
    singles = []
    group_inv = []
    for i in range(B):
        group_idx = torch.unique(token_info[i])
        for idx in group_idx:
            if idx>=0:
                groups.append(x[i,token_info[i]==idx])
                group_inv.append(("x",i,idx))
            else:
                singles.append(x[i,token_info[i]==idx])
                group_inv.append(("y",i,idx))
    len_per_group = [len(g) for g in groups]
    max_len = max(len_per_group)
    assert max_len<=group_size, "Group size must be consistent"
    groups = [F.pad(g,(0,0,0,max_len-len(g))) for g in groups]
    groups = torch.stack(groups)
    return groups, singles, group_inv

def ungroup_tokens(groups: torch.Tensor, 
                   singles: torch.Tensor,
                   token_info: torch.Tensor,
                   group_inv: list) -> torch.Tensor:
    B = token_info.shape[0]
    x = torch.zeros(list(token_info.shape)+[groups.shape[-1]],device=groups.device)
    for g, (t, i, idx) in zip(groups,group_inv):
        if t=="x":
            mask = token_info[i]==idx
            singles[i][mask,:] = g
        else:
            mask = token_info[i]==idx
            singles[i][mask,:] = g

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

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def main():
    import argparse
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
    from utils.utils import set_random_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_test", type=int, default=0)
    args = parser.parse_args()
    if args.unit_test==0:
        print("UNIT TEST 1: Testing default ver")
        set_random_seed(123)
        model = sam_vit_from_idx(-1)
        model.to('cuda')
        model.eval()
        #set seed
        img = torch.randn(2, 3, 1024, 1024).to('cuda')
        pred = model(img)
        print(f"input shape: {img.shape}, output shape: {pred.shape}")
        print(f"input absum val: {img.abs().sum()}")
        print(f"output absum val: {pred.abs().sum()}")
        #expected result:
        #output absum val: 1675137.25
    elif args.unit_test==1:
        print("UNIT TEST 1: fancy")
        set_random_seed(123)
        model = fancy_vit_from_idx(-1)
        model.to('cuda')
        model.eval()
        img = torch.randn(2, 3, 1024, 1024).to('cuda')
        pred = model(img)
        print(f"input absum val: {img.abs().sum()}")
        print(f"output absum val: {pred.abs().sum()}")
    elif args.unit_test==2:
        imsize = 128
        patch_size = 16
        print("UNIT TEST 2: fancy with lots of args")
        inputs = {"image":            {"input_type": "image", "img_size": imsize, "patch_size": patch_size, "in_chans": 3},
                    "same_vol":       {"input_type": "image", "img_size": imsize, "patch_size": patch_size, "in_chans": 3+6},
                    "same_classes":   {"input_type": "image", "img_size": imsize, "patch_size": patch_size, "in_chans": 3+6},
                    "same_dataset":   {"input_type": "image", "img_size": imsize, "patch_size": patch_size, "in_chans": 3+6},
                    "adjecant":       {"input_type": "image", "img_size": imsize, "patch_size": patch_size, "in_chans": 3+6},
                    "time":           {"input_type": "scalar_continuous", "min": 0.0, "max": 1.0},
                    "num_classes":    {"input_type": "scalar_discrete", "size": 64},
                    "class_names":    {"input_type": "vocabulary", "size": 8096},}
        set_random_seed(123)
        dim = 256
        block_types = ["global","window","grouped"]*2
        depth = len(block_types)
        model = FancyViT(embed_dim=dim,block_types=block_types,depth=depth,input_dict=inputs)
        import jlc
        jlc.num_of_params(model)
        bs = 5
        class_names = torch.randint(0,8096,(bs,5))
        padding_mask = 0.4>torch.rand(bs,5)
        class_names[padding_mask] = -1
        input_dict = {"image": torch.rand(bs,3,imsize,imsize),
                      "same_vol": torch.rand(bs,3+6,imsize,imsize),
                    "same_classes": torch.rand(bs,3+6,imsize,imsize),
                    "same_dataset": torch.rand(bs,3+6,imsize,imsize),
                    "adjecant": torch.rand(bs,3+6,imsize,imsize),
                    "time": torch.rand(bs,1),
                    "num_classes": torch.randint(0,64,(bs,)),
                    "class_names": class_names}
        pred = model(input_dict)

    else:
        raise ValueError("Invalid unit test")
    
if __name__=="__main__":
    main()
import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math
from timm.models.regnet import RegStage
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first

    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim ##这里的embed_dim应该是输出的dim， kv_dim则是vision_backbone_id的dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, x.size(1)) 

        x = self.kv_proj(x) 
        x = self.ln_kv(x).permute(1, 0, 2) 

        N = x.shape[1]
        q = self.ln_q(self.query) 
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),  
            x + pos_embed.unsqueeze(1), 
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class ResamplerWhole(nn.Module):
    def __init__(self, nquery, embed_dim, out_dim):
        super().__init__()

        nheads = out_dim//128
        self.nquery = nquery
        self.out_dim = out_dim
        grid_size = int(math.sqrt(nquery))
        self.attn_pool = Resampler(grid_size, out_dim, nheads, kv_dim=embed_dim)
        self.ln_post = nn.LayerNorm(out_dim)
        self.proj = nn.Parameter((out_dim ** -0.5) * torch.randn(out_dim, out_dim))

    def forward(self, x):
        x = x[0] # original single-level 
        x1 = self.attn_pool(x)
        x1 = self.ln_post(x1)
        x1 = x1 @ self.proj

        return x1




class TokenPacker(nn.Module):
    def __init__(
            self,
            raw_grid=24,
            embed_dim=1024,
            num_heads=1024//128,
            kv_dim=1024,
            hidden_size=4096,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        if raw_grid%scale_factor!=0:
            raise ValueError("scale_factor must be divisible by grid size")
        self.raw_grid = raw_grid
        self.grid_size = raw_grid//scale_factor
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.q_proj_1 = nn.Linear(kv_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(4096, 1024)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(1024, 1024))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(4096, 1024)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(1024, 1024))
        self.v_proj_1 = nn.Sequential(*v_modules)

        self.ln_q_1 = norm_layer(embed_dim)
        self.ln_k_1 = norm_layer(embed_dim)
        self.ln_v_1 = norm_layer(embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        modules = [nn.Linear(1024, hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def divide_feature(self, x, kernel_size, token_num, N, c):
        h = w = int(token_num**0.5)

        reshape_x = x.reshape(h, w, N, c).reshape(h//kernel_size, kernel_size, w, N, c)
        reshape_x = reshape_x.permute(0,2,1,3,4)
        reshape_x = reshape_x.reshape(h//kernel_size, w//kernel_size, kernel_size, kernel_size, N, c)
        reshape_x = reshape_x.permute(0,1,3,2,4,5).reshape(h//kernel_size, w//kernel_size, kernel_size*kernel_size, N, c)
        reshape_x = reshape_x.permute(2,0,1,3,4).reshape(kernel_size*kernel_size, -1, c)

        return reshape_x

    def forward(self, x, attn_mask=None):

        x_multi = x[1] # mulit-level
        x_multi = torch.cat(x_multi, dim=2)
        x = x[0] # original single-level

        key = self.ln_k_1(self.k_proj_1(x_multi)).permute(1, 0, 2)
        value = self.ln_v_1(self.v_proj_1(x_multi)).permute(1, 0, 2)

        token_num, N, c = key.shape

        q = F.interpolate(x.reshape(x.shape[0],self.raw_grid,self.raw_grid,-1).float().permute(0,3,1,2), size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1) ## fix
        q = q.reshape(q.shape[0], -1, q.shape[-1]).to(x.dtype)

        query = self.ln_q_1(self.q_proj_1(q)).permute(1, 0, 2)

        reshape_query = self.divide_feature(query, 1, self.num_queries, N, c)
        reshape_key = self.divide_feature(key, self.scale_factor, token_num, N, c)
        reshape_value = self.divide_feature(value, self.scale_factor, token_num, N, value.shape[-1])

        out = self.clip_attn(
            reshape_query,
            reshape_key,
            reshape_value,
            attn_mask=attn_mask)[0]

        x = out
        x = x.reshape(self.num_queries, N, -1)
        x = x.permute(1, 0, 2)

        x = self.mlp(x)


    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)





class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim, bias=True),
        )
    def forward(self, inputx: torch.Tensor) -> torch.Tensor:
        return self.projector(inputx)





def build_pos_embeds(
    config, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config):
    if getattr(config, "prenorm", False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)



class CAbstractorConfig:
    depth = 3
    mlp_depth = 2
    num_eos_tokens = 0
    pos_emb = True
    feature_layer_index=-1
    prenorm = False
    num_query_tokens = 144
    hidden_size = 1024
    encoder_hidden_size = 1024
    output_hidden_size = 4096


class CAbstractor(nn.Module):
    """C-Abstractor based on RegBlock
    
    copied from the scoure code of Cabstractor https://github.com/khanrc/honeybee
    """
    def __init__(
        self,
        config,
        num_input_tokens: int,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens

        # think tokens
        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth # 3
        mlp_depth = self.config.mlp_depth # 2

        n_queries = self.config.num_query_tokens # 144
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        #sampler = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )
        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw) # bs, embed_dim, 24, 24
        x = self.net(x) ## bs, embed_dim, 12, 12
        x = rearrange(x, "b d h w -> b (h w) d") ## bs, 144, embed_dim
        x = self.readout(x) ## bs, 144, embed_dim

        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        x = x[0] ## tokepacker中对clip的输出做了调整
        if self.prenorm is not None: ## False
            x = self.prenorm(x)

        if self.pos_emb is not None: # True
            #x += self.pos_emb ## 这样处理会报错
            x = self._forward(x + self.pos_emb)  # (B, L, output_hidden_size)
        else:
            x = self._forward(x)

        B = x.size(0)
        if self.eos_tokens is not None: ## False
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        
        return x     



class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SElayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x) 
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape) ## npu的adaptiveavgpool2d不支持bf16
            #nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

class LDPConfig:
    mm_hidden_size = 1024
    hidden_size = 4096


class LDPNetProjector(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)

class LDPNetV2Projector(nn.Module):
    def __init__(self, config=None):
        """
        copy from the source code of the mobilevlm v2, https://github.com/Meituan-AutoML/MobileVLM
        """
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((12, 12))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = x[0]
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x) 

        return x


   
class SAEProjector(nn.Module):
    def __init__(self, ntoken=576, nlayer=4, embed_dim=1024, llm_dim=4096, kernel_size=2, stride=2, add_pool=True, apply_init=True, add_mid_norm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.ntoken = ntoken    
        self.llm_dim = llm_dim
        self.nlayer = nlayer
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_pool = add_pool
        self.apply_init = apply_init
        self.add_mid_norm = add_mid_norm

        self.h = int(ntoken**0.5)
        self.w = int(ntoken**0.5)
        self.point_conv = nn.Linear(embed_dim*nlayer, llm_dim, bias=True)
        if add_mid_norm: 
            self.norm1 = nn.LayerNorm(llm_dim)

        ## arch_specificer "none"
        self.conv1 = nn.Conv2d(in_channels=llm_dim, out_channels=llm_dim, kernel_size=self.kernel_size, stride=self.stride, groups=llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        if self.add_pool:
            self.pool = nn.AvgPool2d(kernel_size=(self.kernel_size, self.kernel_size), stride=(self.stride, self.stride))        
            self.norm3 = nn.LayerNorm(llm_dim)
        
        if self.apply_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 
    
    def forward(self, x):
        x = x[1]
        x = torch.cat(x, dim=2) ## bs, ntoken, embed_dim*nlayer
        assert x.shape[2] == self.nlayer*self.embed_dim
        x = self.point_conv(x) ## bs, ntoken, embed_dim
        if self.add_mid_norm:
            x = self.norm1(x)

        BS, ntoken, llm_dim = x.shape
        assert ntoken == self.ntoken and llm_dim == self.llm_dim
        x = x.permute(0,2,1).reshape(BS, llm_dim, self.h, self.w) ## bs, embed, h, w

        x1 = self.conv1(x) ## bs, embed, h/2, w/2
        x1 = x1.permute(0,2,3,1).reshape(BS, -1, self.llm_dim)
        x1 = self.norm2(x1)
        
        if self.add_pool:
            x2 = self.pool(x) 
            x2 = x2.permute(0,2,3,1).reshape(BS, -1, self.llm_dim)
            x2 = self.norm3(x2)

            x3 = x1 + x2
            
            return x3
        else:
            return x1


        return x2




def build_vision_projector(config):
    projector_type = config.mm_projector_type

    if projector_type == 'tokenpacker':
        print("****")
        print("build a tokenpacker-64token")
        print("****")
        return TokenPacker(hidden_size=config.hidden_size, scale_factor=config.scale_factor)

    elif projector_type == "SAEProjector":
        print("***")
        print("build a SAEProjector projector")
        print("***")
        return SAEProjector(ntoken=576, nlayer=5, embed_dim=1024, llm_dim=4096)
    
    elif projector_type == "SAEProjector-13b":
        print("***")
        print("build a SAEProjector projector")
        print("***")
        return SAEProjector(ntoken=576, nlayer=5, embed_dim=1024, llm_dim=5120)


    elif projector_type == "resampler":
        print("***")
        print("build a resampler projector")
        print("***")
        return ResamplerWhole(nquery=144, embed_dim=1024, out_dim=4096)
    

    elif projector_type == "cabstractor":
        print("***")
        print("build a cabstractor projector")
        print("***")
        cabs_config = CAbstractorConfig()
        return CAbstractor(cabs_config, 576)

    
    elif projector_type == "ldpv2":
        print("***")
        print("build a LDPv2 projector")
        print("***")
        ldpv2_config = LDPConfig()
        return LDPNetV2Projector(ldpv2_config)


    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')



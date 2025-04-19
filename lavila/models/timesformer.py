# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from https://github.com/m-bain/frozen-in-time/blob/main/model/video_transformer.py
# Modified by Yue Zhao
# The original code is under MIT License

"""
Implementations of Video Transformers in PyTorch
A PyTorch implementation of space-time transformer as described in
'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval' - https://arxiv.org/abs/2104.00650
A PyTorch implementation of timesformer as described in
'Is Space-Time Attention All You Need for Video Understanding?' - https://arxiv.org/abs/2102.05095
Acknowledgments:
- This code builds on Ross Wightman's vision_transformer code in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
- It is also inspired by lucidrains timesformer implementation:
https://github.com/lucidrains/TimeSformer-pytorch
Hacked together by Max Bain
"""

from collections import OrderedDict, defaultdict
from functools import partial, reduce
import operator
import copy

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn
import torch.nn.functional as F
import pdb

from lavila.models.prompt_tuning import VisualPromptLearner, CMM


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, ln_pre=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        # ln_pre is inserted to be compatible with CLIP-style model
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F <= self.num_frames
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        return x


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random', num_tokens=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_tokens = num_tokens

    def forward(self, x, einops_from, einops_to, einops_dims, cfg):
        style = cfg.get('style', 'default')
        pt_att = cfg.get('pt_att', True)
        n_seg = cfg.get('n_seg', 4)
        if 'VoP' in style:
            return self.forward_VoP(x, einops_from, einops_to, einops_dims, n_seg)
        elif style == 'attall':
            return self.forward_attall(x, pt_att)
        else:
            return self.forward_features(x, einops_from, einops_to, einops_dims, pt_att)

    def forward_features(self, x, einops_from, einops_to, einops_dims, pt_att=True):
        h = self.num_heads
        num_tokens = self.num_tokens
        if self.num_tokens > 0 and not pt_att:
            prompts = x[:, 1:self.num_tokens+1, :]
            x = torch.cat((
                x[:, :1, :], # cls_token
                x[:, self.num_tokens+1:, :] # patch embeddings
            ), dim=1)
            num_tokens = 0

        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out CLS token at index 1 (and prompts)
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:num_tokens+1], t[:, num_tokens+1:]), (q, k, v)) # Bh x () x d

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v) # Bh x (1 + p) x d
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_)) # Bh x NT x d -> Bhr x s x d

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b p d -> (b r) p d', r=r), (cls_k, cls_v))  # Bhr x (1 + p) x d
        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims) # Bh x NT x d

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1) # Bh x (1 + p + NT) x d

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # B x (1 + p + NT) x hd
        if self.num_tokens > 0 and not pt_att:
            out = torch.cat((
                out[:, :1, :], # cls_tokens
                prompts,
                out[:, 1:, :]  # patch embeddings
            ), dim=1)
         
        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

    def forward_VoP(self, x, einops_from, einops_to, einops_dims, n_seg=4):
        # position-specific prompts for spatial attention
        h = self.num_heads
        num_tokens = self.num_tokens

        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1) # B x (1+p+NT) x hd
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)) # Bh x (1+p+NT) x d

        q *= self.scale

        # splice out CLS token at index 1 and prompts
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:num_tokens+1], t[:, num_tokens+1:]), (q, k, v)) # Bh x () x d
        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q[:, :1, :], k, v) # cls token: Bh x 1 x d

        # segment prompts into s segments in time
        pstep = num_tokens // n_seg
        pseg = [range(st, en) for st, en in zip(range(1, num_tokens+1, pstep), range(pstep+1, num_tokens+2, pstep))]
        p_q, p_k, p_v = map(lambda t: rearrange(t[:, pseg, :], 'b s p d -> (b s) p d'), (cls_q, cls_k, cls_v)) # prompt query: (Bh x n_seg) x p_per_seg x d

        # segment patch embeddings into s segments in time
        q_, k_, v_ = map(lambda t: rearrange(t, 'b (f n) d -> b f n d', **einops_dims), (q_, k_, v_)) # Bh x T x N x d
        num_frames = k_.size(1)
        tstep = num_frames // n_seg
        tseg = [range(st, en) for st, en in zip(range(0, num_frames, tstep), range(tstep, num_frames+1, tstep))]
        q_, k_, v_ = map(lambda t: t[:, tseg, ...], (q_, k_, v_)) # Bh x n_seg x f_per_seg x n x d
        q_, k_, v_ = map(lambda t: rearrange(t, 'b s f n d -> (b s) (f n) d'), (q_, k_, v_)) # (Bh x n_seg) x (f_per_seg x n) x d

        # concatenate prompts and patch embeddings
        k_, v_ = map(lambda t: torch.cat((t[0], t[1]), dim=1), ((p_k, k_), (p_v, v_)))
        p_out = attn(p_q, k_, v_) # (Bh x n_seg) x p_per_seg x d
        out = attn(q_, k_, v_)    # (Bh x n_seg) x (f_per_seg x n) x d
        p_out = rearrange(p_out, '(b s) p d -> b (s p) d', s=n_seg) # Bh x p x d
        out = rearrange(out, '(b s) (f n) d -> b (s f n) d', s=n_seg, f=tstep) # Bh x NT x d

        # merge tokens
        out = torch.cat((cls_out, p_out, out), dim=1) # Bh x (1+p+NT) x d
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # B x (NT+1) x hd

        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x
    
    def forward_attall(self, x, pt_att=True):
        h = self.num_heads
        if self.num_tokens > 0 and not pt_att:
            prompts = x[:, 1:self.num_tokens+1, :]
            x = torch.cat((
                x[:, :1, :], # cls_token
                x[:, self.num_tokens+1:, :] # patch embeddings
            ), dim=1)

        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # all tokens attend to all tokens
        out = attn(q, k, v)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # B x (1 + p + NT) x hd
        if self.num_tokens > 0 and not pt_att:
            out = torch.cat((
                out[:, :1, :], # cls_tokens
                prompts,
                out[:, 1:, :]  # patch embeddings
            ), dim=1)

        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', is_tanh_gating=False, num_tokens=0, split_st=False):
        super().__init__()

        self.split_st = split_st # split spatial and temporal prompts
        if split_st:
            num_tokens = num_tokens // 2
            self.num_tokens = num_tokens # learnable prompts

        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_tokens=num_tokens)

        self.timeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_tokens=num_tokens,
            initialize=time_init)

        if is_tanh_gating:
            self.alpha_timeattn = nn.Parameter(torch.zeros([]))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style

    def forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
                time_n, space_f, use_checkpoint=False, pt_spt=True, pt_tmp=True, style='default', n_seg=4):
        if self.split_st:
            spatial_prompts = x[:, 1:self.num_tokens+1, :]
            x = torch.cat((
                x[:, :1, :], # cls_token
                x[:, self.num_tokens+1:, :] # temporal prompts and patch embeddings
            ), dim=1)

        if use_checkpoint:
            time_output = checkpoint.checkpoint(
                self.timeattn, self.norm3(x), einops_from_time, einops_to_time, {"n": time_n}, {'pt_att': pt_tmp}
            )
        else:
            time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, {"n": time_n}, {'pt_att': pt_tmp})
        if hasattr(self, "alpha_timeattn"):
            time_output = torch.tanh(self.alpha_timeattn) * time_output
        time_residual = x + time_output

        if self.split_st:
            temporal_prompts = time_residual[:, 1:self.num_tokens+1, :]
            time_residual = torch.cat((
                time_residual[:, :1, :], # cls_token
                spatial_prompts,
                time_residual[:, self.num_tokens+1:, :] # patch embeddings
            ), dim=1)

        cfg = {'style': style, 'pt_att': pt_spt, 'n_seg': n_seg}
        if use_checkpoint:
            space_output = checkpoint.checkpoint(
                self.attn, self.norm1(time_residual), einops_from_space, einops_to_space, {"f": space_f}, cfg
            )
        else:
            space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                     einops_to_space, {"f": space_f}, cfg)
        if self.attention_style == 'frozen-in-time':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        if self.split_st:
            space_residual = torch.cat((
                space_residual[:, :self.num_tokens+1, :], # cls_token and spacial prompts
                temporal_prompts,
                space_residual[:, self.num_tokens+1:, :]  # patch embeddings
            ), dim=1)

        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        return x


class SpaceTimeTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650
    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
                 act_layer=nn.GELU, is_tanh_gating=False, tune_bias=False, prompt_cfg={}):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.tune_bias = tune_bias
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        print("######USING ATTENTION STYLE: ", attention_style)
        self.param_list = []
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
            self.param_list += list(self.patch_embed.parameters())
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.param_list += [self.cls_token, self.pos_embed, self.temporal_embed]

        if ln_pre:
            self.ln_pre = nn.LayerNorm(embed_dim)
            if self.tune_bias:
                self.param_list += [m for n, m in self.ln_pre.named_parameters() if 'bias' not in n]
            else:
                self.param_list += list(self.ln_pre.parameters())
        else:
            self.ln_pre = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # config for prompts
        self.num_tokens = prompt_cfg.get('num_tokens', 0)
        self.prompt_dim = prompt_cfg.get('prompt_dim', 768)
        self.pt_spt = prompt_cfg.pop('pt_spt', True)
        self.pt_tmp = prompt_cfg.pop('pt_tmp', True)
        self.style = prompt_cfg.pop('style', 'default')
        self.query = prompt_cfg.pop('query', 'cls')
        self.n_seg = prompt_cfg.pop('n_seg', 4)
        self.k_s = prompt_cfg.pop('K_s', depth)
        self.st = prompt_cfg.pop('st', 0)
        self.end = prompt_cfg.pop('end', depth)
        assert self.st <= self.end
        if self.style == 'default':
            print(f'Prompting {self.st}-{self.end} layer of the visual backbone')
        elif self.style == 'VoP_c' and self.k_s < depth:
            self.prompt_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        elif self.style == 'VoP_c_basis':
            self.prompt_temp_embed = nn.Parameter(torch.zeros(1, self.n_seg, embed_dim))
            trunc_normal_(self.prompt_temp_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for i in range(depth):
            stblk_cfg = {}
            if self.num_tokens > 0:
                stblk_cfg = {'num_tokens': prompt_cfg['num_tokens'], 'split_st': prompt_cfg.get('split_st', False)}
            blocks.append(
                SpaceTimeBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                    attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating, **stblk_cfg)
            )

        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)
        if self.tune_bias:
            self.param_list += reduce(operator.add, [[m for n, m in x.named_parameters() if 'bias' not in n] for x in self.blocks])
            self.param_list += [m for n, m in self.norm.named_parameters() if 'bias' not in n]
        else:
            self.param_list += reduce(operator.add, [list(x.parameters()) for x in self.blocks])
            self.param_list += list(self.norm.parameters())

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
            if self.tune_bias:
                self.param_list += [m for n, m in self.pre_logits.named_parameters() if 'bias' not in n]
            else:
                self.param_list += list(self.pre_logits.parameters())
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'
         
        # freeze the backbone and only learn the prompts
        self.prompt_learner = None
        if self.num_tokens > 0:
            if 'VoP_c' in self.style:
                basis = prompt_cfg.pop('basis', {}) if 'basis' in self.style else {}
                if self.k_s > 0:
                    self.prompt_generator = CMM(self.num_tokens // self.n_seg, self.n_seg, embed_dim, self.prompt_dim, num_layer=self.k_s, \
                        shared=prompt_cfg.get('deep_shared', False), basis=basis)
                n_prompt_layer = depth - self.k_s

            else:
                n_prompt_layer = self.end - self.st

            if n_prompt_layer > 0:
                prompt_cfg['num_layers'] = n_prompt_layer
                prompt_cfg['prompt_dim'] = embed_dim
                self.prompt_learner = VisualPromptLearner(patch_size, embed_dim, **prompt_cfg)

            for p in self.param_list:
                p.requies_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, use_checkpoint=False, cls_at_last=True, istrain=False, gamma=1.0):
        # print(x.shape)
        b, curr_frames, channels, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1) # 1 x (NT + 1) x D

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches] # B x (NT + 1) x D
        ps_loss = x.new_zeros([1])
        # incorporate prompts
        if self.num_tokens > 0:
            if 'VoP_c' in self.style and self.k_s > 0:
                ctx, ps = self.prompt_generator(x[:, 1:, :], 0, istrain=istrain, gamma=gamma)
                ps_loss += ps
                if self.prompt_generator.use_basis:
                    prompt_temp_embed = self.prompt_temp_embed.repeat_interleave(self.num_tokens // self.n_seg, 1)
                    ctx = ctx + prompt_temp_embed

            elif self.prompt_learner is not None:
                ctx, ps = self.prompt_learner(x[:, :1, :], 0, istrain=istrain, gamma=gamma)
                ps_loss += ps
                if ctx.size(0) != BF:
                    ctx = ctx.expand(BF, -1, -1)

            x = torch.cat((
                x[:, :1, :], # cls_token
                ctx,
                x[:, 1:, :]
            ), dim=1)

        if self.ln_pre is not None:
            x = self.ln_pre(x)
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for i, blk in enumerate(self.blocks):
            if self.num_tokens > 0 and i > 0 and i >= self.st and i < self.end:
                if 'VoP_c' in self.style:
                    if i < self.k_s:
                        ctx, ps = self.prompt_generator(x[:, self.num_tokens+1:, :], i, istrain=istrain, gamma=gamma)
                        ps_loss += ps
                        if self.prompt_generator.use_basis:
                            prompt_temp_embed = self.prompt_temp_embed.repeat_interleave(self.num_tokens // self.n_seg, 1)
                            ctx = ctx + prompt_temp_embed
                    else:
                        ctx, ps = self.prompt_learner(x[:, :1, :], i-self.k_s, istrain=istrain, gamma=gamma)
                        ps_loss += ps

                        if 'basis' in self.style:
                            prompt_embed = self.prompt_temp_embed.repeat_interleave(self.num_tokens // self.n_seg, 1)
                        else:
                            prompt_embed = self.prompt_embed.repeat_interleave(self.num_tokens // self.num_frames, 1)
                        ctx = ctx + prompt_embed
                        if ctx.size(0) != BF:
                            ctx = ctx.expand(BF, -1, -1)

                elif (i - self.st) < self.prompt_learner.num_layers:
                    ctx, ps = self.prompt_learner(x[:, :1, :], i-self.st, istrain=istrain, gamma=gamma)
                    ps_loss += ps
                    if ctx.size(0) != BF:
                        ctx = ctx.expand(BF, -1, -1)

                x = torch.cat((
                    x[:, :1, :], # cls_token
                    ctx,
                    x[:, self.num_tokens+1:, :]
                ), dim=1)
                
            style = 'default' if i >= self.k_s else self.style
            pt_tmp = self.pt_tmp if i >= self.st and i < self.end else False
            pt_spt = self.pt_spt if i >= self.st and i < self.end else False
            x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f, use_checkpoint=use_checkpoint, pt_spt=pt_spt,
                    pt_tmp=pt_tmp, style=style, n_seg=self.n_seg)

        if cls_at_last:
            x = self.norm(x)
            x = x[:, 0]
            x = self.pre_logits(x)

            return x, ps_loss
        else:
            return self.norm(x), ps_loss

    def forward(self, x, use_checkpoint=False, istrain=False, gamma=1.0):
        # Note:  B C T H W => B T C H W
        # The default input order is different from the one in Frozen-in-Time
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x, ps_loss = self.forward_features(x, use_checkpoint=use_checkpoint, istrain=istrain, gamma=gamma)
        x = self.head(x)

        return x, ps_loss

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for m in self.modules():
            m.training = mode

        if mode and self.num_tokens > 0:
            for n, m in self.named_modules():
                if 'prompt' not in n:
                    m.training = False


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import lavila.models.loss as loss
from lavila.models.openai_clip import load as load_openai_clip
from lavila.models.openai_model import QuickGELU, Transformer
from lavila.models.timesformer import SpaceTimeTransformer
from lavila.models.utils import remap_keys, rsetattr
from lavila.models.prompt_tuning import PromptLearner



class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(vision_model.num_features, num_classes, bias=True)

        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, image, use_checkpoint=False, istrain=False, gamma=1.0):
        image_embed, ps_loss = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit, ps_loss


class VideoClassifierMultiHead(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.ModuleList(
            [nn.Linear(vision_model.num_features, num_classes, bias=True) for num_classes in num_classes_list]
        )

        for m in self.fc_cls:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

    def forward(self, image, use_checkpoint=False, istrain=False, gamma=1.0):
        image_embed, ps_loss = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
        return logit_list, ps_loss


class CLIP(nn.Module):
    def __init__(self,
                 cfg,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width
        self.tune_bias = cfg.get('tune_bias', False)
        self.freeze_vis_backbone = cfg.get('freeze_vis_backbone', False)
        self.freeze_txt_backbone = cfg.get('freeze_txt_backbone', False)

        self.visual = vision_model
        self.t_step = cfg.get('t_step', self.visual.num_frames)
        txt_prompt_cfg = cfg.get('text_prompt', {})
        self.n_ctx = txt_prompt_cfg.get('n_ctx', 0)
        self.txt_use_basis = txt_prompt_cfg.get('use_basis', False)
        if self.txt_use_basis:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompt_cfg=txt_prompt_cfg,
                prompt_learner=PromptLearner(transformer_width, self.n_ctx),
                prompt_generator=self.visual.prompt_generator
            )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompt_cfg=txt_prompt_cfg,
                prompt_learner=PromptLearner(transformer_width, self.n_ctx)
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

        freeze_list = []
        if self.freeze_vis_backbone:
            print("=> Freeze visual backbone")
            freeze_list += self.visual.param_list + [self.image_projection]
       
        if self.freeze_txt_backbone:
            print("=> Freeze text backbone")
            if self.tune_bias:
                freeze_list += [m for n, m in self.transformer.named_parameters() if 'prompt' not in n and 'bias' not in n]
                freeze_list += [m for n, m in self.ln_final.named_parameters() if 'bias' not in n]
            else:
                freeze_list += [m for n, m in self.transformer.named_parameters() if 'prompt' not in n]
                freeze_list += list(self.ln_final.parameters())
            freeze_list += list(self.token_embedding.parameters())
            freeze_list += [self.positional_embedding] + [self.text_projection]

        for p in freeze_list:
            p.requires_grad = False

        # text prompts
        if self.n_ctx > 0:
            if self.txt_use_basis:
                prompt_dim = self.visual.prompt_dim
                if prompt_dim != transformer_width:
                    self.transformer.prompt_inproj = nn.Linear(transformer_width, prompt_dim, bias=False)
                else:
                    self.transformer.prompt_inproj = nn.Identity()
                self.transformer.prompt_outproj = nn.Linear(prompt_dim, transformer_width, bias=False)
                nn.init.kaiming_normal_(
                    self.transformer.prompt_outproj.weight, a=0, mode='fan_out')
       
        params_to_update = [n for n, m in self.named_parameters() if m.requires_grad]
        num_opt_params = sum([m.numel() for m in self.parameters() if m.requires_grad])
        num_fz_params = sum([m.numel() for m in self.parameters() if not m.requires_grad])
        print("=> Params to update: {}".format(params_to_update))
        print("=> Update/Frozen: {}/{}".format(num_opt_params, num_fz_params))

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True, istrain=False, gamma=1.0):
        x, ps_loss = self.visual(image, use_checkpoint=use_checkpoint, istrain=istrain, gamma=gamma)

        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if apply_project:
            x = x @ self.image_projection

        return x, ps_loss

    def encode_text(self, text, use_checkpoint=False, istrain=False, gamma=1.0):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        B = x.shape[0]
        eot = text.argmax(dim=-1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, ps_loss = self.transformer(x, self.positional_embedding, use_checkpoint=use_checkpoint, istrain=istrain, gamma=gamma, eot=eot)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.n_ctx + eot] @ self.text_projection

        return x, ps_loss

    def forward(self, image, text, use_checkpoint=False, norm_embed=False, istrain=False, gamma=1.0):
        image_embed, ps_loss_img = self.encode_image(image, use_checkpoint=use_checkpoint, istrain=istrain, gamma=gamma)
        text_embed, ps_loss_txt = self.encode_text(text, use_checkpoint=use_checkpoint, istrain=istrain, gamma=gamma)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'ps_loss': ps_loss_img + ps_loss_txt}

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for m in self.modules():
            m.training = mode

        if mode:
            if self.freeze_vis_backbone and not self.tune_bias:
                for n, m in self.visual.named_modules():
                    if 'prompt' not in n:
                        m.training = False

            if self.freeze_txt_backbone and not self.tune_bias:
                for n, m in self.transformer.named_modules():
                    if 'prompt' not in n:
                        m.training = False
                        
                self.token_embedding.training = False
                self.ln_final.training = False


def get_loss(model, args, tokenizer=None):
    if model.startswith('CLIP'):
        return loss.CLIPLoss(
            use_vissl=args.contrastive_use_vissl,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        raise NotImplementedError


def get_metric_names(model):
    if model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    else:
        raise NotImplementedError


def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, temperature_init=0.07,
    project_embed_dim=256, **kwargs
):
    cfg = kwargs.pop('model_cfg', {})
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=cfg.get('drop_path_rate', 0),
        tune_bias=cfg.get('tune_bias', False),
        prompt_cfg=cfg.get('visual_prompt', {})
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        cfg,
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict(), strict=False)
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model



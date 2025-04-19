
import math
from functools import reduce
from operator import mul
from einops import rearrange, repeat
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLearner(nn.Module):
    def __init__(self, ctx_dim=512, n_ctx=16):
        super(PromptLearner, self).__init__()
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

        # initialize prompts
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

    def forward(self):
        return self.ctx


class PromptBasisLearner(nn.Module):
    def __init__(self, prompt_dim=256, size=8):
        super(PromptBasisLearner, self).__init__()
        self.prompt_dim = prompt_dim
        self.size = size

        # initialize prompts
        self.prompt_values = nn.Parameter(torch.zeros(size, 1, prompt_dim))
        self.id_table = torch.ones([size]).cuda()

        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_values.data, -1, 1)

    def forward(self, query, k=0, istrain=False, gamma=1.0):
        BZ = query.shape[0]
        out = dict()
        query = F.normalize(query.squeeze(1), p=2, dim=1)
        keys = self.prompt_values.mean(dim=1)
        keys = F.normalize(keys, p=2, dim=1)
        similarity = torch.matmul(query, keys.t())

        if k > 0 and k < self.size:

            if istrain:
                inv_freq = self.id_table.sum() / self.id_table.float()
                weights = (similarity + 1) / 2 * gamma + (1 - gamma) * torch.softmax(inv_freq, dim=-1)
                idx = torch.multinomial(weights, k, replacement=False)
            else:
                idx = torch.argsort(similarity, dim=-1, descending=True)[:, :k]

            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            self.id_table[prompt_id] += id_counts
            prompts = self.prompt_values[idx.flatten(), ...].view(BZ, k, self.prompt_dim)
        else:
            idx = torch.arange(self.size).unsqueeze(0).expand(BZ, -1)
            prompts = self.prompt_values.flatten(0, 1).unsqueeze(0).expand(BZ, -1, -1)

        prompts = F.normalize(prompts, p=2, dim=-1)
        out['prompts'] = prompts
        sel_sim = similarity[torch.arange(BZ).view(-1, 1), idx]
        sel_key = keys[idx.flatten(), ...].view(BZ, k, self.prompt_dim)
        diff = F.mse_loss((sel_sim.unsqueeze(1) @ sel_key).squeeze(1), query.detach(), reduction='sum') / BZ
        ksim = torch.sum(torch.abs(torch.matmul(keys, keys.t()) - torch.eye(self.size).to(keys.device))) / BZ
        out['ps_loss'] = diff + ksim

        return out


class VisualPromptLearner(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_layers=12, prompt_dim=256, num_tokens=5, deep=False,
            deep_shared=False, split_st=False, dropout=0.1, basis={}):
        super(VisualPromptLearner, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.prompt_dim = prompt_dim
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(dropout)
        basis_size = basis.get('size', 0)
        self.use_basis = True if basis_size > 0 and num_tokens <= basis_size else False
        if self.use_basis:
            print(f'Using {self.basis_size} basis prompts (dimension: {prompt_dim})')

        if prompt_dim != embed_dim:
            self.prompt_inproj = nn.Linear(embed_dim, prompt_dim, bias=False)
        else:
            self.prompt_inproj = nn.Identity()

        if self.use_basis:
            self.prompt_outproj = nn.Linear(prompt_dim, embed_dim, bias=False)
            nn.init.kaiming_normal_(
                self.prompt_outproj.weight, a=0, mode='fan_out')
        else:
            self.prompt_outproj = nn.Identity()

        self.split_st = split_st # split spatial and temporal prompts

        # initialize prompts
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))
        if split_st:
            if self.use_basis:
                basis['size'] //= 2
                self.spatial_prompt_basis = PromptBasisLearner(prompt_dim, **basis)
                self.temporal_prompt_basis = PromptBasisLearner(prompt_dim, **basis)
            else:
                self.spatial_prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens // 2, prompt_dim))
                self.temporal_prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens // 2, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.spatial_prompt_embeddings.data, -val, val)
                nn.init.uniform_(self.temporal_prompt_embeddings.data, -val, val)
        else:
            if self.use_basis:
                self.prompt_basis = PromptBasisLearner(prompt_dim, **basis)
            else:
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.deep = deep or deep_shared
        self.deep_shared = deep_shared
        if deep and (not deep_shared):
            total_d_layer = num_layers - 1
            if split_st:
                if self.use_basis:
                    self.spatial_deep_prompt_basis = nn.ModuleList([
                        PromptBasisLearner(prompt_dim, **basis)
                        for i in range(total_d_layer)])
                    self.temporal_deep_prompt_basis = nn.ModuleList([
                        PromptBasisLearner(prompt_dim, **basis)
                        for i in range(total_d_layer)])
                else:
                    self.spatial_deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens // 2, prompt_dim))
                    self.temporal_deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens // 2, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.spatial_deep_prompt_embeddings.data, -val, val)
                    nn.init.uniform_(self.temporal_deep_prompt_embeddings.data, -val, val)
            else:
                if self.use_basis:
                    self.deep_prompt_basis = nn.ModuleList([
                        PromptBasisLearner(prompt_dim, **basis)
                        for i in range(total_d_layer)])
                else:
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def forward(self, query=None, layer=0, istrain=False, gamma=1.0):
        query = query.detach()
        query = self.prompt_inproj(query)
        ps_loss = query.new_zeros([1])
        if self.split_st:
            if self.deep and (not self.deep_shared) and layer > 0:
                if self.use_basis:
                    k = (self.num_tokens // 2)
                    spatial_out = self.spatial_deep_prompt_basis[layer-1](query, k, istrain, gamma)
                    spatial_prompts = spatial_out['prompts']
                    temporal_out = self.temporal_deep_prompt_basis[layer-1](query, k, istrain, gamma)
                    temporal_prompts = temporal_out['prompts']
                    ps_loss += spatial_out.get('ps_loss', 0) + temporal_out.get('ps_loss', 0)
                else:
                    spatial_prompts = self.spatial_deep_prompt_embeddings[layer-1]
                    temporal_prompts = self.temporal_deep_prompt_embeddings[layer-1]
            else:
                if self.use_basis:
                    k = (self.num_tokens // 2)
                    spatial_out = self.spatial_prompt_basis(query, k, istrain, gamma)
                    spatial_prompts = spatial_out['prompts']
                    temporal_out = self.temporal_prompt_basis(query, k, istrain, gamma)
                    temporal_prompts = temporal_out['prompts']
                    ps_loss += spatial_out.get('ps_loss', 0) + temporal_out.get('ps_loss', 0)
                else:
                    spatial_prompts = self.spatial_prompt_embeddings
                    temporal_prompts = self.temporal_prompt_embeddings

            prompts = torch.cat((spatial_prompts, temporal_prompts), dim=1)

        else:
            if self.deep and (not self.deep_shared) and layer > 0:
                if self.use_basis:
                    k = self.num_tokens
                    out = self.deep_prompt_basis[layer-1](query, k, istrain, gamma)
                    prompts = out['prompts']
                    ps_loss += out.get('ps_loss', 0)
                else:
                    prompts = self.deep_prompt_embeddings[layer-1]
            else:
                if self.use_basis:
                    k = self.num_tokens
                    out = self.prompt_basis(query, k, istrain, gamma)
                    prompts = out['prompts']
                    ps_loss += out.get('ps_loss', 0)
                else:
                    prompts = self.prompt_embeddings

        prompts = self.prompt_dropout(self.prompt_outproj(prompts))
        return prompts, ps_loss


class CMM(nn.Module):
    '''Context modeling module'''
    def __init__(self, num_tokens=8, num_frames=16, embed_dim=768, prompt_dim=256, dropout=0., num_layer=1, shared=False, basis={}):
        super(CMM, self).__init__()
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.prompt_dim = prompt_dim
        self.basis_size = basis.get('size', 0)
        self.use_basis = True if self.basis_size > 0 else False
        self.use_rnn = not self.use_basis
        if self.use_rnn:
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.shared = shared
        self.prompt_dropout = nn.Dropout(dropout)

        if self.use_basis:
            print(f'Using {self.basis_size} basis prompts (dimension: {prompt_dim})')
            if self.use_rnn:
                self.prompt_inproj = nn.Linear(embed_dim * 2, prompt_dim)
                nn.init.kaiming_normal_(
                    self.prompt_inproj.weight, a=0, mode='fan_out')
            else:
                if embed_dim != prompt_dim:
                    self.prompt_inproj = nn.Linear(embed_dim, prompt_dim, bias=False)
                else:
                    self.prompt_inproj = nn.Identity()

            self.prompt_outproj = nn.Linear(prompt_dim, embed_dim, bias=False)
            nn.init.kaiming_normal_(
                self.prompt_outproj.weight, a=0, mode='fan_out')

            if shared:
                self.prompt_basis = PromptBasisLearner(prompt_dim, **basis)
            else:
                self.prompt_basis = nn.ModuleList([
                    PromptBasisLearner(prompt_dim, **basis)
                    for i in range(num_layer)])
        else:
            self.fc = nn.Linear(embed_dim * 2, embed_dim * num_tokens)

    def forward(self, x, layer=0, istrain=False, gamma=1.0):
        BZ = x.size(0)
        x = x.detach()
        x = rearrange(x, 'b (f n) d -> b f n d', f=self.num_frames)
        x = torch.mean(x, dim=2)

        if self.use_rnn:
            x, _ = self.rnn(x)

        ps_loss = x.new_zeros([1])
        if self.use_basis:
            query = self.prompt_inproj(x).flatten(0, 1)
            k = self.num_tokens
            if self.shared:
                out = self.prompt_basis(query, k, istrain, gamma)
            else:
                out = self.prompt_basis[layer](query, k, istrain, gamma)

            prompts = rearrange(out['prompts'], '(b f) p d -> b (f p) d', f=self.num_frames)
            prompts = self.prompt_outproj(prompts)
            ps_loss += out.get('ps_loss', 0) * self.num_frames

        else:
            prompts = self.fc(x)
            prompts = rearrange(prompts, 'b f (p d) -> b (f p) d', p=self.num_tokens)

        return prompts, ps_loss



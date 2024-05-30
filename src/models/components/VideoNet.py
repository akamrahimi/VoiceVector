# Copyright (c) VGG, University of Oxford.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: akamrahimi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torch.cuda.amp import autocast
from einops.layers.torch import Rearrange


def video_backbone():
    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[
                        8, 8], patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)
    model = VTP_wrapper(CNN_3d_featextractor(d_model=512, till=24), linear_pooler, in_dim=128, out_dim=512)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

#### visual parts

class Conv3d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=True, residual=False, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

### VTP modules:

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats):
        super().__init__()
        self.row_embed = nn.Embedding(64, num_pos_feats)
        self.col_embed = nn.Embedding(64, num_pos_feats)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos

class CNN_3d_featextractor(nn.Module):
    def __init__(self, d_model, till):
        super().__init__()
        layers = [Conv3d(3, 64, kernel_size=5, stride=(1, 2, 2), padding=2)]  # 48, 48

        if till <= 24:
            layers.extend([Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 24, 24
                Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])
        
        if till <= 12:
            layers.extend([Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 12, 12
                    Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
                    Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])

        if till == 6:
            layers.extend([Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 6, 6
                        Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
                        Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])

        self.encoder = nn.Sequential(*layers)

    @autocast()
    def forward(self, faces, mask):
        assert faces.size(3) == 96
        assert faces.size(-1) == 96

        face_embeddings = self.encoder(faces) # (B, C, T, H, W)

        return face_embeddings

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                            act_layer=nn.ReLU, drop=0.1):
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

class VTP_wrapper(nn.Module):
    def __init__(self, feat_extrator, encoder, in_dim, out_dim):
        super().__init__()
        self.feat_extrator = feat_extrator

        self.hwposition = PositionEmbeddingLearned(in_dim//2)

        self.pooler = nn.Linear(out_dim, 1)

        self.encoder = encoder

    @autocast()
    def forward(self, faces, mask):
        faces = self.feat_extrator(faces, mask)
        faces = faces.transpose(1, 2) # (B, T, C, H, W)
        lens = mask.long().sum((1, 2)) # (B,)
        indiv_faces = []
        for l, fs in zip(lens, faces):
            indiv_faces.append(fs[:l])

        face_tokens = torch.cat(indiv_faces, dim=0) # (B*, C, H, W)

        face_tokens = self.hwposition(face_tokens)

        face_embeddings = self.encoder(face_tokens) # (B, hw, C)

        pooling_weights = F.softmax(self.pooler(face_embeddings), dim=1) # (B, hw, 1)
        self.face_attn_weights = pooling_weights

        face_embeddings = (face_embeddings * pooling_weights).sum(1) # (B, C)

        video_embeddings = []
        max_len = faces.size(1)
        start = 0

        for l in lens:
            cur_face_embeddings = face_embeddings[start : start + l]
            if l != max_len:
                cur_face_embeddings = torch.cat([cur_face_embeddings, torch.zeros((max_len - l, 
                                    cur_face_embeddings.size(1)), 
                                    device=cur_face_embeddings.device)], dim=0)
            start += l
            video_embeddings.append(cur_face_embeddings)
            
        video_embeddings = torch.stack(video_embeddings, dim=0)
        return video_embeddings # (B, T, C)

class VTP(nn.Module):   
    def __init__(self, num_layers, dims, heads, patch_sizes, initial_resolution=48, 
                initial_dim=64, trans_block=LinearAttentionTransformer):
        '''
            Num layers per block, dim and #heads for the layers of each block, 
                patch sizes for downsampling
        ''' 
        super().__init__()

        self.transformer_blocks = nn.ModuleList([])
        self.patch_projectors = nn.ModuleList([]) # converts spatial maps to patches

        cur_dim = initial_dim
        cur_res = initial_resolution
        for l, dim, h, p in zip(num_layers, dims, heads, patch_sizes):
            input_dim = (cur_dim * p * p)
            if input_dim == dim:
                self.patch_projectors.append(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                        p1 = p, p2 = p))
            else:
                self.patch_projectors.append(nn.Sequential(
                                                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                    p1 = p, p2 = p),
                                                 Mlp(input_dim, dim, dim)))
            cur_res /= p
            self.transformer_blocks.append(
                                    trans_block(dim=dim, 
                                    max_seq_len=(cur_res * cur_res),
                                    heads=h, depth=l, ff_dropout = 0.1, 
                                    attn_layer_dropout = 0.1, attn_dropout = 0.1))
            cur_dim = dim

    @autocast()
    def forward(self, face_tokens, cls_token=None):
        x = face_tokens
        for i, (patch_maker, transformer) in enumerate(zip(self.patch_projectors, 
                                                            self.transformer_blocks)):
            x = patch_maker(x)
            if i == len(self.patch_projectors) - 1 and cls_token is not None:
                x = torch.cat([cls_token, x], dim=1)

            x = transformer(x)

            if i != len(self.patch_projectors) - 1:
                r = int(x.size(1) ** 0.5)
                feature_map_projector = Rearrange('b (h w) c -> b c h w', 
                                                    h=r, w=r)
                x = feature_map_projector(x)

        return x

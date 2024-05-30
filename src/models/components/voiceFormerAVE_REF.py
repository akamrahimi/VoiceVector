# Copyright (c) University of Oxford, VGG Group.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: akamrahimi

import math
import torch as th
from torch import nn
import torchaudio
from torch.cuda.amp import autocast
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from models.components.VideoNet import video_backbone

class PositionalEncoding2(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 400, resample=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        
        if resample > 1:
            pe = pe.permute(2,1,0)
            upsample = nn.Upsample(scale_factor=resample, mode='nearest')
            pe = upsample(pe)
            pe = pe.permute(2,1,0)
            
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, ,_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, out_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder_vid = PositionalEncoding2(d_model, dropout, max_len=1500, resample=1)
        self.pos_encoder_aud = PositionalEncoding2(d_model, dropout, max_len=1500, resample=2)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, aud_feat, v_feat, src_mask=None):
        """
        Args:
            encoded_src: Tensor, shape [seq_len, batch_size, features]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, out_size]
        """
        
        encoded_aud_feat = self.pos_encoder_vid(aud_feat)
        # encoded_v_feat = self.pos_encoder_aud(v_feat)
        encoded_src = th.cat([v_feat, encoded_aud_feat], 0)
        output = self.transformer_encoder(encoded_src, src_mask)
        output = self.decoder(output)
        return output

class VoiceFormerAVE_REF(nn.Module):
    """
   
    """
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 padding=2,
                 resample=3.2,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 floor=1e-3,
                 video_chin=512,
                 d_hid = 532, 
                 num_encoder_layers = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 num_heads = 8,  # number of heads in nn.MultiheadAttention
                 ): 

        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
  
        self.video_chin = video_chin
        self.norm_layers = nn.ModuleList()
        self.norm_layers2 = nn.ModuleList()
        self.adaptive_layer = nn.ModuleList()
        self.refinement_layer = nn.ModuleList()
        self.upsample = torchaudio.transforms.Resample(orig_freq= 16000, new_freq= 51200)
        self.downsample = torchaudio.transforms.Resample(orig_freq= 51200, new_freq= 16000)
   
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride, padding),
            ]
            if index > 0:
                decode.append(nn.ReLU())
                
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden) 
        
        refinement = [
            nn.Conv1d(1, 1,520,1,259), nn.ReLU(),
            nn.Conv1d(1, 1,520,1,260)
        ]
        self.refinement_layer.append(nn.Sequential(*refinement))
        
        norm = [ 
            nn.Conv1d(self.chin,self.chin,kernel_size=1,stride=1),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
        ]
        self.norm_layers.append(nn.Sequential(*norm))
        
        adaptive = [ 
            nn.Conv1d(192,384,kernel_size=1,stride=1),
            nn.GroupNorm(1, 384),
            nn.Conv1d(384,768,kernel_size=1,stride=1),
            nn.ReLU(), 
        ]
        self.adaptive_layer.append(nn.Sequential(*adaptive))  
        
        
        self.transformer = TransformerModel(chin, chin, num_heads, d_hid, num_encoder_layers)
              
    def forward(self, audio, visual, speaker_embedding=None, use_video_backbone = False):
         
        with th.no_grad():
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)

            if self.normalize:
                mono = audio.mean(dim=1, keepdim=True)
                std = mono.std(dim=-1, keepdim=True)
                audio = audio / (self.floor + std)
            else:
                std = 1
                
        length = audio.shape[-1]
        x = audio  
        x = self.upsample(x)
        
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        feat = speaker_embedding
        # print('feat', feat.shape)
        
        for adapt in self.adaptive_layer:
            feat = adapt(feat)
            # print('feat', feat.shape)

        x = x.permute(2, 0, 1)
        # print('x', x.shape)
        feat = feat.permute(2, 0, 1)  
        # print('feat', feat.shape)
        x = self.transformer(x, feat)
        # print('x transformer', feat.shape)
        
        x = x.permute(1, 2, 0) 
        x = x[...,feat.shape[0]:]
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
            
        for refinement in self.refinement_layer:
            x = refinement(x)
            
        x = self.downsample(x)            
        x = x[..., :length]
        return x * std

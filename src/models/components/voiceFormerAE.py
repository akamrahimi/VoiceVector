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
        self.pos_encoder = PositionalEncoding2(d_model, dropout, max_len=1500, resample=1)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, aud_feat, sp_em_feat, src_mask=None):
        """
        Args:
            encoded_src: Tensor, shape [seq_len, batch_size, features]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, out_size]
        """
        
        encoded_aud_feat = self.pos_encoder(aud_feat)
        encoded_src = th.cat([sp_em_feat, encoded_aud_feat], 0)
        output = self.transformer_encoder(encoded_src, src_mask)
        output = self.decoder(output)
        return output

class VoiceFormerAE(nn.Module):
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
                 num_encoder_layers = 3,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
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
        self.adaptive_layer = nn.ModuleList()
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
        self.positive_negative_embedding = nn.Embedding(2, chin)
        self.transformer = TransformerModel(chin, chin, num_heads, d_hid, num_encoder_layers)
              
    def forward(self, mixture, audios, video_features, speaker_embedding=None, emmbeding_ids=None, face_embedding=None, target_feats=None):
        # this function accepts all available modalities and pass the relevant ones to the model
        # this is to simplify the code in conjuction with the lightning module
        return self.step(mixture, speaker_embedding, emmbeding_ids) 
    
    def step(self, audio, speaker_embedding, emmbeding_ids):
        with th.no_grad():
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
        # print('x init', audio.shape)  
        
        x = self.upsample(audio)
        # print('x upsample', x.shape)  
        
        for norm in self.norm_layers:
            x = norm(x)
        # print('x norm', x.shape)  
        
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
            
        # print('x encoded', x.shape)  
            
        x = x.permute(2, 0, 1)
        # print('x permute', x.shape)  
          
        # print('speaker_embedding init', speaker_embedding.shape)
        for adapt in self.adaptive_layer:
            speaker_embedding = adapt(speaker_embedding)
            # print('speaker_embedding adapt', speaker_embedding.shape)
            
        # print('speaker_embedding layers', speaker_embedding.shape)

        speaker_embedding = speaker_embedding.permute(2, 0, 1)  
        # print('speaker_embedding reshaped', speaker_embedding.shape)
        
        if emmbeding_ids is not None:
            # print('speaker_embedding emmbeding_ids', emmbeding_ids.shape)
            pne = self.positive_negative_embedding(emmbeding_ids).permute(1,0,2)
            # print('pne init', pne.shape)
            speaker_embedding = pne + speaker_embedding
            # print('speaker_embedding pne', speaker_embedding.shape)
        # print('speaker_embedding', speaker_embedding.shape)
       
        x = self.transformer(x, speaker_embedding)
        # print('x transformer', x.shape)
        
        x = x.permute(1, 2, 0) 
        x = x[...,speaker_embedding.shape[0]:]
        # print('x porstion of transformer', x.shape)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        # print('x decoded', x.shape)
        x = self.downsample(x)[..., :audio.shape[-1]]        
        # print('x downsample', x.shape)    
        return x

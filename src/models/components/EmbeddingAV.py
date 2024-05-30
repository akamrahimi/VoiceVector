import math
import torch as th
from torch import nn
import torchaudio
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from torch.cuda.amp import autocast
from src.models.components.ecapatdn import ECAPA_TDNN


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)
                
class PositionalEncoding2(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1500, resample=1):
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
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, out_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder_aud = PositionalEncoding2(d_model, dropout, max_len=1500, resample=1)
        self.pos_encoder_vid = PositionalEncoding2(d_model, dropout, max_len=1500, resample=2)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=True)
        
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
        encoded_aud_feat = self.pos_encoder_aud(aud_feat)
        encoded_v_feat = self.pos_encoder_vid(v_feat)
        encoded_src = th.cat([encoded_aud_feat, encoded_v_feat], 0)
        output = self.transformer_encoder(encoded_src, src_mask)
        output = self.decoder(output)
        return output


class EmbeddingAV(nn.Module):
    """ 
    VoiceFormer speech separation model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - floor (float): stability flooring when normalizing.
        - video_chin (int): number of input channels for video.
        - d_hid (int): hidden dimension in the transformer encoder layer.
        - nlayers (int): number of layers in the transformer encoder layer.
        - nhead (int): number of heads in the transformer encoder layer.
    
    """
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 padding=2,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 floor=1e-3,
                 video_chin=512,
                 d_hid = 532,
                 nlayers = 3, 
                 nhead = 8,
                 use_refine=True): 

        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.floor = floor
        self.normalize = normalize
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
  
        # ADDED
        self.use_refine = use_refine
        self.video_chin = video_chin
        self.norm_layers = nn.ModuleList()
        self.adaptive_layer = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.upsample = torchaudio.transforms.Resample(orig_freq= 16000, new_freq= 51200)
        self.downsample = torchaudio.transforms.Resample(orig_freq= 51200, new_freq= 16000)
        
        # -----
        
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
            
        # ADDED
        adaptive = [ 
            nn.Conv1d(self.video_chin,chin,kernel_size=1,stride=1),
            nn.GroupNorm(1,chin),
            nn.Dropout(0.1)
        ]
        self.adaptive_layer.append(nn.Sequential(*adaptive))

        norm = [
            nn.GroupNorm(1,chin)
        ]

        linear = [ 
            nn.Conv1d(chin,chin//2,kernel_size=1,stride=1,padding=0),
            nn.Dropout(0.1),
            nn.ReLU()
        ]
         
        self.linear_layers.append(nn.Sequential(*linear))

        self.norm_layers.append(nn.Sequential(*norm))
        self.transformer = TransformerModel(chin, chin, nhead, d_hid, nlayers)
        self.speaker_embedding = ECAPA_TDNN(chin//2, lin_neurons=192)
        rescale_module(self, reference=0.1)
    
    def forward(self, mixture, audios, video_features, speaker_embedding, emmbeding_ids):
        # this function accepts all available modalities and pass the relevant ones to the model
        # this is to simplify the code in conjuction with the lightning module
        return self.step(audios, video_features) 
    
    def step(self, mix, feat):

        with th.no_grad():
            if mix.dim() == 2:
                mix = mix.unsqueeze(1)

            if self.normalize:
                mono = mix.mean(dim=1, keepdim=True)
                std = mono.std(dim=-1, keepdim=True)
                mix = mix / (self.floor + std)
            else:
                std = 1
                
        x = mix  
        x = self.upsample(x)
                
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        for adapt in self.adaptive_layer:
            feat = adapt(feat)
        # print(x.shape, feat.shape)   
        for norm in self.norm_layers:
            x = norm(x)
   
        x = x.permute(2, 0, 1)
        feat = feat.permute(2, 0, 1) 
        x = self.transformer(x, feat)
        x = x.permute(1, 2, 0) 
        
        x = x[...,:x.shape[2] - feat.shape[0]]
        for linear in self.linear_layers:
          x = linear(x)
        x = x.permute(0, 2, 1)
    
        em = self.speaker_embedding(x)
        return em
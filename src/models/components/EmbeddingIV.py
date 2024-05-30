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
        self.pos_encoder = PositionalEncoding2(d_model, dropout, max_len=1500, resample=1)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=True)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, face_feat, vid_feat, src_mask=None):
        """
        Args:
            encoded_src: Tensor, shape [seq_len, batch_size, features]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, out_size]
        """
        # encoded_face_feat = self.pos_encoder(face_feat)
        encoded_vid_feat = self.pos_encoder(vid_feat)
        encoded_src = th.cat([face_feat, encoded_vid_feat], 0)
        output = self.transformer_encoder(encoded_src, src_mask)
        output = self.decoder(output)
        return output


class EmbeddingIV(nn.Module):
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
                 use_refine=True,
                 image_chin=128): 

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
        self.image_chin = image_chin
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
  
        # ADDED
        self.use_refine = use_refine
        self.video_chin = video_chin
        self.norm_layers = nn.ModuleList()
        self.adaptive_layer = nn.ModuleList()
        self.face_embedding_adaptive_layer = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        # -----
        
        chin = 768
        # ADDED
        adaptive = [ 
            nn.Conv1d(self.video_chin,chin,kernel_size=1,stride=1),
            nn.GroupNorm(1,chin),
            nn.Dropout(0.1)
        ]
        self.adaptive_layer.append(nn.Sequential(*adaptive))

        face_embedding_adaptive = [ 
            nn.Conv1d(self.image_chin,chin,kernel_size=1,stride=1),
            nn.GroupNorm(1,chin),
            nn.Dropout(0.1)
        ]
        self.face_embedding_adaptive_layer.append(nn.Sequential(*face_embedding_adaptive))
        
        linear = [ 
            nn.Conv1d(chin,chin//2,kernel_size=1,stride=1,padding=0),
            nn.Dropout(0.1),
            nn.ReLU()
        ]
        self.face_positional_embedding = nn.Embedding(1, chin)
         
        self.linear_layers.append(nn.Sequential(*linear))
        self.transformer = TransformerModel(chin, chin, nhead, d_hid, nlayers)
        self.speaker_embedding = ECAPA_TDNN(chin//2, lin_neurons=192)
        rescale_module(self, reference=0.1)
    
    def forward(self, mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embeddings):
        # this function accepts all available modalities and pass the relevant ones to the model
        # this is to simplify the code in conjuction with the lightning module
        return self.step(video_features, face_embeddings) 
    
    def step(self, visual_feat, face_embeddings):

        for i_adapt in self.face_embedding_adaptive_layer:
            face_embeddings = i_adapt(face_embeddings)
        # print('face embeddings',face_embeddings.shape)
        face_embeddings = face_embeddings.permute(2, 0, 1)
        # print('face embeddings permute',face_embeddings.shape)
        
        face_embeddings_idx = th.zeros(face_embeddings.shape[1], face_embeddings.shape[0], dtype=th.long).to(face_embeddings.device)
        # print('face face_embeddings_idx ',face_embeddings.shape)
       
        pne = self.face_positional_embedding(face_embeddings_idx).permute(1,0,2)
        # print('pne init', pne.shape)
        face_embeddings = pne + face_embeddings
        
        
        for adapt in self.adaptive_layer:
            visual_feat = adapt(visual_feat)
        visual_feat = visual_feat.permute(2, 0, 1) 
        # randomly change visual_feat to zeros to test if the model is working
        # if th.rand(1) > 0.2:
        #     visual_feat = th.zeros_like(visual_feat)
        x = self.transformer(face_embeddings, visual_feat)
        x = x.permute(1, 2, 0) 
        for linear in self.linear_layers:
          x = linear(x)
        x = x.permute(0, 2, 1)
    
        return self.speaker_embedding(x)
# Copyright (c) University of Oxford, VGG Group.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: akamrahimi
import torch
from src.models.components.EmbeddingIV2 import EmbeddingIV
from src.models.components.voiceFormerAE import VoiceFormerAE

class VoiceFormerIVEM(torch.nn.Module):
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
                num_encoder_layers = 3,
                num_heads = 8,  
                ): 

        super().__init__()
       
        self.iv_speaker_embedding = EmbeddingIV()
        self.speaker_separation = VoiceFormerAE()
        self.iv_speaker_embedding.eval()
    def forward(self, mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_feat=None):   
        with torch.no_grad():  
            bs = mixture.shape[0]
            iv_speaker_embedding = self.iv_speaker_embedding(mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding)
        iv_speaker_embedding = iv_speaker_embedding.reshape(bs, iv_speaker_embedding.shape[0]//bs, -1).permute(0,2,1)
        return self.speaker_separation(mixture, audios, video_features, iv_speaker_embedding, emmbeding_ids)

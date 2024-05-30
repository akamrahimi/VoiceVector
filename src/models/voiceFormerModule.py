from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.nn import functional as F
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, SignalDistortionRatio
from pathlib import Path
from src.utils.io import write, save_json
from src.utils.utils import get_metrics, eval_track, get_sdr
import numpy as np 
class VoiceFormerModule(LightningModule):
    """LightningModule for VoiceFormer model.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.sdr = SignalDistortionRatio()
        self.iii = 0
        self.net = net

        # loss function
        self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.nn.MSELoss()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        self.train_loss_best = MinMetric()
        
    def prepare_batch(self, batch: Any):
        mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_audio, speaker_id, target_feat, mixture_data = batch
        
        if not isinstance(speaker_embedding, list):
            speaker_embedding = speaker_embedding.squeeze(2).squeeze(2).permute(0, 2, 1)
        
        if not isinstance(face_embedding, list):
            batch_size = mixture.shape[0]
            face_embedding = face_embedding.reshape(batch_size*face_embedding.shape[1],face_embedding.shape[2], face_embedding.shape[-1]).permute(0,2,1)
        
        if not isinstance(video_features, list):
            video_features = video_features.reshape(-1, video_features.shape[2], video_features.shape[3])
            audios = audios.reshape(-1, audios.shape[-1]).unsqueeze(1)
        
        return mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_audio, speaker_id, target_feat, mixture_data
    
    def forward(self, mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_feat):        
        return self.net(mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_feat)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()
        self.val_loss.reset()
    
    def step(self, batch: Any):
       
        mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_audio, speaker_id, target_feat, mixture_data = self.prepare_batch(batch)
        predictions = self.forward(mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_feat)     
        loss = self.criterion(predictions, target_audio)
 
        # noise = mixture - target_audio
        # noise_p = mixture - predictions
        # loss2 = self.criterion(noise_p, noise)
        # loss = loss + loss2
        return loss, predictions, target_audio, speaker_id


    def training_step(self, batch: Any, batch_idx: int):
       
        loss, predictions, targets, speaker_id = self.step(batch)
      
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, predictions, targets, speaker_id = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # del self.batch
        self.batch = batch

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        
        loss, predictions, targets, speaker_id = self.step(batch)
        mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embedding, target_audio, speaker_id, target_feat, mixture_data = self.prepare_batch(batch)
        
        # self.batch = batch
        sr = 16000
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        sdr_,  new_sdr, pesq, stoi, sdr_torchmetric = 0, 0, 0, 0, 0
        
        # print('predictions', predictions.shape)
        # sdr_ = self.sdr(targets, predictions)
        # diir = 'LRS3TestSet2'
        # Path.mkdir(Path(diir), exist_ok=True)
        # filename_mix = diir+'/'+str(self.iii).zfill(5)+'_mix.wav'
        # filename_target = diir+'/'+str(self.iii).zfill(5)+'_target.wav'
        # filename_enhanced = diir+'/'+str(self.iii).zfill(5)+'_enhanced.wav'
        # filename_feat = diir+'/'+str(self.iii).zfill(5)+'_feat.npy'
        # filename_embd = diir+'/'+str(self.iii).zfill(5)+'_embd.npy'
        # filename_embd_id = diir+'/'+str(self.iii).zfill(5)+'_embd_id.npy'
        # self.iii += 1
        # mix_data = {}
        # for k, v in mixture_data.items():
            
        #     if isinstance(v, torch.Tensor):
        #         mix_data[k] = v.cpu().data.numpy().tolist()
        #     else:
        #         mix_data[k] = v
         
        # save_json(mix_data, filename_mix.replace('wav', 'json'))
        # write(mixture.squeeze(0), filename_mix, sr=16000)
        # write(targets.squeeze(0), filename_target, sr=16000)
        # write(predictions.squeeze(0), filename_enhanced, sr=16000)
        # np.save(filename_feat, target_feat.squeeze(0).cpu().numpy())
        # np.save(filename_embd, speaker_embedding.squeeze(0).cpu().numpy())
        # np.save(filename_embd_id, emmbeding_ids.squeeze(0).cpu().numpy())
            
        sdr_,  new_sdr, pesq, stoi, sdr_torchmetric = eval_track(targets, predictions, win=sr, hop=sr, compute_sdr=True)
       
        # print({'sdr_': sdr_, 'new_sdr': new_sdr})
    
        self.log('sdr_', sdr_, prog_bar=True, on_step=False, on_epoch=True)   
        self.log('pesq', pesq, prog_bar=True,on_step=False,  on_epoch=True) 
        self.log('stoi', stoi, prog_bar=True,on_step=False,  on_epoch=True)
        self.log('new_sdr', new_sdr, prog_bar=True,on_step=False,  on_epoch=True)       
        self.log('sdr_torchmetric', sdr_torchmetric, prog_bar=True,on_step=False,  on_epoch=True)       
        
        return {"loss": loss, 'sdr_': sdr_, 'new_sdr': new_sdr, 'pesq': pesq, 'stoi': stoi, 'sdr_torchmetric': sdr_torchmetric}

    def on_test_epoch_end(self):
        # self.log("SDR", outputs['new_sdr'].mean(), prog_bar=True, sync_dist=True)
        pass


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # for parameter in self.net.av_speaker_embedding.parameters():
        #     parameter.requires_grad = False
        # for parameter in self.net.speaker_separation.encoder.parameters():
        #     parameter.requires_grad = False
            
        # for parameter in self.net.speaker_separation.norm_layers.parameters():
        #     parameter.requires_grad = False
            
        # for paramenter in self.net.speaker_separation.adaptive_layer.parameters():
        #     paramenter.requires_grad = False
            
        # for parameter in self.net.speaker_separation.decoder.parameters():
        #     parameter.requires_grad = False
        
        # modules = [ *self.net.speaker_separation.transformer.transformer_encoder.layers] 
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False
        
        # for parameter in self.net.encoder.parameters():
        #     parameter.requires_grad = False
            
        # for paramenter in self.net.adaptive_layer.parameters():
        #     paramenter.requires_grad = False
            
        # for parameter in self.net.decoder.parameters():
        #     parameter.requires_grad = False
        
        # for parameter in self.net.audio_video_embedding.encoder.parameters():
        #     parameter.requires_grad = False
            
        # for paramenter in self.net.audio_video_embedding.adaptive_layer.parameters():
        #     paramenter.requires_grad = False
            
        # for paramenter in self.net.audio_video_embedding.linear_layers.parameters():
        #     paramenter.requires_grad = False
      
        # for paramenter in self.net.audio_video_embedding.speaker_embedding.parameters():
        #     paramenter.requires_grad = False
        
       
        # modules = [ *self.net.audio_video_embedding.transformer.transformer_encoder.layers] 
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False
        
        # modules = [ *self.net.transformer.transformer_encoder.layers] 
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False
                
        # modules = [ *self.net.transformer.transformer_encoder.layers[:2] ] 
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

            
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = VoiceFormerModule(None, None, None)
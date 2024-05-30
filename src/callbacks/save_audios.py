from lightning.pytorch.callbacks import Callback
from src.utils.io import save_json, save_wavs
from pathlib import Path

class AudioSampleLogger(Callback):
    def __init__(self, output_dir, num_samples=32):
        super().__init__()
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
         self.log_data(pl_module, 'val')
         
    def log_data(self, pl_module, prefix):
        batch = pl_module.batch
        mixture, audios, video_features, speaker_embedding, emmbeding_ids,face_embeddings, target_audio, speaker_id, target_feat,_ = pl_module.prepare_batch(batch)
       
        mixture.to(device=pl_module.device)
        if not isinstance(speaker_embedding, list):
            speaker_embedding.to(device=pl_module.device)
            emmbeding_ids.to(device=pl_module.device)
        
        if not isinstance(face_embeddings, list):
            face_embeddings.to(device=pl_module.device)
            
        if not isinstance(video_features, list): 
            video_features.to(device=pl_module.device) 
            audios.to(device=pl_module.device)
        target_feat.to(device=pl_module.device)  
        predictions = pl_module(mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embeddings, target_feat)
        
        if predictions.shape[-1] > 16000:
            noise = mixture - target_audio
            
            if isinstance(predictions, tuple):
                enhanced = predictions[0] 
            else:
                enhanced = predictions
                
            if enhanced.shape[-1] > mixture.shape[-1]:
                enhanced = enhanced[..., :mixture.shape[-1]]
            else:
                mixture = mixture[..., :enhanced.shape[-1]]
                
            noise_p = mixture - enhanced
            audios = {'mixture': mixture, 'enhanced': enhanced, 'audio': target_audio, 'noise': noise, 'noise_p': noise_p}
            save_wavs(audios, sr=16000, output_dir=self.output_dir)
            
from operator import imod
import os
import numpy as np 
import mir_eval
from pesq import pesq
from pystoi import stoi
from src.utils.io import read_audio, read_feature, write
from tqdm import tqdm
from os.path import exists
from scipy.io import wavfile
import torch
import torchaudio
from src.models.components.AVEmbedding import AVEmbedding
from src.models.components.voiceFormerAE3 import VoiceFormerAE3
from src.models.components.voiceFormerAE import VoiceFormerAE
from src.models.components.voiceFormerAVE import VoiceFormerAVE


def getfiles(path, ext='.mp4'):
    files_list = []
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name).replace(path, ''))
    return files_list

def save_audio(wav, filename, sr=16000):
    # Normalize audio if it prevents clipping
    m = np.max(np.abs(wav))
    wav = (wav/m).astype(np.float32)
    wav = np.transpose(wav, (1, 0))
    wavfile.write(filename, sr, wav)


def main():    
    enh_model = VoiceFormerAVE(num_encoder_layers=3)
    # ckpt = torch.load('logs/train/runs/2023-05-03/23-24-42/checkpoints/last.ckpt', map_location='cpu')
    # ckpt = torch.load('logs/train/runs/2023-07-13/07-57-44/checkpoints/epoch_024.ckpt', map_location='cpu')
    # ckpt = torch.load('logs/train/runs/2023-04-23_01-24-33/checkpoints/epoch_000.ckpt', map_location='cpu')
    ckpt = torch.load('logs/train/runs/2023-07-19/18-07-13/checkpoints/last.ckpt', map_location='cpu')
    print(ckpt['state_dict'].keys())
    for name, param in enh_model.state_dict().items():
        
        if 'net.'+name in ckpt['state_dict'].keys():
            enh_model.state_dict()[name].copy_(ckpt['state_dict']['net.'+name])
            print(name+' loaded')
        else:
            print(f'Not found: {name}')
            
    # for name, param in ckpt['state_dict'].items():
    #     name = name.replace('net.', '')
    #     if name in enh_model.state_dict().keys():
    #         enh_model.state_dict()[name].copy_(param)
    #         print(name+' loaded')
    #     else:
    #         print(f'Not found: {name}')
    enh_model.eval()
    
    data_path = 'jfk/audios/' 
    
    files_ = getfiles('jfk/audios/', ext='.wav')
    emb_files_ = getfiles('jfk/se/', ext='.npy')
     
    vid = []
    i = 1
    for file in tqdm(files_):
        i = i + 1  
        print(file)
        if '16khz' not in file: continue
        print('Processing: ', file)
        mix = read_audio(data_path+file, augment=False)
        embeddings = []
        for i in range(1):
            random = np.random.randint(0, len(emb_files_))
            emb = np.load('jfk/se/'+emb_files_[random])
            em = torch.from_numpy(emb).type(torch.FloatTensor).squeeze(0)
            embeddings.append(em)
        em = torch.stack(embeddings).permute(1,2,0)
        # mix = mix[...,:64000]
        enhanced = enh_model(mix, vid, em)
     
        noise = mix - enhanced
        write(noise.squeeze(0), 'jfk/enhanced/noise_'+file, 16000)
        write(enhanced.squeeze(0), 'jfk/enhanced/enhanced_'+file, 16000)
            

if __name__ == '__main__':
	main()

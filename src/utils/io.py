
# Copyright (c) VGG, University of Oxford.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: akamrahimi

import os, random
from pathlib import Path
import glob
import itertools
from re import L
import subprocess
import json
import numpy as np
import torch as th
import torchvision
import torchaudio
from torch.nn import functional as F
from natsort import natsorted
from src.utils.utils import normalize, clip_audio, augment_audio
from scipy.io import wavfile
from decord import VideoReader

def load_json(json_dir, file=None):
    if file is None: 
        json_file = json_dir
    else:
        json_file = os.path.join(json_dir, file)
  
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
                       
def find_files(path, exts=[]):
    files = []
    for ext in exts:
        found_files = natsorted(glob.glob(path + '*/*'+ext))
        files.extend([p.replace(path, '') for p in found_files])
    return files

def read_video_torch(file, offset, length, fps=None):  
    
    reader = torchvision.io.VideoReader(str(file), "video")
    reader.set_current_stream("video:0")
    
    info = reader.get_metadata()
   
    fps_ = info["video"]["fps"][0]

    if fps is not None:
        if fps_ != fps:
            raise RuntimeError(f"Expected {file} to have fps of "
                                f"{fps}, but got {fps_}")     
    frames = []
    offset = int(offset/fps_)
    length = length if length > 0 else int(info["video"]["duration"][0] * 25 - offset)
    for frame in itertools.islice(reader.seek(offset), length or None):
        frames.append(frame['data'])

    out = th.stack(frames)
    
    while length > out.shape[0]:
        shortage = int(length - out.shape[0])
        out = th.cat((out, out[-shortage:,:,:,:]))
    
    return out
    
def read_video(file, offset=0, length=None, frame_size=160):
    offset = max(offset - 4, 0)
    if length is not None:
        length += 4  # to read til length + 3
    else:
        length = 1e15  # some large finite num

    with open(file, 'rb') as f:
        video_stream = VideoReader(
            f, width=frame_size, height=frame_size)
       
        length = min(length, len(video_stream))

        frames = video_stream.get_batch(list(range(offset, offset+length))).asnumpy()
    
    return th.FloatTensor(frames / 255. ).unsqueeze(0)


def read_audio(file: str, offset: int = 0, length: int = -1, sample_rate: int = 16000, normalize: bool = True, augment: bool = True, random_offset: bool = False):
    length_ = length
    if random_offset and offset == 0: length_ = -1
    
    try:
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            out, sr = torchaudio.load(str(file), normalize=normalize, frame_offset=offset, num_frames=length_)
        else:
            out, sr = torchaudio.load(str(file), normalize=normalize,  offset=offset, num_frames=length_)

        if random_offset and offset == 0: 
            if out.shape[-1] > length:
                offset = random.randint(0, out.shape[-1] - length)
            else:
                offset = random.randint(5, 400)
            
            out = out[..., offset:length+offset]
    except Exception as e:
        print('could not get {}'.format(file))
        print(e)
        return None
    
    if sample_rate is not None and sr != sample_rate:
            raise RuntimeError(f"Expected {file} to have sample rate of {sample_rate}, but got {sr}")
        
    while length > out.shape[-1]: # if the audio is shorter than the length, pad it
        shortage = length - out.shape[-1]
        out = th.cat((out,out[:,-shortage:]),1) 
        
    if out.dim() == 2 and out.shape[0] > 1: 
        out = out.mean(0, keepdim=True)
    elif out.dim() == 1:
        out = out.unsqueeze(0)
        
    return out.type(th.FloatTensor)


def read_feature(file, offset=0, length=-1):

    try:
        features = np.load(file)
        features = features[0].transpose(1,0) if len(features.shape) > 2 else features
        out = features[offset:offset+length or -1,...]
        
        while length > out.shape[0]:
            out = np.pad(out, ((0,length - out.shape[0]),(0, 0)), 'wrap')
        
        return th.from_numpy(out).type(th.FloatTensor).permute(1,0)
    
    except Exception as e:
        print('could not get {}'.format(file))
        print(e)
        return None
    

def save_wavs(audios, sr=44100, output_dir='', limit=40, i=0):
    # Write result
   
    n = 0
    for key in audios:
        n +=1
        if n > limit: break
        dir_ = output_dir + '/'
        save_wav_bulk(audios[key], dir_,key, sr)

def save_wav_bulk(wavs, dir_, filename, sr=44100):
    for i, wav in enumerate(wavs):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        write(wav, dir_+str(i+1).zfill(4)+'_'+filename+'.wav', sr=sr)


def write(wav, filename, sr=44100):
    # Normalize audio if it prevents clipping
    try:
        wav = wav / max(wav.abs().max().item(), 1)
        
    except:
        print('could not write {}'.format(filename))
        print(wav.shape)
    
    torchaudio.save(filename, wav.type(th.FloatTensor).cpu(), sr)  
    
def save_videos(noisy, enhanced, clean, videos, limit=40, i=0):
    Path("samples").mkdir(parents=True, exist_ok=True)
    n = 0
    for noisy_, enhanced_, clean_, vid_ in zip(noisy, enhanced, clean, videos):
        n +=1
        if n > limit: break
        filename = 'samples/'+str(i)+str(n).zfill(4)
        save_video(vid_, noisy_, filename + "_noisy.mp4")
        save_video(vid_, enhanced_, filename + "_enhanced.mp4")
        

def read_video_all(video_path, offset=0, length=-1, fps=None):
    
    reader = torchvision.io.VideoReader(video_path, "video")
    frames = []
   
    i = 0
    for frame in reader:
        if length == -1:
            frames.append(np.array(frame['data']))
        else:
            if i >= offset and i < length + offset:
                frames.append(np.array(frame['data']))
            i +=1
            if i  > length + offset:
                break
        
    return th.as_tensor(np.array(frames))
    
def save_video(video, audio, filename):
    audio = audio / max(audio.abs().max().item(), 1)
    
    # frames = read_video_all(video_path[0])
    video = video.permute(0,2,3,1)
    torchvision.io.write_video(filename=filename, video_array=video.cpu(), video_codec='libx264', audio_codec='aac', fps=25, audio_fps=16000, audio_array=audio.cpu()) 
    
def resample_video_audio(video_file, fps=25, sample_rate=16000, start=None, to=None, frame_size=160):
    video_path_ = video_file.split('/')

    filename = os.path.splitext(video_path_[-1])[0]
    video_ = filename+'_f25.mp4'
    audio_ = filename+'_16khz.wav'

    convert_video = [
        "ffmpeg",
        "-threads","1",
        "-loglevel","error",
        "-y"
       ]
    
    if start is not None:
        convert_video.extend(["-ss",str(start)])
    
    if to is not None:
        convert_video.extend(["-to",str(to)])

    convert_video.extend([
        "-i",video_file,video_,
        "-an",
        "-r",str(fps)
    ])
    
    subprocess.run(convert_video)

    extract_audio = [
        "ffmpeg",
        "-threads","1",
        "-loglevel","error",
        "-y",
        "-i",video_,
        "-async","1",
        "-ac", "1",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        audio_]
    subprocess.run(extract_audio)
    
    video = read_video(os.path.join(os.getcwd(),video_), offset=0, length=None, frame_size=frame_size)
    audio = read_audio(os.path.join(os.getcwd(),audio_), offset= 0, length=-1, normalize=True, augment=False, sample_rate=sample_rate)
    return video, audio, filename

def read_text(file): 
    with open(file, 'r') as f:
        text = f.read()
    return text

def center_crop(frames, img_size=96):
    crop_x = (frames.size(3) - img_size) // 2
    crop_y = (frames.size(4) - img_size) // 2
    faces = frames[:, :, :, crop_x:crop_x + img_size,
                   crop_y:crop_y + img_size]

    return faces
from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import shutil, os
import random
from pathlib import Path

lrs3_dirs = {
    #'tt' : '../../../scratch/lrs_praj_full/lrs3/',
    'cv' : '../../../scratch/lrs_praj_full/lrs3/',
    'tr' : '../../../scratch/lrs_praj_full/lrs3/'
    }
for split in lrs3_dirs:
    num = 100 if split == 'cv' else 500
    
    files = load_json('../data/lrs3_cleaned_imp/'+split+'/data.json')
    random.shuffle(files)
    files = files[:num]
    meta = []
    for file in tqdm(files):
        video_file = file[3]
        v_file_name = video_file.split('/')[-1]
        dir_name = video_file.split('/')[-2]
        a_file_name = file[0].split('/')[-1]
       
        
        video_path = os.path.join(lrs3_dirs[split], video_file)
        audio_path = os.path.join('../../../../../../datasets/lrs3/wav/pretrain/', file[0])
        em_path = os.path.join('../../../scratch/lrs3_embeddings/', file[3])
        
        Path.mkdir(Path('data/vid/'+dir_name), exist_ok=True)
        Path.mkdir(Path('data/wav/'+dir_name), exist_ok=True)
        Path.mkdir(Path('data/em/'+dir_name), exist_ok=True)
        shutil.copyfile(video_path, 'data/vid/'+file[3])
        shutil.copyfile(audio_path, 'data/wav/'+file[0])
        shutil.copyfile(em_path, 'data/em/'+file[3])
        
        meta.append(file)

    save_json(meta, 'data/map/'+split+'/data.json')
    print(split+' done!')
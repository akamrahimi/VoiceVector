from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import shutil, os

lrs3_dirs = {
    'tt' : '../data/lrs3/tt/',
    'cv' : '../data/lrs3/cv/',
    'tr' : '../data/lrs3/tr/'
    }

# --------------------------------------------
# Get the meta data for all the npy fiels and save them in a json file
# --------------------------------------------
for split in lrs3_dirs:
    files = load_json('../data/lrs3/'+split+'/data.json')
    meta = {}
    for i, file in enumerate(files):
        speaker = file[2]
        if speaker not in meta:
            meta[speaker] = []
            
        meta[speaker].append(i)
        
    save_json(meta, '../data/lrs3/'+split+'/speaker.json')
    print(split+' done!')

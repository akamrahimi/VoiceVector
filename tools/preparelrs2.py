from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import shutil, os

lrs2_root = '../../../scratch/lrs2_new/'
lrs2_dirs = {
    'tt' : '../../../scratch/lrs2_new/',
    'cv' : '../../../scratch/lrs2_new/',
    'tr' : '../../../scratch/lrs2_new/'
    }
# --------------------------------------------
# find all the npy files that exists in the LRS3 test set directory 
# --------------------------------------------

# files = find_files(lrs2_root,['.npy'])
# save_json(files, '../data/lrs2_cleaned/video.json')
# print('Found '+str(len(files))+' files')
# print('done!')


# --------------------------------------------
# Get the meta data for all the wav fiels include duration and save them in a json file
# --------------------------------------------
for split in lrs2_dirs:
    files = load_json('../data/lrs2_cleaned/'+split+'/video.json')
    meta = []
    missing_files = []
    for file in tqdm(files):
        file = file[0]
        try:
            audio_file = file.replace('.mp4.npy','.wav')
            audio_path = '../../../scratch/lrs2_wavs/'+audio_file
            sample_rate, audio = wavfile.read(audio_path)
            video_file = file.replace('.mp4.npy','.npy')
            file_np = np.load('../../../scratch/lrs_imp/lrs2/pretrain/'+video_file)  
            meta.append([audio_file, audio.shape[0],sample_rate, file, file_np.shape[0]])
        except Exception as e:
            print('missing file: ', file)
            print(e)
            
    save_json(meta, '../data/lrs2_cleaned/'+split+'/clean2.json')
    print(split+' done!')

# for split in lrs2_dirs:
#     files = load_json('../data/lrs2_cleaned_imp/'+split+'/data_old.json')
#     meta = []
#     for file in tqdm(files):
#         video_file = file[0].replace('wav', 'npy')
#         video_path = os.path.join(lrs2_dirs[split], video_file)
#         try:
#             file_np = np.load(video_path)
#         except:
#             # print('missing file: ', video_path)
#             continue
            
#         meta.append([file[0], file[1],16000,file[0].replace('.wav', '.npy'), file_np.shape[-1]])

#     save_json(meta, '../data/lrs2_cleaned_imp/'+split+'/data.json')
#     print(split+' done!')

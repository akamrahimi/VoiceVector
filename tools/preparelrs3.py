from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import shutil, os

lrs3_dirs = {
    #'tt' : '../../../scratch/lrs_praj_full/lrs3/',
    # 'cv' : '../../../scratch/lrs_praj_full/lrs3/',
    'tr' : '../../../scratch/lrs_praj_full/lrs3/'
    }
# --------------------------------------------
# find all the npy files that exists in the LRS3 test set directory 
# --------------------------------------------
# for split in lrs3_dirs:
#     print(split, lrs3_dirs[split])
#     files = find_files(lrs3_dirs[split],['npy'])
#     save_json(files, '../data/lrs3_imp/'+split+'/video.json')
#     print(split+' done!')


# --------------------------------------------
# Get the meta data for all the npy fiels and save them in a json file
# --------------------------------------------
# for split in lrs3_dirs:
#     files = load_json('../data/lrs3_imp/'+split+'/video.json')
#     meta = []
#     for file in tqdm(files):
#         file_np = np.load(lrs3_dirs[split]+file)       
#         meta.append(['../scratch/lrs_imp/lrs3/pretrain/'+file, file_np.shape[0]])
#     save_json(meta, '../data/lrs3_imp/'+split+'/video2.json')
#     print(split+' done!')

# --------------------------------------------
# Get the file names for all the wav fiels
# --------------------------------------------
# for split in lrs3_dirs:
#     files = load_json('../data/lrs3_imp/'+split+'/video.json')
#     audio_files = []
#     for file in tqdm(files):
#         file_name = file[0].replace('npy', 'wav')
#         file_name = file_name.replace('../scratch/lrs_imp/lrs3/test/', '../../../scratch/lrs3_wavs/')
#         audio_files.append(file_name)
#     save_json(audio_files, '../data/lrs3_imp/'+split+'/clean.json')
#     print(split+' done!')

# --------------------------------------------
# Get the meta data for all the wav fiels include duration and save them in a json file
# --------------------------------------------
# for split in lrs3_dirs:
#     files = load_json('../data/lrs3_imp/'+split+'/clean.json')
#     meta = []
#     missing_files = []
#     for file in tqdm(files):
#         # write the code for loading the audio file duration 
#         file_path = '../../'+file
#         dest_dir = os.path.dirname(file_path)

#         if not os.path.exists(dest_dir):
#             os.makedirs(dest_dir)

#         try:
#             sample_rate, audio = wavfile.read(file_path)
#             meta.append([file, audio.shape[0],sample_rate])
#         except:
#             try:
#                 if split == 'cv':
#                     file_path_src = file.replace('../scratch/lrs3_wavs/', '../../../../../../datasets/lrs3/wav/pretrain/')
#                 else:    
#                     file_path_src = file.replace('../scratch/lrs3_wavs/', '../../../../../../datasets/lrs3/wav/test/')
                
#                 shutil.copyfile(file_path_src, file_path)
#                 sample_rate, audio = wavfile.read(file_path)
#                 meta.append([file, audio.shape[0],sample_rate])
#             except:
#                 missing_files.append(file)
#                 print('missing file: ', file_path_src)
 
#     save_json(meta, '../data/lrs3_imp/'+split+'/clean2.json')
#     save_json(missing_files, '../data/lrs3_imp/'+split+'/missing_clean2.json')
#     print(split+' done!')

for split in lrs3_dirs:
    files = load_json('../data/lrs3_cleaned_imp/'+split+'/data_old.json')
    meta = []
    for file in tqdm(files):
        video_file = file[0].replace('wav', 'npy')
        video_path = os.path.join(lrs3_dirs[split], video_file)
        try:
            file_np = np.load(video_path)
        except:
            # print('missing file: ', video_path)
            continue
            
        meta.append([file[0], file[1],16000,file[0].replace('.wav', '.npy'), file_np.shape[-1]])

    save_json(meta, '../data/lrs3_cleaned_imp/'+split+'/data.json')
    print(split+' done!')

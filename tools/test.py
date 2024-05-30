
from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
import os
# samples_train = load_json('samples_train.json')
# samples_val = load_json('samples_val.json')
# samples_test = load_json('samples_test.json')
# speaker_idx = load_json('speaker_idx.json')

# samples_single = []
# for sample in tqdm(samples_test):
#     talk_id = sample[0].split('/')[0]
#     for sp_idx in speaker_idx:
#         if talk_id == sp_idx[0]:
#             sample.append(sp_idx[1])
#             samples_single.append([sample[0],sample[1], sp_idx[1]])
#             break
# save_json(samples_single, 'samples_tt_single.json')


split = {
    'tr': '../data/lrs3/tr/data.json',
    'cv': '../data/lrs3/cv/data.json'
}
aduio_path = '../../../scratch/lrs3_wavs/'
video_path = '../../../scratch/vtp_features/lrs3/pretrain/'

for key in split:
    data = load_json(split[key])
    for sample in tqdm(data):
        a_path = os.path.join(aduio_path, sample[0]+'.wav')
        v_path = os.path.join(video_path, sample[0]+'.npy')
       
        # check if file exists
        # if os.path.isfile(a_path):
        #     pass
        # else:
        #     print(a_path)
        
        if os.path.isfile(v_path):
            try:
                np.load(v_path)
            except Exception as e:
                print(v_path)
                print(e)
        else:
            print(v_path)
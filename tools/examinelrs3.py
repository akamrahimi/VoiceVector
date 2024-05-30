from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm
lrs3_dir = "../../../../../scratch/shared/beegfs/prajwal/lrs3/"

split = {
    'train': 'pretrain/feats/', 
    'val': 'trainval/feats/', 
    'test': 'test/feats/',
    }

# for sp in split.keys():
#     dir_to_search = lrs3_dir + split[sp]
#     files = find_files(dir_to_search, exts='*/*.npy')
#     save_json(files, sp+'_files.json')
#     print(dir_to_search, len(files))

#---------------------------------------------
# train_files = load_json('train_files.json')
# val_files = load_json('val_files.json')
# test_files = load_json('test_files.json')

# #convert lists to numpy arrays
# train_files = np.array(train_files)
# val_files = np.array(val_files)
# test_files = np.array(test_files)


# unqiue_train_files = np.unique(train_files)
# unqiue_val_files = np.unique(val_files)
# unqiue_test_files = np.unique(test_files)
# save_json(unqiue_train_files.tolist(), 'unique_train_files.json')
# save_json(unqiue_val_files.tolist(), 'unique_val_files.json')
# save_json(unqiue_test_files.tolist(), 'unique_test_files.json')
#---------------------------------------------

# unique_train_files = load_json('unique_train_files.json')
# unique_val_files = load_json('unique_val_files.json')
# unique_test_files = load_json('unique_test_files.json')

# train_files = []
# for f in tqdm(unique_train_files):
#     if f not in unique_test_files and f not in unique_val_files:
#         train_files.append(f)

# train_files = np.array(train_files)

# save_json(train_files.tolist(), 'cleaned_train_files.json')
# print(len(train_files), len(unique_test_files))
#---------------------------------------------

# sample_train = load_json('samples_train.json')
# sample_test = load_json('samples_test.json')

# test_speakers = []
# for f in sample_test:
#     speaker_id = f[0].split('/')[0]
#     test_speakers.append(speaker_id)

# test_speakers = np.array(test_speakers)

# train_speakers = []
# for f in sample_train:
#     speaker_id = f[0].split('/')[0]
#     train_speakers.append(speaker_id)

# train_speakers = np.array(train_speakers)
# print(len(train_speakers), len(test_speakers))

# unique_test_speakers = np.unique(test_speakers)
# unique_train_speakers = np.unique(train_speakers)
# print(len(unique_train_speakers), len(unique_test_speakers))

# duplicates = np.intersect1d(unique_train_speakers, unique_test_speakers)
# print(duplicates[:10])

# print(len(duplicates))
#---------------------------------------------

# unique_val_files = load_json('unique_val_files.json')
# unique_test_files = load_json('unique_test_files.json')

# test_speakers = []
# for f in unique_test_files:
#     speaker_id = f.split('/')[0]
#     test_speakers.append(speaker_id)

# test_speakers = np.array(test_speakers)

# val_speakers = []
# for f in unique_val_files:
#     speaker_id = f.split('/')[0]
#     val_speakers.append(speaker_id)

# val_speakers = np.array(val_speakers)
# print(len(val_speakers), len(test_speakers))

# unique_test_speakers = np.unique(test_speakers)
# unique_val_speakers = np.unique(val_speakers)
# print(len(unique_val_speakers), len(unique_test_speakers))

# duplicates = np.intersect1d(unique_val_speakers, unique_test_speakers)

# print(len(duplicates))

#---------------------------------------------

# cleaned_train_files = load_json('samples_train.json')
# unique_val_files = load_json('unique_val_files.json')

# val_speakers = []
# for f in unique_val_files:
#     speaker_id = f.split('/')[0]
#     val_speakers.append(speaker_id)

# val_speakers = np.array(val_speakers)

# train_speakers = []
# for f in cleaned_train_files:
#     speaker_id = f.split('/')[0]
#     train_speakers.append(speaker_id)

# train_speakers = np.array(train_speakers)
# print(len(train_speakers), len(val_speakers))

# unique_val_speakers = np.unique(val_speakers)
# unique_train_speakers = np.unique(train_speakers)
# print(len(unique_train_speakers), len(unique_val_speakers))

# duplicates = np.intersect1d(unique_train_speakers, unique_val_speakers)
# print(duplicates[:10])
# print(len(duplicates))


    



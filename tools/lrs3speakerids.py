from helpers import find_files, load_json, save_json
import numpy as np
from tqdm import tqdm

# csv_file = 'lrs3spids.csv'

# #load the csv file
# spids = np.loadtxt(csv_file, dtype=str, delimiter=',')
# speaker_ids = {}
# for i in range(len(spids)):
#     talk_id = spids[i][0]
#     speaker = spids[i][1]
    
#     if speaker not in speaker_ids:
#         speaker_ids[speaker] = []
#     speaker_ids[speaker].append(talk_id)

# #save the speaker ids
# save_json( speaker_ids, 'lrs3spids.json')


json_file = 'lrs3spids.json'
speaker_ids = load_json(json_file)


test_talks_ = load_json('samples_test.json')
test_talks = []
for talk in test_talks_:
    t = talk[0].split('/')[0]
    test_talks.append(t)


train_talks_ = load_json('samples_train.json')
train_talks = []
for talk in train_talks_:
    t = talk[0].split('/')[0]
    train_talks.append(t)

unique_train_files = np.unique(train_talks)
unique_val_files = load_json('unique_val_files.json')
unique_test_files = np.unique(test_talks)

test_talks = []
for f in unique_test_files:
    speaker_id = f.split('/')[0]
    test_talks.append(speaker_id)

test_talks = np.array(test_talks)

train_talks = []
for f in unique_train_files:
    speaker_id = f.split('/')[0]
    train_talks.append(speaker_id)

train_talks = np.array(train_talks)

val_talks = []
for f in unique_val_files:
    speaker_id = f.split('/')[0]
    val_talks.append(speaker_id)

val_talks = np.array(val_talks)

unique_test_talks = np.unique(test_talks)
unique_train_talks = np.unique(train_talks)
unique_val_talks = np.unique(val_talks)

unique_training_speakers = []
for speaker in tqdm(speaker_ids):
    talks = speaker_ids[speaker]
    for talk in talks:
        if talk in unique_train_talks:
            unique_training_speakers.append(speaker)
            break

unique_test_speakers = []
for speaker in tqdm(speaker_ids):
    talks = speaker_ids[speaker]
    for talk in talks:
        if talk in unique_test_talks:
            unique_test_speakers.append(speaker)
            break

duplicated_speakers = []
for speaker in tqdm(speaker_ids):
    talks = speaker_ids[speaker]
    for talk in talks:
        if talk in unique_train_talks and talk in unique_test_talks:
            duplicated_speakers.append(talk)
            break
            
duplicated_speakers = list(set(duplicated_speakers))

print('Total unique speakers', len(speaker_ids))
print('Unique speakers in training set', len(unique_training_speakers))
print('Total tracks in the test set', len(test_talks))
print('Unique speakers in the test set', len(unique_test_speakers))
print('Overlapping speakers', len(duplicated_speakers))


duplicated_speakers = []
for speaker in tqdm(speaker_ids):
    talks = speaker_ids[speaker]
    for talk in talks:
        if talk in unique_train_talks and talk in unique_test_talks:
            duplicated_speakers.append(talk)

save_json( duplicated_speakers, 'duplicated_speakers.json')
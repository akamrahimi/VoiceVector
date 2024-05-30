import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import os
import random
from decord import VideoReader
import cv2
import face_recognition

def extract_random_frames_decord(video_path, output_dir, num_frames=10):
    try:
        vr = VideoReader(video_path)
    except:
        print('Error failed to read video: ', video_path)
        return
    total_frames = len(vr)
    
    # Randomly sample frame indices
    frame_indices = random.sample(range(total_frames), num_frames)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 1
    for idx in frame_indices:
        frame = vr[idx].asnumpy()  # Get the frame as a numpy array
        my_face_encoding = face_recognition.face_encodings(frame)
        attempts = 0
        while len(my_face_encoding) == 0 and attempts < 20:
            attempts = attempts + 1
            idx = random.sample(range(total_frames), 1)[0]
            frame = vr[idx].asnumpy()
            my_face_encoding = face_recognition.face_encodings(frame)
        if attempts == 20: continue
        output_file = os.path.join(output_dir, str(i).zfill(4)+'.npy')
        np.save(output_file, my_face_encoding)
        i = i + 1
        # embedding_objs = DeepFace.represent(img_path = output_file, enforce_detection=False)

def load_json(json_dir, file=None):
    if file is None: 
        json_file = json_dir
    else:
        json_file = os.path.join(json_dir, file)
  
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


lrs3_video_path = '../../../../scratch/shared/beegfs/shared-datasets/Lip_Reading/lrs3/mp4/pretrain/mp4/'
splits = ['tt','cv','tr']

for split in splits:
    data = load_json(f'data/lrs3/{split}/data.json')
    j = 0
    for file in tqdm(data):
        j = j + 1
        video_path = lrs3_video_path+file[0]
        face_embeddings_dir = file[0].replace(".mp4", "")
        output_dir = f'../../scratch/lrs3_face_embeddings/{face_embeddings_dir}'
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)
        extract_random_frames_decord(video_path+'.mp4', output_dir, num_frames=10)


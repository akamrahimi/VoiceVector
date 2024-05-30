import os, random, copy, math
import torch
import wget
import numpy as np
from src.utils.io import load_json, read_audio, read_feature, save_json
from src.utils.utils import add_click_pink_noise, apply_random_aumentation_effects, scale_audio
from torch.utils.data import Dataset
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
from torch_audiomentations import Compose, Gain,  AddBackgroundNoise, PeakNormalization, PitchShift, Shift, LowPassFilter, HighPassFilter
import warnings
import json
class LRS23(Dataset):
    """`LRS2 and LRS3 <https://www.robots.ox.ac.uk/~vgg/data/lip_reading/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where the data stored.
        train (bool, True): If True creates dataset from training set, otherwise from test set.
        download (bool, optional): If True, downloads the dataset split from the internet and
            puts it in root directory (not the actual audio-video files but the paths). If it is already downloaded, it is not
            downloaded again. The actual audio-video should be downloaded which requires gaining permission - check the above link.
        transform (callable, optional): A function/transform that  takes in an audio file
            and returns a transformed version.
    """
    
    download_url = [
        'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3/train.json',
        'https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3/test.json'
    ]

    sample_rate = 16000
    
    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    def __init__(
        self,
        data_root: list = ["data/lrs3/"],
        audio_root: list = ['../../scratch/lrs3_wavs/'],
        video_root: list = ['../../scratch/vtp_features/lrs3/'],
        speaker_emb_root: list = ['../../scratch/lrs_imp/lrs3/pretrain_speaker_embeddings/'],
        face_emb_root: list = ['../../scratch/lrs3_face_embeddings/'],
        background_noise_root: str = '../../scratch/DNS/noise/',
        train: str = 'train',
        download: bool = False,
        transform: bool = False,
        speakers = 2,
        max_data: int = -1,
        load_features: bool = True,
        load_speaker_embedding: bool = True,
        frame_rate: int = 25,
        sample_rate: int = 16000,
        max_duration: int = 6,
        add_background_noise: bool = False, 
        add_pink_noise: bool = False,
        generate_speaker_dict: bool = False,
        positive_embeddings: int = 3,
        negative_embeddings: int = 3,
        face_embeddings: int = 3,
    ) -> None:
        self.transform = transform
        self.data_root = data_root
        self.audio_root = audio_root
        self.add_background_noise = add_background_noise
        self.generate_speaker_dict = generate_speaker_dict
        self.speaker_emb_root = speaker_emb_root
        self.face_emb_root = face_emb_root
        self.add_pink_noise = add_pink_noise
        self.max_data = max_data
        self.speakers = speakers
        self.train = train  # training set or test set
        self.positive_embeddings = positive_embeddings
        self.negative_embeddings = negative_embeddings
        self.enrolment_vectors = positive_embeddings + negative_embeddings
        self.face_embeddings = face_embeddings
        self.audio_root = audio_root
        self.mixture_date = {}
        if self.train == 'train':
            self.data_file = "tr/data.json"
            self.speaker_dict = "tr/speaker.json"
            self.video_root = [vr+'pretrain/' for vr in video_root]

        elif self.train == 'val':
            self.data_file = "cv/data.json"
            self.speaker_dict = "tr/speaker.json"
            self.video_root = [vr+'pretrain/' for vr in video_root]
            
        elif self.train == 'test':
            self.data_file = "tt/data.json"
            self.speaker_dict = "tr/speaker.json"
            self.video_root = [vr+'test/' for vr in video_root]
            
        else:
            raise ValueError("train must be either 'train', 'val' or 'test'")
        
        self.noise_files = 'DNS.json'
        self.load_features = load_features
        self.load_speaker_embedding = load_speaker_embedding
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_duration = self.max_duration * self.sample_rate
        self.video_duration = int(self.max_duration * self.frame_rate)
        if download:
            self.download()

        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

        # Initialize augmentation callable
        
        transforms=[
            Gain(min_gain_in_db=-4.0, max_gain_in_db=5.0, p=0.3),
            # LowPassFilter(p=0.1, min_cutoff_freq=120.0, max_cutoff_freq=900.0),
            # HighPassFilter(p=0.1, min_cutoff_freq=20.0, max_cutoff_freq=1500.0),
            # PeakNormalization(p=0.2),
            Shift(min_shift = 500, max_shift = 1500, shift_unit= "samples", p=0.2),
            # PitchShift(p=0.2),
        ]
        self.apply_augmentation = Compose(transforms)
  
        if self.add_background_noise: 
            background_noise=[
                AddBackgroundNoise(background_noise_root, min_snr_in_db=8, max_snr_in_db=10, p=0.5),
            ]
            self.apply_noise = Compose(background_noise)
 
    def apply_audio_augmentation(self, audio: torch.tensor) -> torch.Tensor:
          
        if self.transform:
            audio = self.apply_augmentation(audio, sample_rate=self.sample_rate)
            
        if self.add_background_noise:
            audio = self.apply_noise(audio.unsqueeze(0), sample_rate=self.sample_rate).squeeze(0)

        return audio
    
    def _load_data(self):
        #Load the json file containing the paths to the audio files and 
        # the corresponding transcript including phonemes
        data = []
        for i in range(len(self.data_root)):
            data_ = load_json(self.data_root[i], self.data_file)
            for j, file in enumerate(data_):
                
                file.append(os.path.join(self.video_root[i],file[0]+'.npy'))
                file.append(os.path.join(self.speaker_emb_root[i],file[0]+'.npy'))
                face_emb_dir_path = os.path.join(self.face_emb_root[i],file[0])
                if not self.face_embedding_exists(face_emb_dir_path): continue        
                file.append(face_emb_dir_path)       
                file[0] = os.path.join(self.audio_root[i],file[0]+'.wav')
                 
                data.append(file)
                                
        random.shuffle(data)
                      
        print(f"Loaded {len(data)} samples from {self.data_file}")
        self.data = data
        
        self._load_background_noise()
        self._generate_speaker_dict()
    
    def get_list_of_face_embedding_files(self, dir_path: str, number: int) -> list:
        # determine if any .npy file exists in the directory path 
        face_embedding_files = []
        for file in os.listdir(dir_path):
            if file.endswith(".npy"):
               face_embedding_files.append(os.path.join(dir_path, file))
        if len(face_embedding_files) < number: face_embedding_files = face_embedding_files * number
        return face_embedding_files
    
    def face_embedding_exists(self, dir_path: str) -> bool:
        # determine if any .npy file exists in the directory path 
        for file in os.listdir(dir_path):
            if file.endswith(".npy"):
                return True
      
    def _load_face_embeddings(self, dir_path: int, number: int = 3):
        def load_embedding(file_path):
            """Load embedding and convert to torch tensor."""
            embedding = np.load(file_path)
            return torch.from_numpy(embedding).type(torch.FloatTensor)
        
        face_embedding_files = self.get_list_of_face_embedding_files(dir_path, number)
        
        # Shuffle the list to get randomness instead of using random.sample multiple times
        random.shuffle(face_embedding_files)
        
        face_embeddings = []
        for file in face_embedding_files:
            face_embedding = load_embedding(file)
            if face_embedding.shape[0] == 1:# If the embedding is for a single face
                face_embeddings.append(face_embedding.squeeze(0))
                if len(face_embeddings) == number: break
            
        # If we have fewer than the desired number of embeddings, we'll fill up by duplicating the signle faces we have
        if len(face_embeddings) < number and len(face_embeddings) > 0:
            warnings.warn(f"Couldn't find enough face embeddings in {dir_path}.")
            shortage = number - len(face_embeddings)
            face_embeddings = face_embeddings + [face_embedding[0]] * shortage
        
        if len(face_embeddings) == 0:
            warnings.warn(f"Couldn't find any face embedding in {dir_path}.")
            face_embeddings = [torch.zeros(128)] * number
            
        return torch.stack(face_embeddings, dim=0)
        
    
    def _load_background_noise(self):
        if self.add_background_noise:
            noise_data = load_json(self.data_root[0], self.noise_files)
            random.shuffle(noise_data)
            if self.max_data != -1: noise_data = noise_data[:self.max_data]
            self.noise_data = noise_data
        
    def _generate_speaker_dict(self):
        if self.max_data != -1: self.data = self.data[:self.max_data]
        if self.generate_speaker_dict:
            self.speaker_dict = defaultdict(list)
            for i, d in enumerate(self.data):
                speaker_id = d[2]
                self.speaker_dict[speaker_id].append(i)

            # data_refined = []
            # for i, d in enumerate(self.data):
            #     if d[2] in self.speaker_dict and len(self.speaker_dict[d[2]]) >= max(self.positive_embeddings, self.negative_embeddings):
            #         data_refined.append(d)
        
            # # if max_data is set, only use the first max_data samples for debugging purposes
            # if self.max_data != -1: data_refined = data_refined[:self.max_data]
            # self.data = data_refined
            
            # self.speaker_dict = defaultdict(list)
            # for i, d in enumerate(self.data):
            #     speaker_id = d[2]
            #     self.speaker_dict[speaker_id].append(i)

              
    def fetch_sample_indcies(self, index: int, number: int = 3, positive: bool = True):
        if number is None or number < 1:
            return [], []
        same_speaker = self.data[index][2]
        speaker_indices = self.speaker_dict[same_speaker]
 
        # Get samples (same speaker)
        available_indices = [i for i in speaker_indices if i != index]
        if not available_indices:  # if after removing the current index, there are no available indices
            available_indices = speaker_indices  # revert to using all available indices
        
        if len(available_indices) < number:
            # repeat the indices if there are not enough
            available_indices = available_indices * math.ceil(number / len(available_indices))
            
        sample_indecis = random.sample(available_indices, number)
        embedding_ids = [1] * number if positive else [0] * number
        
        return sample_indecis, embedding_ids
       
    def get_data_info(self, index, number_of_samples: int = 3, positive: bool = True):
        file = self.data[index]
        emmbeding_ids = torch.tensor([1 if positive else 0]).type(torch.LongTensor)
        indecis = [index]
        if self.generate_speaker_dict:
            indecis, emmbeding_ids = self.fetch_sample_indcies(index, number_of_samples, positive)
            
        return {
            'audio_file': file[0],
            'audio_duration': file[1] * 640,
            'indecis': indecis,
            'emmbeding_ids': emmbeding_ids,
            'speaker_id': file[2],
        }
    
    def get_random_index(self, index: int, max_retries: int = 1000):
        """
        Get a random index corresponding to a different speaker.

        This function retrieves a random index from the dataset such that the corresponding entry
        is from a different speaker than the entry at the provided index. If the function fails to 
        find such an index after a specified number of retries, it increments the original index and 
        returns it, wrapping around to 0 if the index is at the end of the dataset.

        Parameters:
        - index (int): The original index from which we want to find another speaker.
        - max_retries (int, optional): The maximum number of attempts to find a different speaker.
        Default is 1000.

        Returns:
        - int: A random index of a different speaker or the incremented original index.

        Warnings:
        - If the function fails to find a different speaker after max_retries, it issues a warning 
        and returns the incremented original index.

        Examples:
        >>> self.data = [
        ...    ["/path/to/speaker1/file1", ...],
        ...    ["/path/to/speaker1/file2", ...],
        ...    ["/path/to/speaker2/file1", ...],
        ... ]
        >>> get_random_index(self, 0)
        2  # Assuming it randomly selects the third entry, which is from a different speaker.
        """

        original_speaker_folder = self.data[index][0].split('/')[-2]
        original_speaker_id = self.data[index][2]
        data_length = len(self.data)

        for _ in range(max_retries):
            random_index = random.randrange(data_length)
            
            random_speaker_folder = self.data[random_index][0].split('/')[-2]
            random_speaker_id = self.data[random_index][2]
            
            if random_index != index and original_speaker_folder != random_speaker_folder and original_speaker_id != random_speaker_id:
                return random_index

        warnings.warn(f"Couldn't find a different speaker index after {max_retries} attempts. Returning the original index incremented.")
        return (index + 1) % data_length

    def create_mixture(self, index: int, audio: torch.tensor, target_meta: dict = None ) -> torch.Tensor:
        """
        Create mixture of two or more speakers added to the target audio.

        Args:
            audio (torch.tensor): Audio of shape (c,T)

        Returns:
            mixture (torch.tensor): Mixture of shape (c,T)
        """
        # positive speaker indecis
        sample_indecis = target_meta['indecis'] if target_meta else []
        embedding_ids = target_meta['emmbeding_ids'] if target_meta else []
        mixture = copy.deepcopy(audio)
        if self.speakers > 1:
            speech_power = audio.norm(p=2)
            for _ in range(self.speakers - 1):
                r = self.get_random_index(index) # get a random index of a different speaker
                meta = self.get_data_info(r, self.negative_embeddings if target_meta else None, positive=False)
                # get negative speaker indecis
                sample_indecis += meta['indecis']
                embedding_ids += meta['emmbeding_ids']
                audio_offset, _ = self.get_random_offset(meta['audio_duration'])
                added_data = {
                    'negative_audio_file': meta['audio_file'],
                    'negative_audio_offset': audio_offset,
                    'negative_audio_duration': meta['audio_duration'],
                 }
                
                audio_ = read_audio(meta['audio_file'], audio_offset, length=mixture.shape[-1])
                audio_ = scale_audio(audio_, speech_power) # scale the audio to match the power of the target audio
                mixture = mixture + audio_

         
        if self.add_pink_noise and mixture.dim() > 1 and mixture.shape[0] > 1:
            mixture = mixture.mean(0, keepdim=True)
            audio, effects = apply_random_aumentation_effects(audio, self.sample_rate)
            mixture = apply_random_aumentation_effects(mixture, self.sample_rate, effects=effects)
            mixture = add_click_pink_noise(mixture, self.sample_rate) 
        else:
            mixture = self.apply_audio_augmentation(mixture)

        return mixture, audio, sample_indecis, torch.tensor(embedding_ids).type(torch.LongTensor), added_data
    
    
    def get_audio_feature_pairs(self, smaple_indecis: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load the audio and matching features from the disk.
        
        Args:
            smaple_indecis (list): List of sample indecis
            
        Returns:
            audio (torch.tensor): Audio of shape (c,T)
            video_features (torch.tensor): Video features of shape (c,T)
        """

        audios = []
        video_features = []
        for index in smaple_indecis:
            audio_path = self.data[index][0]
            audio_offset, video_offset = self.get_random_offset(self.data[index][1] * 640)
            video_features.append(read_feature(self.data[index][3], video_offset, self.video_duration))
            audio = read_audio(audio_path, audio_offset, self.audio_duration)
            mixture, _, _, _,_ = self.create_mixture(index, audio, None )
            audios.append(mixture)
            
        return torch.stack(audios, dim=0).squeeze(1), torch.stack(video_features, dim=0)
    
    def get_speaker_embedding(self, sample_indecis: list) -> torch.Tensor:
        """
        Load the speaker embedding from the disk.
        
        Args:
            embeding_path (str): Path to the embeding file
        
        Returns:
            embedding (torch.tensor): Speaker embedding of shape (c,T)
        """
        embedding_paths = [self.data[i][4] for i in sample_indecis]
        embeddings = []
        for emb_path in embedding_paths:  
            embedding = np.load(emb_path)
            embedding = torch.from_numpy(embedding).type(torch.FloatTensor)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)

    def get_face_embeddings(self, sample_indecis: list) -> torch.Tensor:
        """
        Load the face embedding from the disk.
        
        Args:
            embeding_path (str): Path to the embeding file
        
        Returns:
            embedding (torch.tensor): Speaker embedding of shape (c,T)
        """
        embedding_paths = [self.data[i][5] for i in sample_indecis]
        embeddings = []
        for emb_path in embedding_paths:  
            embedding = self._load_face_embeddings(emb_path, number=self.face_embeddings)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)
    
    def get_audio(self, audio_path: str, audio_offset: int) -> torch.Tensor:
        """
        Load the audio from the disk.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            audio (torch.tensor): Audio of shape (c,T)
        """
        length = self.max_duration * self.sample_rate if self.max_duration else -1
        
        audio = read_audio(audio_path, audio_offset, length=length)
        return audio
         
    def get_random_offset(self, audio_duration: int) -> tuple:
        """
        Calculate the audio and video offsets based on audio duration and maximum duration.

        Parameters:
        - audio_duration (int): Duration of the audio in samples.

        Returns:
        - tuple: Audio and video offsets.
        """

        max_audio_duration = self.max_duration * self.sample_rate
        if audio_duration - 4000 > max_audio_duration:
            extra = audio_duration - max_audio_duration
            try:
                audio_offset = random.randrange(0, extra, 640)
                video_offset = int((audio_offset / self.sample_rate) * self.frame_rate)
                return audio_offset, video_offset
                
            except ValueError:
                warnings.warn(f"Couldn't find a valid offset for audio duration {audio_duration} and max duration {max_audio_duration}.")
                
        return 0, 0

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mixture, phonemes, target) where target is the clean audio.
        """
        if self.train != 'test':
            self.positive_embeddings = random.randint(1, self.enrolment_vectors)
            self.negative_embeddings = abs(self.enrolment_vectors - self.positive_embeddings)

        meta = self.get_data_info(index, self.positive_embeddings)
        audio_duration = meta['audio_duration']
        audio_offset, feat_offset = self.get_random_offset(audio_duration)
        
        target_audio = self.get_audio(meta['audio_file'], audio_offset)
        target_feature = read_feature(self.data[index][3], feat_offset, self.video_duration)
        
        
        
        mixture, target_audio, sample_indecis, emmbeding_ids, added_data  = self.create_mixture(index, target_audio, target_meta=meta)
        
        speaker_embedding = self.get_speaker_embedding(sample_indecis) if self.load_speaker_embedding else []
        audios, video_features = self.get_audio_feature_pairs(sample_indecis) if self.load_features else ([], [])
        face_embeddings = self.get_face_embeddings(sample_indecis) if self.load_features else []
        # emmbeding_ids = (emmbeding_ids -1) * -1 # flip the emmbeding_ids 
        # target_audio = target_audio * random.uniform(0.6, 1.1) # scale the target audio to prevent the model from learning to seprate the speakers based on volume
        mixture_data = {
            'target_audio_file': meta['audio_file'],
            'target_audio_duration': audio_duration,
            'target_audio_offset': audio_offset,
            'target_feat_file': self.data[index][3],
            'target_feat_offset': feat_offset,
            'target_feat_duration': self.video_duration,
            'target_speaker_id': meta['speaker_id'],
            'negative_audio_file': added_data['negative_audio_file'],
            'negative_audio_offset': added_data['negative_audio_offset'],
            'negative_audio_duration': added_data['negative_audio_duration'],
        }
        self.mixture_date[index] = mixture_data
        # if this is the last index save the self.mixture_date to the disk
        if index == len(self.data) - 1:
            save_json(self.mixture_date, 'mixture.json')
       
        return [mixture, audios, video_features, speaker_embedding, emmbeding_ids, face_embeddings, target_audio, meta['speaker_id'], target_feature, mixture_data]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.audio_root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.audio_root, self.__class__.__name__, "processed")

    def download(self) -> None:
        """Download the LRS3 data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return

        # download files
        for url in self.download_url:
            try:
                response = wget.download(url, out=self.data_root[0])
                print('Downloaded '+url+' successfully!')
            except Exception as e:
                print(e)
                raise RuntimeError("Error downloading dataset")
            
    def _check_exists(self) -> bool:
        
        return os.path.exists(os.path.join(self.data_root[0], self.data_file[0]))

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    
    # def fetch_positive_negative_indices(self, index: int, positive: int = 3, negative: int = 3):
    #     if positive < 1:
    #         raise ValueError(f"There should be at aleast one positive example, got {positive}.")
        
    #     speaker_indices = self.speaker_dict[self.data[index][2]]
    #     same_speaker = self.data[index][2]
  
    #     # Get positive samples (same speaker)
    #     available_indices = [i for i in speaker_indices if i != index]
    #     if not available_indices:  # if after removing the current index, there are no available indices
    #         available_indices = speaker_indices  # revert to using all available indices
        
    #     if len(available_indices) < positive:
    #         # repeat the indices if there are not enough
    #         available_indices = available_indices * math.ceil(positive / len(available_indices))
            
    #     pos_samples = random.sample(available_indices, positive)
       
    #     emmbeding_ids = [1] * positive
    #     indices = pos_samples
        
    #     if negative > 0:
    #          # Get negative samples (different speaker)
    #         other_speakers = set(self.speaker_dict.keys()) - {same_speaker}
    #         if len(other_speakers) < negative:
    #             other_speakers = list(other_speakers) * math.ceil(negative / len(other_speakers))
    #         neg_speakers = random.sample(other_speakers, negative)
    #         neg_samples = [random.choice(self.speaker_dict[s]) for s in neg_speakers]
    #         emmbeding_ids += [0] * negative
    #         indices += neg_samples
    #     emmbeding_ids = torch.tensor(emmbeding_ids).type(torch.LongTensor)
        
    #     #return the file paths
    #     return [self.data[i][4] for i in indices], emmbeding_ids
import os, random, copy, math
import torch
import wget
import numpy as np
from src.utils.io import load_json, read_audio, read_feature
from src.utils.utils import add_click_pink_noise, apply_random_aumentation_effects, scale_audio
from torch.utils.data import Dataset
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
from torch_audiomentations import Compose, Gain,  AddBackgroundNoise, PeakNormalization, PitchShift, Shift, LowPassFilter, HighPassFilter
import warnings

class LibiriSpeech(Dataset):
    """`LibriSpeech ASR corpus <https://www.openslr.org/12>`_ Dataset.

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
        data_root: list = ["data/librispeech/"],
        audio_root: list = ['../../scratch/Librispeech/'],
        speaker_emb_root: list = ['../../scratch/Librispeech/speaker_embedding/'],
        background_noise_root: str = '../../scratch/DNS/noise/',
        train: str = 'train',
        download: bool = False,
        transform: bool = False,
        speakers = 2,
        max_data: int = -1,
        load_speaker_embedding: bool = True,
        sample_rate: int = 16000,
        max_duration: int = 6,
        add_background_noise: bool = False, 
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
        self.max_data = max_data
        self.speakers = speakers
        self.train = train  # training set or test set
        self.positive_embeddings = positive_embeddings
        self.negative_embeddings = negative_embeddings
        self.face_embeddings = face_embeddings
        self.audio_root = audio_root
        
        if self.train == 'train':
            split = "test-clean"
        elif self.train == 'val':
            split = "test-clean"
        elif self.train == 'test':
            split = "test-clean"
        else:
            raise ValueError("self.train must be either 'train', 'val' or 'test'")
        
        self.data_file = split+".json"
        self.split = split
        self.noise_files = 'DNS.json'

        self.load_speaker_embedding = load_speaker_embedding
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_duration = self.max_duration * self.sample_rate
        
        # if download:
        #     self.download()

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
                
                file.append(os.path.join(self.speaker_emb_root[i],self.split,file[0].replace('.flac','.npy')))     
                file[0] = os.path.join(self.audio_root[i],self.split,self.split,file[0])
                 
                data.append(file)
                                
        random.shuffle(data)
        
        # if max_data is set, only use the first max_data samples for debugging purposes
        if self.max_data != -1: data = data[:self.max_data]
                           
        print(f"Loaded {len(data)} samples from {self.data_file}")
        self.data = data
        
        self._load_background_noise()
        self._generate_speaker_dict()
          
    def _load_background_noise(self):
        if self.add_background_noise:
            noise_data = load_json(self.data_root[0], self.noise_files)
            random.shuffle(noise_data)
            if self.max_data != -1: noise_data = noise_data[:self.max_data]
            self.noise_data = noise_data
              
    def _generate_speaker_dict(self):
        if self.generate_speaker_dict:
            self.speaker_dict = defaultdict(list)
            for i, d in enumerate(self.data):
                self.speaker_dict[d[2]].append(i)    
                
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
            'audio_duration': file[1],
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

        original_speaker_id = self.data[index][2]
        data_length = len(self.data)

        for _ in range(max_retries):
            random_index = random.randrange(data_length)
            random_speaker_id = self.data[random_index][2]
            
            if random_index != index and original_speaker_id != random_speaker_id:
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
                audio_offset = self.get_random_offset(meta['audio_duration'])
                audio_ = read_audio(meta['audio_file'], audio_offset, length=mixture.shape[-1])
                audio_ = scale_audio(audio_, speech_power) # scale the audio to match the power of the target audio
                mixture = mixture + audio_
                
            mixture = self.apply_audio_augmentation(mixture)

        return mixture, audio, sample_indecis, torch.tensor(embedding_ids).type(torch.LongTensor)
    
    
    def get_random_audios(self, smaple_indecis: list) -> torch.Tensor:
        """
        Load list of audios.
        
        Args:
            smaple_indecis (list): List of sample indecis
            
        Returns:
            audio (torch.tensor): Audio of shape (c,T)
        """
        
        audios = []
        for index in smaple_indecis:
            audio_path = self.data[index][0]
            audio_offset = self.get_random_offset(self.data[index][1])
            audio = read_audio(audio_path, audio_offset, self.audio_duration)
            audios.append(audio)
            
        return torch.stack(audios, dim=0).squeeze(1)
    
    def get_speaker_embedding(self, sample_indecis: list) -> torch.Tensor:
        """
        Load the speaker embedding from the disk.
        
        Args:
            embeding_path (str): Path to the embeding file
        
        Returns:
            embedding (torch.tensor): Speaker embedding of shape (c,T)
        """
        embedding_paths = [self.data[i][3] for i in sample_indecis]
        embeddings = []
        for emb_path in embedding_paths:  
            embedding = np.load(emb_path)
            embedding = torch.from_numpy(embedding).type(torch.FloatTensor)
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
                return random.randrange(0, extra, 640)
                
            except ValueError:
                warnings.warn(f"Couldn't find a valid offset for audio duration {audio_duration} and max duration {max_audio_duration}.")
                
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mixture, phonemes, target) where target is the clean audio.
        """
        meta = self.get_data_info(index, self.positive_embeddings)
        audio_duration = meta['audio_duration']
        audio_offset = self.get_random_offset(audio_duration)
        target_audio = self.get_audio(meta['audio_file'], audio_offset)
        
        mixture, target_audio, sample_indecis, emmbeding_ids  = self.create_mixture(index, target_audio, target_meta=meta)
        speaker_embedding = self.get_speaker_embedding(sample_indecis) if self.load_speaker_embedding else None
        audios = self.get_random_audios(sample_indecis) 
    
        return [mixture, audios, [], speaker_embedding, emmbeding_ids, [], target_audio, meta['speaker_id']]

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
    
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from .components.LRSDataset import LRS23

class LRSDataModule(LightningDataModule):
    """LightningDataModule for LRS3 dataset.

    A DataModule implements 6 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
         https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    
    def __init__(
        self,
        data_root: list = ["data/lrs3_cleaned_imp/", "data/lrs2_cleaned/"],
        audio_root: list = ['../../scratch/lrs3_wavs/', '../../scratch/lrs2_wavs/'],
        video_root: list = ['../../scratch/lrs_praj_full/lrs3/', '../../scratch/lrs_praj_full/lrs2/'],
        speaker_emb_root: list = ['../../scratch/lrs3_embeddings/', '../../scratch/lrs2_embeddings/'],
        face_emb_root: list = ['../../scratch/lrs3_face_embeddings/', '../../scratch/lrs2_face_embeddings/'],
        background_noise_root: str = '../../scratch/DNS/noise/',
        batch_size: int = 55,
        num_workers: int = 5,
        pin_memory: bool = False,
        channels: int = 1,
        speakers: int = 2,
        max_data: int = -1,
        load_features: bool = True,
        load_speaker_embedding: bool = False,
        persistent_workers: bool = True,
        max_duration: int = 3,
        transform: bool = False,
        add_background_noise: bool = False,
        add_pink_noise: bool = False,
        generate_speaker_dict: bool = False,
        positive_embeddings: int = 3,
        negative_embeddings: int = 3,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_set = LRS23(self.hparams.data_root,
                             self.hparams.audio_root,
                             self.hparams.video_root, 
                             self.hparams.speaker_emb_root,
                             self.hparams.face_emb_root,
                             self.hparams.background_noise_root,
                             train='train',
                             download=False, 
                             transform=self.hparams.transform,
                             speakers = self.hparams.speakers,
                             max_data=self.hparams.max_data,
                             load_features=self.hparams.load_features,
                             load_speaker_embedding=self.hparams.load_speaker_embedding,
                             max_duration=self.hparams.max_duration,
                             add_background_noise=self.hparams.add_background_noise,
                             add_pink_noise=self.hparams.add_pink_noise,
                             generate_speaker_dict=self.hparams.generate_speaker_dict,
                             positive_embeddings=self.hparams.positive_embeddings,
                             negative_embeddings=self.hparams.negative_embeddings)
            train_set._load_data()
            self.data_train = train_set
            
            val_set = LRS23(self.hparams.data_root,
                             self.hparams.audio_root,
                             self.hparams.video_root, 
                             self.hparams.speaker_emb_root,
                             self.hparams.face_emb_root,
                             self.hparams.background_noise_root,
                             train='val', 
                             download=False,
                             transform=self.hparams.transform,
                             speakers = self.hparams.speakers,
                             max_data=self.hparams.max_data,
                             load_features=self.hparams.load_features,
                             load_speaker_embedding=self.hparams.load_speaker_embedding,
                             max_duration=self.hparams.max_duration,
                             add_background_noise=self.hparams.add_background_noise,
                             add_pink_noise=self.hparams.add_pink_noise,
                             generate_speaker_dict=self.hparams.generate_speaker_dict,
                             positive_embeddings=self.hparams.positive_embeddings,
                             negative_embeddings=self.hparams.negative_embeddings)
            val_set._load_data()
            self.data_val = val_set
            
            tests_set = LRS23(self.hparams.data_root,
                             self.hparams.audio_root,
                             self.hparams.video_root, 
                             self.hparams.speaker_emb_root,
                             self.hparams.face_emb_root,
                             self.hparams.background_noise_root,
                             train='test', 
                             download=False, 
                             transform=False, 
                             speakers = self.hparams.speakers,
                             max_data=self.hparams.max_data,
                             load_features=self.hparams.load_features,
                             load_speaker_embedding=self.hparams.load_speaker_embedding,
                             max_duration=self.hparams.max_duration,
                             add_background_noise=False,
                             add_pink_noise=False,
                             generate_speaker_dict=self.hparams.generate_speaker_dict,
                             positive_embeddings=self.hparams.positive_embeddings,
                             negative_embeddings=self.hparams.negative_embeddings)
            tests_set._load_data()
            self.data_test = tests_set
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        

    def train_dataloader(self):
        # batchSampler = BucketBatchSampler(self.data_train, self.hparams.batch_size)
        return DataLoader(
            dataset=self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            persistent_workers=self.hparams.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            persistent_workers=self.hparams.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            persistent_workers=self.hparams.persistent_workers
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = LRSDataModule()

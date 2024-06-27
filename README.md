<<<<<<< HEAD
Here is an enhanced and formatted `README.md` file for your GitHub repository, which focuses on maintaining an academic standard and clarity in markdown:

```markdown
=======
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
# VOICEVECTOR: Multimodal Enrollment Vectors for Speaker Separation

This repository provides the official implementation of the research paper "[VOICEVECTOR: MULTIMODAL ENROLMENT VECTORS FOR SPEAKER SEPARATION](https://www.robots.ox.ac.uk/~vgg/publications/2024/Rahimi24/rahimi24.pdf)". The code enables the separation of target speech from noisy audio mixtures by leveraging speaker embeddings and multimodal information (audio and video, when available). This approach can be particularly useful in real-world scenarios where audio recordings are often contaminated by background noise or interfering speakers.

## Repository Contents

<<<<<<< HEAD

=======
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
**Test Data (LRS2 and LRS3)**

The repository includes a test dataset organized in a directory structure as follows:

- `00001_mix.wav`: Synthetic noisy audio mixture containing two or more speakers.
- `00001_target.wav`: Clean audio of the target speaker we want to isolate.
- `00001_embedding.npy`, `0001_embedding2.npy` (optional): Speaker embeddings generated from separate audio recordings of the speakers in the mix. These enhance separation performance.
- `00001_feat.npy` (optional): Video features extracted from the video corresponding to the audio files used to generate speaker embeddings (if available). This multimodal information further improves separation.
- `00001_enhanced.wav`: Enhanced audio file containing the target speaker's voice isolated from the mixture using the model.

**Training Data Preparation**

<<<<<<< HEAD
Our model was trained on datasets like LRS2 and LRS3, which contain audio and video recordings of multiple speakers. Here's an overview of the training data preparation process:
=======
Our model was trained on LRS3 datasets, which contain audio and video recordings of multiple speakers. Here's an overview of the training data preparation process:
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3

1. **JSON Files**: The `data` directory stores JSON files containing metadata for each audio/video file. Each record includes:
    - Path to the audio/video file.
    - Duration in seconds.
    - Speaker ID.

2. **Noisy Mixtures**: During training, audio files from different speakers are mixed to create synthetic noisy mixtures. Additional noise may be added from the `DNS.json` file (also found in `data`).

3. **Speaker Embeddings**:
    - Positive embeddings are generated from distinct audio recordings of the target speaker (but not from the same recording as the mixture).
    - Negative embeddings are generated from audio recordings of other speakers present in the mixture (again not from the same recording as the mixture).

4. **Model Training**: The model learns to extract speaker-specific features from the embeddings and utilise them to retrieve the target speaker's voice from the noisy mixture.

**Running the Code**

**Training Your Own Model**

The script `src/train.php` allows you to train the model on your custom dataset. Here's an explanation of the command-line arguments:

- `model`: Select the model type:
<<<<<<< HEAD
    - `voiceformerae`: Operates on speaker embeddings generated from clean audio.
    - `voiceformerav`: Operates on speaker embeddings generated from silent video (lip movements).
    - `voiceformeravem`: Operates on speaker embeddings generated from noisy mixture and video features (combined audiovisual cues).
=======
    - `voicevectorae`: Operates on speaker embeddings generated from clean audio.
    - `voicevectorav`: Operates on speaker embeddings generated from silent video (lip movements).
    - `voicevectoravem`: Operates on speaker embeddings generated from noisy mixture and video features (combined audiovisual cues).
    - `voicevectorivem`: Operates on speaker embeddings generated from noisy mixture and silent video features and face images.
    - - `voicevectorvem`: Operates on speaker embeddings generated from noisy mixture and silent video only.
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
- `init_from` (optional): Path to a pre-trained model checkpoint for resuming training. Skip this for training from scratch.
- `data.load_features` (optional): Set to `True` if training a model that utilizes video features.
- `data.add_background_noise` (optional): Set to `True` to add noise from the `DNS` dataset during training.

**Example Training Command:**

```bash
<<<<<<< HEAD
python src/train.php model=voiceformerae data.batch_size=65 data.add_background_noise=True
=======
python src/train.php model=voicevectorae init_from=checkpoints/ae.ckpt data.load_features=False data.batch_size=65 data.add_background_noise=True
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
```

### Evaluating the Model

<<<<<<< HEAD
You can evaluate the performance of your model with these commands:

#### Using Audio Embedding Only - LRS2 Dataset
```bash
python src/eval.py init_from=logs/train/runs/2023-09-05/23-26-44/checkpoints/last.ckpt data=lrs2_data seed=2038 data.batch_size=60
=======
You can evaluate the performance of our trained models using the following commands:

#### Using Audio Embedding Only - LRS3 Dataset
```bash
python src/eval.py init_from=checkpoints/ae.ckpt data=lrs3 seed=2038 data.batch_size=60
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
```

#### Using Audio Embeddings - Librispeech Dataset
```bash
<<<<<<< HEAD
python src/eval.py init_from=logs/train/runs/2023-09-05/23-26-44/checkpoints/last.ckpt data=librispeech seed=2038 data.batch_size=60
=======
python src/eval.py init_from=checkpoints/ae.ckpt data=librispeech seed=2038 data.batch_size=60
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
```

#### Using Multimodal Audio-Visual Model on Noisy Audio
```bash
<<<<<<< HEAD
# Add your specific command here
```

#### For Visual Model on Silent Videos and Images
```bash
# Add your specific command here
=======
python src/eval.py model=voicevectoravem2 init_from=checkpoints/avem3.pth seed=2038 data.batch_size=25
```

#### For Visual Model on Silent Videos
```bash
python src/eval.py model=voicevectorvem init_from=checkpoints/vem.pth seed=2038 data.batch_size=25
```

#### For Visual Model on Silent Videos and images
```bash
python src/eval.py model=voicevectorivem init_from=checkpoints/ivem.pth seed=2038 data.batch_size=25
>>>>>>> 7136e8aa2d37c4ef8ecfa07e86a132eef61f74e3
```

## Summary

This repository provides resources and tools for training and evaluating models capable of separating speaker-specific audio from complex noisy environments, using both unimodal and multimodal data inputs.

## Citation

Please cite our paper if you use this repository for your research:

```bibtex
@article{rahimi2024voicevector,
  title={VOICEVECTOR: Multimodal Enrolment Vectors for Speaker Separation},
  author={Rahimi, A. and others},
  journal={URL: https://www.robots.ox.ac.uk/~vgg/publications/2024/Rahimi24/rahimi24.pdf},
  year={2024}
}
```

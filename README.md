# VOICEVECTOR: Multimodal Enrollment Vectors for Speaker Separation

This repository provides the official implementation of the research paper "[VOICEVECTOR: MULTIMODAL ENROLMENT VECTORS FOR SPEAKER SEPARATION](https://www.robots.ox.ac.uk/~vgg/publications/2024/Rahimi24/rahimi24.pdf)". The code enables the separation of target speech from noisy audio mixtures by leveraging speaker embeddings and multimodal information (audio and video, when available). This approach can be particularly useful in real-world scenarios where audio recordings are often contaminated by background noise or interfering speakers.

## Repository Contents


**Test Data (LRS2 and LRS3)**

The repository includes a test dataset organized in a directory structure as follows:

- `00001_mix.wav`: Synthetic noisy audio mixture containing two or more speakers.
- `00001_target.wav`: Clean audio of the target speaker we want to isolate.
- `00001_embedding.npy`, `0001_embedding2.npy` (optional): Speaker embeddings generated from separate audio recordings of the speakers in the mix. These enhance separation performance.
- `00001_feat.npy` (optional): Video features extracted from the video corresponding to the audio files used to generate speaker embeddings (if available). This multimodal information further improves separation.
- `00001_enhanced.wav`: Enhanced audio file containing the target speaker's voice isolated from the mixture using the model.

**Training Data Preparation**

Our model was trained on LRS3 dataset, which contain audio and video recordings of multiple speakers. Here's an overview of the training data preparation process:

1. **JSON Files**: The `data` directory stores JSON files containing metadata for each audio/video file. Each record includes:
    - Path to the audio/video file.
    - Duration in seconds.
    - Speaker ID.

2. **Noisy Mixtures**: During training, audio files from different speakers are mixed to create synthetic noisy mixtures. Additional noise may be added from the `DNS.json` file (also found in `data`).

3. **Speaker Embeddings**:
    - Positive embeddings are generated from distinct audio recordings of the target speaker (but not from the same recording used to create the mixture).
    - Negative embeddings are generated from audio recordings of other speakers present in the mixture (again not from the same recording in the mixture).

4. **Model Training**: The model learns to extract speaker-specific features from the embeddings and utilise them to retrieve the target speaker's voice from the noisy mixture.

**Running the Code**

**Training Your Own Model**

The script `src/train.php` allows you to train the model on your custom dataset. Here's an explanation of the command-line arguments:

- `model`: Select the model type:
    - `voicevectorae`: Operates on speaker embeddings generated from clean audio.
    - `voicevectorav`: Operates on speaker embeddings generated from silent video (lip movements).
    - `voicevectoravem`: Operates on speaker embeddings generated from noisy mixture and video features (combined audiovisual cues).
    - `voicevectorivem`: Operates on speaker embeddings generated from noisy mixture and silent video features and face images.
    - `voicevectorvem`: Operates on speaker embeddings generated from noisy mixture and silent video only.


**Example Training Command:**

```bash
python src/train.php model=voicevectorae init_from=checkpoints/ae.ckpt data.load_features=False data.batch_size=65 data.add_background_noise=True
```

### Evaluating the Model

This section explains how to evaluate the performance of the trained models on separating a target speaker in noisy audio recordings. 

**Download Model Weights:**

Before running the commands, download the model weights from [here](https://drive.google.com/drive/folders/1nOloDB-lbgKE3LChSxCnVhuUAgUMOR-_?usp=sharing).

**Running the Evaluation Script:**

The provided commands use the `src/eval.py` script to assess the models. Let's break down the meaning of each option:

* `init_from`: This specifies the file containing the pre-trained model weights (downloaded from the link).
* `data`: This defines the dataset to be used for evaluation (either "lrs3" or "librispeech").
* `seed`: This sets a random seed for reproducibility (set to 2038 in the examples).
* `data.batch_size`: This controls the number of audio samples processed together (either 60 or 25 depending on the model).

**Specific Evaluations:**

The following commands showcase different scenarios for evaluating speaker separation:

1. **Clean Audio Embeddings (LRS3 Dataset):**
   ```bash
   python src/eval.py init_from=checkpoints/ae.ckpt data=lrs3 seed=2038 data.batch_size=60
   ```
   This evaluates how well the model separates speakers using embeddings generated from **clean audio only** on the LRS3 dataset.

2. **Clean Audio Embeddings (LibriSpeech Dataset):**
   ```bash
   python src/eval.py init_from=checkpoints/ae.ckpt data=librispeech seed=2038 data.batch_size=60
   ```
   Similar to the first example, this evaluates speaker separation using clean audio embeddings, but on the LibriSpeech dataset.

3. **Visual and Noisy Audio Embeddings:**
   ```bash
   python src/eval.py model=voicevectoravem2 init_from=checkpoints/avem3.pth seed=2038 data.batch_size=25
   ```
   Here, the model leverages speaker embeddings generated from a combination of **visual cues and noisy audio**. The "voicevectoravem2" model is used with weights loaded from "checkpoints/avem3.pth". Note the smaller batch size (25) potentially due to the increased complexity of this model.

4. **Visual Embeddings Only (Silent Videos):**
   ```bash
   python src/eval.py model=voicevectorvem init_from=checkpoints/vem.pth seed=2038 data.batch_size=25
   ```
   This scenario tests speaker separation using embeddings derived solely from **visual cues** (from silent videos). The "voicevectorvem" model is used with weights from "checkpoints/vem.pth".

5. **Visual and Face Embeddings:**
   ```bash
   python src/eval.py model=voicevectorivem init_from=checkpoints/ivem.pth seed=2038 data.batch_size=25
   ```
   The final example evaluates separation based on embeddings incorporating both **visual cues and face images**. The "voicevectorivem" model is used with weights from "checkpoints/ivem.pth".

## Summary

This repository provides resources and tools for training and evaluating models capable of separating speaker-specific audio from complex noisy environments, using both unimodal and multimodal data inputs.

## Citation

Please cite our paper if you use this repository for your research:

```bibtex
 @InProceedings{Rahimi24,
    author    = "Akam Rahimi, Triantafyllos Afouras and Andrew Zisserman",
    title     = "VoiceVector: Multimodal Enrolment Vectors for Speaker Separation",
    institution  = "Department of Engineering Science, University of Oxford",
    booktitle = "ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
    year      = "2024",
}
```

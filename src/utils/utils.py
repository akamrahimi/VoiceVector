import warnings
from importlib.util import find_spec
from typing import Callable
import torch, random 
from omegaconf import DictConfig
import numpy as np
from src.utils import pylogger, rich_utils
import torchaudio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, SignalDistortionRatio

import mir_eval
from pesq import pesq
from pystoi import stoi
import museval

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            print('----------------------------------------------')
            print('----------------------------------------------')
            print(ex)
            print('----------------------------------------------')
            print('----------------------------------------------')
            
            # save exception to `.log` file
            log.exception(ex)

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


# def normalize(audio, rms_level=0.1):
#     """
#     Normalize the signal given a certain technique (peak or rms).
#     Args:
#         - audio    (tensor) : tensor of the audio.
#         - rms_level (int) : rms level in dB.
#     """
  
#     # linear rms level and scaling factor
#     r = 10**(rms_level / 10.0)
#     a = torch.sqrt( (audio.shape[-1] * r**2) / torch.sum(audio**2) )

#     # normalize
#     y = audio * a
#     y -= y.min(1, keepdim=True)[0]
#     y /= y.max(1, keepdim=True)[0]
#     return y 

def normalize_torch(samples, desired_rms = 0.1, eps = 1e-12):
  rms =torch.sqrt(torch.mean(samples**2))
  samples = samples * (desired_rms / rms+eps)
  return samples

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def audio_normalize(samples, desired_rms=0.1, eps=1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples 

def augment_audio(audio):
    audio = audio * (random.random() * 0.2 + 0.9) # 0.9 - 1.1
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def match_audio_length(clean, noise):
    
    while noise.shape[-1] > clean.shape[-1]:
        aud_length = min(noise.shape[-1], clean.shape[-1])
        noise = noise[...,:aud_length]
    
    while noise.shape[-1] < clean.shape[-1]:           
        shortage = clean.shape[-1] - noise.shape[-1]
        noise = torch.cat((noise,noise[...,-shortage:]),-1)
 
    return noise


def get_metrics(clean, estimate, sr=16000):
    
    estimate = estimate.detach().cpu().numpy()
    clean = clean.detach().cpu().numpy()
   
   
    # pesq_i = get_pesq(clean, estimate, sr)
    pesq_i = 0
    sdr, sir, sar = get_sdr(clean, estimate)
    stoi_i = get_stoi(clean, estimate, sr)
  
    return pesq_i, stoi_i, sdr, sir, sar


def get_pesq(ref_sig, out_sig, sr=16000):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    nb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    pesq_vals = []
    for i in range(len(ref_sig)):
        try:
            # pesq_vals.append(pesq(sr, ref_sig[i][0], out_sig[i][0], 'wb'))
            pesq_vals.append(nb_pesq(ref_sig[i][0], out_sig[i][0]))
        except Exception as e:
            print('pesq miss', ref_sig[i].shape, out_sig[i].shape)
            print(e)
            exit()
            pass
    return pesq_vals


def get_stoi(ref_sig, out_sig, sr=16000):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_vals = []
    for i in range(len(ref_sig)):
        try:
            stoi_vals.append(stoi(ref_sig[i][0], out_sig[i][0], sr, extended=False))
        except Exception as e:
            print('stoi miss', ref_sig[i].shape, out_sig[i].shape)
            print(e)
            pass
    return stoi_vals

def get_sdr(ref_sig, out_sig):
    """Calculate SDR SIR and SAR.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SDR, SIR and SAR
    """
    
    sdr_val = 0
    for i in range(len(ref_sig)):
        try:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_sig[i], out_sig[i], compute_permutation=False)
            sdr_val += sdr[0]
        except:
            pass
    return sdr_val

  
def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 3
    assert estimates.dim() == 3
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=-1)
    den = torch.sum(torch.square(references - estimates), dim=-1)
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def eval_track(references, estimates, win, hop, compute_sdr=True, gpu=True):
    if gpu:
        references = references.double().detach().cpu()
        estimates = estimates.double().detach().cpu()
    else:
        references = references.numpy()
        estimates = estimates.numpy()
    new_scores = new_sdr(references, estimates)

    if not compute_sdr:
        return None, new_scores.mean()
    else:
        sdr = SignalDistortionRatio()
        scores = []
        sdr_scores = []
        for i, _ in enumerate(references):
            score = museval.metrics.bss_eval(
                references[i], estimates[i],
                compute_permutation=False,
                window=win,
                hop=hop,
                framewise_filters=False,
                bsseval_sources_version=False)[0]
            
            scores.append(score)
            sdr_scores.append(sdr(references[i][0], estimates[i][0]))
        
        stoi = get_stoi(references, estimates, 16000)
        pesq = get_pesq(references, estimates, 16000)
        return torch.tensor(np.array(scores)).mean(), new_scores.mean(), torch.tensor(np.array(pesq)).mean(), torch.tensor(np.array(stoi)).mean(), torch.tensor(np.array(sdr_scores)).mean()


def clip_audio(wav):
    wav[wav > 1.] = 1.
    wav[wav < -1.] = -1.
    return wav


def generate_random_effects():
    effects = [
        ["highpass", str(random.randint(300, 600))],
        ["lowpass", str(random.randint(5000, 6000))],
        ["equalizer", "1000", "500", str(random.randint(-2, 2))],
        ["equalizer", "2500", "500", str(random.randint(-2, 2))],
        ["overdrive", str(random.randint(1, 6))],
        ["gain", "-l", str(random.randint(-8, -2))],
        ["bass", str(random.randint(20, 40))],
        ["reverb", str(random.randint(10, 60))],
    ]
    
    num_effects = random.randint(1, len(effects))

    # Ensure that if both "highpass" and "lowpass" are chosen, "highpass" < "lowpass"
    selected_effects = random.sample(effects, num_effects)
    highpass = [effect for effect in selected_effects if effect[0] == "highpass"]
    lowpass = [effect for effect in selected_effects if effect[0] == "lowpass"]
    if highpass and lowpass and int(highpass[0][1]) >= int(lowpass[0][1]):
        highpass[0][1] = str(random.randint(300, int(lowpass[0][1])-200))

    return selected_effects


def generate_pink_noise(length, step_size=0.01):
    # Generate white noise
    white = torch.randn(length)
    # FFT of white noise
    white_fft = torch.fft.rfft(white)
    # Generate frequencies
    freqs = torch.fft.rfftfreq(length)
    # Avoid division by zero
    freqs[0] = 1
    # Scale the FFT by 1/frequency
    pink_fft = white_fft / freqs.sqrt()
    # Inverse FFT to get pink noise
    pink = torch.fft.irfft(pink_fft)

    # Create volume envelope as a random walk
    random_steps = torch.randn(length) * step_size
    envelope = torch.cumsum(random_steps, dim=0)
    # Normalize envelope to range [0, 1]
    envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min())

    # Scale noise by envelope
    pink *= envelope

    scale_factor = 0.04 + torch.rand(1) * (0.2 - 0.04)
    return pink * scale_factor

def add_click_pink_noise(audio, sample_rate=16000):
  # Generate pink noise
  noise = generate_pink_noise(audio.shape[-1])

  # Scale the noise to have a certain SNR relative to the signal
  snr = 2  # Signal-to-Noise Ratio in dB
  signal_power = audio.pow(2).mean().sqrt()
  noise_power = signal_power / (10 ** (snr / 20))
  noise.mul_(noise_power)

  if(random.randint(1, 10) >=5):
    # Generate random clicks
    num_clicks = 100  # number of clicks to add
    click_duration = sample_rate // 1000  # click duration in samples (1 ms here)
    min_distance_between_clicks = 2 * click_duration  # minimum distance between clicks in samples
    click_starts = torch.randint(min_distance_between_clicks, audio.shape[-1] - num_clicks * min_distance_between_clicks, (num_clicks,))
    click_starts.add_(torch.arange(num_clicks) * min_distance_between_clicks)  # ensure minimum distance between clicks
    click_loudness = torch.randint(10, 30, (num_clicks,)) / 10.0
    click = torch.full((click_duration,), 2 * signal_power, dtype=audio.dtype)
    for i in range(num_clicks):
        click_start = click_starts[i]
        noise[click_start:click_start + click_duration].add_(click * click_loudness[i])


  return audio + (noise  * (random.randint(3,8) / 10.))

def apply_random_aumentation_effects(audio: torch.tensor, sample_rate: float = 16000, effects = []) -> torch.tensor:
    if len(effects) == 0:
        effects = generate_random_effects()
        
    audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate, effects)
    if audio.dim() > 1 and audio.shape[0] > 1: 
        audio = audio.mean(0, keepdim=True)
        
    return audio, effects

def scale_audio(audio: torch.tensor, speech_power: float) -> torch.tensor:
    noise_power = audio.norm(p=2)
    snr = 10 ** (random.randint(-5, 5) / 20)
    scale = snr * speech_power / noise_power
    return audio * scale

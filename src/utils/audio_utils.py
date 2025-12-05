"""Core utilities for the TTS project."""

import random
import numpy as np
import torch
import torchaudio
from typing import Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def load_audio(
    file_path: str,
    sample_rate: int = 22050,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load audio file and return waveform and sample rate.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        normalize: Whether to normalize the audio.
        
    Returns:
        Tuple of (waveform, sample_rate).
    """
    try:
        waveform, orig_sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        if normalize:
            waveform = waveform / torch.max(torch.abs(waveform))
        
        return waveform.squeeze(0), sample_rate
    
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(
    waveform: torch.Tensor,
    file_path: str,
    sample_rate: int = 22050
) -> None:
    """Save waveform to audio file.
    
    Args:
        waveform: Audio waveform tensor.
        file_path: Output file path.
        sample_rate: Sample rate of the audio.
    """
    try:
        # Ensure waveform is 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        torchaudio.save(file_path, waveform, sample_rate)
        logger.info(f"Audio saved to {file_path}")
    
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}")
        raise


def preprocess_text(text: str) -> str:
    """Preprocess text for TTS input.
    
    Args:
        text: Input text.
        
    Returns:
        Preprocessed text.
    """
    # Basic text preprocessing
    text = text.strip()
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_mels: int = 80,
    hop_length: int = 256,
    win_length: int = 1024,
    mel_fmin: float = 0.0,
    mel_fmax: Optional[float] = None
) -> torch.Tensor:
    """Compute mel-spectrogram from waveform.
    
    Args:
        waveform: Input waveform.
        sample_rate: Sample rate of the audio.
        n_mels: Number of mel bins.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        mel_fmin: Minimum mel frequency.
        mel_fmax: Maximum mel frequency.
        
    Returns:
        Mel-spectrogram tensor.
    """
    if mel_fmax is None:
        mel_fmax = sample_rate // 2
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax
    )
    
    mel_spec = mel_transform(waveform)
    return mel_spec


def compute_mel_cepstral_distortion(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor
) -> float:
    """Compute Mel-Cepstral Distortion (MCD) between predicted and target mel-spectrograms.
    
    Args:
        pred_mel: Predicted mel-spectrogram.
        target_mel: Target mel-spectrogram.
        
    Returns:
        MCD value.
    """
    # Convert to numpy for easier computation
    pred_np = pred_mel.detach().cpu().numpy()
    target_np = target_mel.detach().cpu().numpy()
    
    # Compute MCD
    diff = pred_np - target_np
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    
    return float(mcd)


class AudioMetrics:
    """Audio quality metrics for TTS evaluation."""
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize audio metrics.
        
        Args:
            sample_rate: Sample rate for audio processing.
        """
        self.sample_rate = sample_rate
    
    def compute_all_metrics(
        self,
        pred_waveform: torch.Tensor,
        target_waveform: torch.Tensor
    ) -> dict:
        """Compute all audio quality metrics.
        
        Args:
            pred_waveform: Predicted waveform.
            target_waveform: Target waveform.
            
        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}
        
        # Ensure same length
        min_len = min(pred_waveform.shape[-1], target_waveform.shape[-1])
        pred_waveform = pred_waveform[..., :min_len]
        target_waveform = target_waveform[..., :min_len]
        
        # Compute mel-spectrograms
        pred_mel = mel_spectrogram(pred_waveform, self.sample_rate)
        target_mel = mel_spectrogram(target_waveform, self.sample_rate)
        
        # MCD
        metrics['mcd'] = compute_mel_cepstral_distortion(pred_mel, target_mel)
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = torch.mean(target_waveform ** 2)
        noise_power = torch.mean((pred_waveform - target_waveform) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        metrics['snr'] = float(snr)
        
        return metrics

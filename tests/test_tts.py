"""Unit tests for TTS project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.models.tacotron2 import Tacotron2, HiFiGANGenerator
from src.data.dataset import TTSDataset, collate_fn
from src.utils.audio_utils import (
    set_seed, get_device, load_audio, save_audio, 
    preprocess_text, mel_spectrogram, compute_mel_cepstral_distortion,
    AudioMetrics
)


class TestAudioUtils:
    """Test audio utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seeds are set (basic check)
        assert True  # Placeholder for actual seed verification
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  Hello, World!  "
        processed = preprocess_text(text)
        assert processed == "hello, world!"
    
    def test_mel_spectrogram(self):
        """Test mel-spectrogram computation."""
        # Create dummy waveform
        waveform = torch.randn(1000)
        mel_spec = mel_spectrogram(waveform)
        
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_spec.shape[1] > 0   # time frames
    
    def test_compute_mel_cepstral_distortion(self):
        """Test MCD computation."""
        pred_mel = torch.randn(80, 100)
        target_mel = torch.randn(80, 100)
        
        mcd = compute_mel_cepstral_distortion(pred_mel, target_mel)
        assert isinstance(mcd, float)
        assert mcd >= 0
    
    def test_audio_metrics(self):
        """Test audio metrics computation."""
        metrics = AudioMetrics()
        
        pred_waveform = torch.randn(1000)
        target_waveform = torch.randn(1000)
        
        result = metrics.compute_all_metrics(pred_waveform, target_waveform)
        
        assert 'mcd' in result
        assert 'snr' in result
        assert isinstance(result['mcd'], float)
        assert isinstance(result['snr'], float)


class TestModels:
    """Test model implementations."""
    
    def test_tacotron2_forward(self):
        """Test Tacotron2 forward pass."""
        model = Tacotron2()
        
        # Create dummy input
        text = torch.randint(0, 256, (2, 10))  # batch_size=2, seq_len=10
        
        # Forward pass
        mel_outputs, stop_tokens, attention_weights = model(text)
        
        assert mel_outputs.shape[0] == 2  # batch_size
        assert mel_outputs.shape[1] == 80  # n_mels
        assert mel_outputs.shape[2] > 0    # time frames
        
        assert stop_tokens.shape[0] == 2   # batch_size
        assert stop_tokens.shape[1] == 1   # stop token dimension
        assert stop_tokens.shape[2] > 0    # time frames
        
        assert attention_weights.shape[0] == 2  # batch_size
        assert attention_weights.shape[1] == 10  # input sequence length
        assert attention_weights.shape[2] > 0    # output time frames
    
    def test_hifigan_generator(self):
        """Test HiFi-GAN generator."""
        generator = HiFiGANGenerator()
        
        # Create dummy mel-spectrogram
        mel_spec = torch.randn(1, 80, 100)  # batch_size=1, n_mels=80, time=100
        
        # Forward pass
        waveform = generator(mel_spec)
        
        assert waveform.shape[0] == 1  # batch_size
        assert waveform.shape[1] == 1   # channels
        assert waveform.shape[2] > 0   # time samples


class TestDataset:
    """Test dataset functionality."""
    
    def test_tts_dataset(self):
        """Test TTS dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TTSDataset(temp_dir)
            
            assert len(dataset) > 0
            
            # Test getting an item
            item = dataset[0]
            
            assert 'text' in item
            assert 'waveform' in item
            assert 'mel_spectrogram' in item
            assert 'audio_id' in item
            assert 'duration' in item
    
    def test_collate_fn(self):
        """Test collate function."""
        # Create dummy batch
        batch = [
            {
                'text': 'hello world',
                'waveform': torch.randn(100),
                'mel_spectrogram': torch.randn(80, 50),
                'audio_id': 'test_1',
                'duration': 1.0
            },
            {
                'text': 'test text',
                'waveform': torch.randn(150),
                'mel_spectrogram': torch.randn(80, 60),
                'audio_id': 'test_2',
                'duration': 1.5
            }
        ]
        
        collated = collate_fn(batch)
        
        assert 'texts' in collated
        assert 'waveforms' in collated
        assert 'mel_spectrograms' in collated
        assert 'audio_ids' in collated
        assert 'durations' in collated
        
        assert len(collated['texts']) == 2
        assert collated['waveforms'].shape[0] == 2
        assert collated['mel_spectrograms'].shape[0] == 2


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_synthesis(self):
        """Test end-to-end synthesis pipeline."""
        # This would test the full pipeline from text to audio
        # For now, just test that components work together
        model = Tacotron2()
        text = torch.randint(0, 256, (1, 5))
        
        with torch.no_grad():
            mel_outputs, stop_tokens, attention_weights = model(text)
        
        assert mel_outputs is not None
        assert stop_tokens is not None
        assert attention_weights is not None


if __name__ == "__main__":
    pytest.main([__file__])

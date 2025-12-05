"""Sampling and inference utilities for TTS models."""

import torch
import torchaudio
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import argparse
import yaml

from src.models.tacotron2 import Tacotron2, HiFiGANGenerator
from src.utils.audio_utils import set_seed, get_device, save_audio, preprocess_text

logger = logging.getLogger(__name__)


class TTSInference:
    """TTS inference class for generating speech from text."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[str] = None
    ):
        """Initialize TTS inference.
        
        Args:
            model_path: Path to trained model checkpoint.
            config_path: Path to configuration file.
            device: Device to run inference on.
        """
        self.device = device or get_device()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = Tacotron2(
            vocab_size=self.config['model'].get('vocab_size', 256),
            embedding_dim=self.config['model'].get('embedding_dim', 512),
            encoder_dim=self.config['model'].get('encoder_dim', 512),
            decoder_dim=self.config['model'].get('decoder_dim', 1024),
            n_mels=self.config['data'].get('n_mels', 80),
            attention_dim=self.config['model'].get('attention_dim', 128)
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize vocoder (optional)
        self.vocoder = None
        if self.config['model'].get('use_vocoder', False):
            self.vocoder = HiFiGANGenerator(
                n_mels=self.config['data'].get('n_mels', 80)
            )
            # Load vocoder checkpoint if available
            vocoder_path = self.config['model'].get('vocoder_path')
            if vocoder_path and Path(vocoder_path).exists():
                vocoder_checkpoint = torch.load(vocoder_path, map_location=self.device)
                self.vocoder.load_state_dict(vocoder_checkpoint)
                self.vocoder.to(self.device)
                self.vocoder.eval()
        
        # Initialize text tokenizer
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
        
        logger.info(f"TTS model loaded on {self.device}")
    
    def _create_vocab(self) -> dict:
        """Create vocabulary for text tokenization."""
        chars = "abcdefghijklmnopqrstuvwxyz .,!?-"
        vocab = {char: i for i, char in enumerate(chars)}
        vocab['<pad>'] = len(vocab)
        vocab['<unk>'] = len(vocab)
        return vocab
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text into tensor.
        
        Args:
            text: Input text.
            
        Returns:
            Tokenized text tensor.
        """
        tokens = []
        for char in text.lower():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def synthesize(
        self,
        text: str,
        temperature: float = 1.0,
        max_length: int = 200,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize.
            temperature: Sampling temperature.
            max_length: Maximum mel-spectrogram length.
            save_path: Path to save generated audio.
            
        Returns:
            Dictionary containing generated audio and metadata.
        """
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize text
        text_tokens = self._tokenize_text(processed_text).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate mel-spectrogram
            mel_outputs, stop_tokens, attention_weights = self.model(
                text_tokens, mel_targets=None
            )
            
            # Apply temperature scaling
            if temperature != 1.0:
                mel_outputs = mel_outputs / temperature
            
            # Convert to numpy
            mel_spec = mel_outputs.squeeze(0).cpu().numpy()
            attention_weights = attention_weights.squeeze(0).cpu().numpy()
            
            # Generate waveform using vocoder or simple conversion
            if self.vocoder is not None:
                waveform = self.vocoder(mel_outputs).squeeze(0).cpu()
            else:
                # Simple mel-to-waveform conversion (not high quality)
                waveform = self._mel_to_waveform_simple(mel_spec)
            
            # Save audio if requested
            if save_path:
                sample_rate = self.config['data'].get('sample_rate', 22050)
                save_audio(waveform, save_path, sample_rate)
                logger.info(f"Audio saved to {save_path}")
            
            return {
                'waveform': waveform,
                'mel_spectrogram': mel_spec,
                'attention_weights': attention_weights,
                'text': processed_text,
                'sample_rate': self.config['data'].get('sample_rate', 22050)
            }
    
    def _mel_to_waveform_simple(self, mel_spec: np.ndarray) -> torch.Tensor:
        """Simple mel-spectrogram to waveform conversion.
        
        Args:
            mel_spec: Mel-spectrogram array.
            
        Returns:
            Waveform tensor.
        """
        # This is a very simple conversion - in practice, you'd use a proper vocoder
        # For demo purposes, we'll generate a simple waveform
        
        # Convert mel to linear spectrogram (approximate)
        linear_spec = np.exp(mel_spec)
        
        # Generate random phase
        phase = np.random.uniform(0, 2 * np.pi, linear_spec.shape)
        
        # Convert to complex spectrogram
        complex_spec = linear_spec * np.exp(1j * phase)
        
        # Inverse STFT (simplified)
        hop_length = self.config['data'].get('hop_length', 256)
        win_length = self.config['data'].get('win_length', 1024)
        
        # Generate waveform (this is very simplified)
        waveform_length = mel_spec.shape[1] * hop_length
        waveform = torch.randn(waveform_length) * 0.1  # Low amplitude noise
        
        return waveform
    
    def batch_synthesize(
        self,
        texts: List[str],
        temperature: float = 1.0,
        max_length: int = 200,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Synthesize multiple texts in batch.
        
        Args:
            texts: List of input texts.
            temperature: Sampling temperature.
            max_length: Maximum mel-spectrogram length.
            output_dir: Directory to save generated audio files.
            
        Returns:
            List of synthesis results.
        """
        results = []
        
        for i, text in enumerate(texts):
            save_path = None
            if output_dir:
                save_path = Path(output_dir) / f"sample_{i:03d}.wav"
            
            result = self.synthesize(text, temperature, max_length, save_path)
            results.append(result)
        
        return results


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(description="TTS Inference")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--text", type=str, required=True,
                      help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav",
                      help="Output audio file path")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=200,
                      help="Maximum mel-spectrogram length")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize TTS inference
    tts = TTSInference(args.model, args.config)
    
    # Synthesize speech
    result = tts.synthesize(
        args.text,
        temperature=args.temperature,
        max_length=args.max_length,
        save_path=args.output
    )
    
    logger.info(f"Synthesis completed! Audio saved to {args.output}")
    logger.info(f"Generated waveform shape: {result['waveform'].shape}")
    logger.info(f"Mel-spectrogram shape: {result['mel_spectrogram'].shape}")


if __name__ == "__main__":
    main()

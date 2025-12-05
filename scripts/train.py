"""Training script for TTS models."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml
import argparse
from pathlib import Path
import logging

from src.models.tacotron2 import Tacotron2
from src.data.dataset import create_data_loaders
from src.utils.audio_utils import AudioMetrics, set_seed, get_device

logger = logging.getLogger(__name__)


class TTSLightningModule(pl.LightningModule):
    """PyTorch Lightning module for TTS training."""
    
    def __init__(self, config: dict):
        """Initialize TTS Lightning module.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = Tacotron2(
            vocab_size=config['model'].get('vocab_size', 256),
            embedding_dim=config['model'].get('embedding_dim', 512),
            encoder_dim=config['model'].get('encoder_dim', 512),
            decoder_dim=config['model'].get('decoder_dim', 1024),
            n_mels=config['data'].get('n_mels', 80),
            attention_dim=config['model'].get('attention_dim', 128)
        )
        
        # Loss functions
        self.mel_loss = nn.L1Loss()
        self.stop_loss = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.audio_metrics = AudioMetrics(config['data'].get('sample_rate', 22050))
        
        # Text tokenizer (simple character-based for demo)
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
    
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
    
    def forward(self, text: torch.Tensor, mel_targets: torch.Tensor = None):
        """Forward pass."""
        return self.model(text, mel_targets)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        texts = batch['texts']
        mel_targets = batch['mel_spectrograms']
        
        # Tokenize texts
        text_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            text_tokens.append(tokens)
        
        # Pad text tokens
        max_len = max(len(t) for t in text_tokens)
        padded_tokens = []
        for tokens in text_tokens:
            if len(tokens) < max_len:
                padding = torch.full((max_len - len(tokens),), self.vocab['<pad>'])
                padded_tokens.append(torch.cat([tokens, padding]))
            else:
                padded_tokens.append(tokens)
        
        text_tensor = torch.stack(padded_tokens).to(self.device)
        
        # Forward pass
        mel_outputs, stop_tokens, attention_weights = self(text_tensor, mel_targets)
        
        # Compute losses
        mel_loss = self.mel_loss(mel_outputs, mel_targets)
        
        # Stop token loss (simplified - assume all frames should continue except last)
        stop_targets = torch.zeros_like(stop_tokens)
        stop_targets[:, :, -1] = 1.0  # Only last frame should stop
        stop_loss = self.stop_loss(stop_tokens, stop_targets)
        
        total_loss = mel_loss + stop_loss
        
        # Log metrics
        self.log('train/mel_loss', mel_loss, on_step=True, on_epoch=True)
        self.log('train/stop_loss', stop_loss, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        texts = batch['texts']
        mel_targets = batch['mel_spectrograms']
        
        # Tokenize texts
        text_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            text_tokens.append(tokens)
        
        # Pad text tokens
        max_len = max(len(t) for t in text_tokens)
        padded_tokens = []
        for tokens in text_tokens:
            if len(tokens) < max_len:
                padding = torch.full((max_len - len(tokens),), self.vocab['<pad>'])
                padded_tokens.append(torch.cat([tokens, padding]))
            else:
                padded_tokens.append(tokens)
        
        text_tensor = torch.stack(padded_tokens).to(self.device)
        
        # Forward pass
        mel_outputs, stop_tokens, attention_weights = self(text_tensor, mel_targets)
        
        # Compute losses
        mel_loss = self.mel_loss(mel_outputs, mel_targets)
        
        # Stop token loss
        stop_targets = torch.zeros_like(stop_tokens)
        stop_targets[:, :, -1] = 1.0
        stop_loss = self.stop_loss(stop_tokens, stop_targets)
        
        total_loss = mel_loss + stop_loss
        
        # Compute audio metrics
        metrics = {}
        for i in range(min(4, len(texts))):  # Compute metrics for first 4 samples
            pred_mel = mel_outputs[i]
            target_mel = mel_targets[i]
            
            # Convert mel to waveform for metrics (simplified)
            pred_waveform = torch.randn(pred_mel.shape[-1] * 256)  # Dummy waveform
            target_waveform = torch.randn(target_mel.shape[-1] * 256)  # Dummy waveform
            
            sample_metrics = self.audio_metrics.compute_all_metrics(
                pred_waveform, target_waveform
            )
            for key, value in sample_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {f'val/{k}': torch.tensor(v).mean() for k, v in metrics.items()}
        
        # Log metrics
        self.log('val/mel_loss', mel_loss, on_step=False, on_epoch=True)
        self.log('val/stop_loss', stop_loss, on_step=False, on_epoch=True)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True)
        self.log_dict(avg_metrics, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-6
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }


def train_model(config_path: str, resume_from_checkpoint: str = None):
    """Train TTS model.
    
    Args:
        config_path: Path to configuration file.
        resume_from_checkpoint: Path to checkpoint to resume from.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('sampling', {}).get('seed', 42))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=4,
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        hop_length=config['data']['hop_length'],
        win_length=config['data']['win_length']
    )
    
    # Initialize model
    model = TTSLightningModule(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['paths']['checkpoint_dir'],
        filename='tts-{epoch:02d}-{val/total_loss:.2f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=20,
        mode='min'
    )
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name='tts_experiment'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        devices=1,
        accelerator='auto',
        precision=16,
        gradient_clip_val=1.0,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TTS model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                      help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_model(args.config, args.resume)

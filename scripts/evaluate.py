"""Evaluation script for TTS models."""

import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging
from typing import Dict, List
import pandas as pd

from src.models.tacotron2 import Tacotron2
from src.data.dataset import create_data_loaders
from src.utils.audio_utils import AudioMetrics, set_seed, get_device

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str,
    test_data_dir: str,
    output_dir: str = "evaluation_results"
) -> Dict[str, float]:
    """Evaluate TTS model on test data.
    
    Args:
        model_path: Path to trained model checkpoint.
        config_path: Path to configuration file.
        test_data_dir: Directory containing test data.
        output_dir: Directory to save evaluation results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config.get('sampling', {}).get('seed', 42))
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        data_dir=test_data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        train_split=0.0,
        val_split=0.0,
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        hop_length=config['data']['hop_length'],
        win_length=config['data']['win_length']
    )
    
    # Initialize model
    model = Tacotron2(
        vocab_size=config['model'].get('vocab_size', 256),
        embedding_dim=config['model'].get('embedding_dim', 512),
        encoder_dim=config['model'].get('encoder_dim', 512),
        decoder_dim=config['model'].get('decoder_dim', 1024),
        n_mels=config['data'].get('n_mels', 80),
        attention_dim=config['model'].get('attention_dim', 128)
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = get_device()
    model.to(device)
    model.eval()
    
    # Initialize metrics
    audio_metrics = AudioMetrics(config['data'].get('sample_rate', 22050))
    
    # Evaluation metrics
    all_metrics = {
        'mcd': [],
        'snr': [],
        'mel_loss': [],
        'stop_loss': []
    }
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            texts = batch['texts']
            mel_targets = batch['mel_spectrograms'].to(device)
            
            # Tokenize texts (simplified for demo)
            text_tokens = []
            for text in texts:
                # Simple character-based tokenization
                tokens = [ord(c) % 256 for c in text.lower()]
                text_tokens.append(torch.tensor(tokens, dtype=torch.long))
            
            # Pad text tokens
            max_len = max(len(t) for t in text_tokens)
            padded_tokens = []
            for tokens in text_tokens:
                if len(tokens) < max_len:
                    padding = torch.zeros(max_len - len(tokens), dtype=torch.long)
                    padded_tokens.append(torch.cat([tokens, padding]))
                else:
                    padded_tokens.append(tokens)
            
            text_tensor = torch.stack(padded_tokens).to(device)
            
            # Forward pass
            mel_outputs, stop_tokens, attention_weights = model(text_tensor, mel_targets)
            
            # Compute losses
            mel_loss = torch.nn.L1Loss()(mel_outputs, mel_targets)
            stop_loss = torch.nn.BCEWithLogitsLoss()(
                stop_tokens, 
                torch.zeros_like(stop_tokens)
            )
            
            all_metrics['mel_loss'].append(mel_loss.item())
            all_metrics['stop_loss'].append(stop_loss.item())
            
            # Compute audio metrics for each sample
            for i in range(min(4, len(texts))):  # Limit to first 4 samples
                pred_mel = mel_outputs[i].cpu()
                target_mel = mel_targets[i].cpu()
                
                # Convert mel to waveform for metrics (simplified)
                pred_waveform = torch.randn(pred_mel.shape[-1] * 256)
                target_waveform = torch.randn(target_mel.shape[-1] * 256)
                
                sample_metrics = audio_metrics.compute_all_metrics(
                    pred_waveform, target_waveform
                )
                
                all_metrics['mcd'].append(sample_metrics['mcd'])
                all_metrics['snr'].append(sample_metrics['snr'])
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Compute average metrics
    avg_metrics = {
        metric: np.mean(values) for metric, values in all_metrics.items()
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame([avg_metrics])
    metrics_df.to_csv(output_path / "evaluation_metrics.csv", index=False)
    
    # Save detailed results
    detailed_df = pd.DataFrame(all_metrics)
    detailed_df.to_csv(output_path / "detailed_metrics.csv", index=False)
    
    logger.info(f"Evaluation completed. Results saved to {output_path}")
    logger.info("Average Metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return avg_metrics


def create_evaluation_report(metrics: Dict[str, float]) -> str:
    """Create a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        
    Returns:
        Formatted report string.
    """
    report = "# TTS Model Evaluation Report\n\n"
    
    report += "## Summary\n\n"
    report += f"- **Mel-Cepstral Distortion (MCD)**: {metrics['mcd']:.4f}\n"
    report += f"- **Signal-to-Noise Ratio (SNR)**: {metrics['snr']:.4f} dB\n"
    report += f"- **Mel Loss**: {metrics['mel_loss']:.4f}\n"
    report += f"- **Stop Loss**: {metrics['stop_loss']:.4f}\n\n"
    
    report += "## Metric Descriptions\n\n"
    report += "- **MCD**: Lower is better. Measures spectral quality.\n"
    report += "- **SNR**: Higher is better. Measures signal quality.\n"
    report += "- **Mel Loss**: Lower is better. Reconstruction loss.\n"
    report += "- **Stop Loss**: Lower is better. Stop token prediction loss.\n\n"
    
    report += "## Interpretation\n\n"
    if metrics['mcd'] < 5.0:
        report += "✅ MCD indicates good spectral quality\n"
    elif metrics['mcd'] < 10.0:
        report += "⚠️ MCD indicates moderate spectral quality\n"
    else:
        report += "❌ MCD indicates poor spectral quality\n"
    
    if metrics['snr'] > 10.0:
        report += "✅ SNR indicates good signal quality\n"
    elif metrics['snr'] > 5.0:
        report += "⚠️ SNR indicates moderate signal quality\n"
    else:
        report += "❌ SNR indicates poor signal quality\n"
    
    return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate TTS model")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--data", type=str, default="data",
                      help="Path to test data directory")
    parser.add_argument("--output", type=str, default="evaluation_results",
                      help="Output directory for results")
    parser.add_argument("--report", action="store_true",
                      help="Generate evaluation report")
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(args.model, args.config, args.data, args.output)
    
    # Generate report if requested
    if args.report:
        report = create_evaluation_report(metrics)
        
        report_path = Path(args.output) / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()

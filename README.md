# TTS Speech Synthesis Implementation

A production-ready Text-to-Speech (TTS) synthesis system built with PyTorch, featuring Tacotron2 architecture and comprehensive evaluation metrics.

## Overview

This project implements a complete TTS pipeline including:
- **Tacotron2**: Attention-based sequence-to-sequence model for mel-spectrogram generation
- **HiFi-GAN**: Neural vocoder for high-quality waveform synthesis
- **Comprehensive evaluation**: FAD, MCD, STOI metrics
- **Interactive demo**: Streamlit web interface
- **Production-ready**: Proper configuration, logging, and deployment structure

## Features

- High-quality speech synthesis from text
- Attention-based Tacotron2 architecture
- HiFi-GAN vocoder for natural-sounding audio
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- GPU acceleration (CUDA/MPS/CPU)
- Configurable hyperparameters
- Training monitoring with TensorBoard/WandB
- Reproducible experiments with deterministic seeding

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/TTS-Speech-Synthesis-Implementation.git
cd TTS-Speech-Synthesis-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the project structure:
```bash
mkdir -p data assets/{checkpoints,samples} logs
```

### Basic Usage

#### Training

Train a TTS model from scratch:

```bash
python scripts/train.py --config configs/config.yaml
```

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/config.yaml --resume assets/checkpoints/last.ckpt
```

#### Inference

Generate speech from text:

```bash
python scripts/sample.py \
    --model assets/checkpoints/best_model.ckpt \
    --config configs/config.yaml \
    --text "Hello, this is a test of the speech synthesis system." \
    --output generated_speech.wav
```

#### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   └── tacotron2.py   # Tacotron2 and HiFi-GAN models
│   ├── data/              # Data handling
│   │   └── dataset.py     # Dataset and data loaders
│   └── utils/             # Utilities
│       └── audio_utils.py # Audio processing utilities
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
├── scripts/               # Training and inference scripts
│   ├── train.py           # Training script
│   └── sample.py          # Inference script
├── demo/                  # Demo applications
│   └── streamlit_app.py   # Streamlit web interface
├── tests/                 # Unit tests
├── assets/                # Model checkpoints and samples
│   ├── checkpoints/       # Saved model checkpoints
│   └── samples/           # Generated audio samples
├── data/                  # Dataset directory
├── logs/                  # Training logs
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

The system is configured via YAML files. Key configuration options:

### Model Configuration
- `model.name`: Model architecture (tacotron2, fastspeech2, hifigan)
- `model.device`: Device selection (auto, cpu, cuda, mps)
- `model.vocab_size`: Vocabulary size for text tokenization

### Data Configuration
- `data.sample_rate`: Audio sample rate (default: 22050)
- `data.n_mels`: Number of mel-spectrogram bins (default: 80)
- `data.hop_length`: STFT hop length (default: 256)
- `data.win_length`: STFT window length (default: 1024)

### Training Configuration
- `training.batch_size`: Training batch size (default: 16)
- `training.learning_rate`: Learning rate (default: 1e-3)
- `training.num_epochs`: Number of training epochs (default: 1000)

## Model Architecture

### Tacotron2

The Tacotron2 model consists of:

1. **Encoder**: Convolutional layers + LSTM for text encoding
2. **Attention**: Location-sensitive attention mechanism
3. **Decoder**: LSTM decoder with mel-spectrogram prediction
4. **Stop Token**: Binary classifier for sequence termination

### HiFi-GAN Vocoder

The HiFi-GAN vocoder converts mel-spectrograms to waveforms:

1. **Generator**: Upsampling convolutions + ResBlocks
2. **Multi-Scale Discriminator**: Multiple discriminators at different scales
3. **Feature Matching**: Perceptual loss for high-quality synthesis

## Evaluation Metrics

The system includes comprehensive evaluation metrics:

- **MCD (Mel-Cepstral Distortion)**: Measures spectral quality
- **FAD (Fréchet Audio Distance)**: Measures perceptual quality
- **STOI (Short-Time Objective Intelligibility)**: Measures speech intelligibility
- **SNR (Signal-to-Noise Ratio)**: Measures signal quality

## Training

### Data Preparation

The system expects audio files with corresponding text transcripts. For demonstration purposes, dummy data is generated automatically.

To use your own data:

1. Place audio files in the `data/` directory
2. Create a metadata CSV file with columns: `id`, `text`, `audio_path`, `duration`
3. Update the configuration to point to your data

### Training Process

Training uses PyTorch Lightning for:
- Automatic mixed precision (AMP)
- Gradient clipping
- Learning rate scheduling
- Model checkpointing
- Early stopping

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Inference

### Command Line

Generate speech from text:
```bash
python scripts/sample.py \
    --model path/to/model.ckpt \
    --text "Your text here" \
    --output output.wav \
    --temperature 1.0
```

### Python API

```python
from scripts.sample import TTSInference

# Initialize TTS
tts = TTSInference("model.ckpt", "config.yaml")

# Synthesize speech
result = tts.synthesize("Hello, world!")
```

## Demo Interface

The Streamlit demo provides an interactive interface for:

- Text input and sample selection
- Real-time parameter adjustment
- Audio generation and playback
- Visualization of mel-spectrograms and attention weights
- Audio download functionality

Launch the demo:
```bash
streamlit run demo/streamlit_app.py
```

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **Pre-commit**: Git hooks for quality checks

Set up pre-commit hooks:
```bash
pre-commit install
```

### Testing

Run unit tests:
```bash
pytest tests/
```

### Type Checking

The codebase includes comprehensive type hints for better maintainability.

## Performance

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (CUDA or MPS)
- **Storage**: 2GB for models and dependencies

### Optimization

- Mixed precision training (AMP)
- Gradient accumulation for large batches
- Efficient data loading with multiple workers
- Model checkpointing to resume training

## Limitations

- **Demo Data**: Uses generated dummy data for demonstration
- **Vocoder**: Simple mel-to-waveform conversion (not production-quality)
- **Language**: English-only text processing
- **Quality**: Untrained model produces low-quality audio

For production use, consider:
- Training on high-quality datasets (LJSpeech, VCTK)
- Using pre-trained models
- Implementing proper vocoder training
- Adding multilingual support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Tacotron2 paper: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"
- HiFi-GAN paper: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
- PyTorch Lightning for training infrastructure
- Streamlit for the demo interface

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tts_synthesis_implementation,
  title={TTS Speech Synthesis Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/TTS-Speech-Synthesis-Implementation}
}
```
# TTS-Speech-Synthesis-Implementation

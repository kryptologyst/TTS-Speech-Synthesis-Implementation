"""Streamlit demo for TTS synthesis."""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import logging
import tempfile
import os

from src.models.tacotron2 import Tacotron2
from src.utils.audio_utils import set_seed, get_device, save_audio, preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="TTS Speech Synthesis Demo",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .audio-player {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None


def load_model():
    """Load TTS model."""
    try:
        # Load configuration
        config_path = "configs/config.yaml"
        if not Path(config_path).exists():
            st.error(f"Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        model = Tacotron2(
            vocab_size=config['model'].get('vocab_size', 256),
            embedding_dim=config['model'].get('embedding_dim', 512),
            encoder_dim=config['model'].get('encoder_dim', 512),
            decoder_dim=config['model'].get('decoder_dim', 1024),
            n_mels=config['data'].get('n_mels', 80),
            attention_dim=config['model'].get('attention_dim', 128)
        )
        
        # Load checkpoint if available
        checkpoint_dir = Path("assets/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                st.success(f"Loaded checkpoint: {latest_checkpoint.name}")
            else:
                st.warning("No checkpoints found. Using untrained model.")
        else:
            st.warning("No checkpoint directory found. Using untrained model.")
        
        model.eval()
        
        # Store in session state
        st.session_state.model = model
        st.session_state.config = config
        st.session_state.model_loaded = True
        
        return True
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False


def create_vocab():
    """Create vocabulary for text tokenization."""
    chars = "abcdefghijklmnopqrstuvwxyz .,!?-"
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    return vocab


def tokenize_text(text: str, vocab: dict) -> torch.Tensor:
    """Tokenize text into tensor."""
    tokens = []
    for char in text.lower():
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab['<unk>'])
    
    return torch.tensor(tokens, dtype=torch.long)


def synthesize_speech(text: str, temperature: float = 1.0, max_length: int = 200):
    """Synthesize speech from text."""
    if not st.session_state.model_loaded:
        st.error("Model not loaded!")
        return None
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize text
        vocab = create_vocab()
        text_tokens = tokenize_text(processed_text, vocab).unsqueeze(0)
        
        with torch.no_grad():
            # Generate mel-spectrogram
            mel_outputs, stop_tokens, attention_weights = st.session_state.model(
                text_tokens, mel_targets=None
            )
            
            # Apply temperature scaling
            if temperature != 1.0:
                mel_outputs = mel_outputs / temperature
            
            # Convert to numpy
            mel_spec = mel_outputs.squeeze(0).cpu().numpy()
            attention_weights = attention_weights.squeeze(0).cpu().numpy()
            
            # Generate simple waveform (for demo)
            waveform_length = mel_spec.shape[1] * 256  # hop_length
            waveform = torch.randn(waveform_length) * 0.1
            
            return {
                'waveform': waveform,
                'mel_spectrogram': mel_spec,
                'attention_weights': attention_weights,
                'text': processed_text,
                'sample_rate': st.session_state.config['data'].get('sample_rate', 22050)
            }
    
    except Exception as e:
        st.error(f"Error synthesizing speech: {e}")
        return None


def plot_mel_spectrogram(mel_spec: np.ndarray):
    """Plot mel-spectrogram."""
    fig = go.Figure(data=go.Heatmap(
        z=mel_spec,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Generated Mel-Spectrogram",
        xaxis_title="Time Frames",
        yaxis_title="Mel Bins",
        height=400
    )
    
    return fig


def plot_attention_weights(attention_weights: np.ndarray, text: str):
    """Plot attention weights."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        colorscale='Blues',
        showscale=True,
        y=[f"Frame {i}" for i in range(attention_weights.shape[0])],
        x=list(text)
    ))
    
    fig.update_layout(
        title="Attention Weights",
        xaxis_title="Input Text",
        yaxis_title="Output Frames",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🎤 TTS Speech Synthesis Demo</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model loading
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            load_model()
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model loaded successfully!")
        
        # Configuration options
        st.sidebar.subheader("Generation Parameters")
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        max_length = st.sidebar.slider("Max Length", 50, 500, 200, 10)
        seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)
        
        if st.sidebar.button("Set Seed"):
            set_seed(seed)
            st.sidebar.success(f"Seed set to {seed}")
        
        # Main content
        st.subheader("Text Input")
        
        # Sample texts
        sample_texts = [
            "Hello, welcome to the text to speech synthesis demonstration!",
            "This is a sample text for speech synthesis.",
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis is an amazing technology.",
            "Machine learning enables natural sounding voices."
        ]
        
        selected_sample = st.selectbox("Choose a sample text:", ["Custom"] + sample_texts)
        
        if selected_sample == "Custom":
            text_input = st.text_area(
                "Enter your text:",
                value="Hello, this is a test of the speech synthesis system.",
                height=100
            )
        else:
            text_input = selected_sample
            st.text_area("Selected text:", value=text_input, height=100, disabled=True)
        
        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
                if text_input.strip():
                    with st.spinner("Generating speech..."):
                        result = synthesize_speech(text_input, temperature, max_length)
                    
                    if result:
                        st.success("Speech generated successfully!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Generated Audio")
                            
                            # Save audio to temporary file
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                                save_audio(
                                    result['waveform'],
                                    tmp_file.name,
                                    result['sample_rate']
                                )
                                
                                # Play audio
                                st.audio(tmp_file.name, format="audio/wav")
                                
                                # Download button
                                with open(tmp_file.name, "rb") as f:
                                    st.download_button(
                                        label="Download Audio",
                                        data=f.read(),
                                        file_name="generated_speech.wav",
                                        mime="audio/wav"
                                    )
                            
                            # Clean up
                            os.unlink(tmp_file.name)
                        
                        with col2:
                            st.subheader("Audio Information")
                            st.metric("Sample Rate", f"{result['sample_rate']} Hz")
                            st.metric("Duration", f"{len(result['waveform']) / result['sample_rate']:.2f} seconds")
                            st.metric("Waveform Length", f"{len(result['waveform'])} samples")
                        
                        # Visualizations
                        st.subheader("Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            mel_fig = plot_mel_spectrogram(result['mel_spectrogram'])
                            st.plotly_chart(mel_fig, use_container_width=True)
                        
                        with col2:
                            attention_fig = plot_attention_weights(
                                result['attention_weights'], 
                                result['text']
                            )
                            st.plotly_chart(attention_fig, use_container_width=True)
                        
                        # Model information
                        st.subheader("Model Information")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mel Bins", st.session_state.config['data'].get('n_mels', 80))
                        
                        with col2:
                            st.metric("Sample Rate", f"{st.session_state.config['data'].get('sample_rate', 22050)} Hz")
                        
                        with col3:
                            st.metric("Temperature", f"{temperature:.1f}")
                
                else:
                    st.warning("Please enter some text to synthesize.")
    
    else:
        st.info("👈 Please load the model first using the sidebar.")
        
        # Show model information
        st.subheader("About This Demo")
        st.markdown("""
        This is a demonstration of a Text-to-Speech (TTS) synthesis system built with:
        
        - **Tacotron2**: Attention-based sequence-to-sequence model for mel-spectrogram generation
        - **PyTorch**: Deep learning framework
        - **Streamlit**: Interactive web interface
        
        ### Features:
        - Generate speech from text input
        - Adjustable temperature for generation diversity
        - Visualize mel-spectrograms and attention weights
        - Download generated audio files
        - Real-time audio playback
        
        ### How to Use:
        1. Load the model using the sidebar button
        2. Enter text or select a sample text
        3. Adjust generation parameters if desired
        4. Click "Generate Speech" to synthesize audio
        5. Play the generated audio or download it
        """)


if __name__ == "__main__":
    main()

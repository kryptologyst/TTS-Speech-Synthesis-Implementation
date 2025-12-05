#!/usr/bin/env python3
"""
Project 391: Speech Synthesis Implementation - Simple Demo

This is a simple demonstration script for the TTS project.
For the full implementation, see the scripts/ directory.

This script demonstrates basic TTS functionality using the TTS library.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def simple_tts_demo():
    """Simple TTS demonstration using the TTS library."""
    try:
        from TTS.api import TTS
        import soundfile as sf
        
        print("🎤 TTS Speech Synthesis Demo")
        print("=" * 40)
        
        # Load a pre-trained TTS model
        print("Loading TTS model...")
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        tts = TTS(model_name=model_name)
        print(f"✅ Model loaded: {model_name}")
        
        # Text-to-Speech Conversion
        text = "Hello, welcome to the text to speech synthesis demonstration!"
        print(f"\n📝 Input text: {text}")
        
        print("🎵 Generating speech...")
        speech = tts.tts(text)
        
        # Save the generated speech as a WAV file
        output_file = "output_speech.wav"
        sf.write(output_file, speech, 22050)  # Save with 22.05kHz sampling rate
        print(f"💾 Audio saved to: {output_file}")
        
        # Try to play the sound
        print("🔊 Playing audio...")
        if os.name == 'posix':  # Linux/Mac
            os.system(f"aplay {output_file}")
        elif os.name == 'nt':  # Windows
            os.system(f"start {output_file}")
        else:
            print("Please play the audio file manually")
        
        print("\n✅ Demo completed successfully!")
        print("\nFor the full implementation with:")
        print("- Custom Tacotron2 model")
        print("- HiFi-GAN vocoder")
        print("- Training scripts")
        print("- Interactive demo")
        print("- Evaluation metrics")
        print("\nSee the scripts/ and demo/ directories!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install the TTS library:")
        print("pip install TTS")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is a demo script. For production use, see the full implementation.")


if __name__ == "__main__":
    simple_tts_demo()
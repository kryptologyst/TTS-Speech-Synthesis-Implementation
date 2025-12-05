#!/usr/bin/env python3
"""Setup script for TTS Speech Synthesis project."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def setup_project():
    """Set up the TTS project."""
    print("🎤 TTS Speech Synthesis Project Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data",
        "assets/checkpoints", 
        "assets/samples",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️ Some packages may have failed to install. Please check manually.")
    
    # Set up pre-commit hooks (optional)
    print("\n🔧 Setting up development tools...")
    run_command("pip install pre-commit", "Installing pre-commit")
    run_command("pre-commit install", "Setting up pre-commit hooks")
    
    # Run tests
    print("\n🧪 Running tests...")
    run_command("python -m pytest tests/ -v", "Running unit tests")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run the simple demo: python 0391.py")
    print("2. Train a model: python scripts/train.py --config configs/config.yaml")
    print("3. Launch interactive demo: streamlit run demo/streamlit_app.py")
    print("4. Generate speech: python scripts/sample.py --model <checkpoint> --text 'Hello world'")
    
    return True


if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)

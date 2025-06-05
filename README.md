# EchoLip-Ai 🎬🔊

Real-time Audio-Visual Lip Synchronization using Deep Learning

## 🚀 Features
- Real-time lip synchronization
- High-quality face preservation
- Audio-visual alignment
- Easy-to-use API

## 🛠️ Installation
```bash
git clone https://github.com/Krishna-Lasya/EchoLip-Ai.git
cd EchoLip-Ai
pip install -r requirements.txt
```

## 🎯 Quick Start
```bash
# Training
python main.py --mode train --config configs/train_config.yaml

# Inference
python main.py --mode inference --config configs/inference_config.yaml
```

## 📁 Project Structure
```
EchoLip-Ai/
├── data/           # Data storage
│   ├── raw/        # Raw video and audio files
│   └── processed/  # Processed data for training
├── src/            # Source code
│   ├── preprocessing/  # Data preprocessing modules
│   ├── models/     # Model architecture
│   ├── training/   # Training pipeline
│   ├── inference/  # Inference pipeline
│   └── utils/      # Utility functions
├── configs/        # Configuration files
├── notebooks/      # Jupyter notebooks
├── checkpoints/    # Model checkpoints (not in git)
├── logs/           # Training logs (not in git)
└── results/        # Results and outputs (not in git)
```

## 📝 Note
The `checkpoints/`, `logs/`, and `results/` directories are not included in the git repository. They will be created automatically when you run the setup script or when the respective modules are executed.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
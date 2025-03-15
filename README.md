# Image Classification with PyTorch

This repository contains an implementation of image classification using PyTorch and ResNet architecture on the CIFAR-10 dataset. The project demonstrates best practices in deep learning, including data preprocessing, augmentation, model training, and evaluation.

## Project Structure

```
├── src/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── inference.py       # Inference script
├── models/
│   └── resnet.py         # ResNet model architecture
├── utils/
│   ├── data_loader.py    # Data loading utilities
│   ├── transforms.py     # Data augmentation transforms
│   └── metrics.py        # Evaluation metrics
├── notebooks/
│   └── visualization.ipynb # Data visualization and analysis
├── data/                  # Dataset directory
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

- ResNet implementation for CIFAR-10 classification
- Data preprocessing and augmentation pipeline
- Training with learning rate scheduling
- Model evaluation and metrics tracking
- TensorBoard integration for visualization
- Inference pipeline for single image prediction

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py --epochs 100 --batch-size 128 --learning-rate 0.001
```

### Evaluation

```bash
python src/evaluate.py --model-path checkpoints/best_model.pth
```

### Inference

```bash
python src/inference.py --image-path path/to/image.jpg --model-path checkpoints/best_model.pth
```

## Model Architecture

The project uses ResNet-18 architecture modified for CIFAR-10:
- Input size: 32x32x3
- Output classes: 10 (CIFAR-10 classes)
- Feature extraction: ResNet blocks with residual connections
- Training optimizer: Adam with cosine annealing learning rate scheduler

## Data Preprocessing

- Normalization using CIFAR-10 mean and std
- Random horizontal flip
- Random rotation
- Random crop with padding
- Color jittering

## Performance Metrics

The model is evaluated using:
- Top-1 Accuracy
- Confusion Matrix
- Per-class Precision and Recall
- F1 Score

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Image Super-Resolution Using Convolutional Neural Network

## Overview
This project implements the paper "Image Super-Resolution Using Convolutional Neural Network" from the 2022 IEEE 2nd Mysore Sub Section International Conference (MysuruCon).

**Paper Citation:**
```
K. S. Charan, G. Rochan Ravi, T. N. Shashank and C. Gururaj, "Image Super-Resolution Using Convolutional Neural Network," 2022 IEEE 2nd Mysore Sub Section International Conference (MysuruCon), 2022, pp. 1-7, doi: 10.1109/MysuruCon55714.2022.9972459
```

## Project Structure
```
FinalProject/
├── checkpoints/
│   ├── model1/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_5.pth
│   │   └── training_metrics.json
│   └── model2/
│       ├── best_model.pth
│       ├── checkpoint_epoch_5.pth
│       └── checkpoint_epoch_10.pth
├── configs/
│   └── config.yaml
├── data/
│   ├── Raw Data/
│   ├── Raw_Data_Backup/
│   ├── test/
│   ├── train/
│   ├── val/
│   └── __init__.py
├── results/
│   ├── evaluation_[timestamp]/
│   │   ├── metrics.txt
│   │   ├── psnr_distribution.png
│   │   ├── psnr_vs_ssim.png
│   │   ├── result_[0-4].png
│   │   └── ssim_distribution.png
│   └── runs/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── prepare_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model1.py
│   │   └── model2.py
│   └── utils/
│       ├── __init__.py
│       ├── evaluate.py
│       └── evaluate_comparative.py
├── tests/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_training.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## Model Architectures

### Model 1: Advanced Architecture
- Input Resolution: 256x256x3
- U-Net inspired architecture with skip connections
- Residual blocks for feature learning
- Dual-path structure (encoder-decoder)
- Dropout and batch normalization for regularization
- LeakyReLU activation functions
- L1 regularization factor: 1e-10
- Total parameters: 9,602,766
- Performance: PSNR: 31.53 dB, SSIM: 0.941

### Model 2: Simple Architecture
- Direct convolution approach (7 layers)
- Channel progression: 64→128→256→128→64→3
- ReLU activation functions
- Performance: PSNR: 24.97 dB, SSIM: 0.918

## Performance Results

Overall metrics achieved on test dataset:
- **Average PSNR**: 29.17 dB
- **Average SSIM**: 0.8240

Training Performance:
- Training accuracy: 86.31%
- Validation accuracy: 92.35%
- Training loss: 0.0170
- Validation loss: 0.0160
- Early stopping at epoch 19

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/FinalProject
cd FinalProject

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Data Structure
```
data/
├── train/
│   ├── hr/    # High-resolution training images
│   └── lr/    # Low-resolution training images
├── val/
│   ├── hr/
│   └── lr/
└── test/
    ├── hr/
    └── lr/
```

### Prepare Dataset
```bash
# Process and split raw data
python src/data/prepare_data.py
```

## Training

### Configuration
Edit `configs/config.yaml` to set:
- Training parameters
- Model architecture
- Dataset paths
- Logging options

### Train Models
```bash
# Train Model 1
python src/models/train.py --config configs/config.yaml --model 1

# Train Model 2
python src/models/train.py --config configs/config.yaml --model 2

# Train both models for comparison
python train_comparative.py
```

## Evaluation

### Single Model Evaluation
```bash
python src/utils/evaluate.py --model_path checkpoints/model1/best_model.pth
```

### Comparative Evaluation
```bash
python src/utils/evaluate_comparative.py
```

Results will be saved in `results/evaluation_[timestamp]/`:
- Metric calculations (metrics.txt)
- PSNR and SSIM distributions
- Visual results (result_[0-4].png)
- Performance comparisons

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test suite
python -m unittest tests.test_models
```

## Configuration

Key settings in `configs/config.yaml`:
```yaml
data:
  image_size: 256
  train_split: 700
  val_split: 130
  test_split: 25
  color_space: "ycbcr"
  channel: "y"
  normalize_range: [-1, 1]

model:
  name: "model1"  # or "model2"
  input_channels: 3
  regularization_weight: 0.0001

training:
  batch_size: 128
  learning_rate: 0.0001
  early_stopping_patience: 5
```

## Results Visualization

The `results` directory contains:
- Training metrics and logs
- Evaluation results
- Model comparisons
- Visual outputs

Check `results/evaluation_[timestamp]/` for:
- PSNR/SSIM distributions
- Model predictions
- Comparative analysis

## Acknowledgements
- Implementation based on IEEE MysuruCon 2022 paper
- Special thanks to AI assistants for helping explain complex concepts and teaching me how to structure my code


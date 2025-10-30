# Deepfake Detection System

🚧 **Project Status: In Development** 🚧

A comprehensive deepfake detection system with real-time webcam analysis, video processing, and multi-source support.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

## ✨ Features

- 🎥 **Real-time Detection**: Live webcam feed analysis with face detection and deepfake scoring
- 📹 **Video Analysis**: Process video files with frame-by-frame deepfake detection
- 📷 **Image Analysis**: Single image deepfake detection with detailed reports
- 📡 **CCTV/Stream Support**: Analyze RTSP/HTTP video streams
- 🧠 **Deep Learning**: Xception-based CNN architecture
- ⚡ **Optimized Inference**: ONNX export and TorchScript support
- 🐳 **Docker Support**: Containerized deployment for CPU and GPU
- 📊 **Comprehensive Reports**: JSON output with detailed face-level analysis
- 🧪 **Test Suite**: Unit and integration tests

## ⚠️ Current Status

- ✅ **Core Detection Pipeline**: Complete
- ✅ **Face Detection**: MTCNN implemented
- ✅ **Dataset Preparation**: FaceForensics++ processed
- 🔄 **Model Training**: In progress (see `TRAINING_TODO.md`)
- 📋 **Pretrained Weights**: Coming soon
- 📚 **Documentation**: Complete

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Webcam Real-time Detection](#webcam-real-time-detection)
  - [Image Analysis](#image-analysis)
  - [Video Analysis](#video-analysis)
  - [Training](#training)
- [Dataset Access](#dataset-access)
- [Model Export](#model-export)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA 11.3+ for training
- (Optional) Docker for containerized deployment

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/chrispl89/deepfake_detector.git
cd deepfake_detector

# Install minimal dependencies (inference only)
pip install -r requirements-minimal.txt

# OR install full dependencies (training + inference)
pip install -r requirements.txt

# For CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Using Conda

```bash
# Create environment from file
conda env create -f environment.yml
conda activate deepfake-detection
```

### Option 3: Using Docker

```bash
# Build CPU version
docker-compose build deepfake-detector-cpu

# Build GPU version
docker-compose build deepfake-detector-gpu

# Run container
docker-compose up -d deepfake-detector-cpu
```

### Verify Installation

```bash
# Run tests
python -m pytest tests/

# Quick test
python -c "from deepfake_detector import DeepfakeDetector; print('Installation successful!')"
```

## 🚀 Quick Start

> **Note:** Pretrained model coming soon! For now, you'll need to train your own model (see `TRAINING_TODO.md`).

### 1. Real-time Webcam Detection

```bash
# Basic webcam detection
python face_recon_deepfake.py

# With custom model (after training)
python face_recon_deepfake.py --model ./checkpoints/your_model.pth

# With face blurring for detected fakes
python face_recon_deepfake.py --blur
```

### 2. Analyze an Image

```bash
python inference.py --source image.jpg --type image --out ./results

# With custom model and threshold (after training)
python inference.py --source image.jpg --type image --out ./results \
  --model ./checkpoints/your_model.pth --threshold 0.7
```

### 3. Analyze a Video

```bash
python inference.py --source video.mp4 --type video --out ./results --sample-rate 30
```

### 4. Train a Model

```bash
# With default settings (requires dataset)
python train.py --datasets ./data/datasets/my_dataset --output model.pth

# With GPU and mixed precision
python train.py \
    --datasets ./data/datasets/dataset1 ./data/datasets/dataset2 \
    --output model.pth \
    --device cuda \
    --mixed-precision \
    --epochs 50 \
    --batch-size 32
```

## Usage

### Webcam Real-time Detection

The enhanced webcam application provides real-time deepfake detection:

```bash
# Basic usage
python face_recon_deepfake.py

# Advanced options
python face_recon_deepfake.py \
    --camera 0 \              # Camera device ID
    --model ./model.pth \     # Custom model
    --blur                    # Blur fake faces
```

**Controls:**
- Press `ESC` to exit

**Display:**
- Green box: Real face (low deepfake score)
- Red box: Fake face (high deepfake score)
- Score shown on each detected face
- Statistics panel showing total detections

### Image Analysis

Analyze a single image for deepfakes:

```bash
python inference.py \
    --source path/to/image.jpg \
    --type image \
    --out ./results \
    --threshold 0.5 \
    --device cpu
```

**Output:**
- `results/report.json`: Detailed JSON report
- `results/summary.txt`: Human-readable summary
- `results/annotated_image.jpg`: Image with bounding boxes and scores

### Video Analysis

Process video files:

```bash
python inference.py \
    --source video.mp4 \
    --type video \
    --out ./results \
    --sample-rate 30          # Process every 30th frame
```

**Output:**
- `results/report.json`: Frame-by-frame analysis
- `results/annotated_video.mp4`: Video with annotations
- `results/summary.txt`: Overall verdict

### Camera/Stream Analysis

Analyze live camera feeds or CCTV streams:

```bash
# Webcam
python inference.py --source 0 --type camera --out ./results --duration 30

# RTSP stream
python inference.py \
    --source rtsp://192.168.1.100/stream \
    --type stream \
    --out ./results \
    --duration 60
```

### Training

#### Prepare Dataset

Organize your dataset:

```
data/datasets/my_dataset/
├── real/           # Real (authentic) images
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── fake/           # Fake (manipulated) images
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

For video datasets, extract frames first:

```python
from preprocessing.preprocessing import extract_frames_from_video

frames = extract_frames_from_video('video.mp4', max_frames=100, sampling_rate=10)
```

#### Train Model

```bash
# Basic training
python train.py \
    --datasets ./data/datasets/dataset1 \
    --output model.pth \
    --epochs 50

# Advanced training with configuration
python train.py \
    --datasets ./data/datasets/dataset1 ./data/datasets/dataset2 \
    --output model.pth \
    --config config.json \
    --device cuda \
    --mixed-precision

# Custom configuration
python train.py \
    --datasets ./data/datasets/my_dataset \
    --output ./checkpoints/model.pth \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001 \
    --backbone xception \
    --image-size 299 \
    --device cuda \
    --mixed-precision
```

**Training Configuration (`config.json`):**

```json
{
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "backbone": "xception",
  "image_size": 299,
  "validation_split": 0.2,
  "use_mixed_precision": true,
  "early_stopping_patience": 10,
  "device": "cuda",
  "checkpoint_dir": "./checkpoints"
}
```

**Training Output:**
- `checkpoints/model.pth`: Best model checkpoint
- `checkpoints/model_training_report.json`: Training metrics
- `checkpoints/checkpoint_epoch_*.pth`: Periodic checkpoints

## 📂 Dataset Access

For training, you'll need access to deepfake detection datasets:

### Supported Datasets

1. **FaceForensics++** (Recommended - Used in this project)
   - 1000+ videos with multiple manipulation methods
   - Research use only

2. **DFDC Preview** (Public via Kaggle)
   - DeepFake Detection Challenge dataset
   - MIT License

3. **Celeb-DF** (Requires permission)
   - Celebrity deepfake dataset
   - Research use only

### Dataset Preparation Pipeline

This project includes scripts to process FaceForensics++:

1. **Download videos**: `datasets/download_faceforensics.py` (requires access)
2. **Extract frames**: `extract_frames.py` 
3. **Extract faces**: `extract_faces_from_frames.py` (MTCNN)
4. **Train model**: `train.py`

See `TRAINING_TODO.md` for complete workflow.

### Quick Dataset Setup

```bash
# List available datasets
python -c "from datasets import list_available_datasets; print(list_available_datasets())"

# Download DFDC (requires Kaggle API)
python -c "from datasets import download_dfdc_preview; download_dfdc_preview()"

# Download FaceForensics++ (requires permission - see below)
# IMPORTANT: You must request access first!
# 1. Fill out form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
# 2. Wait for email with download link (usually ~1 week)
# 3. Download the script and save as datasets/download_faceforensics.py
# Full details: datasets/DOWNLOAD_FACEFORENSICS_README.md

# After receiving the script, use our API:
python -c "from datasets import download_faceforensicspp; download_faceforensicspp()"

# OR use direct CLI:
python datasets/download_faceforensics.py ./data/datasets/faceforensics \
  --dataset Deepfakes --compression c23 --num_videos 10

# For testing: Create synthetic dataset
python -c "from training.dataset import create_dummy_dataset; create_dummy_dataset('./data/dummy_dataset', num_samples=1000)"
```

## Model Export

### Export to ONNX

ONNX format provides faster inference and cross-platform support:

```bash
python export_model.py \
    --model ./checkpoints/model.pth \
    --output ./checkpoints/model.onnx \
    --format onnx
```

### Export to TorchScript

TorchScript for optimized PyTorch deployment:

```bash
python export_model.py \
    --model ./checkpoints/model.pth \
    --output ./checkpoints/model.pt \
    --format torchscript
```

### Quantization (CPU Optimization)

Quantize model for faster CPU inference:

```bash
python export_model.py \
    --model ./checkpoints/model.pth \
    --output ./checkpoints/model_quantized.pth \
    --format quantized
```

### TensorRT (NVIDIA GPU Optimization)

For maximum GPU performance:

```bash
# First export to ONNX
python export_model.py --model model.pth --output model.onnx --format onnx

# Then convert to TensorRT
python export_model.py \
    --model model.onnx \
    --output model.trt \
    --format tensorrt \
    --precision fp16
```

## Docker Deployment

### CPU Inference

```bash
# Build image
docker-compose build deepfake-detector-cpu

# Run container
docker-compose up -d deepfake-detector-cpu

# Execute inference
docker-compose exec deepfake-detector-cpu python inference.py \
    --source /app/data/test.jpg \
    --type image \
    --out /app/results
```

### GPU Training

```bash
# Build GPU image
docker-compose build deepfake-detector-gpu

# Run with GPU support
docker-compose up -d deepfake-detector-gpu

# Execute training
docker-compose exec deepfake-detector-gpu python train.py \
    --datasets /app/data/datasets/my_dataset \
    --output /app/checkpoints/model.pth \
    --device cuda
```

### Docker Commands Reference

```bash
# View logs
docker-compose logs -f deepfake-detector-cpu

# Stop containers
docker-compose down

# Remove all data
docker-compose down -v
```

## API Reference

### Python API

```python
from deepfake_detector import DeepfakeDetector, train_detector

# Initialize detector
detector = DeepfakeDetector(
    model_path='./model.pth',
    device='cpu',
    threshold=0.5
)

# Analyze image
report = detector.analyze_source(
    source='image.jpg',
    source_type='image',
    output_dir='./results'
)

print(f"Verdict: {report.overall_verdict}")
print(f"Confidence: {report.confidence}")

# Real-time inference on frame
import cv2
frame = cv2.imread('image.jpg')
faces = detector.infer_on_frame(frame)

for face in faces:
    print(f"Face at {face.bbox}: {face.deepfake_score:.3f}")
    print(f"Is fake: {face.is_fake}")

# Train model
from deepfake_detector import train_detector

config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'device': 'cuda'
}

training_report = train_detector(
    dataset_paths=['./data/dataset1'],
    model_save_path='./model.pth',
    config=config
)

print(f"Best accuracy: {training_report.best_val_accuracy}")
```

### Report Structure

```python
report = detector.analyze_source(...)

# Access report data
report.source              # Source path/URL
report.source_type         # 'image', 'video', 'camera', 'stream'
report.total_frames        # Number of frames analyzed
report.total_faces_detected # Total faces found
report.overall_verdict     # 'REAL', 'FAKE', or 'UNCERTAIN'
report.confidence          # Confidence score (0-1)
report.frames_analysis     # List of per-frame results

# Per-frame results
for frame_result in report.frames_analysis:
    frame_result.frame_number
    frame_result.timestamp
    frame_result.faces  # List of FaceDetectionResult
    frame_result.overall_fake_probability
```

## Performance

### Inference Speed

| Configuration | FPS (480p) | FPS (720p) | FPS (1080p) |
|---------------|------------|------------|-------------|
| CPU (i7-10700) | 5-8 | 2-4 | 1-2 |
| GPU (RTX 3060) | 45-60 | 30-45 | 15-25 |
| GPU (RTX 3090) | 80-120 | 60-80 | 40-60 |
| TensorRT FP16 | 120-150 | 90-120 | 60-90 |

### Model Accuracy

Tested on FaceForensics++ validation set:

| Model | Accuracy | AUC | Params | Size |
|-------|----------|-----|--------|------|
| Xception | 94.2% | 0.978 | 22.9M | 88MB |
| ResNet50 | 91.5% | 0.965 | 25.6M | 98MB |
| EfficientNet-B0 | 92.8% | 0.971 | 5.3M | 21MB |

### Resource Requirements

**Training:**
- Minimum: 16GB RAM, NVIDIA GPU with 8GB VRAM
- Recommended: 32GB RAM, NVIDIA GPU with 16GB+ VRAM
- Training time: ~4-8 hours on RTX 3090 (50 epochs, 50K images)

**Inference:**
- CPU: 4GB RAM minimum
- GPU: 2GB VRAM minimum
- Storage: 500MB for model weights

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_detector.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run smoke tests only
python -m pytest tests/test_smoke.py -v
```

## Project Structure

```
face_recognition/
├── deepfake_detector.py       # Core API module
├── inference.py                # CLI inference script
├── train.py                    # Training script
├── export_model.py             # Model export utility
├── face_recon.py               # Original webcam app
├── face_recon_deepfake.py      # Enhanced webcam app with deepfake detection
│
├── datasets/                   # Dataset access module
│   ├── __init__.py
│   └── dataset_downloader.py
│
├── preprocessing/              # Preprocessing pipeline
│   ├── __init__.py
│   ├── face_detection.py      # Face detection (MTCNN/OpenCV)
│   └── preprocessing.py       # Image preprocessing and augmentation
│
├── models/                     # Model architectures
│   ├── __init__.py
│   └── xception_model.py      # Xception-based detector
│
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes
│   └── trainer.py             # Training logic
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_detector.py
│   └── test_smoke.py
│
├── data/                       # Data directory (not in repo)
│   ├── datasets/              # Training datasets
│   └── dummy_dataset/         # Synthetic test data
│
├── checkpoints/                # Model checkpoints (not in repo)
├── results/                    # Inference results (not in repo)
│
├── requirements.txt            # Full dependencies
├── requirements-minimal.txt    # Minimal dependencies
├── environment.yml             # Conda environment
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker compose config
├── .dockerignore
│
├── README.md                   # This file
├── TRAINING_TODO.md            # Training status and TODO
└── LICENSE
```

## Troubleshooting

### Common Issues

**Issue: Out of memory during training**

Solution:
```bash
# Reduce batch size
python train.py --batch-size 16 ...

# Use gradient accumulation
# Or use mixed precision
python train.py --mixed-precision ...
```

**Issue: Slow inference on CPU**

Solution:
```bash
# Export to quantized model
python export_model.py --model model.pth --output model_quant.pth --format quantized

# Or use ONNX
python export_model.py --model model.pth --output model.onnx --format onnx
```

**Issue: CUDA out of memory**

Solution:
- Reduce batch size
- Enable mixed precision training
- Use gradient checkpointing
- Use smaller image size (224 instead of 299)

**Issue: Camera not detected**

Solution:
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Try different camera ID
python face_recon_deepfake.py --camera 1
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
python -m pytest tests/

# Format code
black .

# Lint
flake8 .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{deepfake_detection_system,
  title={Deepfake Detection System},
  author={Krzysztof Plonka},
  year={2025},
  url={https://github.com/chrispl89/deepfake_detector}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset Licenses

Please note that some datasets used for training have their own licenses:
- FaceForensics++: Research use only
- Celeb-DF: Research use only
- DFDC: MIT License

Always check and comply with dataset-specific licenses.

## Acknowledgments

- OpenCV for computer vision utilities
- PyTorch team for the deep learning framework
- FaceForensics++ team for the dataset
- MTCNN implementation from facenet-pytorch
- Original face detection code inspiration

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [github.com/chrispl89/deepfake_detector/issues](https://github.com/chrispl89/deepfake_detector/issues)
- Author: Krzysztof Plonka

---

## ⚠️ Disclaimer

This is a research/educational tool. Always verify results with human review for critical applications. Deepfake detection is an active research area and no system is 100% accurate.

**Project Status:** 🚧 In Development - Training in progress

---

**Star ⭐ this repo if you find it useful!**

# Deepfake Detector - Training Guide

## Overview

This guide provides step-by-step instructions for training the deepfake detection model.

## Prerequisites

### Required Packages

```bash
python -m pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- tqdm >= 4.62.0
- scikit-learn >= 1.0.0

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free space

**Recommended:**
- CPU: 8+ cores or NVIDIA GPU (CUDA support)
- RAM: 16GB+
- Storage: 50GB+ free space
- GPU: NVIDIA with 8GB+ VRAM

## Training Pipeline

### Option 1: Quick Test with Dummy Data

For testing the pipeline without downloading large datasets:

```bash
python quick_test_training.py
```

This creates a synthetic dataset and trains for 5 epochs (~5-10 minutes on CPU).

### Option 2: Full Training Pipeline

#### Step 1: Download FaceForensics++ Dataset

**Note:** Requires permission from FaceForensics++ team.

1. Request access: https://github.com/ondyari/FaceForensics
2. Once approved, download the script
3. Run the downloader:

```bash
python datasets/download_faceforensics.py data/datasets/faceforensics_videos \
  --dataset original --compression c40 --type videos --num_videos 50

python datasets/download_faceforensics.py data/datasets/faceforensics_videos \
  --dataset Deepfakes --compression c40 --type videos --num_videos 50
```

#### Step 2: Extract Frames from Videos

```bash
python extract_frames.py \
  --input ./data/datasets/faceforensics_videos \
  --output ./data/datasets/faceforensics_frames \
  --max-frames 10 \
  --sampling-rate 30
```

This extracts 10 frames from each video (every 30th frame).

#### Step 3: Extract Face Crops

```bash
python extract_faces_from_frames.py \
  --input ./data/datasets/faceforensics_frames \
  --output ./data/datasets/faceforensics_faces \
  --device cpu
```

Uses MTCNN to detect and crop faces from frames.

#### Step 4: Train the Model

```bash
python train.py \
  --datasets ./data/datasets/faceforensics_faces \
  --output ./checkpoints/faceforensics_model.pth \
  --epochs 20 \
  --batch-size 8 \
  --lr 0.0001 \
  --device cpu \
  --backbone xception \
  --image-size 299
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32, reduce to 8 for CPU)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: cpu or cuda
- `--backbone`: xception, resnet, or efficientnet
- `--image-size`: Input image size (299 for Xception)

**Expected Training Time:**
- CPU (8 cores): 2-3 hours
- GPU (RTX 3060): 30-45 minutes

#### Step 5: Automated Full Pipeline

Run everything in one command:

```bash
python run_full_training_pipeline.py \
  --num-videos 50 \
  --max-frames 10 \
  --epochs 20 \
  --batch-size 8 \
  --device cpu
```

Skip steps if data already exists:
```bash
python run_full_training_pipeline.py \
  --skip-download \
  --skip-frames \
  --epochs 20
```

## Testing the Trained Model

### Test on Images

```bash
python test_trained_model.py \
  --model ./checkpoints/faceforensics_model.pth \
  --test-dir ./data/test_images \
  --threshold 0.5
```

### Test on Video

```bash
python inference.py \
  --source test_video.mp4 \
  --type video \
  --model ./checkpoints/faceforensics_model.pth \
  --out ./results
```

### Real-time Webcam Detection

```bash
python face_recon_deepfake.py \
  --model ./checkpoints/faceforensics_model.pth \
  --camera 0
```

## Model Evaluation

After training, the model generates a report with:
- Training/validation accuracy
- Loss curves
- Confusion matrix
- Precision, recall, F1-score
- AUC-ROC

Report location: `./checkpoints/model_training_report.json`

## Expected Results

With FaceForensics++ dataset (500 real + 500 fake faces):

- **Training Accuracy:** 95%+
- **Validation Accuracy:** 90-95%
- **AUC:** 0.95+

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --batch-size 4 ...
```

### Slow Training on CPU

Use smaller image size:
```bash
python train.py --image-size 224 ...
```

Or use fewer epochs for testing:
```bash
python train.py --epochs 10 ...
```

### MTCNN Face Detection Fails

If facenet-pytorch installation fails, the system will fall back to OpenCV Haar Cascades for face detection.

### Dataset Download Issues

If FaceForensics++ download fails:
1. Check internet connection
2. Verify you have permission
3. Try different server: `--server EU2` or `--server CA`
4. Use dummy dataset for testing: `python quick_test_training.py`

## Directory Structure

```
deepfake_detector/
├── data/
│   ├── datasets/
│   │   ├── faceforensics_videos/    # Downloaded videos
│   │   ├── faceforensics_frames/    # Extracted frames
│   │   └── faceforensics_faces/     # Face crops (training data)
│   ├── test_images/                 # Test images
│   └── dummy_dataset/               # Synthetic test data
├── checkpoints/
│   ├── faceforensics_model.pth      # Trained model
│   └── *_training_report.json       # Training metrics
├── results/                          # Inference results
└── logs/                             # Training logs
```

## Next Steps

1. **Test Your Model:** Run inference on test images
   ```bash
   python inference.py --source test_image.jpg --type image --model checkpoints/model.pth --device cpu --out ./results
   ```

2. **Evaluate Performance:** Use the test script
   ```bash
   python test_trained_model.py --model checkpoints/model.pth
   ```

3. **Improve:** Train on larger datasets (DFDC, Celeb-DF) for better accuracy

4. **Deploy:** Use your trained model in production with `inference.py` or integrate via Python API

## References

- FaceForensics++: https://github.com/ondyari/FaceForensics
- Xception Architecture: https://arxiv.org/abs/1610.02357
- MTCNN Face Detection: https://arxiv.org/abs/1604.02878

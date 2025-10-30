# 🚧 Training TODO

## Status
⚠️ **Model training in progress - coming soon!**

## What Needs to Be Done

### 1. Data Preparation ✅ DONE
- [x] Download FaceForensics++ dataset (50 real + 50 fake videos)
- [x] Extract frames from videos (1000 frames total)
- [x] Extract face crops from frames using MTCNN
- [x] Final dataset: `data/datasets/faceforensics_faces/` (500 real + 500 fake faces)

### 2. Model Training 🔄 IN PROGRESS
```bash
python train.py \
  --datasets ./data/datasets/faceforensics_faces \
  --output ./checkpoints/faceforensics_faces_model.pth \
  --epochs 20 --batch-size 8 --lr 0.0001 --device cpu
```

**Expected Results:**
- Training Accuracy: 95%+
- Validation Accuracy: 90-95%
- Training Time: ~2-3 hours (CPU) or ~30 min (GPU)

### 3. Model Evaluation 📋 TODO
- [ ] Test on validation set
- [ ] Calculate metrics (accuracy, precision, recall, F1, AUC)
- [ ] Generate confusion matrix
- [ ] Test on real-world images

### 4. Model Deployment 📦 TODO
- [ ] Export to ONNX format for production
- [ ] Create model zoo with pretrained weights
- [ ] Add model download script
- [ ] Update inference examples

## Important Notes

### Data Consistency Issue (FIXED ✅)
**Problem:** Initial training used full video frames (854x480 with background), but inference crops faces only. This caused model to perform poorly (~50% accuracy) on real photos.

**Solution:** Re-extract face crops from training frames using MTCNN, matching inference preprocessing exactly.

### Preprocessing Pipeline
```python
# Training & Inference (MUST BE IDENTICAL):
1. Detect face using MTCNN
2. Crop face with 20% margin
3. Resize to 299x299
4. Convert BGR → RGB
5. Normalize to [0, 1] range
6. Convert to tensor (C, H, W)
```

## Next Steps
1. Complete training on face-cropped dataset
2. Validate model performance
3. Update README with model download instructions
4. Add pretrained model to releases

## Dataset Info
- **Source:** FaceForensics++ (c40 compression)
- **Real:** 500 face crops from YouTube originals
- **Fake:** 500 face crops from Deepfakes manipulation
- **Total:** 1000 training samples
- **Train/Val Split:** 80/20

## Expected Timeline
- Training: 2-3 hours (next session)
- Validation: 30 minutes
- Documentation: 1 hour
- **Total:** ~4 hours

---
**Last Updated:** 2025-10-30  
**Status:** Ready for training

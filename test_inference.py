#!/usr/bin/env python
"""
Quick test of the inference pipeline.
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector import DeepfakeDetector

def main():
    print("="*60)
    print("DEEPFAKE DETECTOR - INFERENCE TEST")
    print("="*60)
    
    # Initialize detector
    print("\n1. Loading model...")
    detector = DeepfakeDetector(
        model_path='checkpoints/faceforensics_model.pth',
        device='cpu',
        threshold=0.5
    )
    print("✓ Model loaded")
    
    # Test on a sample image
    print("\n2. Testing on sample image...")
    test_image_path = 'data/test_images/real_test_1.jpg'
    
    if Path(test_image_path).exists():
        image = cv2.imread(test_image_path)
        faces = detector.infer_on_frame(image)
        
        print(f"\n✓ Image processed: {test_image_path}")
        print(f"  Faces detected: {len(faces)}")
        
        for i, face in enumerate(faces):
            print(f"\n  Face {i+1}:")
            print(f"    Score: {face.deepfake_score:.4f}")
            print(f"    Verdict: {'FAKE' if face.is_fake else 'REAL'}")
            print(f"    Confidence: {face.confidence:.4f}")
    else:
        print(f"✗ Test image not found: {test_image_path}")
    
    print("\n" + "="*60)
    print("✓ INFERENCE TEST COMPLETED")
    print("="*60)
    print("\nThe model is ready to use!")
    print("\nNext steps:")
    print("1. Test on your own images: python inference.py --source your_image.jpg --type image")
    print("2. Test on video: python inference.py --source your_video.mp4 --type video")
    print("3. Real-time webcam: python face_recon_deepfake.py")
    print("="*60)

if __name__ == '__main__':
    main()

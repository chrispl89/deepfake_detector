"""
End-to-end smoke tests for the deepfake detection system.
"""

import unittest
import numpy as np
import cv2
import sys
import os
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepfake_detector import DeepfakeDetector
from training.dataset import create_dummy_dataset, DeepfakeDataset
from models.xception_model import XceptionDeepfake


class SmokeTest(unittest.TestCase):
    """End-to-end smoke tests."""
    
    def test_full_pipeline_image(self):
        """Test complete pipeline on a single image."""
        # Create detector
        detector = DeepfakeDetector(model_path=None, device='cpu')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_image_path = f.name
            cv2.imwrite(temp_image_path, test_image)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run analysis
                report = detector.analyze_source(
                    source=temp_image_path,
                    source_type='image',
                    output_dir=temp_dir
                )
                
                # Verify report was created
                self.assertIsNotNone(report)
                
                # Verify output files exist
                report_file = os.path.join(temp_dir, 'report.json')
                self.assertTrue(os.path.exists(report_file))
                
                # Verify JSON is valid
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    self.assertIn('overall_verdict', report_data)
        finally:
            os.unlink(temp_image_path)
    
    def test_dataset_creation_and_loading(self):
        """Test dataset creation and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy dataset
            create_dummy_dataset(temp_dir, num_samples=20)
            
            # Verify directories were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'real')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'fake')))
            
            # Load dataset
            dataset = DeepfakeDataset(temp_dir)
            
            # Verify dataset has samples
            self.assertGreater(len(dataset), 0)
            
            # Test getting a sample
            image, label = dataset[0]
            self.assertIsNotNone(image)
            self.assertIn(label, [0, 1])
    
    def test_model_creation(self):
        """Test model can be created and run."""
        model = XceptionDeepfake(pretrained=False, num_classes=2)
        
        # Test forward pass
        dummy_input = np.random.randn(2, 3, 299, 299).astype(np.float32)
        
        import torch
        dummy_input_tensor = torch.from_numpy(dummy_input)
        
        with torch.no_grad():
            output = model(dummy_input_tensor)
        
        # Verify output shape
        self.assertEqual(output.shape, (2, 2))
    
    def test_inference_on_video_frame_sequence(self):
        """Test inference on a sequence of frames (simulating video)."""
        detector = DeepfakeDetector(model_path=None, device='cpu')
        
        # Create a sequence of frames
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Process each frame
        all_results = []
        for frame in frames:
            results = detector.infer_on_frame(frame)
            all_results.append(results)
        
        # Verify we got results for all frames
        self.assertEqual(len(all_results), 5)


class IntegrationTest(unittest.TestCase):
    """Integration tests for combined components."""
    
    def test_detector_with_face_detection(self):
        """Test that detector properly integrates with face detection."""
        from preprocessing.face_detection import FaceDetector
        
        face_detector = FaceDetector(use_mtcnn=False)
        deepfake_detector = DeepfakeDetector(model_path=None, device='cpu')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect faces
        faces = face_detector.detect_faces(test_image)
        
        # This should work even if no faces detected
        self.assertIsInstance(faces, list)
        
        # Run deepfake detection
        results = deepfake_detector.infer_on_frame(test_image)
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

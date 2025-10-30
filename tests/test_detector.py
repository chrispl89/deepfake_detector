"""
Integration tests for deepfake detector.
"""

import unittest
import numpy as np
import cv2
import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepfake_detector import DeepfakeDetector, FaceDetectionResult, Report


class TestDeepfakeDetector(unittest.TestCase):
    """Test cases for deepfake detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = DeepfakeDetector(model_path=None, device='cpu')
        
        # Create a test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.device, 'cpu')
        self.assertEqual(self.detector.threshold, 0.5)
    
    def test_infer_on_frame_returns_list(self):
        """Test that infer_on_frame returns a list."""
        results = self.detector.infer_on_frame(self.test_image)
        self.assertIsInstance(results, list)
    
    def test_infer_on_frame_result_structure(self):
        """Test that results have correct structure."""
        results = self.detector.infer_on_frame(self.test_image)
        
        for result in results:
            self.assertIsInstance(result, FaceDetectionResult)
            self.assertIsInstance(result.bbox, tuple)
            self.assertEqual(len(result.bbox), 4)
            self.assertIsInstance(result.confidence, float)
            self.assertIsInstance(result.deepfake_score, float)
            self.assertIsInstance(result.is_fake, bool)
            
            # Check score range
            self.assertGreaterEqual(result.deepfake_score, 0.0)
            self.assertLessEqual(result.deepfake_score, 1.0)
    
    def test_infer_on_empty_frame(self):
        """Test inference on empty frame."""
        results = self.detector.infer_on_frame(None)
        self.assertEqual(len(results), 0)
        
        results = self.detector.infer_on_frame(np.array([]))
        self.assertEqual(len(results), 0)
    
    def test_analyze_image(self):
        """Test image analysis."""
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_image_path = f.name
            cv2.imwrite(temp_image_path, self.test_image)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                report = self.detector.analyze_source(
                    source=temp_image_path,
                    source_type='image',
                    output_dir=temp_dir
                )
                
                self.assertIsInstance(report, Report)
                self.assertEqual(report.source_type, 'image')
                self.assertEqual(report.total_frames, 1)
                self.assertIn(report.overall_verdict, ['REAL', 'FAKE', 'UNCERTAIN'])
        finally:
            os.unlink(temp_image_path)
    
    def test_report_serialization(self):
        """Test that report can be serialized to dict."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_image_path = f.name
            cv2.imwrite(temp_image_path, self.test_image)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                report = self.detector.analyze_source(
                    source=temp_image_path,
                    source_type='image',
                    output_dir=temp_dir
                )
                
                # Convert to dict
                report_dict = report.to_dict()
                
                self.assertIsInstance(report_dict, dict)
                self.assertIn('source', report_dict)
                self.assertIn('source_type', report_dict)
                self.assertIn('overall_verdict', report_dict)
                self.assertIn('confidence', report_dict)
        finally:
            os.unlink(temp_image_path)


class TestFaceDetectionResult(unittest.TestCase):
    """Test cases for FaceDetectionResult."""
    
    def test_creation(self):
        """Test creating a FaceDetectionResult."""
        result = FaceDetectionResult(
            bbox=(10, 20, 100, 100),
            confidence=0.95,
            deepfake_score=0.3,
            is_fake=False
        )
        
        self.assertEqual(result.bbox, (10, 20, 100, 100))
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.deepfake_score, 0.3)
        self.assertEqual(result.is_fake, False)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = FaceDetectionResult(
            bbox=(10, 20, 100, 100),
            confidence=0.95,
            deepfake_score=0.3,
            is_fake=False
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['bbox'], (10, 20, 100, 100))
        self.assertEqual(result_dict['confidence'], 0.95)


if __name__ == '__main__':
    unittest.main()

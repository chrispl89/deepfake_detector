"""
Unit tests for preprocessing module.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.preprocessing import preprocess_face, augment_image
from preprocessing.face_detection import FaceDetector


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample image
        self.test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_preprocess_face_shape(self):
        """Test that preprocessing produces correct output shape."""
        processed = preprocess_face(self.test_image, target_size=(299, 299))
        self.assertEqual(processed.shape, (299, 299, 3))
    
    def test_preprocess_face_normalization(self):
        """Test that normalization works correctly."""
        processed = preprocess_face(self.test_image, normalize=True)
        # Values should be in [-1, 1] range
        self.assertGreaterEqual(processed.min(), -1.0)
        self.assertLessEqual(processed.max(), 1.0)
    
    def test_preprocess_face_no_normalization(self):
        """Test preprocessing without normalization."""
        processed = preprocess_face(self.test_image, normalize=False)
        # Values should be in [0, 1] range
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)
    
    def test_augment_image_shape(self):
        """Test that augmentation preserves image shape."""
        augmented = augment_image(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
    
    def test_augment_image_type(self):
        """Test that augmentation preserves image type."""
        augmented = augment_image(self.test_image)
        self.assertEqual(augmented.dtype, np.uint8)
    
    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid input."""
        with self.assertRaises(ValueError):
            preprocess_face(None)
        
        with self.assertRaises(ValueError):
            preprocess_face(np.array([]))


class TestFaceDetector(unittest.TestCase):
    """Test cases for face detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(use_mtcnn=False)  # Use OpenCV for testing
        
        # Create a test image
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.cascade)
    
    def test_detect_faces_returns_list(self):
        """Test that detect_faces returns a list."""
        faces = self.detector.detect_faces(self.test_image)
        self.assertIsInstance(faces, list)
    
    def test_detect_faces_empty_image(self):
        """Test detection on empty image."""
        faces = self.detector.detect_faces(np.array([]))
        self.assertEqual(len(faces), 0)
    
    def test_face_detection_result_structure(self):
        """Test that detection results have correct structure."""
        # Create image with a simple "face" pattern
        face_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        faces = self.detector.detect_faces(face_image)
        
        # Even if no faces detected, result should be a list
        self.assertIsInstance(faces, list)
        
        # If faces are detected, check structure
        for face in faces:
            self.assertIn('bbox', face)
            self.assertIn('confidence', face)
            self.assertEqual(len(face['bbox']), 4)


if __name__ == '__main__':
    unittest.main()

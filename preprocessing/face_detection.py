"""
Face detection module using MTCNN or OpenCV fallback.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector with MTCNN primary and OpenCV fallback."""
    
    def __init__(self, use_mtcnn: bool = True, min_face_size: int = 30):
        """
        Initialize face detector.
        
        Args:
            use_mtcnn: Try to use MTCNN if available
            min_face_size: Minimum face size to detect
        """
        self.min_face_size = min_face_size
        self.mtcnn = None
        self.cascade = None
        
        if use_mtcnn:
            self.mtcnn = self._load_mtcnn()
        
        if self.mtcnn is None:
            logger.info("Falling back to OpenCV Haar Cascade detector")
            self.cascade = self._load_cascade()
    
    def _load_mtcnn(self) -> Optional[object]:
        """Try to load MTCNN detector."""
        try:
            from facenet_pytorch import MTCNN as MTCNN_Facenet
            
            mtcnn = MTCNN_Facenet(
                keep_all=True,
                min_face_size=self.min_face_size,
                device='cpu'
            )
            logger.info("MTCNN detector loaded successfully")
            return mtcnn
        except ImportError:
            logger.warning("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
            return None
        except Exception as e:
            logger.warning(f"Failed to load MTCNN: {e}")
            return None
    
    def _load_cascade(self) -> cv2.CascadeClassifier:
        """Load OpenCV Haar Cascade classifier."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        
        if cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        logger.info("OpenCV Haar Cascade loaded successfully")
        return cascade
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of dictionaries with face detection results:
                - bbox: (x, y, width, height)
                - confidence: detection confidence
                - landmarks: facial landmarks (if available)
        """
        if image is None or image.size == 0:
            return []
        
        if self.mtcnn is not None:
            return self._detect_mtcnn(image)
        else:
            return self._detect_cascade(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=True)
            
            if boxes is None:
                return []
            
            results = []
            for box, prob, landmark in zip(boxes, probs, landmarks):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                # Convert to integer coordinates
                x, y, w, h = int(x1), int(y1), int(w), int(h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                
                results.append({
                    'bbox': (x, y, w, h),
                    'confidence': float(prob),
                    'landmarks': [(int(x), int(y)) for x, y in landmark] if landmark is not None else None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []
    
    def _detect_cascade(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': (int(x), int(y), int(w), int(h)),
                'confidence': 1.0,
                'landmarks': None
            })
        
        return results
    
    def align_face(self, image: np.ndarray, landmarks: List[Tuple[int, int]], 
                   output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Align face based on eye landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            output_size: Desired output size
            
        Returns:
            Aligned face image
        """
        if landmarks is None or len(landmarks) < 2:
            # No alignment possible, just resize
            return cv2.resize(image, output_size)
        
        try:
            # Get eye coordinates (assuming first two landmarks are eyes)
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            
            # Calculate angle
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Calculate center point between eyes
            eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                          (left_eye[1] + right_eye[1]) // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            
            # Apply affine transformation
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                    flags=cv2.INTER_CUBIC)
            
            # Resize to output size
            aligned = cv2.resize(aligned, output_size)
            
            return aligned
            
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return cv2.resize(image, output_size)
    
    def extract_face_crops(self, image: np.ndarray, 
                          output_size: Tuple[int, int] = (224, 224),
                          align: bool = True) -> List[Dict]:
        """
        Detect and extract face crops from image.
        
        Args:
            image: Input image
            output_size: Size for cropped faces
            align: Whether to align faces based on landmarks
            
        Returns:
            List of dictionaries with 'crop', 'bbox', 'confidence'
        """
        faces = self.detect_faces(image)
        
        results = []
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Crop face with some margin
            margin = int(0.2 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Align if landmarks available
            if align and face['landmarks'] is not None:
                face_crop = self.align_face(face_crop, face['landmarks'], output_size)
            else:
                face_crop = cv2.resize(face_crop, output_size)
            
            results.append({
                'crop': face_crop,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'landmarks': face['landmarks']
            })
        
        return results


def test_face_detector():
    """Test face detector on a sample image."""
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (255, 255, 255)  # White background
    
    detector = FaceDetector()
    faces = detector.detect_faces(test_image)
    
    print(f"Detected {len(faces)} faces")
    for i, face in enumerate(faces):
        print(f"Face {i+1}: bbox={face['bbox']}, confidence={face['confidence']:.3f}")


if __name__ == "__main__":
    test_face_detector()

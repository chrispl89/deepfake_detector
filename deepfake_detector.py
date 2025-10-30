"""
Deepfake Detector - Core API Module

This module provides a comprehensive API for detecting deepfake/manipulated faces
in images, videos, camera feeds, and CCTV streams.

Main Functions:
    - analyze_source(): Analyze media from various sources
    - train_detector(): Train or fine-tune the detector model
    - infer_on_frame(): Real-time inference on single frames
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import cv2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionResult:
    """Results from face detection and deepfake inference on a single face."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Face detection confidence
    deepfake_score: float  # Probability of being fake (0-1)
    is_fake: bool  # True if deepfake_score > threshold
    landmarks: Optional[List[Tuple[int, int]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FrameAnalysisResult:
    """Analysis results for a single frame."""
    frame_number: int
    timestamp: float  # seconds
    faces: List[FaceDetectionResult]
    overall_fake_probability: float  # Max or average of all faces
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'faces': [f.to_dict() for f in self.faces],
            'overall_fake_probability': self.overall_fake_probability
        }


@dataclass
class Report:
    """Comprehensive analysis report for a media source."""
    source: str
    source_type: str  # image, video, camera, stream
    timestamp: str
    total_frames: int
    total_faces_detected: int
    frames_analysis: List[FrameAnalysisResult]
    overall_verdict: str  # "REAL", "FAKE", "UNCERTAIN"
    confidence: float  # Overall confidence in verdict
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'total_frames': self.total_frames,
            'total_faces_detected': self.total_faces_detected,
            'frames_analysis': [f.to_dict() for f in self.frames_analysis],
            'overall_verdict': self.overall_verdict,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def save_json(self, output_path: str):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {output_path}")


@dataclass
class TrainingReport:
    """Report from training/fine-tuning the detector."""
    model_path: str
    training_start: str
    training_end: str
    total_epochs: int
    best_epoch: int
    best_val_accuracy: float
    best_val_auc: float
    final_metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, output_path: str):
        """Save training report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Training report saved to {output_path}")


class DeepfakeDetector:
    """Main deepfake detection class."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu',
                 threshold: float = 0.5):
        """
        Initialize the deepfake detector.
        
        Args:
            model_path: Path to pretrained model weights. If None, uses default.
            device: 'cpu' or 'cuda' for GPU acceleration
            threshold: Threshold for binary classification (default 0.5)
        """
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.model = None
        self.face_detector = None
        
        logger.info(f"Initializing DeepfakeDetector on {device}")
        self._load_model()
        self._load_face_detector()
    
    def _load_model(self):
        """Load the deepfake detection model."""
        try:
            from models.xception_model import load_model
            self.model = load_model(self.model_path, self.device)
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("Model module not found. Model will be loaded when available.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_face_detector(self):
        """Load face detection model (MTCNN or similar)."""
        try:
            from preprocessing.face_detection import FaceDetector
            self.face_detector = FaceDetector()
            logger.info("Face detector loaded successfully")
        except ImportError:
            logger.warning("Face detector module not found. Will use fallback.")
            self.face_detector = None
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
    
    def infer_on_frame(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run face detection + deepfake inference on one frame.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            List of FaceDetectionResult objects, one per detected face
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        results = []
        
        # Detect faces
        if self.face_detector is not None:
            faces = self.face_detector.detect_faces(frame)
        else:
            # Fallback to simple OpenCV detector
            faces = self._detect_faces_opencv(frame)
        
        # Run deepfake detection on each face
        for face_data in faces:
            bbox = face_data['bbox']
            x, y, w, h = bbox
            
            # Crop face region
            face_crop = frame[y:y+h, x:x+w]
            
            # Preprocess and infer
            deepfake_score = self._infer_deepfake(face_crop)
            
            result = FaceDetectionResult(
                bbox=bbox,
                confidence=face_data.get('confidence', 1.0),
                deepfake_score=deepfake_score,
                is_fake=deepfake_score > self.threshold,
                landmarks=face_data.get('landmarks')
            )
            results.append(result)
        
        return results
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[Dict]:
        """Fallback face detection using OpenCV Haar Cascade."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [{'bbox': (x, y, w, h), 'confidence': 1.0} 
                for (x, y, w, h) in faces]
    
    def _infer_deepfake(self, face_crop: np.ndarray) -> float:
        """
        Run deepfake inference on a cropped face.
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            Probability of being fake (0-1)
        """
        if self.model is None:
            # Return random score if model not loaded (for testing)
            logger.debug("Model not loaded, returning random score")
            return np.random.random()
        
        try:
            import torch
            from preprocessing.preprocessing import preprocess_face
            
            # Preprocess face
            processed = preprocess_face(face_crop, target_size=(299, 299), normalize=True)
            
            # Convert to tensor and add batch dimension
            # processed is (H, W, C), need (1, C, H, W)
            if len(processed.shape) == 3:
                processed = np.transpose(processed, (2, 0, 1))  # (C, H, W)
            
            tensor = torch.from_numpy(processed).float().unsqueeze(0)  # (1, C, H, W)
            tensor = tensor.to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                output = self.model(tensor)
                # Get probability of fake class
                probabilities = torch.softmax(output, dim=1)
                fake_prob = probabilities[0, 1].item()  # Class 1 = fake
            
            return float(fake_prob)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return 0.5  # Return uncertain score on error
    
    def analyze_source(self, source: str, source_type: str, 
                      output_dir: str) -> Report:
        """
        Analyze media from various sources for deepfakes.
        
        Args:
            source: Path or URL to media (image, video, camera device, CCTV stream)
            source_type: One of 'image', 'video', 'camera', 'stream'
            output_dir: Folder where to save results
            
        Returns:
            Report object with analysis results
        """
        logger.info(f"Analyzing {source_type} source: {source}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Route to appropriate handler
        if source_type == 'image':
            return self._analyze_image(source, output_dir)
        elif source_type == 'video':
            return self._analyze_video(source, output_dir)
        elif source_type == 'camera':
            return self._analyze_camera(source, output_dir)
        elif source_type == 'stream':
            return self._analyze_stream(source, output_dir)
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
    
    def _analyze_image(self, image_path: str, output_dir: str) -> Report:
        """Analyze a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Analyze frame
        faces = self.infer_on_frame(frame)
        
        frame_result = FrameAnalysisResult(
            frame_number=0,
            timestamp=0.0,
            faces=faces,
            overall_fake_probability=max([f.deepfake_score for f in faces]) if faces else 0.0
        )
        
        # Save annotated image
        annotated = self._annotate_frame(frame, faces)
        output_path = os.path.join(output_dir, 'annotated_image.jpg')
        cv2.imwrite(output_path, annotated)
        
        # Generate verdict
        verdict, confidence = self._generate_verdict([frame_result])
        
        report = Report(
            source=image_path,
            source_type='image',
            timestamp=datetime.now().isoformat(),
            total_frames=1,
            total_faces_detected=len(faces),
            frames_analysis=[frame_result],
            overall_verdict=verdict,
            confidence=confidence,
            metadata={'output_image': output_path}
        )
        
        # Save report
        report.save_json(os.path.join(output_dir, 'report.json'))
        
        return report
    
    def _analyze_video(self, video_path: str, output_dir: str, 
                       sample_rate: int = 30) -> Report:
        """Analyze a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_analysis = []
        frame_count = 0
        total_faces = 0
        
        # Prepare video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_dir, 'annotated_video.mp4')
        out = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames to avoid processing every frame
            if frame_count % sample_rate == 0:
                faces = self.infer_on_frame(frame)
                
                frame_result = FrameAnalysisResult(
                    frame_number=frame_count,
                    timestamp=frame_count / fps if fps > 0 else 0,
                    faces=faces,
                    overall_fake_probability=max([f.deepfake_score for f in faces]) if faces else 0.0
                )
                frames_analysis.append(frame_result)
                total_faces += len(faces)
                
                # Annotate frame
                annotated = self._annotate_frame(frame, faces)
                
                # Initialize video writer
                if out is None:
                    h, w = annotated.shape[:2]
                    out = cv2.VideoWriter(out_path, fourcc, fps/sample_rate, (w, h))
                
                out.write(annotated)
            
            frame_count += 1
        
        cap.release()
        if out is not None:
            out.release()
        
        # Generate verdict
        verdict, confidence = self._generate_verdict(frames_analysis)
        
        report = Report(
            source=video_path,
            source_type='video',
            timestamp=datetime.now().isoformat(),
            total_frames=len(frames_analysis),
            total_faces_detected=total_faces,
            frames_analysis=frames_analysis,
            overall_verdict=verdict,
            confidence=confidence,
            metadata={
                'output_video': out_path,
                'original_fps': fps,
                'total_frames_in_video': total_frames_in_video,
                'sample_rate': sample_rate
            }
        )
        
        # Save report
        report.save_json(os.path.join(output_dir, 'report.json'))
        
        return report
    
    def _analyze_camera(self, camera_id: str, output_dir: str, 
                       duration: int = 30) -> Report:
        """Analyze camera feed for specified duration."""
        # Convert camera_id to int if it's a number
        try:
            camera_id = int(camera_id)
        except ValueError:
            pass
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        fps = 30  # Assume 30 fps for camera
        frames_to_capture = fps * duration
        
        frames_analysis = []
        frame_count = 0
        total_faces = 0
        
        logger.info(f"Capturing from camera for {duration} seconds...")
        
        while frame_count < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.infer_on_frame(frame)
            
            frame_result = FrameAnalysisResult(
                frame_number=frame_count,
                timestamp=frame_count / fps,
                faces=faces,
                overall_fake_probability=max([f.deepfake_score for f in faces]) if faces else 0.0
            )
            frames_analysis.append(frame_result)
            total_faces += len(faces)
            
            frame_count += 1
        
        cap.release()
        
        # Generate verdict
        verdict, confidence = self._generate_verdict(frames_analysis)
        
        report = Report(
            source=str(camera_id),
            source_type='camera',
            timestamp=datetime.now().isoformat(),
            total_frames=len(frames_analysis),
            total_faces_detected=total_faces,
            frames_analysis=frames_analysis,
            overall_verdict=verdict,
            confidence=confidence,
            metadata={'duration': duration, 'fps': fps}
        )
        
        # Save report
        report.save_json(os.path.join(output_dir, 'report.json'))
        
        return report
    
    def _analyze_stream(self, stream_url: str, output_dir: str, 
                       duration: int = 30) -> Report:
        """Analyze RTSP/HTTP stream."""
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames_to_capture = int(fps * duration)
        
        frames_analysis = []
        frame_count = 0
        total_faces = 0
        
        logger.info(f"Analyzing stream for {duration} seconds...")
        
        while frame_count < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Lost connection to stream")
                break
            
            # Sample every Nth frame to reduce processing
            if frame_count % 10 == 0:
                faces = self.infer_on_frame(frame)
                
                frame_result = FrameAnalysisResult(
                    frame_number=frame_count,
                    timestamp=frame_count / fps,
                    faces=faces,
                    overall_fake_probability=max([f.deepfake_score for f in faces]) if faces else 0.0
                )
                frames_analysis.append(frame_result)
                total_faces += len(faces)
            
            frame_count += 1
        
        cap.release()
        
        # Generate verdict
        verdict, confidence = self._generate_verdict(frames_analysis)
        
        report = Report(
            source=stream_url,
            source_type='stream',
            timestamp=datetime.now().isoformat(),
            total_frames=len(frames_analysis),
            total_faces_detected=total_faces,
            frames_analysis=frames_analysis,
            overall_verdict=verdict,
            confidence=confidence,
            metadata={'duration': duration, 'fps': fps}
        )
        
        # Save report
        report.save_json(os.path.join(output_dir, 'report.json'))
        
        return report
    
    def _annotate_frame(self, frame: np.ndarray, 
                       faces: List[FaceDetectionResult]) -> np.ndarray:
        """Draw bounding boxes and scores on frame."""
        annotated = frame.copy()
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Choose color based on verdict
            color = (0, 0, 255) if face.is_fake else (0, 255, 0)  # Red for fake, green for real
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{'FAKE' if face.is_fake else 'REAL'}: {face.deepfake_score:.2f}"
            cv2.putText(annotated, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def _generate_verdict(self, frames_analysis: List[FrameAnalysisResult]) -> Tuple[str, float]:
        """Generate overall verdict from frame analysis."""
        if not frames_analysis:
            return "UNCERTAIN", 0.0
        
        # Calculate average fake probability across all frames
        fake_scores = []
        for frame in frames_analysis:
            if frame.faces:
                fake_scores.append(frame.overall_fake_probability)
        
        if not fake_scores:
            return "UNCERTAIN", 0.0
        
        avg_score = np.mean(fake_scores)
        confidence = np.std(fake_scores)  # Lower std = higher confidence
        
        if avg_score > 0.7:
            verdict = "FAKE"
            confidence = 1 - confidence
        elif avg_score < 0.3:
            verdict = "REAL"
            confidence = 1 - confidence
        else:
            verdict = "UNCERTAIN"
            confidence = 0.5
        
        return verdict, float(np.clip(confidence, 0, 1))


def train_detector(dataset_paths: List[str], model_save_path: str, 
                   config: dict) -> TrainingReport:
    """
    Train or fine-tune the deepfake detector.
    
    Args:
        dataset_paths: List of paths to training datasets
        model_save_path: Where to save the trained model
        config: Training configuration dictionary with keys:
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - learning_rate: Initial learning rate
            - backbone: Model backbone ('xception', 'resnet', 'efficientnet')
            - image_size: Input image size (default 299 for Xception)
            - validation_split: Fraction of data for validation
            - use_mixed_precision: Enable mixed precision training
            - early_stopping_patience: Epochs to wait before early stopping
            
    Returns:
        TrainingReport with training metrics and history
    """
    logger.info("Starting model training...")
    logger.info(f"Datasets: {dataset_paths}")
    logger.info(f"Config: {config}")
    
    from training.trainer import Trainer
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    training_start = datetime.now().isoformat()
    history = trainer.train(dataset_paths, model_save_path)
    training_end = datetime.now().isoformat()
    
    # Generate training report
    report = TrainingReport(
        model_path=model_save_path,
        training_start=training_start,
        training_end=training_end,
        total_epochs=history['total_epochs'],
        best_epoch=history['best_epoch'],
        best_val_accuracy=history['best_val_accuracy'],
        best_val_auc=history['best_val_auc'],
        final_metrics=history['final_metrics'],
        training_history=history['history'],
        config=config
    )
    
    # Save training report
    report_path = model_save_path.replace('.pth', '_training_report.json')
    report.save_json(report_path)
    
    return report


# Convenience function for quick testing
def quick_test():
    """Quick test of the deepfake detector on a sample image."""
    detector = DeepfakeDetector()
    
    # Create a sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = detector.infer_on_frame(frame)
    
    print(f"Detected {len(results)} faces")
    for i, result in enumerate(results):
        print(f"Face {i+1}: Deepfake score = {result.deepfake_score:.3f}")


if __name__ == "__main__":
    quick_test()

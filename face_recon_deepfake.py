"""
Enhanced face recognition webcam application with real-time deepfake detection.

This integrates the original face detection with deepfake detection capabilities.
"""

import cv2
import numpy as np
import logging
from deepfake_detector import DeepfakeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakeWebcamApp:
    """Real-time webcam application with deepfake detection."""
    
    def __init__(self, camera_id=0, model_path=None, enable_blur=False):
        """
        Initialize the webcam application.
        
        Args:
            camera_id: Camera device ID (default 0)
            model_path: Path to deepfake detection model
            enable_blur: Enable face blurring feature
        """
        self.camera_id = camera_id
        self.enable_blur = enable_blur
        
        # Initialize video capture
        self.capture = cv2.VideoCapture(camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize deepfake detector
        logger.info("Initializing deepfake detector...")
        self.detector = DeepfakeDetector(model_path=model_path, device='cpu', threshold=0.5)
        
        # Original face cascade for comparison
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Statistics
        self.frame_count = 0
        self.total_faces = 0
        self.total_fake_detections = 0
        
        logger.info("Webcam application initialized")
    
    def run(self):
        """Run the main webcam loop."""
        logger.info("Starting webcam feed... Press ESC to exit")
        
        try:
            while True:
                ret, frame = self.capture.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add statistics overlay
                self._add_stats_overlay(processed_frame)
                
                # Display frame
                cv2.imshow('Deepfake Detection - Press ESC to exit', processed_frame)
                
                # Check for ESC key
                key = cv2.waitKey(50)
                if key == 27:  # ESC
                    break
                
                self.frame_count += 1
        
        finally:
            self.cleanup()
    
    def process_frame(self, frame):
        """
        Process a single frame with deepfake detection.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with annotations
        """
        # Run deepfake detection
        face_results = self.detector.infer_on_frame(frame)
        
        # Update statistics
        self.total_faces += len(face_results)
        self.total_fake_detections += sum(1 for f in face_results if f.is_fake)
        
        # Annotate frame
        for face in face_results:
            x, y, w, h = face.bbox
            
            # Determine color based on detection
            if face.is_fake:
                color = (0, 0, 255)  # Red for fake
                label = "FAKE"
            else:
                color = (0, 255, 0)  # Green for real
                label = "REAL"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with score
            score_text = f"{label}: {face.deepfake_score:.2f}"
            label_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(frame, score_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Optional: Apply blur to fake faces
            if self.enable_blur and face.is_fake:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y:y+h, x:x+w] = blurred
        
        return frame
    
    def _add_stats_overlay(self, frame):
        """Add statistics overlay to frame."""
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Stats panel
        panel_height = 100
        cv2.rectangle(overlay, (0, 0), (300, panel_height), (0, 0, 0), -1)
        
        # Blend with original
        alpha = 0.6
        frame[:panel_height, :300] = cv2.addWeighted(
            overlay[:panel_height, :300], alpha, 
            frame[:panel_height, :300], 1 - alpha, 0
        )
        
        # Add text
        y_offset = 25
        cv2.putText(frame, f"Frames: {self.frame_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Total Faces: {self.total_faces}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Fake Detected: {self.total_fake_detections}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection rate
        if self.total_faces > 0:
            fake_rate = (self.total_fake_detections / self.total_faces) * 100
            y_offset += 25
            cv2.putText(frame, f"Fake Rate: {fake_rate:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total faces detected: {self.total_faces}")
        logger.info(f"Total fake detections: {self.total_fake_detections}")
        
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Deepfake Detection Webcam')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to deepfake detection model')
    parser.add_argument('--blur', action='store_true',
                       help='Enable face blurring for detected fakes')
    
    args = parser.parse_args()
    
    # Create and run application
    app = DeepfakeWebcamApp(
        camera_id=args.camera,
        model_path=args.model,
        enable_blur=args.blur
    )
    
    app.run()


if __name__ == "__main__":
    main()

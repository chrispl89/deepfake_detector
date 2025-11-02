"""
Inference CLI for deepfake detection.

Usage:
    python inference.py --source <path_or_stream> --type <image|video|camera|stream> --out <dir>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector import DeepfakeDetector
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python inference.py --source image.jpg --type image --model checkpoints/faceforensics_model.pth --device cpu --out ./results

  # Analyze a video file
  python inference.py --source video.mp4 --type video --model checkpoints/faceforensics_model.pth --device cpu --out ./results

  # Analyze with custom threshold
  python inference.py --source image.jpg --type image --model checkpoints/faceforensics_model.pth --threshold 0.7 --device cpu --out ./results

  # Analyze webcam feed (30 seconds)
  python inference.py --source 0 --type camera --model checkpoints/faceforensics_model.pth --device cpu --out ./results --duration 30

  # Analyze RTSP stream
  python inference.py --source rtsp://camera_ip/stream --type stream --model checkpoints/faceforensics_model.pth --device cpu --out ./results
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Path or URL to media source')
    parser.add_argument('--type', type=str, required=True,
                       choices=['image', 'video', 'camera', 'stream'],
                       help='Type of media source')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0-1)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds for camera/stream analysis')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='Sample every Nth frame for video analysis')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.type in ['image', 'video'] and not os.path.exists(args.source):
        logger.error(f"Source file not found: {args.source}")
        return 1
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize detector
    logger.info("Initializing deepfake detector...")
    detector = DeepfakeDetector(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold
    )
    
    # Run analysis
    logger.info(f"Analyzing {args.type} source: {args.source}")
    
    try:
        report = detector.analyze_source(
            source=args.source,
            source_type=args.type,
            output_dir=args.out
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("DEEPFAKE DETECTION REPORT")
        print("=" * 60)
        print(f"Source: {report.source}")
        print(f"Type: {report.source_type}")
        print(f"Total Frames Analyzed: {report.total_frames}")
        print(f"Total Faces Detected: {report.total_faces_detected}")
        print(f"\nOVERALL VERDICT: {report.overall_verdict}")
        print(f"Confidence: {report.confidence:.2%}")
        print("=" * 60)
        
        # Save detailed report
        report_path = os.path.join(args.out, 'report.json')
        report.save_json(report_path)
        
        # Save summary
        summary_path = os.path.join(args.out, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DEEPFAKE DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Source: {report.source}\n")
            f.write(f"Type: {report.source_type}\n")
            f.write(f"Timestamp: {report.timestamp}\n")
            f.write(f"Total Frames Analyzed: {report.total_frames}\n")
            f.write(f"Total Faces Detected: {report.total_faces_detected}\n")
            f.write(f"\nOVERALL VERDICT: {report.overall_verdict}\n")
            f.write(f"Confidence: {report.confidence:.2%}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            if report.frames_analysis:
                f.write("Per-Frame Analysis:\n")
                f.write("-" * 60 + "\n")
                for frame in report.frames_analysis[:10]:  # First 10 frames
                    f.write(f"\nFrame {frame.frame_number} @ {frame.timestamp:.2f}s:\n")
                    f.write(f"  Faces detected: {len(frame.faces)}\n")
                    f.write(f"  Overall fake probability: {frame.overall_fake_probability:.3f}\n")
                    
                    for i, face in enumerate(frame.faces):
                        f.write(f"  Face {i+1}: {'FAKE' if face.is_fake else 'REAL'} ")
                        f.write(f"(score: {face.deepfake_score:.3f})\n")
        
        logger.info(f"Results saved to {args.out}")
        logger.info(f"Report: {report_path}")
        logger.info(f"Summary: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

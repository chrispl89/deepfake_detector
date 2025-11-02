#!/usr/bin/env python
"""
Extract frames from FaceForensics++ videos.

Usage:
    python extract_frames.py --input ./data/datasets/faceforensics_videos --output ./data/datasets/faceforensics_frames --max-frames 10 --sampling-rate 30
"""

import argparse
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path: Path, output_dir: Path, max_frames: int = 10, sampling_rate: int = 30):
    """
    Extract frames from a single video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract per video
        sampling_rate: Extract every Nth frame
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Processing {video_path.name}: {total_frames} frames @ {fps:.2f} FPS")
    
    frame_count = 0
    extracted_count = 0
    
    while extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % sampling_rate == 0:
            # Save frame
            frame_filename = output_dir / f"{video_path.stem}_frame_{extracted_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {extracted_count} frames from {video_path.name}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description='Extract frames from FaceForensics++ videos')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing videos')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for extracted frames')
    parser.add_argument('--max-frames', type=int, default=10,
                       help='Maximum frames to extract per video')
    parser.add_argument('--sampling-rate', type=int, default=30,
                       help='Extract every Nth frame')
    parser.add_argument('--video-extensions', nargs='+', default=['.mp4', '.avi', '.mov'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Find all video files recursively
    video_files = []
    for ext in args.video_extensions:
        video_files.extend(input_dir.rglob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return 1
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Process videos
    total_extracted = 0
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        # Determine if video is real or fake based on directory structure
        # FaceForensics++ structure: .../original/... or .../manipulated_sequences/...
        if 'original' in str(video_path).lower():
            label = 'real'
        else:
            label = 'fake'
        
        # Create output directory for this label
        label_output_dir = output_dir / label
        
        # Extract frames
        count = extract_frames_from_video(
            video_path, 
            label_output_dir, 
            args.max_frames, 
            args.sampling_rate
        )
        total_extracted += count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos processed: {len(video_files)}")
    logger.info(f"Total frames extracted: {total_extracted}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Count frames per label
    for label in ['real', 'fake']:
        label_dir = output_dir / label
        if label_dir.exists():
            frame_count = len(list(label_dir.glob('*.jpg')))
            logger.info(f"{label.upper()} frames: {frame_count}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

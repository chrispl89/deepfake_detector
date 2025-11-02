#!/usr/bin/env python
"""
Download real FaceForensics++ dataset.
This script downloads 50 real + 50 fake videos from FaceForensics++.
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset(dataset_type, num_videos, output_path, server='EU2'):
    """Download FaceForensics++ dataset using the official script."""
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        'datasets/download_faceforensics.py',
        str(output_path),
        '--dataset', dataset_type,
        '--compression', 'c40',
        '--type', 'videos',
        '--num_videos', str(num_videos),
        '--server', server
    ]
    
    logger.info(f"Downloading {dataset_type} videos...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run with echo to auto-confirm
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Auto-confirm by sending newline
    try:
        process.stdin.write('\n')
        process.stdin.flush()
    except:
        pass
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        logger.info(f"✓ {dataset_type} download completed")
        return True
    else:
        logger.error(f"✗ {dataset_type} download failed with code {process.returncode}")
        return False


def main():
    output_path = Path('./data/datasets/faceforensics_videos')
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DOWNLOADING FACEFORENSICS++ DATASET")
    logger.info("="*60)
    logger.info(f"Output: {output_path.absolute()}")
    logger.info("Target: 50 real + 50 fake videos")
    logger.info("="*60)
    
    # Try different servers
    servers = ['EU2', 'CA', 'EU']
    
    for server in servers:
        logger.info(f"\nTrying server: {server}")
        
        # Download real videos (original YouTube)
        logger.info("\n1. Downloading REAL videos (original sequences)...")
        if download_dataset('original', 50, output_path, server):
            break
        logger.warning(f"Server {server} failed, trying next...")
    else:
        logger.error("All servers failed for real videos")
        logger.info("\nNOTE: FaceForensics++ requires permission. Have you:")
        logger.info("1. Requested access at: https://github.com/ondyari/FaceForensics")
        logger.info("2. Received approval email?")
        logger.info("3. Downloaded the official script?")
        logger.info("\nAlternatively, you can use a public dataset like DFDC from Kaggle.")
        return 1
    
    # Download fake videos (Deepfakes)
    logger.info("\n2. Downloading FAKE videos (Deepfakes manipulation)...")
    for server in servers:
        if download_dataset('Deepfakes', 50, output_path, server):
            break
        logger.warning(f"Server {server} failed, trying next...")
    else:
        logger.error("All servers failed for fake videos")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Videos saved to: {output_path.absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Extract frames: python extract_frames.py")
    logger.info("2. Extract faces: python extract_faces_from_frames.py")
    logger.info("3. Train model: python train.py")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
Dataset download and access manager with safety checks.

This module implements dataset downloading with proper permission checks,
license compliance, and fallback strategies.
"""

import os
import json
import logging
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Manages dataset downloads with safety and permission checks."""
    
    def __init__(self, datasets_root: str = "./data/datasets"):
        """
        Initialize the dataset downloader.
        
        Args:
            datasets_root: Root directory for storing datasets
        """
        self.datasets_root = Path(datasets_root)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.datasets_root / "datasets_metadata.json"
        self.metadata = self._load_metadata()
        
        # Dataset registry with access requirements
        self.registry = {
            'faceforensicspp': {
                'name': 'FaceForensics++',
                'url': 'https://github.com/ondyari/FaceForensics',
                'requires_permission': True,
                'license': 'Custom - Research only',
                'request_url': 'https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md',
                'fallback': 'kaggle_mirror'
            },
            'dfdc_preview': {
                'name': 'DFDC Preview Dataset',
                'url': 'https://www.kaggle.com/c/deepfake-detection-challenge/data',
                'requires_permission': False,
                'license': 'MIT',
                'kaggle_dataset': 'deepfake-detection-challenge/deepfake-detection-challenge',
                'fallback': None
            },
            'celebdf': {
                'name': 'Celeb-DF',
                'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
                'requires_permission': True,
                'license': 'Custom - Research only',
                'request_url': 'https://github.com/yuezunli/celeb-deepfakeforensics/blob/master/README.md',
                'fallback': None
            }
        }
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save dataset metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _log_download(self, dataset_name: str, url: str, checksum: str, status: str):
        """Log download attempt with metadata."""
        log_entry = {
            'dataset': dataset_name,
            'url': url,
            'checksum': checksum,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if dataset_name not in self.metadata:
            self.metadata[dataset_name] = []
        
        self.metadata[dataset_name].append(log_entry)
        self._save_metadata()
    
    def _create_permission_request(self, dataset_name: str) -> str:
        """
        Create a permission request file for datasets requiring access.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the request file
        """
        dataset_info = self.registry.get(dataset_name)
        if not dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        request_dir = self.datasets_root / "access_requests"
        request_dir.mkdir(exist_ok=True)
        
        request_file = request_dir / f"{dataset_name}_request.txt"
        
        with open(request_file, 'w') as f:
            f.write(f"Dataset Access Request: {dataset_info['name']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {dataset_info['name']}\n")
            f.write(f"URL: {dataset_info['url']}\n")
            f.write(f"License: {dataset_info['license']}\n")
            f.write(f"Request URL: {dataset_info.get('request_url', 'N/A')}\n\n")
            f.write("INSTRUCTIONS:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Visit the request URL above\n")
            f.write("2. Follow the dataset provider's access request process\n")
            f.write("3. Fill out any required forms or agreements\n")
            f.write("4. Wait for approval (may take several days)\n")
            f.write("5. Once approved, download the dataset manually\n")
            f.write("6. Place downloaded files in:\n")
            f.write(f"   {self.datasets_root / dataset_name}\n\n")
            f.write("IMPORTANT:\n")
            f.write("-" * 60 + "\n")
            f.write("- Respect the dataset license terms\n")
            f.write("- Do not redistribute without permission\n")
            f.write("- Use only for research/educational purposes\n\n")
            f.write(f"Request created: {datetime.now().isoformat()}\n")
        
        logger.info(f"Permission request created: {request_file}")
        return str(request_file)
    
    def _download_file(self, url: str, output_path: Path, 
                       chunk_size: int = 8192) -> bool:
        """
        Download a file with progress tracking.
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def download_faceforensicspp(self, force: bool = False, 
                                 compression: str = 'c23',
                                 num_videos: int = 100) -> Dict:
        """
        Download FaceForensics++ dataset using official script.
        
        Args:
            force: Force download even if permission not granted
            compression: Compression level ('raw', 'c23', 'c40')
            num_videos: Number of videos to download (None for all)
            
        Returns:
            Dictionary with download status and paths
        """
        dataset_name = 'faceforensicspp'
        dataset_dir = self.datasets_root / dataset_name
        
        logger.info(f"Attempting to download {dataset_name}...")
        
        # Check if already downloaded
        if dataset_dir.exists() and not force and any(dataset_dir.iterdir()):
            logger.info(f"{dataset_name} already exists at {dataset_dir}")
            return {'status': 'already_exists', 'path': str(dataset_dir)}
        
        # Create directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Show TOS and instructions
        logger.info("=" * 60)
        logger.info("FaceForensics++ Dataset Download")
        logger.info("=" * 60)
        logger.info("\nIMPORTANT: You must agree to FaceForensics++ Terms of Service")
        logger.info("TOS: http://kaldir.vc.in.tum.de/faceforensics/")
        logger.info("\nThe official download script will:")
        logger.info("1. Ask you to confirm ToS acceptance")
        logger.info("2. Download videos directly from FaceForensics servers")
        logger.info(f"3. Save to: {dataset_dir}")
        logger.info("\nYou can run the official script manually:")
        logger.info(f"  python datasets/download_faceforensics.py {dataset_dir} \\")
        logger.info(f"    --dataset Deepfakes --compression {compression} --num_videos {num_videos}")
        
        # Ask user if they want to proceed
        logger.info("\n" + "=" * 60)
        response = input("\nDo you want to run the official download script now? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            # Create instruction file
            instruction_file = self._create_faceforensics_instructions(
                dataset_dir, compression, num_videos
            )
            return {
                'status': 'user_declined',
                'instruction_file': instruction_file,
                'target_dir': str(dataset_dir),
                'message': 'User declined automatic download. See instructions file.'
            }
        
        # Check if official script exists
        script_path = Path(__file__).parent / 'download_faceforensics.py'
        
        if not script_path.exists():
            logger.error("=" * 60)
            logger.error("OFFICIAL SCRIPT NOT FOUND")
            logger.error("=" * 60)
            logger.error("\nThe FaceForensics++ download script is NOT included in this")
            logger.error("repository due to licensing restrictions.")
            logger.error("\n⚠️  IMPORTANT: You CANNOT simply download the script!")
            logger.error("You must REQUEST ACCESS first:")
            logger.error("\nStep 1: Fill out the official Google Form:")
            logger.error("  https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform")
            logger.error("\nStep 2: Wait for approval email (~1 week)")
            logger.error("  - Email will contain download link to the script")
            logger.error("  - Check spam folder if no response")
            logger.error("\nStep 3: Download script from link in email")
            logger.error("  - Save it as: datasets/download_faceforensics.py")
            logger.error("\nFull instructions: datasets/DOWNLOAD_FACEFORENSICS_README.md")
            logger.error("Official repo: https://github.com/ondyari/FaceForensics")
            logger.error("=" * 60)
            
            instruction_file = self._create_faceforensics_instructions(
                dataset_dir, compression, num_videos
            )
            return {
                'status': 'script_not_found',
                'instruction_file': instruction_file,
                'message': 'Official FaceForensics++ script not found. See datasets/DOWNLOAD_FACEFORENSICS_README.md'
            }
        
        # Run official download script
        logger.info("\nLaunching official FaceForensics++ download script...")
        logger.info("This will download a sample of the dataset.")
        
        try:
            import subprocess
            import sys
            
            cmd = [
                sys.executable,
                str(script_path),
                str(dataset_dir),
                '--dataset', 'Deepfakes',  # Start with Deepfakes
                '--compression', compression,
                '--type', 'videos',
                '--num_videos', str(num_videos),
                '--server', 'EU2'
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False, capture_output=False)
            
            if result.returncode == 0:
                logger.info("Download completed successfully!")
                return {
                    'status': 'success',
                    'path': str(dataset_dir),
                    'message': f'Downloaded {num_videos} videos to {dataset_dir}'
                }
            else:
                logger.warning("Download may have failed or been interrupted.")
                return {
                    'status': 'partial',
                    'path': str(dataset_dir),
                    'message': 'Download completed with warnings. Check output directory.'
                }
                
        except Exception as e:
            logger.error(f"Error running download script: {e}")
            instruction_file = self._create_faceforensics_instructions(
                dataset_dir, compression, num_videos
            )
            return {
                'status': 'error',
                'error': str(e),
                'instruction_file': instruction_file,
                'message': 'Automatic download failed. See instructions for manual download.'
            }
    
    def _create_faceforensics_instructions(self, dataset_dir: Path,
                                          compression: str,
                                          num_videos: int) -> str:
        """Create instruction file for manual FaceForensics++ download."""
        instruction_file = dataset_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        
        with open(instruction_file, 'w') as f:
            f.write("FaceForensics++ Manual Download Instructions\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STEP 0: Request Access to FaceForensics++ (REQUIRED)\n")
            f.write("-" * 60 + "\n")
            f.write("⚠️  IMPORTANT: You CANNOT simply download the script!\n")
            f.write("The official script requires permission.\n\n")
            f.write("Correct Procedure:\n\n")
            f.write("1. Fill out the official Google Form:\n")
            f.write("   https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform\n\n")
            f.write("2. Wait for approval email (usually ~1 week)\n")
            f.write("   - Email will contain private download link\n")
            f.write("   - Check spam folder if no response\n\n")
            f.write("3. Download script from the link in email\n")
            f.write("   - Save as: datasets/download_faceforensics.py\n\n")
            f.write("Full details: datasets/DOWNLOAD_FACEFORENSICS_README.md\n")
            f.write("Official repo: https://github.com/ondyari/FaceForensics\n\n")
            
            f.write("OPTION 1: Use Official Script\n")
            f.write("-" * 60 + "\n")
            f.write(f"python datasets/download_faceforensics.py {dataset_dir} \\\n")
            f.write(f"  --dataset Deepfakes \\\n")
            f.write(f"  --compression {compression} \\\n")
            f.write(f"  --num_videos {num_videos} \\\n")
            f.write(f"  --server EU2\n\n")
            
            f.write("OPTION 2: Download All Datasets\n")
            f.write("-" * 60 + "\n")
            f.write(f"python datasets/download_faceforensics.py {dataset_dir} \\\n")
            f.write(f"  --dataset all \\\n")
            f.write(f"  --compression {compression}\n\n")
            
            f.write("OPTION 3: Download Specific Dataset\n")
            f.write("-" * 60 + "\n")
            f.write("Available datasets:\n")
            f.write("  - original (real videos)\n")
            f.write("  - Deepfakes\n")
            f.write("  - Face2Face\n")
            f.write("  - FaceSwap\n")
            f.write("  - NeuralTextures\n")
            f.write("  - DeepFakeDetection\n\n")
            
            f.write(f"Example:\n")
            f.write(f"python datasets/download_faceforensics.py {dataset_dir} \\\n")
            f.write(f"  --dataset Face2Face \\\n")
            f.write(f"  --compression c23\n\n")
            
            f.write("COMPRESSION OPTIONS:\n")
            f.write("  - raw: Lossless (largest files)\n")
            f.write("  - c23: Medium compression (recommended)\n")
            f.write("  - c40: High compression (smallest files)\n\n")
            
            f.write("SERVERS:\n")
            f.write("  - EU: Germany (default)\n")
            f.write("  - EU2: Germany (alternative)\n")
            f.write("  - CA: Canada\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("For more information:\n")
            f.write("https://github.com/ondyari/FaceForensics\n")
        
        logger.info(f"Instructions saved to: {instruction_file}")
        return str(instruction_file)
    
    def download_dfdc_preview(self, kaggle_api_key: Optional[str] = None) -> Dict:
        """
        Download DFDC Preview dataset from Kaggle.
        
        Args:
            kaggle_api_key: Kaggle API key (optional, will try environment)
            
        Returns:
            Dictionary with download status and paths
        """
        dataset_name = 'dfdc_preview'
        dataset_dir = self.datasets_root / dataset_name
        
        logger.info(f"Attempting to download {dataset_name}...")
        
        # Check if already downloaded
        if dataset_dir.exists():
            logger.info(f"{dataset_name} already exists at {dataset_dir}")
            return {'status': 'already_exists', 'path': str(dataset_dir)}
        
        # Try to use Kaggle API
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            logger.info("Downloading from Kaggle...")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            kaggle_dataset = self.registry[dataset_name]['kaggle_dataset']
            api.dataset_download_files(
                kaggle_dataset,
                path=str(dataset_dir),
                unzip=True
            )
            
            logger.info(f"Successfully downloaded to {dataset_dir}")
            
            self._log_download(
                dataset_name,
                f"kaggle:{kaggle_dataset}",
                "N/A",
                "success"
            )
            
            return {'status': 'success', 'path': str(dataset_dir)}
            
        except ImportError:
            logger.warning("Kaggle API not installed. Install with: pip install kaggle")
            return self._create_manual_download_instructions(dataset_name)
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            return self._create_manual_download_instructions(dataset_name)
    
    def request_celebdf(self) -> Dict:
        """
        Create access request for Celeb-DF dataset.
        
        Returns:
            Dictionary with request status and instructions
        """
        dataset_name = 'celebdf'
        
        logger.info(f"Creating access request for {dataset_name}...")
        
        request_file = self._create_permission_request(dataset_name)
        
        dataset_dir = self.datasets_root / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        return {
            'status': 'permission_required',
            'request_file': request_file,
            'target_dir': str(dataset_dir),
            'message': 'Manual access request required'
        }
    
    def _create_manual_download_instructions(self, dataset_name: str) -> Dict:
        """Create instructions for manual dataset download."""
        dataset_info = self.registry[dataset_name]
        dataset_dir = self.datasets_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        readme_path = dataset_dir / "MANUAL_DOWNLOAD.txt"
        with open(readme_path, 'w') as f:
            f.write(f"{dataset_info['name']} - Manual Download Instructions\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset URL: {dataset_info['url']}\n")
            f.write(f"License: {dataset_info['license']}\n\n")
            f.write("Instructions:\n")
            f.write("1. Visit the dataset URL above\n")
            f.write("2. Download the dataset files\n")
            f.write(f"3. Extract to: {dataset_dir}\n\n")
        
        logger.info(f"Manual download instructions created: {readme_path}")
        
        return {
            'status': 'manual_required',
            'instructions_file': str(readme_path),
            'target_dir': str(dataset_dir)
        }
    
    def list_available_datasets(self) -> List[Dict]:
        """
        List all registered datasets and their status.
        
        Returns:
            List of dictionaries with dataset information
        """
        datasets = []
        
        for dataset_id, info in self.registry.items():
            dataset_dir = self.datasets_root / dataset_id
            exists = dataset_dir.exists() and any(dataset_dir.iterdir())
            
            datasets.append({
                'id': dataset_id,
                'name': info['name'],
                'url': info['url'],
                'license': info['license'],
                'requires_permission': info['requires_permission'],
                'downloaded': exists,
                'path': str(dataset_dir) if exists else None
            })
        
        return datasets


# Convenience functions
def download_faceforensicspp(datasets_root: str = "./data/datasets") -> Dict:
    """Download FaceForensics++ dataset."""
    downloader = DatasetDownloader(datasets_root)
    return downloader.download_faceforensicspp()


def download_dfdc_preview(datasets_root: str = "./data/datasets") -> Dict:
    """Download DFDC Preview dataset."""
    downloader = DatasetDownloader(datasets_root)
    return downloader.download_dfdc_preview()


def request_celebdf(datasets_root: str = "./data/datasets") -> Dict:
    """Request access to Celeb-DF dataset."""
    downloader = DatasetDownloader(datasets_root)
    return downloader.request_celebdf()


def list_available_datasets(datasets_root: str = "./data/datasets") -> List[Dict]:
    """List all available datasets."""
    downloader = DatasetDownloader(datasets_root)
    return downloader.list_available_datasets()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Available Datasets ===")
    datasets = list_available_datasets()
    for ds in datasets:
        print(f"\n{ds['name']} ({ds['id']})")
        print(f"  License: {ds['license']}")
        print(f"  Requires Permission: {ds['requires_permission']}")
        print(f"  Downloaded: {ds['downloaded']}")

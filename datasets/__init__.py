"""
Dataset access and download module.

This module handles downloading and accessing deepfake detection datasets
with proper safety checks and license compliance.
"""

from .dataset_downloader import (
    download_faceforensicspp,
    download_dfdc_preview,
    request_celebdf,
    list_available_datasets
)

__all__ = [
    'download_faceforensicspp',
    'download_dfdc_preview',
    'request_celebdf',
    'list_available_datasets'
]

"""
load_data.py

This module provides functions for finding the project root, loading image and mask file paths, 
and verifying that image and mask pairs match correctly.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetPaths:
    """Container for dataset paths"""
    train_images: List[str]
    train_masks: List[str]
    valid_images: List[str]
    valid_masks: List[str]
    test_images: List[str]
    test_masks: List[str]

class PathLoader:
    """Handles loading and validation of dataset paths"""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path = root_path or self._find_project_root()
        logger.info(f"Project root is {self.root_path}")

    def _find_project_root(self, marker_file: str = "README.MD", max_depth: int = 10) -> Path:
        """Find project root directory"""
        current_path = Path.cwd().resolve()
        for _ in range(max_depth):
            if (current_path / marker_file).exists():
                return current_path
            if current_path == current_path.parent:
                break
            current_path = current_path.parent
        raise FileNotFoundError(
            f"Root directory containing {marker_file} not found within {max_depth} levels"
        )

    def load_paths(self, directory: Path) -> List[str]:
        """Load all file paths from directory"""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        return [str(p.resolve()) for p in directory.rglob("*") if p.is_file()]

    def get_datasets(self) -> DatasetPaths:
        """Get all dataset paths"""
        datasets = {}
        for split in ['train', 'valid', 'test']:
            images_dir = self.root_path / f'dataset/processed/{split}/images'
            masks_dir = self.root_path / f'dataset/processed/{split}/masks'
            
            images = sorted([
                img for img in self.load_paths(images_dir)
                if "_leftImg8bit.png" in img
            ])
            masks = sorted([
                msk for msk in self.load_paths(masks_dir)
                if "_labelIds.png" in msk
            ])
            
            self._check_paths(images, masks)
            
            datasets[f'{split}_images'] = images
            datasets[f'{split}_masks'] = masks
        
        return DatasetPaths(**datasets)

    @staticmethod
    def _check_paths(images: List[str], masks: List[str]) -> None:
        """Verify image and mask pairs match"""
        if len(images) != len(masks):
            raise ValueError(
                f"Number of images ({len(images)}) does not match masks ({len(masks)})"
            )
        
        for img_path, msk_path in zip(images, masks):
            img_name = Path(img_path).stem.split("_")[:3]
            msk_name = Path(msk_path).stem.split("_")[:3]
            if img_name != msk_name:
                raise ValueError(f"Mismatch: {img_path} and {msk_path}")

# Remove duplicate functions and use the PathLoader class instead
def get_datasets(root_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Legacy function for backward compatibility"""
    loader = PathLoader(root_path)
    paths = loader.get_datasets()
    return {
        "train_images": paths.train_images,
        "train_masks": paths.train_masks,
        "valid_images": paths.valid_images,
        "valid_masks": paths.valid_masks,
        "test_images": paths.test_images,
        "test_masks": paths.test_masks
    }
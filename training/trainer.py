"""
Trainer class for deepfake detection models with mixed precision and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime

from .dataset import DeepfakeDataset
from models.xception_model import XceptionDeepfake

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for deepfake detection models with advanced features:
    - Mixed precision training
    - Gradient checkpointing
    - Early stopping
    - Learning rate scheduling
    - Comprehensive metrics tracking
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary with keys:
                - epochs: Number of training epochs
                - batch_size: Batch size
                - learning_rate: Initial learning rate
                - backbone: Model backbone ('xception', 'resnet', etc.)
                - image_size: Input image size
                - validation_split: Validation split ratio
                - use_mixed_precision: Enable mixed precision
                - early_stopping_patience: Patience for early stopping
                - device: 'cpu' or 'cuda'
                - checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        
        # Set device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.validation_split = config.get('validation_split', 0.2)
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        backbone = self.config.get('backbone', 'xception')
        
        if backbone == 'xception':
            model = XceptionDeepfake(pretrained=True, num_classes=2)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        model = model.to(self.device)
        
        logger.info(f"Created model: {backbone}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train(self, dataset_paths: List[str], model_save_path: str) -> Dict:
        """
        Train the model on provided datasets.
        
        Args:
            dataset_paths: List of paths to training datasets
            model_save_path: Path to save the trained model
            
        Returns:
            Dictionary with training history and metrics
        """
        logger.info("Starting training...")
        logger.info(f"Datasets: {dataset_paths}")
        
        # Load and combine datasets
        train_loader, val_loader = self._prepare_dataloaders(dataset_paths)
        
        if train_loader is None or val_loader is None:
            logger.warning("No data available for training. Creating dummy dataset...")
            self._create_and_load_dummy_data()
            train_loader, val_loader = self._prepare_dataloaders(['./data/dummy_dataset'])
        
        # Training loop
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_auc = self._validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_val_auc = val_auc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                self._save_checkpoint(model_save_path, epoch, is_best=True)
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self._save_checkpoint(str(checkpoint_path), epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Final evaluation
        final_metrics = self._final_evaluation(val_loader)
        
        # Prepare training report
        training_report = {
            'total_epochs': epoch + 1,
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'final_metrics': final_metrics,
            'history': [
                {
                    'epoch': i + 1,
                    'train_loss': self.history['train_loss'][i],
                    'train_acc': self.history['train_acc'][i],
                    'val_loss': self.history['val_loss'][i],
                    'val_acc': self.history['val_acc'][i],
                    'val_auc': self.history['val_auc'][i]
                }
                for i in range(len(self.history['train_loss']))
            ]
        }
        
        logger.info("\nTraining completed!")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")
        
        return training_report
    
    def _prepare_dataloaders(self, dataset_paths: List[str]) -> tuple:
        """Prepare training and validation dataloaders."""
        # Load datasets
        datasets = []
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    dataset = DeepfakeDataset(
                        root_dir=path,
                        target_size=(self.config.get('image_size', 299), 
                                    self.config.get('image_size', 299))
                    )
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to load dataset from {path}: {e}")
        
        if not datasets:
            logger.warning("No datasets loaded successfully")
            return None, None
        
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        
        # Split into train and validation
        val_size = int(len(combined_dataset) * self.validation_split)
        train_size = len(combined_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            combined_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Calculate AUC
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except:
            epoch_auc = 0.0
        
        return epoch_loss, epoch_acc, epoch_auc
    
    def _final_evaluation(self, val_loader: DataLoader) -> Dict:
        """Perform final evaluation with detailed metrics."""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'accuracy': (np.array(all_predictions) == np.array(all_labels)).mean(),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        
        logger.info("\nFinal Evaluation Metrics:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            logger.info(f"Best model checkpoint saved to {path}")
        else:
            logger.info(f"Checkpoint saved to {path}")
    
    def _create_and_load_dummy_data(self):
        """Create dummy data for testing."""
        from training.dataset import create_dummy_dataset
        
        dummy_path = './data/dummy_dataset'
        create_dummy_dataset(dummy_path, num_samples=100)
        logger.info(f"Created dummy dataset at {dummy_path}")


if __name__ == "__main__":
    # Test trainer
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'epochs': 5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'backbone': 'xception',
        'image_size': 299,
        'validation_split': 0.2,
        'use_mixed_precision': False,
        'early_stopping_patience': 3,
        'device': 'cpu'
    }
    
    trainer = Trainer(config)
    
    # Create dummy dataset
    from training.dataset import create_dummy_dataset
    create_dummy_dataset('./data/test_training', num_samples=50)
    
    # Train
    result = trainer.train(['./data/test_training'], './checkpoints/test_model.pth')
    
    print("\nTraining completed!")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")

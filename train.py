"""
Training script for deepfake detection models.

Usage:
    python train.py --datasets <path1> <path2> ... --output model.pth --config config.json
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector import train_detector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train.py --datasets ./data/dataset1 ./data/dataset2 --output model.pth

  # Train with custom config
  python train.py --datasets ./data/dataset1 --output model.pth --config config.json

  # Train with GPU
  python train.py --datasets ./data/dataset1 --output model.pth --device cuda
        """
    )
    
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Paths to training datasets')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save trained model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to training configuration JSON file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--backbone', type=str, default='xception',
                       choices=['xception', 'resnet', 'efficientnet'],
                       help='Model backbone architecture')
    parser.add_argument('--image-size', type=int, default=299,
                       help='Input image size')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Build config from command line arguments
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'backbone': args.backbone,
            'image_size': args.image_size,
            'validation_split': 0.2,
            'use_mixed_precision': args.mixed_precision,
            'early_stopping_patience': 10,
            'device': args.device,
            'checkpoint_dir': './checkpoints'
        }
    
    # Verify datasets exist
    valid_datasets = []
    for dataset_path in args.datasets:
        if Path(dataset_path).exists():
            valid_datasets.append(dataset_path)
            logger.info(f"Found dataset: {dataset_path}")
        else:
            logger.warning(f"Dataset not found: {dataset_path}")
    
    if not valid_datasets:
        logger.error("No valid datasets found!")
        logger.info("Creating dummy dataset for testing...")
        from training.dataset import create_dummy_dataset
        dummy_path = './data/dummy_dataset'
        create_dummy_dataset(dummy_path, num_samples=100)
        valid_datasets = [dummy_path]
    
    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")
    
    # Start training
    logger.info("Starting training...")
    
    try:
        training_report = train_detector(
            dataset_paths=valid_datasets,
            model_save_path=args.output,
            config=config
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Model saved to: {args.output}")
        print(f"Best epoch: {training_report.best_epoch}")
        print(f"Best validation accuracy: {training_report.best_val_accuracy:.4f}")
        print(f"Best validation AUC: {training_report.best_val_auc:.4f}")
        print("=" * 60 + "\n")
        
        # Print final metrics
        print("Final Metrics:")
        for key, value in training_report.final_metrics.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")
        
        # Save training configuration
        config_output = args.output.replace('.pth', '_config.json')
        with open(config_output, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Training configuration saved to {config_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

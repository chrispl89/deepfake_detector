"""
Xception-based deepfake detection model with transfer learning support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)


class XceptionDeepfake(nn.Module):
    """
    Xception model adapted for deepfake detection.
    Uses transfer learning from ImageNet pretrained weights.
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.5):
        """
        Initialize Xception model for deepfake detection.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(XceptionDeepfake, self).__init__()
        
        # Try to use torchvision's Xception-like model or implement simplified version
        try:
            import timm
            # Use timm library for better model support
            self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
            logger.info(f"Loaded Xception from timm, feature dim: {self.feature_dim}")
        except ImportError:
            logger.warning("timm not installed, using simplified architecture")
            # Simplified CNN architecture as fallback
            self.backbone = self._create_simple_cnn()
            self.feature_dim = 512
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        logger.info(f"XceptionDeepfake initialized with {num_classes} classes")
    
    def _create_simple_cnn(self) -> nn.Module:
        """Create a simplified CNN architecture as fallback."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        
        # Flatten if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        features = self.dropout(features)
        output = self.fc(features)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities for each class
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


class SimpleDeepfakeDetector(nn.Module):
    """
    Lightweight deepfake detector for CPU inference.
    """
    
    def __init__(self):
        super(SimpleDeepfakeDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path: Optional[str] = None, 
               device: str = 'cpu',
               model_type: str = 'xception') -> nn.Module:
    """
    Load a deepfake detection model.
    
    Args:
        model_path: Path to model weights. If None, returns untrained model.
        device: Device to load model on ('cpu' or 'cuda')
        model_type: Type of model ('xception' or 'simple')
        
    Returns:
        Loaded model
    """
    # Create model
    if model_type == 'xception':
        model = XceptionDeepfake(pretrained=(model_path is None))
    elif model_type == 'simple':
        model = SimpleDeepfakeDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights if provided
    if model_path is not None and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using untrained model")
    
    model = model.to(device)
    model.eval()
    
    return model


def export_to_onnx(model: nn.Module, 
                   output_path: str,
                   input_shape: tuple = (1, 3, 299, 299),
                   opset_version: int = 11) -> bool:
    """
    Export PyTorch model to ONNX format for optimized inference.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        opset_version: ONNX opset version
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification successful")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def get_model_info(model: nn.Module) -> Dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__,
        'device': next(model.parameters()).device
    }


if __name__ == "__main__":
    # Test model creation
    model = XceptionDeepfake(pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 299, 299)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model info
    info = get_model_info(model)
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

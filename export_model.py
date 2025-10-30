"""
Model export utility for converting PyTorch models to ONNX and TorchScript formats.

Usage:
    python export_model.py --model model.pth --output model.onnx --format onnx
"""

import argparse
import logging
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.xception_model import load_model, export_to_onnx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_to_torchscript(model: torch.nn.Module, output_path: str, 
                           input_shape: tuple = (1, 3, 299, 299)) -> bool:
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        input_shape: Input tensor shape
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save traced model
        traced_model.save(output_path)
        
        logger.info(f"Model exported to TorchScript: {output_path}")
        
        # Verify the exported model
        loaded_model = torch.jit.load(output_path)
        test_output = loaded_model(dummy_input)
        logger.info("TorchScript model verification successful")
        
        return True
        
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        return False


def quantize_model(model: torch.nn.Module, output_path: str) -> bool:
    """
    Quantize PyTorch model for faster CPU inference.
    
    Args:
        model: PyTorch model to quantize
        output_path: Path to save quantized model
        
    Returns:
        True if quantization successful, False otherwise
    """
    try:
        model.eval()
        
        # Dynamic quantization (best for CPU inference)
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        logger.info(f"Quantized model saved: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False


def optimize_for_tensorrt(onnx_path: str, output_path: str, 
                          precision: str = 'fp16') -> bool:
    """
    Optimize ONNX model for TensorRT inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        
    Returns:
        True if optimization successful, False otherwise
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger.info("Building TensorRT engine... This may take several minutes.")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Using INT8 precision")
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved: {output_path}")
        
        return True
        
    except ImportError:
        logger.error("TensorRT not installed. Install with: pip install tensorrt")
        return False
    except Exception as e:
        logger.error(f"TensorRT optimization failed: {e}")
        return False


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description='Export deepfake detection model to various formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  python export_model.py --model model.pth --output model.onnx --format onnx

  # Export to TorchScript
  python export_model.py --model model.pth --output model.pt --format torchscript

  # Quantize for CPU
  python export_model.py --model model.pth --output model_quantized.pth --format quantized

  # Convert ONNX to TensorRT
  python export_model.py --model model.onnx --output model.trt --format tensorrt --precision fp16
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to input model')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save exported model')
    parser.add_argument('--format', type=str, required=True,
                       choices=['onnx', 'torchscript', 'quantized', 'tensorrt'],
                       help='Export format')
    parser.add_argument('--input-shape', type=int, nargs=4, 
                       default=[1, 3, 299, 299],
                       help='Input shape (batch, channels, height, width)')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision for TensorRT (fp32, fp16, int8)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to load model on')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    logger.info(f"Loading model from {args.model}")
    
    # Export based on format
    if args.format == 'tensorrt':
        # TensorRT requires ONNX as input
        if not args.model.endswith('.onnx'):
            logger.error("TensorRT export requires ONNX model as input")
            return 1
        
        success = optimize_for_tensorrt(
            args.model, 
            args.output,
            precision=args.precision
        )
    else:
        # Load PyTorch model
        try:
            model = load_model(args.model, device=args.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 1
        
        # Export
        if args.format == 'onnx':
            success = export_to_onnx(
                model, 
                args.output,
                input_shape=tuple(args.input_shape)
            )
        elif args.format == 'torchscript':
            success = export_to_torchscript(
                model, 
                args.output,
                input_shape=tuple(args.input_shape)
            )
        elif args.format == 'quantized':
            success = quantize_model(model, args.output)
        else:
            logger.error(f"Unknown format: {args.format}")
            return 1
    
    if success:
        logger.info(f"Export successful! Model saved to: {args.output}")
        
        # Print file size
        size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.2f} MB")
        
        return 0
    else:
        logger.error("Export failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

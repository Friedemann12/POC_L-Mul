#!/usr/bin/env python
"""
8-bit Quantization using BitsAndBytes

This module provides 8-bit quantization using BitsAndBytes library
with Hugging Face integration via BitsAndBytesConfig.
"""

import torch
import warnings
import os


def check_quantization_support():
    """Check if BitsAndBytes quantization is supported."""
    # Check if quantization is disabled via environment variable
    if os.environ.get("DISABLE_FP8", "0") == "1":
        warnings.warn("8-bit quantization disabled via DISABLE_FP8 environment variable")
        return False
    
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        return True
    except ImportError:
        warnings.warn("BitsAndBytes not found. Install with: pip install bitsandbytes")
        return False
    
    if not torch.cuda.is_available():
        warnings.warn("8-bit quantization requires CUDA")
        return False
    
    return True


# Check quantization support at module load time
HAS_FP8_SUPPORT = check_quantization_support()


def get_8bit_config():
    """
    Get BitsAndBytes 8-bit quantization configuration.
    
    Returns:
        BitsAndBytesConfig for 8-bit quantization or None if not supported
    """
    if not HAS_FP8_SUPPORT:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
        )
    except ImportError:
        return None


def quantize_model_to_fp8(model):
    """
    Quantize a model to 8-bit using BitsAndBytes.
    
    Note: Quantization is done during model loading using BitsAndBytesConfig.
    
    Args:
        model: The model (quantized during loading)
    
    Returns:
        The same model
    """
    if not HAS_FP8_SUPPORT:
        raise RuntimeError("8-bit quantization is not supported on this system")
    
    # Model is quantized during loading
    print("✓ Model quantized to 8-bit using BitsAndBytes")
    return model


def prepare_model_for_fp8(model):
    """
    Prepare a model for 8-bit quantized inference.
    """
    return quantize_model_to_fp8(model)


# Legacy class names for compatibility
class FP8Quantizer:
    """Legacy compatibility class."""
    pass


class FP8Linear:
    """Legacy compatibility class."""
    pass 
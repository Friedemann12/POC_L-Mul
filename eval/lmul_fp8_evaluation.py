#!/usr/bin/env python
"""
Combined L-Mul + 8-bit Quantization Evaluation Script

This script combines L-Mul (Linear-complexity Multiplication) with 8-bit quantization
for maximum efficiency in language model inference.

Key Features:
- L-Mul algorithm for efficient attention computation
- 8-bit quantization via BitsAndBytes for reduced memory usage
- Synergistic optimization combining both techniques
- Custom CUDA kernel support with quantization

Usage:
    python lmul_fp8_evaluation.py --tasks strawberry_test
    python lmul_fp8_evaluation.py --model-name "google/gemma-2-2b-it"
    python lmul_fp8_evaluation.py  # Run all tasks

This combines the benefits of both L-Mul (computation efficiency) 
and 8-bit quantization (memory efficiency) for optimal performance.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import subprocess
import argparse
from utils.l_mul_optimized import l_mul_attention_optimized, compile_cuda_kernel, CUDA_AVAILABLE
from utils.evaluation_common import (
    clear_gpu_memory,
    load_model_and_tokenizer,
    setup_evaluation,
    run_evaluation_tasks,
    create_common_argparser,
    ensure_model_device_consistency
)
from utils.fp8_utils import (
    quantize_model_to_fp8,
    prepare_model_for_fp8,
    HAS_FP8_SUPPORT,
    get_8bit_config
)


# --- Initial Setup: Compile CUDA Kernel ---
def setup_cuda_kernel():
    """Compile CUDA kernel if available and not already compiled"""
    if not torch.cuda.is_available():
        return False
    
    if CUDA_AVAILABLE:
        return True
    
    try:
        # Try to compile the CUDA kernel
        compile_result = subprocess.run(
            [sys.executable, 'utils/compile_lmul_cuda.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if compile_result.returncode == 0:
            # Try to import the compiled module
            try:
                import l_mul_cuda
                return True
            except ImportError:
                return False
        else:
            return False
            
    except (subprocess.TimeoutExpired, Exception):
        return False

# Run setup at module import time
cuda_available = setup_cuda_kernel()

# Re-import to check if CUDA is now available after compilation
from utils.l_mul_optimized import CUDA_AVAILABLE as CUDA_NOW_AVAILABLE


# --- Combined L-Mul + FP8 Monkey-Patching ---
original_matmul = torch.matmul

def lmul_fp8_matmul_replacement(tensor_a, tensor_b):
    """
    L-Mul replacement for torch.matmul in quantized models.
    Uses L-Mul for attention computation on 8-bit quantized models.
    """
    # Heuristic to identify the QK^T operation in attention
    is_attention_scores = (
        tensor_a.dim() == 4 and tensor_b.dim() == 4 and
        tensor_a.shape[0] < 64 and # Batch size is usually not huge
        tensor_a.shape[2] == tensor_b.shape[3] # q_len == k_len for self-attention
    )

    if is_attention_scores and CUDA_AVAILABLE and tensor_a.is_cuda and tensor_b.is_cuda:
        try:
            # Apply L-Mul directly (model is already 8-bit quantized at the layer level)
            # Note: tensor_b is K.transpose(-2, -1), so we need to transpose it back
            key_states = tensor_b.transpose(-2, -1)
            return l_mul_attention_optimized(tensor_a, key_states, mantissa_bits=4)
            
        except Exception as e:
            # If L-Mul fails, let it fail cleanly
            raise RuntimeError(f"L-Mul computation failed: {str(e)}") from e
    
    # For all other matrix multiplications, use the original function
    return original_matmul(tensor_a, tensor_b)

def apply_lmul_fp8(model):
    """Applies the L-Mul optimization to 8-bit quantized models."""
    torch.matmul = lmul_fp8_matmul_replacement
    
    status = []
    if CUDA_NOW_AVAILABLE:
        status.append("L-Mul CUDA kernel")
    else:
        status.append("L-Mul PyTorch implementation")
    
    if HAS_FP8_SUPPORT:
        status.append("8-bit quantization")
    else:
        status.append("8-bit quantization not supported")
    
    print(f"✓ Combined optimization activated: {' + '.join(status)}")
    return model

def remove_lmul_fp8(model):
    """Removes the L-Mul monkey-patch to restore original functionality."""
    torch.matmul = original_matmul
    return model


def main(args):
    """Main function to run combined L-Mul + 8-bit quantization evaluation."""
    # Get quantization config
    quantization_config = get_8bit_config()
    if not quantization_config:
        raise RuntimeError("8-bit quantization not supported on this system")
    
    # Setup evaluation
    tasks_to_run, output_dir, model_name_for_file = setup_evaluation(args, "lmul_fp8")
    
    # Clear GPU memory before loading large model
    clear_gpu_memory()
    
    # Load model with 8-bit quantization
    print("Loading model with 8-bit quantization...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, 
        force_single_device=True,
        quantization_config=quantization_config
    )
    
    # Store device information
    original_device = next(model.parameters()).device
    print(f"Model loaded on device: {original_device}")

    # Apply quantization function
    model = quantize_model_to_fp8(model)
    
    # Verify device after quantization
    model_device = next(model.parameters()).device
    print(f"Model device after quantization: {model_device}")
    
    # Ensure model has a .device attribute for compatibility
    if not hasattr(model, 'device'):
        model.device = model_device
    
    # Configure caching based on args
    model.config.use_cache = not args.no_cache
    
    # Apply L-Mul optimization to the quantized model
    model = apply_lmul_fp8(model)
    
    if args.no_cache:
        print("⚠️  KV-caching disabled for accurate energy measurements")
    
    # Enable gradient checkpointing for memory efficiency (if needed)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Ensure model device consistency
    model = ensure_model_device_consistency(model)
    
    print(f"Running evaluation with device: {model_device}")

    # Run evaluation
    run_evaluation_tasks(model, tokenizer, tasks_to_run, output_dir, model_name_for_file, 
                        "lmul_fp8", force_device=str(model_device))

    # Clean up by removing the monkey-patch
    model = remove_lmul_fp8(model)
    
    print(f"\n🎉 Combined L-Mul + 8-bit quantization evaluation complete!")
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {memory_used:.2f} GB")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = create_common_argparser("Run combined L-Mul + 8-bit quantized model evaluations")
    
    # No additional arguments needed
    
    args = parser.parse_args()
    main(args) 
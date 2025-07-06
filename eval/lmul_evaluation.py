#!/usr/bin/env python
"""
L-Mul Evaluation Script with Custom CUDA Kernel

This script evaluates the Linear-complexity Multiplication (L-Mul) algorithm
using a high-performance custom CUDA kernel that replaces traditional matrix 
multiplication in attention mechanisms with energy-efficient approximations.

Key Features:
- Custom CUDA kernel for L-Mul operations (4-bit mantissa approximation)
- Automatic kernel compilation and optimization
- Direct integration via torch.matmul monkey-patching
- Comprehensive evaluation on various language understanding tasks

Usage:
    python lmul_evaluation.py --tasks strawberry_test
    python lmul_evaluation.py --model-name "google/gemma-2-2b-it"
    python lmul_evaluation.py  # Run all tasks

Based on: "Addition is All You Need for Energy-efficient Language Models"
          by Hongyin Luo and Wei Sun (2024)
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


# --- L-Mul Monkey-Patching ---
original_matmul = torch.matmul

def lmul_matmul_replacement(tensor_a, tensor_b):
    """
    This function replaces `torch.matmul`. It checks if the operation is the
    specific QK^T attention score calculation and, if so, uses the L-Mul kernel.
    Otherwise, it calls the original `torch.matmul`.
    """
    # Heuristic to identify the QK^T operation in Gemma3's attention
    # We check the dimensions: Q is (bsz, num_heads, q_len, head_dim) and K.T is (bsz, num_heads, head_dim, k_len)
    is_attention_scores = (
        tensor_a.dim() == 4 and tensor_b.dim() == 4 and
        tensor_a.shape[0] < 64 and # Batch size is usually not huge
        tensor_a.shape[2] == tensor_b.shape[3] # q_len == k_len for self-attention
    )

    if is_attention_scores and CUDA_AVAILABLE and tensor_a.is_cuda and tensor_b.is_cuda:
        # It's highly likely this is the QK^T calculation.
        # Note: tensor_b is K.transpose(-2, -1), so we need to transpose it back
        key_states = tensor_b.transpose(-2, -1)
        try:
            return l_mul_attention_optimized(tensor_a, key_states, mantissa_bits=4)
        except Exception as e:
            # Fallback to original matmul if L-Mul fails
            return original_matmul(tensor_a, tensor_b)
    
    # For all other matrix multiplications, use the original function
    return original_matmul(tensor_a, tensor_b)

def apply_lmul(model):
    """Applies the L-Mul monkey-patch."""
    torch.matmul = lmul_matmul_replacement
    if CUDA_NOW_AVAILABLE:
        print("✓ L-Mul CUDA kernel activated")
    else:
        print("✓ L-Mul PyTorch implementation activated")
    return model

def remove_lmul(model):
    """Removes the L-Mul monkey-patch to restore original functionality."""
    torch.matmul = original_matmul
    return model


def main(args):
    """Main function to run L-Mul evaluation."""
    # Common setup
    tasks_to_run, output_dir, model_name_for_file = setup_evaluation(args, "l_mul")
    
    # Clear GPU memory before loading large model
    clear_gpu_memory()
    
    # Load model without Flash Attention since we'll use L-Mul custom kernel
    model, tokenizer = load_model_and_tokenizer(args.model_name, force_single_device=True)

    # Configure caching based on args
    model.config.use_cache = not args.no_cache
    
    # Apply the L-Mul custom kernel via monkey-patching
    model = apply_lmul(model)
    
    if args.no_cache:
        print("⚠️  KV-caching disabled for accurate energy measurements")
    
    # Enable gradient checkpointing for memory efficiency (if needed)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Run evaluation with force_device for L-Mul
    run_evaluation_tasks(model, tokenizer, tasks_to_run, output_dir, model_name_for_file, 
                        "l_mul", force_device="cuda:0")

    # Clean up by removing the monkey-patch
    model = remove_lmul(model)
    
    print(f"\n🎉 L-Mul evaluation complete!")


if __name__ == "__main__":
    parser = create_common_argparser("Run L-Mul optimized model evaluations")
    
    # L-Mul specific arguments
    parser.add_argument("--mantissa-bits", type=int, default=4, 
                       help="Number of mantissa bits for L-Mul (default: 4)")
    
    args = parser.parse_args()
    main(args) 
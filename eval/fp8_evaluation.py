#!/usr/bin/env python
"""
8-bit Quantization Evaluation Script for Language Models

This script evaluates language models using 8-bit quantization via BitsAndBytes.
This provides significant memory savings compared to FP16.

Key Features:
- 8-bit quantization using BitsAndBytes
- Replaces linear layers with 8-bit quantized versions
- Memory-efficient inference
- Compatible with various transformer architectures

Usage:
    python fp8_evaluation.py --tasks strawberry_test
    python fp8_evaluation.py --model-name "google/gemma-2-2b-it"
    python fp8_evaluation.py  # Run all tasks
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
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


def main(args):
    """Main function to run 8-bit quantized evaluation."""
    # Get quantization config
    quantization_config = get_8bit_config()
    if not quantization_config:
        raise RuntimeError("8-bit quantization not supported on this system")
    
    # Setup evaluation
    tasks_to_run, output_dir, model_name_for_file = setup_evaluation(args, "fp8")
    
    # Clear GPU memory before loading large model
    clear_gpu_memory()

    # Load model with 8-bit quantization
    print("Loading model with 8-bit quantization...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
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
    
    if HAS_FP8_SUPPORT:
        print("✓ 8-bit quantization supported")
    else:
        print("⚠️  8-bit quantization not supported on this system")
        raise RuntimeError("8-bit quantization requires BitsAndBytes and CUDA")
    
    # Configure caching based on args
    model.config.use_cache = not args.no_cache
    
    if args.no_cache:
        print("⚠️  KV-caching disabled for accurate energy measurements")
    
    # Ensure model device consistency
    model = ensure_model_device_consistency(model)
    
    print(f"Running evaluation with device: {model_device}")
    
    # Run evaluation
    run_evaluation_tasks(model, tokenizer, tasks_to_run, output_dir, model_name_for_file, "fp8")
    
    print(f"\n🎉 8-bit quantized evaluation complete!")
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {memory_used:.2f} GB")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = create_common_argparser("Run 8-bit quantized model evaluations")
    
    # No 8-bit quantization-specific arguments needed for basic mode
    
    args = parser.parse_args()
    main(args) 
#!/usr/bin/env python
"""
Baseline Evaluation Script for Language Models

This script provides a baseline evaluation of language models using standard
attention mechanisms. It serves as a comparison point for the L-Mul implementation.

Key Features:
- Standard transformer attention mechanisms
- Comprehensive evaluation on various language understanding tasks
- Performance benchmarking and metrics tracking
- Clean baseline for comparison with optimized implementations

Usage:
    python evaluation.py --tasks strawberry_test
    python evaluation.py --model-name "google/gemma-2-2b-it"
    python evaluation.py  # Run all tasks

Run this before lmul_evaluation.py to establish baseline performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from utils.evaluation_common import (
    clear_gpu_memory,
    load_model_and_tokenizer,
    setup_evaluation,
    run_evaluation_tasks,
    create_common_argparser,
    ensure_model_device_consistency
)


def main(args):
    """Main function to run standard evaluation."""
    # Common setup
    tasks_to_run, output_dir, model_name_for_file = setup_evaluation(args, "standard")
    
    # Clear GPU memory before loading large model
    clear_gpu_memory()

    # Load the standard model with the specified attention implementation
    model, tokenizer = load_model_and_tokenizer(args.model_name, attn_implementation=args.attn_impl)
    
    # Configure caching based on args
    model.config.use_cache = not args.no_cache
    
    # Ensure model device consistency
    model = ensure_model_device_consistency(model)
    
    if args.attn_impl:
        print(f"✓ Standard model loaded with {args.attn_impl}")
    else:
        print("✓ Standard model loaded")
    
    if args.no_cache:
        print("⚠️  KV-caching disabled for accurate energy measurements")
    
    # Run evaluation
    run_evaluation_tasks(model, tokenizer, tasks_to_run, output_dir, model_name_for_file, "standard")
    
    print(f"\n🎉 Standard evaluation complete!")


if __name__ == "__main__":
    # Create parser with common arguments
    parser = create_common_argparser("Run standard model evaluations")
    
    # Add standard-specific arguments
    parser.add_argument("--attn-impl", type=str, default=None, 
                       help="Attention implementation (e.g., 'flash_attention_2')")
    
    args = parser.parse_args()
    main(args) 
#!/usr/bin/env python
"""
Common Evaluation Utilities

This module contains shared functionality for both standard and L-Mul evaluations.
It provides a consistent interface for model loading, task execution, and result saving.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.metrics import TaskTracker
from utils.tasks import ALL_TASKS
from utils.config import RESULTS_DIR
import os
import json
from datetime import datetime
import gc
import torch._dynamo
import warnings
import logging
import argparse
import time

# Suppress unnecessary warnings and output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable torch dynamo to avoid "Unsupported: generator" errors with Gemma2
torch._dynamo.config.suppress_errors = True
# Set environment variable to disable dynamo globally
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def clear_gpu_memory():
    """Clear GPU memory to make room for large models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def load_model_and_tokenizer(model_name, attn_implementation=None, force_single_device=False, quantization_config=None):
    """
    Loads the model and tokenizer from Hugging Face.
    
    Args:
        model_name: The Hugging Face model identifier
        attn_implementation: Optional attention implementation (e.g., 'flash_attention_2')
        force_single_device: Whether to force loading on a single device
        quantization_config: Optional BitsAndBytesConfig for quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_args = {
        "device_map": {"": 0}, 
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True
    }
        
    if attn_implementation:
        model_args["attn_implementation"] = attn_implementation
    
    # Add quantization config if provided
    if quantization_config:
        model_args["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_args
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def save_task_result(output_dir, task_name, result_data):
    """
    Saves a single task's result to a JSON file.
    
    Args:
        output_dir: Directory to save results
        task_name: Name of the task
        result_data: Dictionary containing task results
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{task_name}.json")
    with open(filename, "w") as f:
        json.dump(result_data, f, indent=4)


def ensure_model_device_consistency(model):
    """
    Ensures all model parameters are on the same device.
    Handles Accelerate-dispatched models properly.
    
    Args:
        model: The model to check
        
    Returns:
        The model (potentially modified)
    """
    devices = {param.device for param in model.parameters() if param.device.type != 'meta'}
    meta_devices = {param.device for param in model.parameters() if param.device.type == 'meta'}
    
    # Only print warnings for actual issues
    if len(devices) > 1:
        print(f"Warning: Model has parameters on multiple devices: {devices}")
    
    return model


def run_evaluation_tasks(model, tokenizer, tasks_to_run, output_dir, model_name_for_file, 
                        opt_type, force_device=None):
    """
    Helper function to run evaluation tasks for a given model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        tasks_to_run: List of task names to execute
        output_dir: Directory to save results
        model_name_for_file: Sanitized model name for file paths
        opt_type: Optimization type ('standard' or 'l_mul')
        force_device: Optional device to force for task execution (e.g., "cuda:0" for L-Mul)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Running evaluation...")
    print(f"{'='*60}")
    
    # Run tasks and save their results immediately
    for i, task_name in enumerate(tasks_to_run, 1):
        if task_name in ALL_TASKS:
            print(f"\n[{i}/{len(tasks_to_run)}] Executing task: {task_name}")
            print(f"  ├─ Status: Starting...", end="\r")
            
            task_function = ALL_TASKS[task_name]
            try:
                # Get task description (if available)
                task_desc = {
                    "output_length_variance": "Testing output length control",
                    "linguistic_constraints": "Testing linguistic constraints",
                    "hallucination_resistance": "Testing hallucination resistance", 
                    "strawberry_test": "Running strawberry counting test",
                    "code_challenges_basic": "Generating basic code solutions",
                    "code_challenges_advanced": "Generating advanced code solutions",
                    "format_adherence": "Testing format adherence",
                    "reasoning_tasks": "Testing reasoning capabilities",
                    "bias_and_safety": "Testing bias and safety responses"
                }.get(task_name, "Running evaluation")
                
                print(f"  ├─ Status: {task_desc}...", end="\r")
                
                project_name = f"{model_name_for_file}_{opt_type}_{task_name}"
                start_time = time.time()
                
                with TaskTracker(project_name, output_dir) as tracker:
                    # Add a progress callback to the tracker
                    original_add_tokens = tracker.add_generated_tokens
                    tokens_count = [0]  # Use list to allow modification in nested function
                    
                    def add_tokens_with_progress(count):
                        tokens_count[0] += count
                        print(f"  ├─ Status: Generating... ({tokens_count[0]} tokens)", end="\r")
                        return original_add_tokens(count)
                    
                    tracker.add_generated_tokens = add_tokens_with_progress
                    
                    # Pass force_device if provided (for L-Mul)
                    if force_device:
                        task_results = task_function(model, tokenizer, tracker=tracker, force_device=force_device)
                    else:
                        task_results = task_function(model, tokenizer, tracker=tracker)
                
                elapsed_time = time.time() - start_time
                
                # Create result with task-level metrics
                task_result_with_metrics = {
                    "task_name": task_name,
                    "performance_metrics": tracker.metrics,
                    "results": task_results
                }
                
                save_task_result(output_dir, task_name, task_result_with_metrics)
                
                # Final status
                print(f"  ├─ Status: Completed in {elapsed_time:.2f}s ({tokens_count[0]} tokens)")
                print(f"  └─ Result: ✓ Success")

            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"  ├─ Status: Failed after {elapsed_time:.2f}s")
                print(f"  └─ Result: ✗ Error: {type(e).__name__}: {str(e)}")
                
                import traceback
                full_error = traceback.format_exc()
                save_task_result(output_dir, f"{task_name}_error", {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": full_error
                })
        else:
            print(f"\n[{i}/{len(tasks_to_run)}] Task '{task_name}' not found. Skipping.")
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to: {output_dir}")


def setup_evaluation(args, evaluation_type="standard", extra_config=None):
    """
    Common setup for evaluations.
    
    Args:
        args: Parsed command line arguments
        evaluation_type: Type of evaluation ('standard', 'l_mul', 'fp8', 'lmul_fp8')
        extra_config: Optional dict with additional configuration to include in directory name
        
    Returns:
        Tuple of (tasks_to_run, output_dir, model_name_for_file)
    """
    # Set random seeds for reproducibility
    if not args.random_seed:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    model_name_for_file = args.model_name.replace("/", "_")
    
    # Determine which tasks to run
    tasks_to_run = args.tasks.split() if args.tasks else list(ALL_TASKS.keys())
    
    # Print evaluation info
    print(f"\n🚀 {evaluation_type.upper()} Evaluation Starting")
    print(f"Model: {args.model_name}")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    if args.random_seed:
        print(f"Random seed: enabled")
    else:
        print(f"Fixed seed: 42")
    
    # Create output directory with configuration indicators
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build directory name with configuration flags
    dir_parts = [model_name_for_file, evaluation_type, timestamp]
    
    # Add configuration flags
    if args.random_seed:
        dir_parts.append("random_seed")
    else:
        dir_parts.append("fixed_seed")
        
    if args.no_cache:
        dir_parts.append("no_cache")
    else:
        dir_parts.append("with_cache")
    
    # Add extra configuration if provided
    if extra_config:
        for key, value in extra_config.items():
            if value is not None:
                # Sanitize value for filename
                clean_value = str(value).replace("/", "_").replace(" ", "_")
                dir_parts.append(f"{key}_{clean_value}")
    
    # Join with underscores
    output_dir = RESULTS_DIR / "_".join(dir_parts)
    
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return tasks_to_run, str(output_dir), model_name_for_file


def create_common_argparser(description):
    """
    Creates a common argument parser with shared arguments.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b-it", 
                       help="Model name from Hugging Face")
    parser.add_argument("--tasks", type=str, default=None, 
                       help="Space-separated list of tasks to run (e.g., 'code_challenges strawberry_test')")
    parser.add_argument("--random-seed", action="store_true", 
                       help="Use random seed instead of fixed seed (42)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable KV-caching for more accurate energy measurements (slower but shows true L-Mul efficiency)")
    return parser 
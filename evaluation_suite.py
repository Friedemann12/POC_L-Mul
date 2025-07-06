import subprocess
import sys
import argparse
import time
from datetime import datetime
import os
import json
from pathlib import Path
import signal

# Global variable to track current subprocess
current_process = None

def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C) gracefully."""
    global current_process
    print("\n\n⚠️  Evaluation interrupted by user (Ctrl+C)")
    print("Cleaning up and terminating processes...")
    
    # Terminate current subprocess if running
    if current_process and current_process.poll() is None:
        try:
            if sys.platform == "win32":
                # On Windows, use terminate() which sends SIGTERM
                current_process.terminate()
            else:
                # On Unix, send SIGINT first to allow graceful shutdown
                current_process.send_signal(signal.SIGINT)
                # Give it a moment to clean up
                time.sleep(2)
                # If still running, terminate
                if current_process.poll() is None:
                    current_process.terminate()
        except Exception as e:
            print(f"Error terminating subprocess: {e}")
    
    print("Evaluation suite terminated.")
    sys.exit(130)  # Standard exit code for SIGINT

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
if sys.platform == "win32":
    # On Windows, also handle SIGBREAK (Ctrl+Break)
    signal.signal(signal.SIGBREAK, signal_handler)

# Default model configurations for comprehensive mode
COMPREHENSIVE_MODELS = [
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",       
    "google/gemma-7b-it",          
    "mistralai/Ministral-8B-Instruct-2410", 
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-14B",   
]

QUICK_MODELS = [
    "google/gemma-3-1b-it",           # Small model for quick tests
    "llama3/llama-3.1-8b-instruct",  # Alternative architecture
]

def save_error_log(eval_type, result, cmd, extra_info=None):
    """
    Save detailed error information to log file for investigation.
    
    Args:
        eval_type: Type of evaluation that failed
        result: subprocess.CompletedProcess result
        cmd: Command that was executed
        extra_info: Additional context information
    """
    try:
        error_log_dir = Path("results") / "error_logs"
        error_log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = error_log_dir / f"{eval_type}_error_{timestamp}.log"
        
        with open(error_file, 'w') as f:
            f.write(f"Evaluation Type: {eval_type.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exit Code: {result.returncode}\n")
            f.write(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
            if extra_info:
                f.write(f"Extra Info: {extra_info}\n")
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or "No stderr output\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout or "No stdout output\n")
            
        print(f"Error details saved to: {error_file}")
        return error_file
    except Exception as e:
        print(f"Failed to save error log: {e}")
        return None

def handle_subprocess_error(eval_type, result, cmd, extra_info=None, rerun_for_error=True):
    """
    Handle subprocess errors with consistent logging and display.
    
    Args:
        eval_type: Type of evaluation that failed
        result: subprocess.CompletedProcess result
        cmd: Command that was executed
        extra_info: Additional context information
        rerun_for_error: Whether to rerun command to capture error output
    
    Returns:
        bool: False (indicating failure)
    """
    print(f"\n❌ {eval_type.upper()} evaluation failed!")
    print(f"Exit code: {result.returncode}")
    
    # If we were showing real-time output and don't have captured error details,
    # we might want to rerun to capture the error
    if rerun_for_error and not result.stderr and not result.stdout:
        print("Re-running to capture error details...")
        try:
            error_result = run_subprocess_with_interrupt(cmd, eval_type, show_output=False)
            if error_result.stderr or error_result.stdout:
                result = error_result
        except:
            pass
    
    if result.stderr:
        print("Error details:")
        print(result.stderr)
    
    if result.stdout:
        print("Output:")
        print(result.stdout)
    
    # Save detailed error log
    save_error_log(eval_type, result, cmd, extra_info)
    
    return False

def run_subprocess_with_interrupt(cmd, eval_type="evaluation", show_output=True):
    """
    Run a subprocess with proper interrupt handling and real-time output.
    
    Args:
        cmd: Command to execute (list of strings)
        eval_type: Type of evaluation for logging
        show_output: Whether to show real-time output (default: True)
    
    Returns:
        subprocess.CompletedProcess: The result of the subprocess
    """
    global current_process
    
    try:
        if show_output:
            # Use Popen without capturing output for real-time display
            popen_kwargs = {
                'stdout': None,  # Don't capture, let it go to console
                'stderr': None,  # Don't capture, let it go to console
                'text': True
            }
        else:
            # Use Popen with output capture (for error logging)
            popen_kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.PIPE,
                'text': True
            }
        
        # On Windows, create a new process group
        if sys.platform == "win32":
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        
        current_process = subprocess.Popen(cmd, **popen_kwargs)
        
        # Wait for process to complete
        returncode = current_process.wait()
        
        # Create a CompletedProcess object to maintain compatibility
        if show_output:
            # When showing output, we don't have captured text
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=returncode,
                stdout="",  # Empty since we didn't capture
                stderr=""   # Empty since we didn't capture
            )
        else:
            # When capturing output, read from pipes
            stdout, stderr = current_process.communicate()
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr
            )
        
        return result
        
    except KeyboardInterrupt:
        # Handle Ctrl+C during subprocess execution
        print(f"\n⚠️  {eval_type} interrupted by user")
        if current_process and current_process.poll() is None:
            current_process.terminate()
        raise
    finally:
        # Clear the current process reference
        current_process = None

def run_simple_evaluation(args):
    """
    Run simple evaluation mode (replaces run_full_evaluation.py functionality).
    Runs all 4 optimization types on a single configuration.
    """
    model = args.models[0] if args.models else args.model_name
    
    print(f"\n{'='*80}")
    print(f"L-MUL EVALUATION SUITE - Simple Mode")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Evaluations: {', '.join(args.eval_types) if args.eval_types else 'All (standard, lmul, fp8, lmul_fp8)'}")
    print(f"Cache: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"Seed: {'Random' if args.random_seed else 'Fixed (42)'}")
    if args.tasks:
        print(f"Tasks: {args.tasks}")
    print(f"{'='*80}\n")
    
    # Build base command arguments
    base_cmd = [sys.executable]
    base_args = ["--model-name", model]
    
    if args.tasks:
        base_args.extend(["--tasks", args.tasks])
    if args.random_seed:
        base_args.append("--random-seed")
    if args.no_cache:
        base_args.append("--no-cache")
    
    # Determine which evaluations to run
    eval_types = args.eval_types if args.eval_types else ["standard", "lmul", "fp8", "lmul_fp8"]
    
    success = True
    
    # Run each evaluation type
    for eval_type in eval_types:
        print(f"\n{'='*60}")
        print(f"Running {eval_type.upper()} Evaluation...")
        print(f"{'='*60}\n")
        
        if eval_type == "standard":
            cmd = base_cmd + ["eval/evaluation.py"] + base_args
            if args.attn_impl:
                cmd.extend(["--attn-impl", args.attn_impl])
        elif eval_type == "lmul":
            cmd = base_cmd + ["eval/lmul_evaluation.py"] + base_args
        elif eval_type == "fp8":
            cmd = base_cmd + ["eval/fp8_evaluation.py"] + base_args
        elif eval_type == "lmul_fp8":
            cmd = base_cmd + ["eval/lmul_fp8_evaluation.py"] + base_args
        
        result = run_subprocess_with_interrupt(cmd, eval_type)
        if result.returncode != 0:
            success = handle_subprocess_error(eval_type, result, cmd)
        else:
            print(f"\n✅ {eval_type.upper()} evaluation completed!")
        
        # Brief pause between evaluations
        print("")  # Add blank line for readability
        time.sleep(2)
    
    # Run analysis if requested and all succeeded
    if not args.skip_analysis and success:
        print(f"\n{'='*60}")
        print("Running Results Analysis...")
        print(f"{'='*60}")
        
        cmd = [sys.executable, "eval/analyze_results.py"]
        result = run_subprocess_with_interrupt(cmd, "analysis")
        if result.returncode == 0:
            print("\n✅ Results analysis completed!")
        else:
            handle_subprocess_error("analysis", result, cmd)
    
    return success

def run_single_comprehensive_evaluation(model, seed_mode, cache_mode, tasks=None, eval_types=None):
    """
    Run evaluation for a single configuration in comprehensive mode.
    
    Args:
        model: Model name
        seed_mode: "fixed" or "random"
        cache_mode: "with_cache" or "no_cache"
        tasks: Specific tasks to run (optional)
        eval_types: List of evaluation types to run
    """
    # Create configuration identifier
    config_name = f"{model.split('/')[-1]}_{seed_mode}_{cache_mode}"
    
    print(f"\n{'='*80}")
    print(f"STARTING CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Seed: {seed_mode}")
    print(f"Cache: {cache_mode}")
    print(f"Evaluations: {', '.join(eval_types) if eval_types else 'All'}")
    print(f"{'='*80}\n")
    
    # Build base command - use this script in simple mode
    base_cmd = [sys.executable, "evaluation_suite.py", "--model-name", model]
    
    # Add seed configuration
    if seed_mode == "random":
        base_cmd.append("--random-seed")
    
    # Add cache configuration
    if cache_mode == "no_cache":
        base_cmd.append("--no-cache")
    
    # Add tasks if specified
    if tasks:
        base_cmd.extend(["--tasks", tasks])
    
    # Add evaluation types
    if eval_types:
        base_cmd.extend(["--eval-types"] + eval_types)
    
    # Skip analysis for individual runs
    base_cmd.append("--skip-analysis")
    
    # Pass through FP8 disable flag if set
    if os.environ.get("DISABLE_FP8", "0") == "1":
        base_cmd.append("--disable-fp8")
    
    # Run the evaluation
    start_time = time.time()
    result = run_subprocess_with_interrupt(base_cmd, config_name)
    elapsed_time = time.time() - start_time
    
    # Log result with detailed error info if failed
    success = result.returncode == 0
    if not success:
        extra_info = f"Model: {model}, Seed: {seed_mode}, Cache: {cache_mode}"
        handle_subprocess_error("comprehensive", result, base_cmd, extra_info)
    
    status = "✅ SUCCESS" if success else "❌ FAILED"
    
    print(f"\n{status} - {config_name} completed in {elapsed_time/60:.1f} minutes")
    print("=" * 80)  # Add separator for clarity
    
    return {
        "model": model,
        "seed_mode": seed_mode,
        "cache_mode": cache_mode,
        "eval_types": eval_types or ["all"],
        "success": success,
        "elapsed_minutes": elapsed_time / 60
    }

def save_evaluation_summary(results, output_file="evaluation_suite_summary.json"):
    """Save evaluation summary to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluations": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time_hours": sum(r["elapsed_minutes"] for r in results) / 60,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def print_comprehensive_summary(summary):
    """Print evaluation summary for comprehensive mode."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_time_hours']:.1f} hours")
    print(f"\nResults saved to: evaluation_suite_summary.json")
    print(f"{'='*80}")

def run_comprehensive_evaluation(args):
    """Run comprehensive evaluation mode with multiple models and configurations."""
    # Determine models to test
    if args.models:
        models = args.models
    elif args.quick:
        models = QUICK_MODELS
    else:
        models = COMPREHENSIVE_MODELS
    
    # Determine configurations
    seed_modes = []
    if not args.skip_fixed_seed:
        seed_modes.append("fixed")
    if not args.skip_random_seed:
        seed_modes.append("random")
    
    cache_modes = []
    if not args.skip_with_cache:
        cache_modes.append("with_cache")
    if not args.skip_no_cache:
        cache_modes.append("no_cache")
    
    # Determine evaluation types
    eval_types = args.eval_types if args.eval_types else None
    
    # Calculate total evaluations
    total_configs = len(models) * len(seed_modes) * len(cache_modes)
    if eval_types:
        total_evals = total_configs * len(eval_types)
    else:
        total_evals = total_configs * 4  # 4 evaluation types by default
    
    print(f"\n{'='*80}")
    print("L-MUL EVALUATION SUITE - Comprehensive Mode")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    print(f"Seed modes: {seed_modes}")
    print(f"Cache modes: {cache_modes}")
    print(f"Evaluation types: {eval_types if eval_types else 'All (standard, lmul, fp8, lmul_fp8)'}")
    print(f"Total configurations: {total_configs}")
    print(f"Total evaluations: {total_evals}")
    print(f"{'='*80}")
    
    # Confirm with user
    try:
        response = input("\nProceed with evaluation? (y/n): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return False
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user.")
        return False
    
    # Run evaluations
    results = []
    start_time = time.time()
    
    for model in models:
        for seed_mode in seed_modes:
            for cache_mode in cache_modes:
                result = run_single_comprehensive_evaluation(
                    model=model,
                    seed_mode=seed_mode,
                    cache_mode=cache_mode,
                    tasks=args.tasks,
                    eval_types=eval_types
                )
                results.append(result)
                
                # Brief pause between evaluations
                time.sleep(5)
    
    # Save and print summary
    summary = save_evaluation_summary(results)
    print_comprehensive_summary(summary)
    
    # Final message
    total_time = time.time() - start_time
    print(f"\n🎉 Comprehensive evaluation completed in {total_time/3600:.1f} hours!")
    
    # Run final analysis if all evaluations succeeded
    if summary['failed'] == 0 and not args.skip_analysis:
        print("\nRunning final comparative analysis...")
        cmd = [sys.executable, "eval/analyze_results.py"]
        result = run_subprocess_with_interrupt(cmd, "final_analysis")
        if result.returncode == 0:
            print("\n✅ Final analysis completed!")
        else:
            handle_subprocess_error("final_analysis", result, cmd)
    
    return summary['failed'] == 0

def main():
    parser = argparse.ArgumentParser(
        description="L-Mul Evaluation Suite - Complete benchmarking framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode (default) - run all optimizations on one model
  python evaluation_suite.py
  
  # Run specific optimizations only
  python evaluation_suite.py --only lmul
  python evaluation_suite.py --eval-types standard lmul
  
  # Comprehensive mode - full test matrix
  python evaluation_suite.py --comprehensive
  python evaluation_suite.py --comprehensive --quick
  
  # Scientific evaluation (no cache)
  python evaluation_suite.py --comprehensive --skip-with-cache
  
  # Custom model set
  python evaluation_suite.py --models "google/gemma-2-2b-it" "meta-llama/Llama-3.2-3B-Instruct"
  
  # Disable FP8 quantization (use FP16 only)
  python evaluation_suite.py --disable-fp8
  
  # Note: FP8 uses 8-bit E4M3 format
        """
    )
    
    # Mode selection
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive evaluation across multiple models and configurations")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b-it",
                       help="Model to evaluate (for simple mode)")
    parser.add_argument("--models", type=str, nargs="+",
                       help="Models to evaluate (overrides --model-name)")
    
    # Evaluation options
    parser.add_argument("--eval-types", type=str, nargs="+",
                       choices=["standard", "lmul", "fp8", "lmul_fp8"],
                       help="Specific evaluation types to run")
    parser.add_argument("--only", type=str,
                       choices=["standard", "lmul", "fp8", "lmul_fp8"],
                       help="Run only one evaluation type (shorthand for --eval-types)")
    
    # Task configuration
    parser.add_argument("--tasks", type=str,
                       help="Specific tasks to run (default: all)")
    
    # Seed and cache options
    parser.add_argument("--random-seed", action="store_true",
                       help="Use random seed instead of fixed seed (42)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable KV-caching for accurate energy measurements")
    
    # Comprehensive mode options
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer models (comprehensive mode only)")
    parser.add_argument("--skip-fixed-seed", action="store_true",
                       help="Skip fixed seed evaluations (comprehensive mode only)")
    parser.add_argument("--skip-random-seed", action="store_true",
                       help="Skip random seed evaluations (comprehensive mode only)")
    parser.add_argument("--skip-with-cache", action="store_true",
                       help="Skip evaluations with KV-cache (comprehensive mode only)")
    parser.add_argument("--skip-no-cache", action="store_true",
                       help="Skip evaluations without KV-cache (comprehensive mode only)")
    
    # Technical options
    parser.add_argument("--attn-impl", type=str,
                       help="Attention implementation for standard evaluation")
    
    # Other options
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip results analysis after evaluation")
    
    # Disable FP8 quantization option
    parser.add_argument("--disable-fp8", action="store_true",
                       help="Disable FP8 E4M3 quantization and use FP16 only")
    
    args = parser.parse_args()
    
    # Handle --only shorthand
    if args.only:
        args.eval_types = [args.only]
    
    # Handle FP8 disabling
    if args.disable_fp8:
        os.environ["DISABLE_FP8"] = "1"
        print("ℹ️  FP8 E4M3 quantization disabled. Using FP16 only.")
    
    # Override no_cache if skip options are used in simple mode
    if not args.comprehensive:
        if args.skip_with_cache and not args.skip_no_cache:
            args.no_cache = True
        elif args.skip_no_cache and not args.skip_with_cache:
            args.no_cache = False
    
    # Run appropriate mode
    if args.comprehensive:
        success = run_comprehensive_evaluation(args)
    else:
        success = run_simple_evaluation(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation suite interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT 
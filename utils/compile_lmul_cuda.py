#!/usr/bin/env python
"""
Script to compile the L-Mul CUDA kernel.
Run this once after cloning the repository to enable GPU acceleration.
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("L-Mul CUDA Kernel Compilation")
    print("=" * 60)
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available on this system.")
            print("The CUDA kernel will not be compiled.")
            return
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print("Please install PyTorch first: pip install torch")
        return
    
    # Navigate to cuda_kernels directory
    cuda_dir = os.path.join(os.path.dirname(__file__), '../cuda_kernels')
    
    if not os.path.exists(cuda_dir):
        print(f"ERROR: CUDA kernels directory not found at {cuda_dir}")
        return
    
    print(f"Compiling CUDA kernel in {cuda_dir}...")
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to cuda_kernels directory
        os.chdir(cuda_dir)
        
        # Run the setup script
        result = subprocess.run([sys.executable, 'setup.py', 'install'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✓ CUDA kernel compiled successfully!")
            print("\nThe L-Mul implementation will now use GPU acceleration.")
            print("You should see significant performance improvements.")
        else:
            print("\n✗ Compilation failed!")
            print("\nError output:")
            print(result.stderr)
            print("\nThe L-Mul implementation will fall back to PyTorch.")
            
    except Exception as e:
        print(f"\n✗ Compilation failed with exception: {e}")
        print("\nThe L-Mul implementation will fall back to PyTorch.")
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Simple verification script for CUDA kernel compilation and L-Mul functionality.
"""

import torch
import sys

print("=" * 60)
print("CUDA Kernel Verification")
print("=" * 60)

# Check PyTorch and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nTrying to import L-Mul modules...")

try:
    from utils.l_mul import l_mul as l_mul_pytorch
    print("✓ PyTorch L-Mul imported successfully")
except Exception as e:
    print(f"✗ Failed to import PyTorch L-Mul: {e}")
    sys.exit(1)

try:
    from utils.l_mul_optimized import l_mul_optimized, CUDA_AVAILABLE
    print(f"✓ Optimized L-Mul imported successfully")
    print(f"  CUDA kernel available: {CUDA_AVAILABLE}")
except Exception as e:
    print(f"✗ Failed to import optimized L-Mul: {e}")
    sys.exit(1)

# Quick functionality test
print("\nTesting basic L-Mul functionality...")
try:
    # Test with small tensors
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    
    # PyTorch implementation
    result_pytorch = l_mul_pytorch(x, y)
    print(f"✓ PyTorch L-Mul works: {result_pytorch}")
    
    # Standard multiplication for comparison
    standard = x * y
    print(f"  Standard multiplication: {standard}")
    print(f"  Relative error: {torch.abs(result_pytorch - standard) / standard}")
    
    if torch.cuda.is_available():
        print("\nTesting CUDA functionality...")
        x_cuda = x.cuda()
        y_cuda = y.cuda()
        
        # Test optimized version
        result_cuda = l_mul_optimized(x_cuda, y_cuda)

        if not CUDA_AVAILABLE:
             print(f"✓ PyTorch FALLBACK on CUDA tensors works: {result_cuda.cpu()}")
             print("  NOTE: This is NOT using the custom CUDA kernel.")
        else:
             print(f"✓ CUDA L-Mul KERNEL works: {result_cuda.cpu()}")
        
except Exception as e:
    print(f"✗ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60) 
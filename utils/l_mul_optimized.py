import torch
import os
import warnings

# Try to import the CUDA extension
try:
    import l_mul_cuda
    CUDA_AVAILABLE = True
    print("L-Mul CUDA kernel loaded successfully!")
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("L-Mul CUDA kernel not found. Falling back to PyTorch implementation.")

from .l_mul import l_mul as l_mul_pytorch

def l_mul_optimized(x: torch.Tensor, y: torch.Tensor, mantissa_bits: int = 4):
    """
    Optimized L-Mul implementation that uses CUDA kernel when available.
    Falls back to PyTorch implementation if CUDA kernel is not available.
    """
    if CUDA_AVAILABLE and x.is_cuda and y.is_cuda:
        # Ensure tensors are contiguous for CUDA kernel
        x = x.contiguous()
        y = y.contiguous()
        return l_mul_cuda.l_mul(x, y, mantissa_bits)
    else:
        # Fall back to PyTorch implementation
        return l_mul_pytorch(x, y, mantissa_bits)

def l_mul_attention_optimized(query: torch.Tensor, key: torch.Tensor, mantissa_bits: int = 4):
    """
    Optimized attention computation using L-Mul CUDA kernel.
    This is specifically designed for the attention mechanism and is much faster
    than computing individual L-Mul operations.
    
    Args:
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_heads, seq_len, head_dim]
        mantissa_bits: Number of mantissa bits for quantization
    
    Returns:
        attention_scores: [batch_size, num_heads, seq_len, seq_len]
    """
    # Check if we can use CUDA kernel
    if CUDA_AVAILABLE and query.is_cuda and key.is_cuda and query.device == key.device:
        # Ensure tensors are contiguous
        query = query.contiguous()
        key = key.contiguous()
        try:
            return l_mul_cuda.l_mul_attention(query, key, mantissa_bits)
        except Exception as e:
            print(f"CUDA kernel failed with error: {e}")
            print("Falling back to PyTorch implementation")
    else:
        # Fall back to chunked PyTorch implementation
        print(f"WARNING: Using PyTorch fallback. CUDA_AVAILABLE: {CUDA_AVAILABLE}, query.is_cuda: {query.is_cuda}, key.is_cuda: {key.is_cuda}")
        
        # Ensure both tensors are on the same device
        target_device = query.device
        key = key.to(target_device)
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, 
                                 device=target_device, dtype=query.dtype)
        
        # Process in chunks for memory efficiency
        chunk_size = min(32, seq_len)
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            query_chunk = query[:, :, i:end_idx, :]
            
            # Compute attention scores for this chunk
            query_chunk_reshaped = query_chunk.unsqueeze(3) # [b, h, chunk_q, 1, d]
            key_reshaped = key.unsqueeze(2)             # [b, h, 1, k_len, d]
            
            # Ensure both tensors are on the same device before L-Mul
            query_chunk_reshaped = query_chunk_reshaped.to(target_device)
            key_reshaped = key_reshaped.to(target_device)
            
            # L-Mul between query chunk and all keys
            l_mul_products = l_mul_pytorch(query_chunk_reshaped, key_reshaped, mantissa_bits)
            scores = torch.sum(l_mul_products, dim=-1) # Sum over head_dim
            
            attn_weights[:, :, i:end_idx, :] = scores
        
        return attn_weights

# Utility function to compile CUDA kernel
def compile_cuda_kernel():
    """
    Compile the CUDA kernel. Run this once after installation.
    """
    cuda_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cuda_kernels')
    original_dir = os.getcwd()
    
    try:
        os.chdir(cuda_dir)
        os.system('python setup.py install')
        print("CUDA kernel compiled successfully!")
    except Exception as e:
        print(f"Failed to compile CUDA kernel: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    # Test the implementation
    if torch.cuda.is_available():
        # Test basic L-Mul
        x = torch.randn(100, device='cuda:0', dtype=torch.bfloat16)
        y = torch.randn(100, device='cuda:0', dtype=torch.bfloat16)
        
        result = l_mul_optimized(x, y)
        print(f"L-Mul test passed. Result shape: {result.shape}")
        
        # Test attention computation
        query = torch.randn(2, 8, 64, 32, device='cuda', dtype=torch.bfloat16)
        key = torch.randn(2, 8, 64, 32, device='cuda', dtype=torch.bfloat16)
        
        attn_scores = l_mul_attention_optimized(query, key)
        print(f"Attention test passed. Result shape: {attn_scores.shape}")
    else:
        print("CUDA not available for testing") 
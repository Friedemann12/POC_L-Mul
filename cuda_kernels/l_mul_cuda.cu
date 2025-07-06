#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Check CUDA version for bfloat16 support (CUDA 11.0+)
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#define HAS_BFLOAT16 1
#else
#define HAS_BFLOAT16 0
// Define dummy type for older CUDA versions
typedef struct { unsigned short x; } __nv_bfloat16;
#endif

#include <vector>
#include <stdio.h>

// Helper functions for type conversion
template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float(__half val) {
    return __half2float(val);
}

#if HAS_BFLOAT16
template<>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}
#endif

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float(float val) {
    return __float2half(val);
}

#if HAS_BFLOAT16
template<>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) {
    return __float2bfloat16(val);
}
#endif

// CUDA kernel for L-Mul operation
// This implements the L-Mul algorithm from the paper: (1 + x_m + y_m + 2^(-l(m))) * 2^(x_e + y_e)
template <typename scalar_t>
__global__ void l_mul_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int size,
    const int mantissa_bits) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Read inputs and convert to float
        float x_val = to_float(x[idx]);
        float y_val = to_float(y[idx]);
        
        // Handle zero cases
        if (x_val == 0.0f || y_val == 0.0f) {
            out[idx] = from_float<scalar_t>(0.0f);
            return;
        }
        
        // Get signs and work with absolute values
        float sign = (x_val < 0) != (y_val < 0) ? -1.0f : 1.0f;
        x_val = fabsf(x_val);
        y_val = fabsf(y_val);
        
        // Decompose into mantissa and exponent
        int x_exp, y_exp;
        float x_mant = frexpf(x_val, &x_exp);
        float y_mant = frexpf(y_val, &y_exp);
        
        // Convert mantissa to [1, 2) range
        x_mant = x_mant * 2.0f;
        y_mant = y_mant * 2.0f;
        x_exp = x_exp - 1;
        y_exp = y_exp - 1;
        
        // Extract fractional part
        float x_frac = x_mant - 1.0f;
        float y_frac = y_mant - 1.0f;
        
        // Quantize to reduced precision
        float scale = powf(2.0f, mantissa_bits);
        x_frac = roundf(x_frac * scale) / scale;
        y_frac = roundf(y_frac * scale) / scale;
        
        // L-Mul approximation: instead of multiplication, use addition
        float mantissa_sum = x_frac + y_frac;
        
        // Add correction term based on mantissa bits
        float correction;
        if (mantissa_bits <= 3) {
            correction = powf(2.0f, -mantissa_bits);
        } else if (mantissa_bits == 4) {
            correction = powf(2.0f, -3);
        } else {
            correction = powf(2.0f, -4);
        }
        mantissa_sum += correction;
        
        // Compute result exponent
        int result_exp = x_exp + y_exp;
        
        // Handle carry
        if (mantissa_sum >= 1.0f) {
            result_exp += 1;
            mantissa_sum -= 1.0f;
        }
        
        // Reconstruct result
        float result = sign * (1.0f + mantissa_sum) * ldexpf(1.0f, result_exp);
        
        // Write output
        out[idx] = from_float<scalar_t>(result);
    }
}

// Optimized kernel for matrix multiplication using L-Mul
template <typename scalar_t>
__global__ void l_mul_matmul_cuda_kernel(
    const scalar_t* __restrict__ query,  // [batch_size, num_heads, seq_len, head_dim]
    const scalar_t* __restrict__ key,    // [batch_size, num_heads, seq_len, head_dim]
    scalar_t* __restrict__ output,       // [batch_size, num_heads, seq_len, seq_len]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int mantissa_bits) {
    
    // Calculate global thread position
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;
    const int key_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        query_idx >= seq_len || key_idx >= seq_len) {
        return;
    }
    
    // Calculate offsets
    const int query_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;
    const int key_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim;
    
    // Compute dot product using L-Mul
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float q_val = to_float(query[query_offset + d]);
        float k_val = to_float(key[key_offset + d]);
        
        // Apply L-Mul (inline for efficiency)
        if (q_val != 0.0f && k_val != 0.0f) {
            float sign = (q_val < 0) != (k_val < 0) ? -1.0f : 1.0f;
            q_val = fabsf(q_val);
            k_val = fabsf(k_val);
            
            int q_exp, k_exp;
            float q_mant = frexpf(q_val, &q_exp) * 2.0f;
            float k_mant = frexpf(k_val, &k_exp) * 2.0f;
            q_exp--; k_exp--;
            
            float scale = powf(2.0f, mantissa_bits);
            float q_frac = roundf((q_mant - 1.0f) * scale) / scale;
            float k_frac = roundf((k_mant - 1.0f) * scale) / scale;
            
            float mantissa_sum = q_frac + k_frac;
            float correction = mantissa_bits <= 3 ? powf(2.0f, -mantissa_bits) : 
                             (mantissa_bits == 4 ? powf(2.0f, -3) : powf(2.0f, -4));
            mantissa_sum += correction;
            
            int result_exp = q_exp + k_exp;
            if (mantissa_sum >= 1.0f) {
                result_exp++;
                mantissa_sum -= 1.0f;
            }
            
            sum += sign * (1.0f + mantissa_sum) * ldexpf(1.0f, result_exp);
        }
    }
    
    // Write result
    const int output_idx = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * seq_len + key_idx;
    output[output_idx] = from_float<scalar_t>(sum);
}

// C++ interface
torch::Tensor l_mul_cuda(torch::Tensor x, torch::Tensor y, int mantissa_bits) {
    const auto size = x.numel();
    auto output = torch::zeros_like(x);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Manual type dispatch for better compatibility
    if (x.scalar_type() == torch::ScalarType::Float) {
        l_mul_cuda_kernel<float><<<blocks, threads>>>(
            static_cast<float*>(x.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            size,
            mantissa_bits
        );
    } else if (x.scalar_type() == torch::ScalarType::Double) {
        l_mul_cuda_kernel<double><<<blocks, threads>>>(
            static_cast<double*>(x.data_ptr()),
            static_cast<double*>(y.data_ptr()),
            static_cast<double*>(output.data_ptr()),
            size,
            mantissa_bits
        );
    } else if (x.scalar_type() == torch::ScalarType::Half) {
        l_mul_cuda_kernel<__half><<<blocks, threads>>>(
            static_cast<__half*>(x.data_ptr()),
            static_cast<__half*>(y.data_ptr()),
            static_cast<__half*>(output.data_ptr()),
            size,
            mantissa_bits
        );
#if HAS_BFLOAT16
    } else if (x.scalar_type() == torch::ScalarType::BFloat16) {
        l_mul_cuda_kernel<__nv_bfloat16><<<blocks, threads>>>(
            static_cast<__nv_bfloat16*>(x.data_ptr()),
            static_cast<__nv_bfloat16*>(y.data_ptr()),
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            size,
            mantissa_bits
        );
#endif
    } else {
        AT_ERROR("l_mul_cuda not implemented for ", x.scalar_type());
    }
    
    return output;
}

// Optimized attention computation using L-Mul
torch::Tensor l_mul_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    int mantissa_bits) {
    
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_len = query.size(2);
    const auto head_dim = query.size(3);
    
    auto output = torch::zeros({batch_size, num_heads, seq_len, seq_len}, 
                              query.options());
    
    dim3 blocks(seq_len, num_heads, batch_size);
    dim3 threads(std::min(static_cast<int>(seq_len), 1024)); // Limit threads per block
    
    // Manual type dispatch for better compatibility
    if (query.scalar_type() == torch::ScalarType::Float) {
        l_mul_matmul_cuda_kernel<float><<<blocks, threads>>>(
            static_cast<float*>(query.data_ptr()),
            static_cast<float*>(key.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            mantissa_bits
        );
    } else if (query.scalar_type() == torch::ScalarType::Double) {
        l_mul_matmul_cuda_kernel<double><<<blocks, threads>>>(
            static_cast<double*>(query.data_ptr()),
            static_cast<double*>(key.data_ptr()),
            static_cast<double*>(output.data_ptr()),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            mantissa_bits
        );
    } else if (query.scalar_type() == torch::ScalarType::Half) {
        l_mul_matmul_cuda_kernel<__half><<<blocks, threads>>>(
            static_cast<__half*>(query.data_ptr()),
            static_cast<__half*>(key.data_ptr()),
            static_cast<__half*>(output.data_ptr()),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            mantissa_bits
        );
#if HAS_BFLOAT16
    } else if (query.scalar_type() == torch::ScalarType::BFloat16) {
        l_mul_matmul_cuda_kernel<__nv_bfloat16><<<blocks, threads>>>(
            static_cast<__nv_bfloat16*>(query.data_ptr()),
            static_cast<__nv_bfloat16*>(key.data_ptr()),
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            mantissa_bits
        );
#endif
    } else {
        AT_ERROR("l_mul_attention_cuda not implemented for ", query.scalar_type());
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l_mul", &l_mul_cuda, "L-Mul CUDA implementation");
    m.def("l_mul_attention", &l_mul_attention_cuda, "L-Mul Attention CUDA implementation");
} 
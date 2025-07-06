import torch

def l_mul(x: torch.Tensor, y: torch.Tensor, mantissa_bits: int = 4, exponent_bits: int = 8):
    """
    L-Mul implementation using addition-based approximation.
    Uses reduced precision mantissa (default 4-bit) for energy efficiency.
    
    Key L-Mul principles:
    1. Decompose numbers into sign, exponent, mantissa
    2. Quantize mantissa to reduced precision (4-bit instead of 23-bit)
    3. Use ADDITION instead of multiplication: (1+m1)*(1+m2) ≈ 1 + m1 + m2 + correction
    4. This saves energy by replacing expensive multiplication with addition
    """
    # Store original properties
    original_dtype = x.dtype
    original_device = x.device
    
    # Work with float32 for intermediate computations (for numerical stability)
    # but implement the actual L-Mul algorithm with reduced precision
    # IMPORTANT: Preserve the device when converting to float32
    x_work = x.to(dtype=torch.float32, device=original_device)
    y_work = y.to(dtype=torch.float32, device=original_device)
    
    # Handle zero values
    zero_mask = (x_work == 0) | (y_work == 0)
    
    # Get signs
    sign_x = torch.sign(x_work)
    sign_y = torch.sign(y_work)
    result_sign = sign_x * sign_y
    
    # Work with absolute values
    x_abs = torch.abs(x_work)
    y_abs = torch.abs(y_work)
    
    # Decompose into mantissa and exponent using frexp
    mantissa_x, exponent_x = torch.frexp(x_abs)
    mantissa_y, exponent_y = torch.frexp(y_abs)
    
    # Convert mantissa to [1, 2) range (standard IEEE format)
    mantissa_x = mantissa_x * 2.0
    mantissa_y = mantissa_y * 2.0
    exponent_x = exponent_x - 1
    exponent_y = exponent_y - 1
    
    # Extract fractional part of mantissa (the part we quantize)
    frac_x = mantissa_x - 1.0
    frac_y = mantissa_y - 1.0
    
    # CRITICAL: Quantize mantissa to reduced precision (this is the key L-Mul step)
    scale = 2.0 ** mantissa_bits
    frac_x_quantized = torch.round(frac_x * scale) / scale
    frac_y_quantized = torch.round(frac_y * scale) / scale
    
    # Calculate result exponent (this is exact)
    result_exponent = exponent_x + exponent_y
    
    # L-Mul mantissa approximation using ADDITION instead of multiplication
    # Instead of (1 + frac_x) * (1 + frac_y) = 1 + frac_x + frac_y + frac_x*frac_y
    # L-Mul approximates this as: 1 + frac_x + frac_y + correction_term
    mantissa_sum = frac_x_quantized + frac_y_quantized
    
    # Add correction term (from L-Mul paper)
    if mantissa_bits <= 3:
        correction = 2.0 ** (-mantissa_bits)
    elif mantissa_bits == 4:
        correction = 2.0 ** (-3)
    else:
        correction = 2.0 ** (-4)
    
    mantissa_sum += correction
    
    # Handle carry (if mantissa_sum >= 1.0)
    carry = (mantissa_sum >= 1.0).float()
    result_exponent = result_exponent + carry
    mantissa_sum = mantissa_sum - carry
    
    # Reconstruct result: sign * (1 + mantissa_frac) * 2^exponent
    result_mantissa = 1.0 + mantissa_sum
    result = result_sign * result_mantissa * torch.pow(2.0, result_exponent)
    
    # Handle zeros
    result = torch.where(zero_mask, torch.zeros_like(result), result)
    
    # Convert back to original dtype and device
    result = result.to(dtype=original_dtype, device=original_device)
    
    return result

# Example usage for testing:
if __name__ == '__main__':
    # Use float32 for testing
    x_cpu = torch.tensor([1.5, -3.75, 123.456, 0.1])
    y_cpu = torch.tensor([2.0, 5.0, -0.5, 0.2])

    # Standard multiplication
    z_std = x_cpu * y_cpu

    # L-Mul multiplication with 4-bit mantissa
    z_lmul_4bit = l_mul(x_cpu, y_cpu, mantissa_bits=4)

    print("Standard Multiplication Result:", z_std)
    print("L-Mul Approx (4-bit mantissa):", z_lmul_4bit)
    print("Approximation Error (4-bit):  ", torch.abs(z_std - z_lmul_4bit))

    # L-Mul multiplication with 8-bit mantissa
    z_lmul_8bit = l_mul(x_cpu, y_cpu, mantissa_bits=8)
    print("\nL-Mul Approx (8-bit mantissa):", z_lmul_8bit)
    print("Approximation Error (8-bit):  ", torch.abs(z_std - z_lmul_8bit))
    
    if torch.cuda.is_available():
        print("\n--- Running on GPU ---")
        device = torch.device("cuda")
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        z_std_gpu = x_gpu * y_gpu
        z_lmul_gpu_4bit = l_mul(x_gpu, y_gpu, mantissa_bits=4)
        
        print("Standard Multiplication Result (GPU):", z_std_gpu.cpu())
        print("L-Mul Approx (4-bit mantissa, GPU):", z_lmul_gpu_4bit.cpu())
        print("Approximation Error (4-bit, GPU):  ", torch.abs(z_std_gpu - z_lmul_gpu_4bit).cpu())

        z_lmul_gpu_8bit = l_mul(x_gpu, y_gpu, mantissa_bits=8)
        print("\nL-Mul Approx (8-bit mantissa, GPU):", z_lmul_gpu_8bit.cpu())
        print("Approximation Error (8-bit, GPU):  ", torch.abs(z_std_gpu - z_lmul_gpu_8bit).cpu()) 
# Proof-of-Concept: L-Mul Implementation for Energy-Efficient Language Models

Implementation of the **Linear-complexity Multiplication (L-Mul)** algorithm from "Addition is All You Need for Energy-efficient Language Models" (Luo & Sun, 2024). Includes PyTorch and CUDA kernel versions with FP8 quantization support.

## Features

- L-Mul algorithm with 4-bit mantissa approximation
- CUDA kernel optimization for 10-100x speedup
- FP8 quantization for memory efficiency
- Combined L-Mul + FP8 optimizations
- Automatic CUDA compilation with PyTorch fallback

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Complete Evaluation Pipeline
```bash
# Run all optimizations: Standard, L-Mul, FP8, L-Mul+FP8
python evaluation_suite.py
```

### Individual Evaluations
```bash
# Standard baseline
python eval/evaluation.py

# L-Mul optimization
python eval/lmul_evaluation.py

# FP8 quantization
python eval/fp8_evaluation.py

# Combined L-Mul + FP8
python eval/lmul_fp8_evaluation.py

# Results analysis
python eval/analyze_results.py
```

## Project Structure

```
├── evaluation_suite.py         # Main evaluation framework
├── eval/                       # Evaluation scripts
│   ├── evaluation.py           # Standard baseline
│   ├── lmul_evaluation.py      # L-Mul evaluation
│   ├── fp8_evaluation.py       # FP8 evaluation
│   └── lmul_fp8_evaluation.py  # Combined evaluation
├── utils/                      # Core utilities
│   ├── l_mul.py               # PyTorch L-Mul implementation
│   ├── l_mul_optimized.py     # CUDA-optimized wrapper
│   ├── fp8_utils.py           # FP8 quantization
│   └── evaluation_common.py   # Shared evaluation code
├── cuda_kernels/              # CUDA kernel source
│   ├── l_mul_cuda.cu          # Custom CUDA kernel
│   └── setup.py               # CUDA compilation
└── results/                    # Evaluation results
```

## Usage Examples

### Specific Model
```bash
python evaluation_suite.py --model-name "google/gemma-2-2b-it"
```

### Specific Optimizations
```bash
python evaluation_suite.py --eval-types fp8 lmul_fp8
```

### KV-Cache Control
```bash
# With KV-cache (default - faster)
python evaluation_suite.py

# Without KV-cache (shows L-Mul efficiency)
python evaluation_suite.py --no-cache
```

## Technical Details

**L-Mul Algorithm**: Approximates floating-point multiplication using addition with 4-bit mantissa quantization for energy-efficient attention computation.

**FP8 Quantization**: 8-bit floating point representation reducing memory usage by ~50% while maintaining accuracy.

**Combined Optimization**: L-Mul + FP8 provides maximum efficiency gains suitable for edge deployment.

## Evaluation Tasks

The evaluation suite tests 25+ tasks across 9 categories:
- Code Generation
- Instruction Following  
- Format Adherence
- Reasoning & Logic
- Attention Tests
- Hallucination Resistance
- Bias & Safety
- Output Quality
- Robustness

## Performance

| Optimization | Throughput | Energy Usage | Memory Usage |
|--------------|------------|--------------|--------------|
| Standard     | 1.0x       | 1.0x         | 1.0x         |
| L-Mul        | 0.9-1.1x   | 0.3-0.5x     | 1.0x         |
| FP8          | 1.2-1.5x   | 0.8x         | 0.5x         |
| L-Mul + FP8  | 1.1-1.3x   | 0.3-0.4x     | 0.5x         |
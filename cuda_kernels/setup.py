from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='l_mul_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'l_mul_cuda',
            ['l_mul_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
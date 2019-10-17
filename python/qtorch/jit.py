import os

from torch.utils.cpp_extension import load

directory = os.path.dirname(__file__)
# JIT compile module.
# TODO: move to standalone shared library building.

# directory = "/home/aleksander/education/phd/bachelor-diploma/lowbit-cnn/python/qtorch/"
import shutil
shutil.rmtree('/tmp/torch_extensions/qtorch/')
# "--use_fast_math"
cpp = load(
    name='qtorch', verbose=True,
    sources=[
        os.path.join(directory, 'cpp/module.cu'),
    ],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-DNDEBUG", "-gencode=arch=compute_75,code=sm_75", "-std=c++11",
                       "-I/usr/local/cuda/include", "-I"+os.path.join(directory, 'cpp')],
    extra_ldflags=['-lcudnn']
)

add_executable(check
        check.cu
        utils.cuh conv2DForward3x3TensorCores.cuh refConv2DForward.hpp
        # cudnn2DConvolution.cuh
        # conv2DForward3x3Fused.cuh conv2DForward3x3WinogradFused.cuh conv2DForward3x3.cuh
        # conv2DBackwardData3x3.cuh conv2DBackwardWeights3x3.cuh
)

target_compile_features(check PRIVATE cxx_std_11)
target_compile_options(check PRIVATE
        "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
        "-lineinfo" "--ptxas-options=-v" 
        "-Xcompiler" "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(check PRIVATE "${CUDA_LIBRARIES}" "${TORCH_LIBRARIES}" cudnn OpenMP::OpenMP_CXX)
target_include_directories(check PRIVATE "${CUDA_INCLUDE_DIRS}" cudnn.h)
set_target_properties(check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
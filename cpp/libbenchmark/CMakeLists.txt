find_package(CUDA REQUIRED)

add_library(libbenchmark SHARED benchmark.cuh benchmark.cu half.hpp)

target_compile_features(libbenchmark PRIVATE cxx_std_11)
target_compile_options(libbenchmark PRIVATE "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

target_link_libraries(libbenchmark PUBLIC "${CUDA_LIBRARIES}" libcudnn.so)
target_include_directories(libbenchmark 
    PUBLIC "${CUDA_INCLUDE_DIRS}"
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}" 
)

set_target_properties(libbenchmark PROPERTIES 
    CUDA_SEPERABLE_COMPILATION ON
)
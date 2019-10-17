include_directories("/usr/include/python3.6m/")
add_subdirectory(custom_convolution)
add_subdirectory(../int8conv)


add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PUBLIC libbenchmark Boost::boost)
target_compile_features(benchmark PRIVATE cxx_std_14)

# Runtime config example
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.json" "${CMAKE_CURRENT_BINARY_DIR}/config.json")

# # Float 32 benchmark
# # add_executable(float32 utils.h float32.cu)
# # target_compile_features(float32 PRIVATE cxx_std_11)
# # target_link_libraries(float32 "${CUDA_LIBRARIES}" libcudnn.so)
# # target_include_directories(float32 PRIVATE "${CUDA_INCLUDE_DIRS}" cudnn.h)
# # set_target_properties(float32 PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

# # Float 32 benchmark
# # add_executable(int8 utils.h int8.cu)
# # target_compile_features(int8 PRIVATE cxx_std_11)
# # target_link_libraries(int8 "${CUDA_LIBRARIES}" libcudnn.so)
# # target_include_directories(int8 PRIVATE "${CUDA_INCLUDE_DIRS}" cudnn.h)
# # set_target_properties(int8 PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

# find_package(CUDA 10.0 REQUIRED)
# add_executable(conv_int8x32 conv_int8x32.cu)
# target_compile_features(conv_int8x32 PRIVATE cxx_std_11)
# target_compile_options(conv_int8x32 PRIVATE "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

# target_link_libraries(conv_int8x32 PUBLIC "${CUDA_LIBRARIES}" libcudnn.so)

# set_target_properties(conv_int8x32 PROPERTIES 
#     CUDA_SEPERABLE_COMPILATION ON
# )


# add_executable(pooling_int8x32 pooling_int8x32.cu)
# target_compile_features(pooling_int8x32 PRIVATE cxx_std_11)
# target_compile_options(pooling_int8x32 PRIVATE "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

# target_link_libraries(pooling_int8x32 PUBLIC "${CUDA_LIBRARIES}" libcudnn.so)

# set_target_properties(pooling_int8x32 PROPERTIES 
#     CUDA_SEPERABLE_COMPILATION ON
# )

# add_executable(softmax_int8x4 softmax_int8x4.cu)
# target_compile_features(softmax_int8x4 PRIVATE cxx_std_11)
# target_compile_options(softmax_int8x4 PRIVATE "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

# target_link_libraries(softmax_int8x4 PUBLIC "${CUDA_LIBRARIES}" libcudnn.so)

# set_target_properties(softmax_int8x4 PROPERTIES 
#     CUDA_SEPERABLE_COMPILATION ON
# )




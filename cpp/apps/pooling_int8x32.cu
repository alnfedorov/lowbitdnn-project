#include <chrono>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cudnn.h>
#include <cuda_device_runtime_api.h>

void validate_call(const cudaError_t& err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error occurred: " << err << " " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

void validate_call(const cudnnStatus_t& err)
{
    if (err != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "cuDNN error occurred: " << err << " " << cudnnGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

void log(int verbose, std::ostream& ostream, std::string str)
{
    validate_call(cudaDeviceSynchronize());
    if (verbose)
        ostream << str << std::endl;
}

template <typename T>
__global__ void fill_with_constant(T *px, T k)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k;
}

int main()
{
    // Prerequisits
    int verbose = 2;
    
    // Data types and formats

    auto inputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
    auto inputDataType = CUDNN_DATA_INT8x32;

    auto outDataType = CUDNN_DATA_INT8x32;
    auto outputTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;

    // Tensors
    auto poolH = 4;
    auto poolW = 4;

    auto B = 1;
    auto C = 32;
    auto H = 4;
    auto W = 4;

    size_t padH, padW;
    padH = padW = 0;

    size_t strideH, strideW;
    strideH = strideW = 1;

    const int8_t CONSTANT_DATA = 127;


    validate_call(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    // Create cudnn
    cudnnHandle_t cudnn;
    validate_call(cudnnCreate(&cudnn));
    log(verbose, std::clog, "Cudnn created");
    
    // Create input tensor
    cudnnTensorDescriptor_t inputDescriptor;
    validate_call(cudnnCreateTensorDescriptor(&inputDescriptor));
    validate_call(cudnnSetTensor4dDescriptor(inputDescriptor, inputTensorFormat, inputDataType, B, C, H, W));

    int8_t *inputData;
    validate_call(cudaMalloc(&inputData, B * C * H * W * sizeof(int8_t)));
    log(verbose, std::clog, "Input tensor allocated");

    // Create pooling descriptor
    cudnnPoolingDescriptor_t poolingDescriptor;
    validate_call(cudnnCreatePoolingDescriptor(&poolingDescriptor));
    validate_call(cudnnSetPooling2dDescriptor(poolingDescriptor, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN, 
                                              poolH, poolW, padH, padW, strideH, strideW));

    int outB, outC, outH, outW;
    validate_call(cudnnGetPooling2dForwardOutputDim(poolingDescriptor, inputDescriptor, &outB, &outC, &outH, &outW));
    log(verbose, std::clog, "Computed convolution output shape");
    log(verbose, std::clog, std::to_string(outB)+"x"+std::to_string(outC)+"x"+std::to_string(outH)+"x"+std::to_string(outW));

    // Output tensor
    cudnnTensorDescriptor_t outDescriptor;
    validate_call(cudnnCreateTensorDescriptor(&outDescriptor));
    validate_call(cudnnSetTensor4dDescriptor(outDescriptor, outputTensorFormat, outDataType, outB, outC, outH, outW));

    int8_t *outData;
    validate_call(cudaMalloc(&outData, outB * outC * outH * outW * sizeof(int8_t)));
    log(verbose, std::clog, "Output tensor allocated");

    // Performing pooling
    float alpha = 1.0f;
    float beta = 0.0f;

    // Dummy values
    ::fill_with_constant<<<W * H, B * C>>>(inputData, (int8_t)CONSTANT_DATA);
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Filled with dummy values");

    validate_call(cudaDeviceSynchronize());
    auto begin = std::chrono::high_resolution_clock::now();

    validate_call(cudnnPoolingForward(
            cudnn, poolingDescriptor,
            &alpha, inputDescriptor, inputData, &beta, outDescriptor, outData));

    validate_call(cudaDeviceSynchronize());
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin);

    auto cpuOutData = new int8_t[outB * outC * outH * outW * sizeof(int8_t)];
    validate_call(cudaMemcpy(cpuOutData, outData, outB * outC * outH * outW * sizeof(int8_t), cudaMemcpyDeviceToHost));
    validate_call(cudaDeviceSynchronize());

    std::clog << "RESULT : " << std::endl;
    for (size_t b=0; b < outB; ++b)
    {   
	    for (size_t c=0; c < outC; ++c)
        {
            for (size_t h=0; h < outH; ++h)
                for (size_t w=0; w < outW; ++w)
                    std::clog << std::setw(4) << static_cast<int>(cpuOutData[b*(outH*outW*outC) + c*(outW*outH) + h*outW + w]) << " ";
        }
        std::clog << std::endl;
    }        
    std::clog << "END RESULT" << std::endl;
            
    delete[] cpuOutData;

    log(verbose, std::clog, "Finalizing");

    // Finalizing
        
    validate_call(cudaFree(outData));
    validate_call(cudnnDestroyTensorDescriptor(outDescriptor));
    log(verbose, std::clog, "Out tensor destroyed");
    
    validate_call(cudnnDestroyPoolingDescriptor(poolingDescriptor));
    log(verbose, std::clog, "Conv descriptor destroyed");

    validate_call(cudaFree(inputData));
    validate_call(cudnnDestroyTensorDescriptor(inputDescriptor));
    log(verbose, std::clog, "Input tensor destroyed");

    validate_call(cudnnDestroy(cudnn));
    log(verbose, std::clog, "Cudnn destroyed");
    log(verbose, std::clog, "Elapsed time is " + std::to_string(elapsed.count()));
    return EXIT_SUCCESS;
}
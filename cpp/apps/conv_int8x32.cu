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

    auto filterDataType = CUDNN_DATA_INT8x32;
    auto filterTensorFormat = CUDNN_TENSOR_NCHW_VECT_C;

    auto convAccumulatorDataType = CUDNN_DATA_INT32;

    auto outDataType = CUDNN_DATA_FLOAT;
    auto outputTensorFormat = CUDNN_TENSOR_NCHW;

    // Tensors
    auto numFilters = 32;
    auto filterH = 3;
    auto filterW = 3;

    auto B = 1;
    auto C = 32;
    auto H = 3;
    auto W = 3;

    size_t padH, padW;
    padH = padW = 0;

    size_t strideH, strideW, dilationH, dilationW;
    strideH = strideW = dilationH = dilationW = 1;

    const int8_t CONSTANT_DATA = 127;


    validate_call(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    // Create cudnn
    cudnnHandle_t cudnn;
    validate_call(cudnnCreate(&cudnn));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Cudnn created");
    
    // Create input tensor
    cudnnTensorDescriptor_t inputDescriptor;
    validate_call(cudnnCreateTensorDescriptor(&inputDescriptor));
    validate_call(cudnnSetTensor4dDescriptor(inputDescriptor, inputTensorFormat, inputDataType, B, C, H, W));

    int8_t *inputData;
    validate_call(cudaMalloc(&inputData, B * C * H * W * sizeof(int8_t)));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Input tensor allocated");

    // Create filter descriptor
    cudnnFilterDescriptor_t filterDescriptor;
    validate_call(cudnnCreateFilterDescriptor(&filterDescriptor));
    validate_call(cudnnSetFilter4dDescriptor(filterDescriptor, filterDataType, filterTensorFormat, numFilters, C, filterH, filterW));

    int8_t *filterData;
    validate_call(cudaMalloc(&filterData, numFilters * C * filterH * filterW * sizeof(int8_t)));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Filter tensor allocated");

    // Convolution descriptor
    cudnnConvolutionDescriptor_t convDescriptor;
    validate_call(cudnnCreateConvolutionDescriptor(&convDescriptor));
    validate_call(cudnnSetConvolution2dDescriptor(convDescriptor, padH, padW, strideH, strideW,
                                                  dilationH, dilationW, CUDNN_CONVOLUTION, convAccumulatorDataType));
    validate_call(cudnnSetConvolutionMathType(convDescriptor, CUDNN_TENSOR_OP_MATH));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Convolution descriptor created");
    int outB, outC, outH, outW;
    validate_call(cudnnGetConvolution2dForwardOutputDim(convDescriptor, inputDescriptor, filterDescriptor, &outB, &outC, &outH, &outW));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Computed convolution output shape");
    log(verbose, std::clog, std::to_string(outB)+"x"+std::to_string(outC)+"x"+std::to_string(outH)+"x"+std::to_string(outW));

    // Output tensor
    cudnnTensorDescriptor_t outDescriptor;
    validate_call(cudnnCreateTensorDescriptor(&outDescriptor));
    validate_call(cudnnSetTensor4dDescriptor(outDescriptor, outputTensorFormat, outDataType, outB, outC, outH, outW));

    int8_t *outData;
    validate_call(cudaMalloc(&outData, outB * outC * outH * outW * sizeof(float)));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Output tensor allocated");

    // Algorithm
    cudnnConvolutionFwdAlgoPerf_t convAlgo;
    int foundAlgo;
    validate_call(cudnnGetConvolutionForwardAlgorithm_v7(
            cudnn, inputDescriptor, filterDescriptor, convDescriptor, outDescriptor, 1, &foundAlgo, &convAlgo));
    if (foundAlgo == 0 || convAlgo.determinism == CUDNN_NON_DETERMINISTIC || convAlgo.status != CUDNN_STATUS_SUCCESS)
    {
        log(verbose, std::clog, "Best algorithm is non deterministic or not found. Terminating.");
        throw std::runtime_error("Failed to find cudnn algorithm for convolution.");
    }
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Best algorithm is chosen " + std::to_string(convAlgo.algo) + " with math " + std::to_string(convAlgo.mathType));

    if (convAlgo.mathType == CUDNN_TENSOR_OP_MATH)
        log(verbose, std::clog, "Using Tensor CORES!!!");

    // Workspace
    size_t workspaceSize = convAlgo.memory;
    void *workspaceData;
    if (workspaceSize != 0){}
        validate_call(cudaMalloc(&workspaceData, workspaceSize));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Workspace is allocated");

    // Performing convolution
    float alpha = 0.5f;
    float beta = 0.0f;

    // Dummy values
    ::fill_with_constant<<<numFilters*filterW * filterH, C>>>(filterData, (int8_t)CONSTANT_DATA);
    ::fill_with_constant<<<W * H, B * C>>>(inputData, (int8_t)CONSTANT_DATA);
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Filled with dummy values");

    validate_call(cudaDeviceSynchronize());
    auto begin = std::chrono::high_resolution_clock::now();

    validate_call(cudnnConvolutionForward(
            cudnn,
            &alpha, inputDescriptor, inputData, filterDescriptor, filterData,
            convDescriptor, convAlgo.algo, workspaceData, workspaceSize,
            &beta, outDescriptor, outData));

    validate_call(cudaDeviceSynchronize());
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin);

    auto cpuOutData = new float[outB * outC * outH * outW * sizeof(float)];
    validate_call(cudaMemcpy(cpuOutData, outData, outB * outC * outH * outW * sizeof(float), cudaMemcpyDeviceToHost));
    validate_call(cudaDeviceSynchronize());

    std::clog << "RESULT : " << std::endl;
    for (size_t b=0; b < outB; ++b)
    {   
	    for (size_t c=0; c < outC; ++c)
        {
            for (size_t h=0; h < outH; ++h)
                for (size_t w=0; w < outW; ++w)
                    // std::clog << std::setw(4) << static_cast<int>(cpuOutData[b*(outH*outW*outC) + c*(outW*outH) + h*outW + w]) << " ";
                    std::clog << std::setw(4) << cpuOutData[b*(outH*outW*outC) + c*(outW*outH) + h*outW + w] << " ";
        }
        std::clog << std::endl;
    }        
    std::clog << "END RESULT" << std::endl;
            
    delete[] cpuOutData;

    log(verbose, std::clog, "Finalizing");

    // Finalizing
    if (workspaceSize != 0)
        validate_call(cudaFree(workspaceData));
        
    validate_call(cudaFree(outData));
    validate_call(cudnnDestroyTensorDescriptor(outDescriptor));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Out tensor destroyed");
    
    validate_call(cudnnDestroyConvolutionDescriptor(convDescriptor));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Conv descriptor destroyed");
    
    validate_call(cudaFree(filterData));
    validate_call(cudnnDestroyFilterDescriptor(filterDescriptor));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Filter tensor destroyed");

    validate_call(cudaFree(inputData));
    validate_call(cudnnDestroyTensorDescriptor(inputDescriptor));
    log(verbose, std::clog, "Input tensor destroyed");

    validate_call(cudnnDestroy(cudnn));
    validate_call(cudaDeviceSynchronize());
    log(verbose, std::clog, "Cudnn destroyed");
    log(verbose, std::clog, "Elapsed time is " + std::to_string(elapsed.count()));
    return EXIT_SUCCESS;
}
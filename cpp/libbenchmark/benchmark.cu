#include "benchmark.cuh"

void validate_call(const cudaError_t& err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error occurred: " << err << " " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void validate_call(const cudnnStatus_t& err)
{
    if (err != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "cuDNN error occurred: " << err << " " << cudnnGetErrorString(err) << std::endl;
        throw std::runtime_error("cuDNN error");
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

bool first = true;
template <typename InputDataType, typename FilterDataType, typename OutDataType>
std::chrono::microseconds benchmark_convolution(size_t B, size_t C, size_t H, size_t W,
                                                size_t numFilters, size_t filterH, size_t filterW,
                                                size_t padH, size_t padW, size_t strideH, size_t strideW, size_t dilationH, size_t dilationW,
                                                cudnnTensorFormat_t inputTensorFormat, cudnnTensorFormat_t filterTensorFormat, cudnnTensorFormat_t outputTensorFormat, 
                                                cudnnDataType_t inputDataType, cudnnDataType_t filterDataType, 
                                                cudnnDataType_t convAccumulatorDataType, cudnnDataType_t outDataType,
                                                int verbose)
{
    if (first)
    {
        first = false;
        validate_call(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }

    cudnnHandle_t cudnn = nullptr;

    cudnnTensorDescriptor_t inputDescriptor = nullptr;
    InputDataType *inputData = nullptr;

    cudnnFilterDescriptor_t filterDescriptor = nullptr;
    FilterDataType *filterData = nullptr;
    cudnnConvolutionDescriptor_t convDescriptor = nullptr;

    cudnnTensorDescriptor_t outDescriptor = nullptr;
    OutDataType *outData = nullptr;

    cudnnConvolutionFwdAlgoPerf_t convAlgo;
    void *workspaceData = nullptr;

    std::chrono::microseconds elapsed;

    try
    {
        // Create cudnn
        validate_call(cudnnCreate(&cudnn));
        log(verbose, std::clog, "Cudnn created");

        // Create input tensor
        validate_call(cudnnCreateTensorDescriptor(&inputDescriptor));
        validate_call(cudnnSetTensor4dDescriptor(inputDescriptor, inputTensorFormat, inputDataType, B, C, H, W));

        validate_call(cudaMalloc(&inputData, B * C * H * W * sizeof(InputDataType)));
        log(verbose, std::clog, "Input tensor allocated");

        // Create filter descriptor
        validate_call(cudnnCreateFilterDescriptor(&filterDescriptor));
        validate_call(cudnnSetFilter4dDescriptor(filterDescriptor, filterDataType, filterTensorFormat, numFilters, C, filterH, filterW));

        validate_call(cudaMalloc(&filterData, numFilters * C * filterH * filterW * sizeof(FilterDataType)));
        log(verbose, std::clog, "Filter tensor allocated");

        // Convolution descriptor
        validate_call(cudnnCreateConvolutionDescriptor(&convDescriptor));
        validate_call(cudnnSetConvolution2dDescriptor(convDescriptor, padH, padW, strideH, strideW,
                                                    dilationH, dilationW, CUDNN_CONVOLUTION, convAccumulatorDataType));
        validate_call(cudnnSetConvolutionMathType(convDescriptor, CUDNN_TENSOR_OP_MATH));
        log(verbose, std::clog, "Convolution descriptor created");
        int outB, outC, outH, outW;
        validate_call(cudnnGetConvolution2dForwardOutputDim(convDescriptor, inputDescriptor, filterDescriptor, &outB, &outC, &outH, &outW));
        log(verbose, std::clog, "Computed convolution output shape");

        // Output tensor
        validate_call(cudnnCreateTensorDescriptor(&outDescriptor));
        validate_call(cudnnSetTensor4dDescriptor(outDescriptor, outputTensorFormat, outDataType, outB, outC, outH, outW));
        validate_call(cudaMalloc(&outData, outB * outC * outH * outW * sizeof(OutDataType)));
        log(verbose, std::clog, "Output tensor allocated");

        // Algorithm
        int foundAlgo;
        validate_call(cudnnFindConvolutionForwardAlgorithm(
                cudnn, inputDescriptor, filterDescriptor, convDescriptor, outDescriptor, 1, &foundAlgo, &convAlgo));
        if (foundAlgo == 0 || convAlgo.determinism == CUDNN_NON_DETERMINISTIC || convAlgo.status != CUDNN_STATUS_SUCCESS)
        {
            log(verbose, std::clog, "Best algorithm is non deterministic or not found. Terminating.");
            throw std::runtime_error("Failed to find cudnn algorithm for convolution.");
        }
        log(verbose, std::clog, "Best algorithm is chosen " + std::to_string(convAlgo.algo) + " with math " + std::to_string(convAlgo.mathType));

        if (convAlgo.mathType == CUDNN_TENSOR_OP_MATH)
            log(verbose, std::clog, "Using Tensor CORES!!!");

        // Workspace
        size_t workspaceSize = convAlgo.memory;
        if (workspaceSize != 0){}
            validate_call(cudaMalloc(&workspaceData, workspaceSize));
        log(verbose, std::clog, "Workspace is allocated");

        // Convolution
        float alpha = 1.0f;
        float beta = 0.0f;

        // Dummy values
        ::fill_with_constant<<<numFilters*filterW * filterH, C>>>(filterData, (FilterDataType)2);
        ::fill_with_constant<<<W * H, B * C>>>(inputData, (InputDataType)1);
        log(verbose, std::clog, "Filled with dummy values");

        validate_call(cudaDeviceSynchronize());
        auto begin = std::chrono::high_resolution_clock::now();

        validate_call(cudnnConvolutionForward(
                cudnn,
                &alpha, inputDescriptor, inputData, filterDescriptor, filterData,
                convDescriptor, convAlgo.algo, workspaceData, workspaceSize,
                &beta, outDescriptor, outData));

        validate_call(cudaDeviceSynchronize());
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin);
        log(verbose, std::clog, "Finalizing");
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error during convolution forward. Returned value is 0. Releasing resources..." << '\n';
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(0));
    }
    

    // Finalizing
    if (workspaceData != nullptr)
        validate_call(cudaFree(workspaceData));

    if (outData != nullptr)
        validate_call(cudaFree(outData));
    if (outDescriptor != nullptr)
        validate_call(cudnnDestroyTensorDescriptor(outDescriptor));
    log(verbose, std::clog, "Out tensor destroyed");
    
    if (convDescriptor != nullptr)
        validate_call(cudnnDestroyConvolutionDescriptor(convDescriptor));
    log(verbose, std::clog, "Conv descriptor destroyed");
    
    if (filterData != nullptr)
        validate_call(cudaFree(filterData));
    if (filterDescriptor != nullptr)
        validate_call(cudnnDestroyFilterDescriptor(filterDescriptor));
    log(verbose, std::clog, "Filter tensor destroyed");

    if (inputData != nullptr)
        validate_call(cudaFree(inputData));
    if (inputDescriptor != nullptr)
        validate_call(cudnnDestroyTensorDescriptor(inputDescriptor));
    log(verbose, std::clog, "Input tensor destroyed");

    if (cudnn != nullptr)
        validate_call(cudnnDestroy(cudnn));
    log(verbose, std::clog, "Cudnn destroyed");
    
    return elapsed;
};


// FLOAT CONFIG
template std::chrono::microseconds benchmark_convolution<float, float, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// TRUE_HALF CONFIG
template std::chrono::microseconds benchmark_convolution<half, half, half>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// DOUBLE CONFIG
template std::chrono::microseconds benchmark_convolution<double, double, double>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// INT8* CONFIG
template std::chrono::microseconds benchmark_convolution<int8_t, int8_t, int8_t>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// INT8*_EXT CONFIG
template std::chrono::microseconds benchmark_convolution<int8_t, int8_t, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// UINT8* CONFIG
template std::chrono::microseconds benchmark_convolution<uint8_t, int8_t, int8_t>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t,
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// UINT8*_EXT CONFIG
template std::chrono::microseconds benchmark_convolution<uint8_t, int8_t, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cooperative_groups.h>
#include <cfloat>

constexpr uint32_t APPROX_THREADS_PER_BLOCK = 64;
constexpr uint32_t VECT_C = 32;

using c10::IntList;
using at::Tensor;
using std::string;

__device__ __forceinline__ uint32_t DATA_INDEX(const uint32_t& C, const uint32_t& H, const uint32_t& W, 
                                               const uint32_t& tbatch, const uint32_t& tC, const uint32_t& tH, const uint32_t& tW)
{
  return tbatch*C*H*W*VECT_C + tC*H*W*VECT_C + tH*W*VECT_C + tW*VECT_C;
}

__device__ __forceinline__ uint32_t KERNEL_INDEX(const uint32_t& inC, const uint32_t& kH, const uint32_t& kW, 
                                                 const uint32_t& toutC, const uint32_t& tinC, const uint32_t& tkH, const uint32_t& tkW)
{
  return toutC*inC*kH*kW*VECT_C + tinC*kH*kW*VECT_C + tkH*kW*VECT_C + tkW*VECT_C;
}

// template <const uint32_t kernelTileH, const uint32_t kernelTileW, const uint32_t kernelTilesPerLine,
//           const uint32_t kernelTileStepH, const uint32_t kernelTileStepW, const uint32_t kernelTileStepsPerTile,

//           const uint32_t dataTileH, const uint32_t dataTileW, const uint32_t dataTilesPerLine,
//           const uint32_t dataTileStepH, const uint32_t dataTileStepW,

//           const uint32_t inCGroups, const uint32_t outCGroups,
//           const uint32_t inCPerGroup, const uint32_t outCPerGroup>
// __global__ void Conv2DForwardCudaV1(int8_t* idata, int8_t* kernel, int32_t* output, 
//                                          const uint32_t B,  const uint32_t inC, const uint32_t inH, const uint32_t inW,
//                                          const uint32_t outC, const uint32_t outH, const uint32_t outW,
//                                          const uint32_t kH, const uint32_t kW)
// {
//   // Must encode in the grid

//   // batch
//   // inC grid
//   // outC grid
//   // kernel tiles
//   // data tiles


//   // Must encode in the block
//   // kernel tile
//   // inC tile
//   // outC tile

//   // grid
//   //        x - dataTile                      |2^31 - 1| Everything must be nice here
//   //        y - kernelTile                    |2^16 - 1| Each block occupy at least 1 kernel spatial location, 
//   //                                                     so maximum kernel tiles is about 256*256
//   //        z - batch * outCGroup * inCGroup  |2^16 - 1| Batch might be up to the 1024 and will be simple splited into several calls

//   // block
//   //        x - outCGroup offset               |1024|
//   //        y - kernelTile offset              |1024|
//   //        z - inCGroup offset                |64|

//   // determine thread targets

//   // target batch
//   const uint32_t targetB = blockIdx.z / (outCGroups * inCGroups);
  
//   // channels
//   const uint32_t targetOutCGroup = (blockIdx.z - (targetB * outCGroups * inCGroups)) / inCGroups;
//   const uint32_t targetInCGroup = blockIdX.z - (targetB * outCGroups * inCGroups) - (targetOutCGroup * inCGroups);
  

//   uint32_t tmp = targetOutCGroup * outCPerGroup + threadIdx.x;  // target output channel, VECT_C excluded
//   const uint32_t targetOutC = tmp / 32;                         // index in output, without VECT_C
//   const uint32_t targetOutVectC = tmp - targetOutC;             // index in output, together with VECT_C


//   const uint32_t targetInC = targetInCGroup * inCPerGroup + threadIdx.z; // VECT_C included

//   // thread data elements
//   tmp = blockIdx.x / dataTilesPerLine;
//   const uint32_t dataYOffset = tmp * dataTileStepH;
//   const uint32_t dataXOffset = (blockIdx.x - tmp) * dataTileStepW;

//   // thread kernel elements
//   tmp = blockidx.y / kernelTilesPerLine; // line number
//   const uint32_t targetBlockKernelTileHOffset =  tmp * kernelTileH; // tile row begin
//   const uint32_t targetBlockKernelTileWOffset = (blockidx.y - tmp) * kernelTileW; // tile column begin

//   tmp = threadIdx.y / kernelTileStepH; //Tile Y Offset
//   const uint32_t kernelYOffset = threadIdx.y / kernelTileStepH;
//   const uint32_t kernelXOffset = threadIdx.x - kernelYOffset;



//   // Load kernel weights
//   int8_t threadWeights[kernelTileStepH][kernelTileStepW][VECT_C];
//   #pragma unroll
//   for (uint32_t y=0; y<kernelTileStepH; ++y)
//     #pragma unroll
//       for (uint32_t x=0; x<kernelTileStepW; ++x)
//         #pragma unroll
//           for (uint32_t c=0; c<VECT_C; ++c)
//             threadWeights[y][x][c] = kernel[KERNEL_INDEX(inC, kH, kW, targetOutC*targetOutVectC, targetInC, kernelYOffset+y, kernelXOffset+x)][c]


//   // const int8_t const* data = idata[DATA_INDEX(inC, inH, inW, targetB, targetInC, dataYOffset, dataXOffset)]
//   // const int32_t* out = output[DATA_INDEX(outC, outH, outW, targetB, targetOutC, dataYOffset, dataXOffset)]
//   #pragma unroll
//   for (uint32_t y=0; y<dataTileH; ++y)
//     #pragma unroll
//     for (uint32_t x=0; x<dataTileW; ++x)
//       // Data inside range. Should be processed nicely inside a warp due to the thread block tiling strategy
//       if ((dataYOffset + y + kernelTileStepH < inH) && (dataXOffset + x + kernelXOffset < inW))
//       {
//         int32_t cache = 0;
//         #pragma unroll
//         for (uint32_t ky=0; ky<kernelTileStepH; ++ky)
//           #pragma unroll
//           for (uint32_t kx=0; kx<kernelTileStepH; ++kx)
//             #pragma unroll
//             for (uint32_t c=0; c<VECT_C; ++c)
//               cache += int32_t(threadWeights[ky][kx][c]) * int32_t(idata[DATA_INDEX(inC, inH, inW, targetB, targetInC, dataYOffset+y+ky, dataXOffset+x+kx) + c])
        
//         // threads in warp should write to the adjacent blocks, so, everythin MUST be nice
//         atomicAdd(&output[DATA_INDEX(outC, outH, outW, targetB, targetOutC, dataYOffset+y, dataXOffset+x) + targetOutVectC], cache);
//       }
// }

// template <const uint32_t kernelTileH, const uint32_t kernelTileW, const uint32_t kernelTilesPerLine,
//           const uint32_t kernelTileStepH, const uint32_t kernelTileStepW, const uint32_t kernelTileStepsPerTile,

//           const uint32_t dataTileH, const uint32_t dataTileW, const uint32_t dataTilesPerLine,
//           const uint32_t dataTileStepH, const uint32_t dataTileStepW,

//           const uint32_t inCGroups, const uint32_t outCGroups,
//           const uint32_t inCPerGroup, const uint32_t outCPerGroup>
// __global__ void Conv2DForwardCudaV1(int8_t* idata, int8_t* kernel, int32_t* output, 
//                                          const uint32_t B,  const uint32_t inC, const uint32_t inH, const uint32_t inW,
//                                          const uint32_t outC, const uint32_t outH, const uint32_t outW,
//                                          const uint32_t kH, const uint32_t kW)
// {
//   // Must encode in the grid

//   // batch
//   // inC grid
//   // outC grid
//   // kernel tiles
//   // data tiles


//   // Must encode in the block
//   // kernel tile
//   // inC tile
//   // outC tile

//   // grid
//   //        x - dataTile                      |2^31 - 1| Everything must be nice here
//   //        y - kernelTile                    |2^16 - 1| Each block occupy at least 1 kernel spatial location, 
//   //                                                     so maximum kernel tiles is about 256*256
//   //        z - batch * outCGroup * inCGroup  |2^16 - 1| Batch might be up to the 1024 and will be simple splited into several calls

//   // block
//   //        x - outCGroup offset               |1024|
//   //        y - kernelTile offset              |1024|
//   //        z - inCGroup offset                |64|

//   // determine thread targets

//   // target batch
//   const uint32_t targetB = blockIdx.z / (outCGroups * inCGroups);
  
//   // channels
//   const uint32_t targetOutCGroup = (blockIdx.z - (targetB * outCGroups * inCGroups)) / inCGroups;
//   const uint32_t targetInCGroup = blockIdX.z - (targetB * outCGroups * inCGroups) - (targetOutCGroup * inCGroups);
  

//   uint32_t tmp = targetOutCGroup * outCPerGroup + threadIdx.x;  // target output channel, VECT_C excluded
//   const uint32_t targetOutC = tmp / 32;                         // index in output, without VECT_C
//   const uint32_t targetOutVectC = tmp - targetOutC;             // index in output, together with VECT_C


//   const uint32_t targetInC = targetInCGroup * inCPerGroup + threadIdx.z; // VECT_C included

//   // thread data elements
//   tmp = blockIdx.x / dataTilesPerLine;
//   const uint32_t dataYOffset = tmp * dataTileStepH;
//   const uint32_t dataXOffset = (blockIdx.x - tmp) * dataTileStepW;

//   // thread kernel elements
//   tmp = blockidx.y / kernelTilesPerLine; // line number
//   const uint32_t targetBlockKernelTileHOffset =  tmp * kernelTileH; // tile row begin
//   const uint32_t targetBlockKernelTileWOffset = (blockidx.y - tmp) * kernelTileW; // tile column begin

//   tmp = threadIdx.y / kernelTileStepH; //Tile Y Offset
//   const uint32_t kernelYOffset = threadIdx.y / kernelTileStepH;
//   const uint32_t kernelXOffset = threadIdx.x - kernelYOffset;



//   // Load kernel weights
//   int8_t threadWeights[kernelTileStepH][kernelTileStepW][VECT_C];
//   #pragma unroll
//   for (uint32_t y=0; y<kernelTileStepH; ++y)
//     #pragma unroll
//       for (uint32_t x=0; x<kernelTileStepW; ++x)
//         #pragma unroll
//           for (uint32_t c=0; c<VECT_C; ++c)
//             threadWeights[y][x][c] = kernel[KERNEL_INDEX(inC, kH, kW, targetOutC*targetOutVectC, targetInC, kernelYOffset+y, kernelXOffset+x)][c]


//   // const int8_t const* data = idata[DATA_INDEX(inC, inH, inW, targetB, targetInC, dataYOffset, dataXOffset)]
//   // const int32_t* out = output[DATA_INDEX(outC, outH, outW, targetB, targetOutC, dataYOffset, dataXOffset)]
//   #pragma unroll
//   for (uint32_t y=0; y<dataTileH; ++y)
//     #pragma unroll
//     for (uint32_t x=0; x<dataTileW; ++x)
//       // Data inside range. Should be processed nicely inside a warp due to the thread block tiling strategy
//       if ((dataYOffset + y + kernelTileStepH < inH) && (dataXOffset + x + kernelXOffset < inW))
//       {
//         int32_t cache = 0;
//         #pragma unroll
//         for (uint32_t ky=0; ky<kernelTileStepH; ++ky)
//           #pragma unroll
//           for (uint32_t kx=0; kx<kernelTileStepH; ++kx)
//             #pragma unroll
//             for (uint32_t c=0; c<VECT_C; ++c)
//               cache += int32_t(threadWeights[ky][kx][c]) * int32_t(idata[DATA_INDEX(inC, inH, inW, targetB, targetInC, dataYOffset+y+ky, dataXOffset+x+kx) + c])
        
//         // threads in warp should write to the adjacent blocks, so, everythin MUST be nice
//         atomicAdd(&output[DATA_INDEX(outC, outH, outW, targetB, targetOutC, dataYOffset+y, dataXOffset+x) + targetOutVectC], cache);
//       }
// }

template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void Conv2DForward3x3CudaV1(int8_t* idata, int8_t* kernel, int32_t* output)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - inC * outC                    |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  

  // batch  
  const uint32_t& tBatch = blockIdx.y;
  const uint32_t& tOutCVectCOffset = threadIdx.y;          

  // channels
  const uint32_t tInC = blockIdx.z / outC;          // VECT_C included
  const uint32_t tOutC = blockIdx.z - tInC * outC;  // VECT_C included

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  tmp = threadIdx.x % 2; // 0 or 1 to decide where to look for the kernel
                         // first 16 bytes or second 16 bytes. Each 16 bytes are loaded twice by the same warp

  // if (threadIdx.y == 0)
  // {
  // printf("GRID DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // printf("BLOCK DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // }
  // printf("COMPUTED %d %d %d %d %d %d\n", tBatch, tOutCVectCOffset, tInC, tOutC, outputYOffset, outputXOffset);

  // std::cout << tBatch << " " << tOutCVectCOffset << " " << tInC << " " << tOutC << " " << outputYOffset << " " << outputXOffset << std::endl;

  // Load kernel weights
  int8_t threadWeights[kH][kW][VECT_C / 2];
  #pragma unroll
  for (uint32_t y = 0; y < kH; ++y)
    #pragma unroll
      for (uint32_t x = 0; x < kW; ++x)
        #pragma unroll
          for (uint32_t c = 0; c < VECT_C / 2; ++c)
            threadWeights[y][x][c] = kernel[KERNEL_INDEX(inC, kH, kW, tOutC*VECT_C+tOutCVectCOffset, tInC, y, x) + 16*tmp + c];

  // convolve
  const uint32_t columnShift = threadIdx.x / resultTileW; // 0 or 1, for 1 or 2 column to process

  #pragma unroll
  for (uint32_t y = 0; y < resultTileH; ++y)
    #pragma unroll
    for (uint32_t x = columnShift; x < columnShift + (resultTileW / 2); ++x)
    {
      int32_t cache = 0;
      #pragma unroll
        for (uint32_t ky = 0; ky < kH; ++ky)
          #pragma unroll
          for (uint32_t kx = 0; kx < kW; ++kx)
            #pragma unroll
            for (uint32_t c = 0; c < VECT_C / 2; ++c)
              cache += int32_t(threadWeights[ky][kx][c]) * 
                       int32_t(idata[DATA_INDEX(inC, inH, inW, tBatch, tInC, outputYOffset+y+ky, outputXOffset+x+kx) + 16*tmp + c]);

      atomicAdd(&output[DATA_INDEX(outC, outH, outW, tBatch, tOutC, outputYOffset+y, outputXOffset+x) + tOutCVectCOffset], cache);
    }
}

std::tuple<Tensor, float> Conv2DCustomForwardV1(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / 32, 254, 254, 32}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(127*127, 1, 1*1); // output tiles, batch, inCGroups*outCGroups
  dim3 block_size(4, 32, 1); // 4 threads per filter, 32 filters

  THCudaCheck(cudaGetLastError());

  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  Conv2DForward3x3CudaV1<1, 1, 3, 3, 256, 256, 254, 254, 2, 2><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, elapsedTime);
}	
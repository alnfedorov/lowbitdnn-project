#pragma once

#include "utils.cuh"
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_61_intrinsics.h>
#include <vector_types.h>


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW, // == 2x2
  const uint32_t padH, const uint32_t padW
>
__global__ void 
__launch_bounds__(128)
CUDAConv2DBackwardData3x3(const int8_t* __restrict__ ridata, 
                       const int8_t*  __restrict__ rkernel, 
                       int32_t* __restrict__ routput)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - inC * outC                    |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  

  // batch
  const uint32_t& tOutCVectCOffset = threadIdx.y;         

  // channels
  const uint32_t tInC = blockIdx.z / outC;          // VECT_C included
  const uint32_t tOutC = blockIdx.z - tInC * outC;  // VECT_C included

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  tmp = threadIdx.x; // 0, 1, 2, 3 to decide where to look for the kernel
                     // Each 8 bytes are loaded only once
  const int8_t (*idata)[inH][inW][VECT_C] = 
                (const int8_t (*)[inH][inW][VECT_C])(&(ridata[blockIdx.y*inC*inH*inW*VECT_C + tInC*inH*inW*VECT_C]));
  const int8_t (*kernel)[kH][kW][VECT_C] = 
                (const int8_t (*)[kH][kW][VECT_C])(&(rkernel[(tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C]));
  int32_t (*output)[outH][outW][VECT_C] = 
              (int32_t (*)[outH][outW][VECT_C])(&(routput[blockIdx.y * outC * outH * outW * VECT_C + tOutC * outH * outW * VECT_C]));
  
  // Load kernel weights
  int2 threadWeights[kH][kW];
  #pragma unroll
  for (uint32_t y = kH; y > 0; --y)
    #pragma unroll
      for (uint32_t x = kW; x > 0; --x)
            threadWeights[kH-y][kW-x] = *reinterpret_cast<const int2*>(&((*kernel)[y-1][x-1][8*tmp]));

  int2 data;
  #pragma unroll
  for (uint32_t y = 0; y < resultTileH; ++y)
    #pragma unroll
    for (uint32_t x = 0; x < resultTileW; ++x)
    {
      int32_t result = 0;
      #pragma unroll
        for (uint32_t ky = 0; ky < kH; ++ky)
          #pragma unroll
          for (uint32_t kx = 0; kx < kW; ++kx)
            if ((outputYOffset+y+ky) >= padH   && (outputXOffset+x+kx) >= padW && 
                (outputYOffset+(y-padH)+ky) < inH && (outputXOffset+(x-padW)+kx) < inW)
            {
              data = *reinterpret_cast<const int2*>(&((*idata)[outputYOffset+y+ky-padH][outputXOffset+x+kx-padW][8*tmp]));
              result = __dp4a(data.x, threadWeights[ky][kx].x, result);
              result = __dp4a(data.y, threadWeights[ky][kx].y, result);
            }
              
      atomicAdd(&((*output)[outputYOffset+y][outputXOffset+x][tOutCVectCOffset]), result);
    }
}


template<
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW, 
  const uint32_t padH, const uint32_t padW
>
std::tuple<Tensor, float> conv2DBackwardData3x3(const Tensor& rograds, const Tensor& rkernel)
{
  constexpr uint32_t resultTileH = 8;
  constexpr uint32_t resultTileW = 8;
  constexpr uint32_t kH = 3;
  constexpr uint32_t kW = 3;  
  auto ograds = rograds.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({ograds.size(0), outC, outH, outW, VECT_C},
                    at::TensorOptions().dtype(torch::kInt32)
                                       .device(ograds.device())
                                       .is_variable(false));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0);
  dim3 grid(outH / resultTileH * outW / resultTileW, ograds.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // if (true)
  // {
  //   std::cout << ograds.sizes() << std::endl;
  //   std::cout << kernel.sizes() << std::endl;
  //   std::cout << output.sizes() << std::endl;
  //   std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  //   std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;
  // }

  constexpr uint32_t bPadH = kH-1-padH;
  constexpr uint32_t bPadW = kW-1-padW;
  cudaEventRecord(start);
  CUDAConv2DBackwardData3x3<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW, bPadH, bPadW><<<grid, block_size, 0, stream>>>(
    (int8_t*)ograds.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  // cudaDeviceSetSharedMemConfig(conf);
  return {output, elapsedTime};
}	

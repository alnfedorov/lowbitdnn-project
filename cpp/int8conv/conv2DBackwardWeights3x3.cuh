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
  const uint32_t padH, const uint32_t padW
>
__global__ void 
__launch_bounds__(128)
CUDAConv2DBackwardWeights3x3(const int8_t* __restrict__ ridata, 
                             const int8_t*  __restrict__ rograd, 
                             int32_t* __restrict__ rwgrad)
{
  // grid
  //        x - batch                    |2^31 - 1| Everything must be nice here
  //        y - inC                      |2^16 - 1| Simple and straight forward
  //        z - outC                     |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads                                    |1024| 
  //        y - channel offset for 32 output channels group  |1024|  

  // batch  
  const uint32_t& tBatch = blockIdx.x;
  const uint32_t& tInC = blockIdx.y;                  // VECT_C included
  const uint32_t& tOutC = blockIdx.z;                 // VECT_C included
  const uint32_t& tOutCVectCOffset = threadIdx.y;          
  const uint32_t tInCVectCOffset = threadIdx.x * 8;

  const int8_t (*idata)[inH][inW][VECT_C] = 
                (const int8_t (*)[inH][inW][VECT_C])(&(ridata[tBatch*inC*inH*inW*VECT_C + tInC*inH*inW*VECT_C]));
  const int8_t (*ograd)[outH][outW][VECT_C] = 
                (const int8_t (*)[outH][outW][VECT_C])(&(rograd[tBatch*outC*outH*outW*VECT_C + tOutC*outH*outW*VECT_C]));
  int32_t (*wgrad)[kH][kW][VECT_C] = 
              (int32_t (*)[kH][kW][VECT_C])(&(rwgrad[(tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C]));
  
  #pragma unroll
  for (uint32_t ky=0; ky < kH; ++ky)
    #pragma unroll
    for (uint32_t kx=0; kx < kW; ++kx)
      #pragma unroll
      for (uint32_t chn = tInCVectCOffset; chn < (tInCVectCOffset + 8); ++chn)
      {
        int32_t wgrads = 0;
        #pragma unroll
        for (uint32_t y=0; y < outH; ++y)
          #pragma unroll
          for (uint32_t x=0; x < outW; x+=4)
            if ((y+ky) >= padH && ((y-padH)+ky) < inH)
              {
                char4 og, id;
                og.x = 0;
                og.y = 0;
                og.z = 0;
                og.w = 0;

                id.x = 0;
                id.y = 0;
                id.z = 0;
                id.w = 0;
                
                if ((x+kx+0) >= padW && ((x-padW)+kx+0) < inW)
                {
                  og.x = (*ograd)[y][x+0][tOutCVectCOffset];
                  id.x = (*idata)[ky+y-padH][kx+x+0-padW][chn];
                }
                if ((x+kx+1) >= padW && ((x-padW)+kx+1) < inW)
                {
                  og.y = (*ograd)[y][x+1][tOutCVectCOffset];
                  id.y = (*idata)[ky+y-padH][kx+x+1-padW][chn];
                }
                if ((x+kx+2) >= padW && ((x-padW)+kx+2) < inW)
                {
                  og.z = (*ograd)[y][x+2][tOutCVectCOffset];
                  id.z = (*idata)[ky+y-padH][kx+x+2-padW][chn];
                }
                if ((x+kx+3) >= padW && ((x-padW)+kx+3) < inW)
                {
                  og.w = (*ograd)[y][x+3][tOutCVectCOffset];
                  id.w = (*idata)[ky+y-padH][kx+x+3-padW][chn];
                }
                wgrads = __dp4a(og, id, wgrads);
              }

        atomicAdd(&((*wgrad)[ky][kx][chn]), wgrads);
      }
}


template<
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW, 
  const uint32_t padH, const uint32_t padW
>
std::tuple<Tensor, float> conv2DBackwardWeights3x3(const Tensor& rinput, const Tensor& rograds)
{
  constexpr uint32_t kH = 3;
  constexpr uint32_t kW = 3;  
  
  auto input = rinput.contiguous();
  auto ograds = rograds.contiguous();

  
  auto output = at::zeros({outC*VECT_C, inC, kH, kW, VECT_C},
                    at::TensorOptions().dtype(torch::kInt32)
                                       .device(input.device())
                                       .is_variable(false));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  dim3 grid(input.size(0), outC, inC); // batch, outC, inC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (false)
  {
    std::cout << input.sizes() << std::endl;
    std::cout << ograds.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
    std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;
  }
  cudaEventRecord(start);
  CUDAConv2DBackwardWeights3x3<inC, outC, kH, kW, inH, inW, outH, outW, padH, padW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)ograds.data_ptr(), (int32_t*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  return {output, elapsedTime};
}

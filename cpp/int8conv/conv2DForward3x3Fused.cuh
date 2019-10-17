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


inline std::tuple<Tensor, Tensor> quantize_uint8(Tensor tensor)
{
  AT_ASSERT(tensor.size(1) % 4 == 0);
  tensor = tensor.reshape({tensor.size(0), tensor.size(1) / 4, -1});
  Tensor minv = std::get<0>(tensor.min(-1, false));
  Tensor maxv = std::get<0>(tensor.max(-1, false));

  constexpr float qmin = 0.f;
  constexpr float qmax = 255.f;

  // Tensor diff = maxv.sub_(minv).add_(1e-6f);
  Tensor diff = maxv - minv + 1e-6;
  auto invscale = (qmax - qmin) / diff;
  // auto qzero = minv.mul_(invscale).neg_().add_(qmin).round_().clamp_(qmin, qmax);
  auto qzero = (qmin - minv * invscale).round_().clamp_(qmin, qmax);
  return {qzero, invscale};
}


// clamp x to range [minimum, maximum]
__device__ __forceinline__ float clamp(const float& x, const float& minimum, const float& maximum)
{
  return fmaxf(minimum, fminf(maximum, x));
}


__device__ __forceinline__ float quantize(const float& value, const float& inv_scale, const float& zero_point)
{
  return roundf(clamp((value * inv_scale) + zero_point, 0.f, 255.f));
}

__device__ __forceinline__ uint8_t quantize(const float& value, const float& inv_scale)
{
  return uint8_t(max(0, min(__float2int_rn(value * inv_scale), 255)));
}


__device__ __forceinline__ uchar4 quantize(const float4& value, const float& inv_scale)
{
  uchar4 res;
  res.x = quantize(value.x, inv_scale);
  res.y = quantize(value.y, inv_scale);
  res.z = quantize(value.z, inv_scale);
  res.w = quantize(value.w, inv_scale);
  return res;
}


template<
  const uint32_t kH, const uint32_t kW,
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW
>
__global__ void 
__launch_bounds__(128)
CUDAConv2DForward3x3Fused(const float* __restrict__ ridata, 
                          const float* __restrict__ ridataInvScale,   // Batch x inC / 4
                          const float* __restrict__ ridataZeroPoint,  // Batch x inC / 4
                          const float* __restrict__ rkernel, 
                          const float* __restrict__ rkernelInvScale,  // outC x inC / 4
                          const float* __restrict__ rkernelZeroPoint, // outC x inC / 4
                          float* __restrict__ routput)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - outC                          |2^16 - 1| Output channels group. Each group include 16 channels

  // block
  //        x - 8 threads to convolve 8 input lines                  |1024| 
  //        y - channel offset for 16 output channels group          |1024|  

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;
  const uint32_t tOutC = blockIdx.z * uint32_t(16) + threadIdx.y;
  const uint32_t& tBatch = blockIdx.y;

  float (*output)[outH][outW] = (float (*)[outH][outW])(&(routput[tBatch * outC * outH * outW + tOutC * outH * outW]));
  const float (*idata)[inC][inH][inW] = (const float (*)[inC][inH][inW])(&(ridata[tBatch*inC*inH*inW]));
  const float (*idataInvScale)[inC / 4] = (const float (*)[inC /4])(&(ridataInvScale[tBatch * inC / 4]));
  const float (*idataZeroPoint)[inC / 4] = (const float (*)[inC /4])(&(ridataZeroPoint[tBatch * inC / 4]));

  const float (*kernel)[inC][kH][kW] = (const float (*)[inC][kH][kW])(&(rkernel[tOutC*inC*kH*kW]));
  const float (*kernelInvScale)[inC / 4] = (const float (*)[inC /4])(&(rkernelInvScale[tOutC * inC / 4]));
  const float (*kernelZeroPoint)[inC / 4] = (const float (*)[inC /4])(&(rkernelZeroPoint[tOutC * inC / 4]));
  
  __shared__ uchar4 datai[resultTileH + 2][resultTileW + 2];
  float result[resultTileW];
  #pragma unroll
  for (tmp=0; tmp < resultTileW; ++tmp)
    result[tmp] = 0;

  // nvcc bug, unrolling cycle leads to the wrong results
  // #pragma unroll
  for (uint32_t tInC = 0; tInC < inC; tInC += 4)   
  {
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    //   printf("INC %d\n", tInC);

    const float invKernel = (*kernelInvScale)[tInC / 4];
    const float zeroKernel = (*kernelZeroPoint)[tInC / 4];

    const float invData = (*idataInvScale)[tInC / 4];
    const float zeroData = (*idataZeroPoint)[tInC / 4];

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    //   printf("invK %f zK %f  invD %f zD %f \n", invKernel, zeroKernel, invData, zeroData);

    // populate data to shared memory. 
    uint32_t ind = threadIdx.x + threadIdx.y * 8;
    uint32_t y = ind / 10;
    uint32_t x = ind - y * 10;

    // load data
    if (ind < 100)
    {
      uchar4 cv;
      float v;
      v = (*idata)[tInC + 0][outputYOffset + y][outputXOffset + x];
      cv.x = quantize(v, invData);

      v = (*idata)[tInC + 1][outputYOffset + y][outputXOffset + x];
      cv.y = quantize(v, invData);
      
      v = (*idata)[tInC + 2][outputYOffset + y][outputXOffset + x];
      cv.z = quantize(v, invData);

      v = (*idata)[tInC + 3][outputYOffset + y][outputXOffset + x];
      cv.w = quantize(v, invData);

      datai[y][x] = cv;
    }

    __syncthreads();

    uchar4 weights[kH][kW];

    // LOAD KERNEL
    ////////////////////////////////////////////////////////////////////////////
    {
      const float (*fkernel)[4 * kH * kW] = (const float (*)[4 * kH * kW])(&((*kernel)[tInC]));
      uchar4 w4[9];
      
      #pragma unroll
      for (tmp=0; tmp < 9; ++tmp)
      {
        w4[tmp] = quantize(*reinterpret_cast<const float4*>(&((*fkernel)[tmp*4])), invKernel);
        // w4[tmp] = quantize(w4[tmp], invKernel, zeroKernel);
      }
        
      uint8_t (*w)[4][3][3] = (uint8_t (*)[4][3][3])(&w4);
      #pragma unroll
      for (y = 0; y < kH; ++y)
        #pragma unroll
        for (x = 0; x < kW; ++x)
        {
          weights[y][x].x = (*w)[0][y][x];
          weights[y][x].y = (*w)[1][y][x];
          weights[y][x].z = (*w)[2][y][x];
          weights[y][x].w = (*w)[3][y][x];
        }
    }
    // kernel loading is finished
    ////////////////////////////////////////////////////////////////////////////
    
    // SIMD for uchar4 summation
    // uint32_t ones = 1;

    // // Weights addition
    // uint32_t iwcache = 0;
    // #pragma unroll
    // for (uint32_t kx=0; kx < kW; ++kx)
    //   #pragma unroll
    //   for (uint32_t ky=0; ky < kH; ++ky)
    //     // iwcache += __vsadu4(*reinterpret_cast<uint32_t*>(&weights[ky][kx]), zeros); // Computes per-byte sum of abs difference of unsigned. 
    //     iwcache = __dp4a(*reinterpret_cast<uint32_t*>(&weights[ky][kx]), ones, iwcache);

    // float wcache = -zeroData * iwcache;

    #pragma unroll
    for (x = 0; x < resultTileW; ++x)
    {
      // uint32_t idcache = 0;
      uint32_t icache = 0;
      #pragma unroll
      for (uint32_t kx=0; kx < kW; ++kx)
        #pragma unroll
        for (uint32_t ky=0; ky < kH; ++ky)
          {
            icache = __dp4a(weights[ky][kx], datai[threadIdx.x + ky][x + kx], icache);
            // idcache += __vsadu4(*reinterpret_cast<uint32_t*>(&datai[threadIdx.x + ky][x + kx]), zeros);
            // idcache = __dp4a(*reinterpret_cast<uint32_t*>(&datai[threadIdx.x + ky][x + kx]), ones, idcache);
          }
      // float dcache = -zeroKernel * idcache;
      // printf("scale data %f scale kernel %f sum weights %f sum data %f zeros %f conv %d\n", 
      //        1.f/invData, 1.f/invKernel, wcache, dcache, kH*kW*zeroKernel*zeroData, icache);
      // result[x] += 1.f/invData * 1.f/invKernel * (wcache + dcache + kH*kW*zeroKernel*zeroData + icache);
      result[x] += 1.f/invData * 1.f/invKernel * icache;
    }
  }

  #pragma unroll
  for (tmp = 0; tmp < resultTileW; ++tmp)
    (*output)[outputYOffset + threadIdx.x][outputXOffset + tmp] = result[tmp];
}


template<
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t padH, const uint32_t padW
>
std::tuple<Tensor, float> conv2DForward3x3Fused(const Tensor& rinput, const Tensor& rkernel) 
{
  constexpr uint32_t resultTileH = 8;
  constexpr uint32_t resultTileW = 8;
  constexpr uint32_t kH = 3;
  constexpr uint32_t kW = 3;

  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  
  auto output = at::empty({input.size(0), outC, outH, outW},
                          at::TensorOptions().dtype(torch::kFloat32)
                                             .device(input.device())
                                             .is_variable(true));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0 && outC % 16 == 0);

  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC / 16); // output tiles, batch, outC
  dim3 block_size(8, 16, 1); // 8 threads per filter, 16 filters, 

  // cudaFuncSetCacheConfig(CUDAConv2DForward3x3CudaV5<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferL1);

  if (false)
  {
    std::cout << input.sizes() << std::endl;
    std::cout << kernel.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
    std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // compute zero point and scales
  auto tmp = quantize_uint8(kernel);
  auto kernelZero = std::get<0>(tmp);
  auto kernelInvScale = std::get<1>(tmp);

  tmp = quantize_uint8(input);
  auto inputZero = std::get<0>(tmp);
  auto inputInvScale = std::get<1>(tmp);

  inputZero.fill_(0);
  inputInvScale.fill_(1);

  kernelZero.fill_(0);
  kernelInvScale.fill_(1);

  // std::cout << "INPUT" << std::endl;
  // std::cout << inputZero << std::endl;
  // std::cout << inputInvScale << std::endl;

  // std::cout << "KERNEL" << std::endl;
  // std::cout << kernelZero << std::endl;
  // std::cout << kernelInvScale << std::endl;

  CUDAConv2DForward3x3Fused<kH, kW, inC, outC, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (float*)input.data_ptr(), (float*)inputInvScale.data_ptr(), (float*)inputZero.data_ptr(),
    (float*)kernel.data_ptr(), (float*)kernelInvScale.data_ptr(), (float*)kernelZero.data_ptr(),
    (float*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  return {output, elapsedTime};
}	

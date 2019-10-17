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


inline Tensor quantize_int8(Tensor tensor)
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
  return invscale;
}


// clamp x to range [minimum, maximum]
__device__ __forceinline__ float clamp(const float& x, const float& minimum, const float& maximum)
{
  return fmaxf(minimum, fminf(maximum, x));
}


__device__ __forceinline__ int8_t quantize(const float& value, const float& invScale)
{
  // if (clamp(value * invScale, -128.f, 127.f) != roundf(clamp(value * invScale, -128.f, 127.f)))
  //   printf("Value raw %f \n", value * invScale);
  // return int8_t(roundf(clamp(value * invScale, -128.f, 127.f)));
  return int8_t(max(-128, min(__float2int_rn(value * invScale), 127)));
  // return int8_t(value*invScale);
}

// template<
//   const uint32_t inC, const uint32_t outC,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW,
//   const uint32_t resultTileH, const uint32_t resultTileW
// >
// __global__ void 
// __launch_bounds__(256)
// CUDAConv2DForward3x3WinogradFused(const float* __restrict__ ridata,
//                                   const float* __restrict__ ridataInvScale,
//                                   const float* __restrict__ rkernel,
//                                   const float* __restrict__ rkernelInvScale,
//                                   float* __restrict__ routput)
// {
//   // grid
//   //        x - resultTile                    |2^31 - 1| Everything must be nice here
//   //        y - batch                         |2^16 - 1| Simple and straight forward
//   //        z - outC                          |2^16 - 1| Output channels group. Each group include 16 channels

//   // block  (16x16)
//   //        x - 16 threads to convolve 16 input tiles                |1024|
//   //        y - channel offset for 16 output channels group          |1024|

//   // output offsets
//   uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
//   const uint32_t outputYOffset = tmp * resultTileH; 
//   const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;
//   const uint32_t tOutC = blockIdx.z * uint32_t(16) + threadIdx.y;
//   const uint32_t& tBatch = blockIdx.y;

//   float (*output)[outH][outW] = (float (*)[outH][outW])(&(routput[tBatch * outC * outH * outW + tOutC * outH * outW]));
//   const float (*idata)[inC][inH][inW] = (const float (*)[inC][inH][inW])(&(ridata[tBatch*inC*inH*inW]));
//   // const float (*idataInvScale)[inC / 4] = (const float (*)[inC /4])(&(ridataInvScale[tBatch * inC / 4]));

//   const float (*kernel)[outC][inC][3][3] = (const float (*)[outC][inC][3][3])(rkernel);
//   // const float (*kernelInvScale)[32][inC / 4] = (const float (*)[32][inC /32])(&(rkernelInvScale[blockIdx.z * uint32_t(32) * inC / 32]));
  
//   __shared__ char4 dataWinograd[16][16][5]; // channel x tile x winograd column(4-packed)
//   __shared__ char4 kernelWinograd[16][16][5]; // channel out x channel in x winograd column(4-packed)
  
//   float result[2][2] = {
//     {0.f, 0.f}, 
//     {0.f, 0.f},
//   };

//   // nvcc bug, unrolling cycle leads to the wrong results
//   // #pragma unroll
//   for (uint32_t tInC = 0; tInC < inC; tInC += 16)
//   {
//     // populate data to shared memory. One thread, one tile
//     {
//       const float invScale = 4.f;
//       const uint32_t& chn = threadIdx.y;
//       const uint32_t& tileInd = threadIdx.x;
//       const uint32_t tileYOffset = tileInd / 4;
//       const uint32_t tileXOffset = tileInd - tileYOffset * 4;

//       // from global to registers
//       float dataf[4][4];
//       #pragma unroll
//       for (uint32_t ty = 0; ty < 4; ++ty)
//         #pragma unroll
//         for (uint32_t tx = 0; tx < 4; ++tx)
//           dataf[ty][tx] = (*idata)[tInC + chn]
//                                   [outputYOffset + tileYOffset * 2 + ty]
//                                   [outputXOffset + tileXOffset * 2 + tx];
//       char4 vec;
//       vec.x = quantize(dataf[0][0] - dataf[0][2] - dataf[2][0] + dataf[2][2], invScale);
//       vec.y = quantize(dataf[1][0] - dataf[1][2] + dataf[2][0] - dataf[2][2], invScale);
//       vec.z = quantize(-dataf[1][0] + dataf[1][2] + dataf[2][0] - dataf[2][2], invScale);
//       vec.w = quantize(dataf[1][0] - dataf[1][2] - dataf[3][0] + dataf[3][2], invScale);
//       dataWinograd[chn][tileInd][0] = vec;

//       vec.x = quantize(dataf[0][1] + dataf[0][2] - dataf[2][1] - dataf[2][2], invScale);
//       vec.y = quantize(dataf[1][1] + dataf[1][2] + dataf[2][1] + dataf[2][2], invScale);
//       vec.z = quantize(-dataf[1][1] - dataf[1][2] + dataf[2][1] + dataf[2][2], invScale);
//       vec.w = quantize(dataf[1][1] + dataf[1][2] - dataf[3][1] - dataf[3][2], invScale);
//       dataWinograd[chn][tileInd][1] = vec;

//       vec.x = quantize(-dataf[0][1] + dataf[0][2] + dataf[2][1] - dataf[2][2], invScale);
//       vec.y = quantize(-dataf[1][1] + dataf[1][2] - dataf[2][1] + dataf[2][2], invScale);
//       vec.z = quantize(dataf[1][1] - dataf[1][2] - dataf[2][1] + dataf[2][2], invScale);
//       vec.w = quantize(-dataf[1][1] + dataf[1][2] + dataf[3][1] - dataf[3][2], invScale);
//       dataWinograd[chn][tileInd][2] = vec;


//       vec.x = quantize(dataf[0][1] - dataf[0][3] - dataf[2][1] + dataf[2][3], invScale);
//       vec.y = quantize(dataf[1][1] - dataf[1][3] + dataf[2][1] - dataf[2][3], invScale);
//       vec.z = quantize(-dataf[1][1] + dataf[1][3] + dataf[2][1] - dataf[2][3], invScale);
//       vec.w = quantize(dataf[1][1] - dataf[1][3] - dataf[3][1] + dataf[3][3], invScale);
//       dataWinograd[chn][tileInd][3] = vec;
//     }

//     // populate kernel data to shared memory. One thread, two tiles
//     {
//       const uint32_t& filter = threadIdx.y;
//       const uint32_t& chn = threadIdx.x;
//       const float invScale = 4.f;
//       const float halfInvScale = invScale / 2;
//       const float quarterInvScale = invScale / 4;

//       float filterf[3][3];
//       #pragma unroll
//       for (uint32_t kx=0; kx < 3; ++kx)
//         #pragma unroll
//         for (uint32_t ky=0; ky < 3; ++ky)
//           filterf[ky][kx] = (*kernel)[blockIdx.z * uint32_t(16) + filter][tInC + chn][ky][kx];

//       char4 vec;
//       vec.x = quantize(filterf[0][0], invScale);
//       vec.y = quantize(filterf[0][0] + filterf[1][0] + filterf[2][0], halfInvScale);
//       vec.z = quantize(filterf[0][0] - filterf[1][0] + filterf[2][0], halfInvScale);
//       vec.w = quantize(filterf[2][0], invScale);
//       kernelWinograd[filter][chn][0] = vec;

//       vec.x = quantize(filterf[0][0] + filterf[0][1] + filterf[0][2], halfInvScale);
//       vec.y = quantize(filterf[0][0] + filterf[0][1] + filterf[0][2] + 
//                        filterf[1][0] + filterf[1][1] + filterf[1][2] + 
//                        filterf[2][0] + filterf[2][1] + filterf[2][2], quarterInvScale);
//       vec.z = quantize(filterf[0][0] + filterf[0][1] + filterf[0][2] -
//                        filterf[1][0] - filterf[1][1] - filterf[1][2] + 
//                        filterf[2][0] + filterf[2][1] + filterf[2][2], quarterInvScale);
//       vec.w = quantize(filterf[2][0] + filterf[2][1] + filterf[2][2], halfInvScale);
//       kernelWinograd[filter][chn][1] = vec;


//       vec.x = quantize(filterf[0][0] - filterf[0][1] + filterf[0][2], halfInvScale);
//       vec.y = quantize(filterf[0][0] - filterf[0][1] + filterf[0][2] + 
//                        filterf[1][0] - filterf[1][1] + filterf[1][2] + 
//                        filterf[2][0] - filterf[2][1] + filterf[2][2], quarterInvScale);
//       vec.z = quantize(filterf[0][0] - filterf[0][1] + filterf[0][2] - 
//                        filterf[1][0] + filterf[1][1] - filterf[1][2] + 
//                        filterf[2][0] - filterf[2][1] + filterf[2][2], quarterInvScale);
//       vec.w = quantize(filterf[2][0] - filterf[2][1] + filterf[2][2], halfInvScale);
//       kernelWinograd[filter][chn][2] = vec;


//       vec.x = quantize(filterf[0][2], invScale);
//       vec.y = quantize(filterf[0][2] + filterf[1][2] + filterf[2][2], halfInvScale);
//       vec.z = quantize(filterf[0][2] - filterf[1][2] + filterf[2][2], halfInvScale);
//       vec.w = quantize(filterf[2][2], invScale);
//       kernelWinograd[filter][chn][3] = vec;
//     }
//     __syncthreads();
  
//     // __shared__ char4 dataWinograd[16][16][5]; // channel x tile x winograd column(4-packed)
//     // __shared__ char4 kernelWinograd[16][16][5]; // channel out x channel in x winograd column(4-packed)
//     const float scale = 0.0625;
//     char4 (*weights)[16][5] = (char4 (*)[16][5])(&(kernelWinograd[threadIdx.y]));
//     int32_t ATMUL[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

//     // #pragma unroll
//     for (uint32_t chn=0; chn < 2; ++chn)
//     {
//       int8_t rmatData[4][4][8]; // row-col-channel
//       int8_t rmatKernel[4][4][8];
    
//       #pragma unroll
//       for (uint32_t col=0; col < 4; ++col)
//         #pragma unroll
//         for (uint32_t offset=0; offset < 8; ++offset)
//         {
//           char4 d = dataWinograd[chn*8 + offset][threadIdx.x][col];
//           char4 k = (*weights)[chn*8 + offset][col];
//           rmatData[0][col][offset] = d.x;
//           rmatData[1][col][offset] = d.y;
//           rmatData[2][col][offset] = d.z;
//           rmatData[3][col][offset] = d.w;

//           rmatKernel[0][col][offset] = k.x;
//           rmatKernel[1][col][offset] = k.y;
//           rmatKernel[2][col][offset] = k.z;
//           rmatKernel[3][col][offset] = k.w;
//         }
//       char4 (*matData)[4][4][2] = (char4 (*)[4][4][2])(&rmatData);
//       char4 (*matKernel)[4][4][2] = (char4 (*)[4][4][2])(&rmatKernel);

//       #pragma unroll
//       for (uint32_t col=0; col < 4; ++col)
//       {
//         ATMUL[0][col] = __dp4a((*matKernel)[0][col][0], (*matData)[0][col][0], ATMUL[0][col]);
//         ATMUL[0][col] = __dp4a((*matKernel)[0][col][1], (*matData)[0][col][1], ATMUL[0][col]);
//       }

//       #pragma unroll
//       for (uint32_t col=0; col < 4; ++col)
//       {
//         int32_t cache = 0;
//         cache = __dp4a((*matKernel)[1][col][0], (*matData)[1][col][0], cache);
//         cache = __dp4a((*matKernel)[1][col][1], (*matData)[1][col][1], cache);
//         ATMUL[0][col] += cache;
//         ATMUL[1][col] += cache;
//       }

//       #pragma unroll
//       for (uint32_t col=0; col < 4; ++col)
//       {
//         int32_t cache = 0;
//         cache = __dp4a((*matKernel)[2][col][0], (*matData)[2][col][0], cache);
//         cache = __dp4a((*matKernel)[2][col][1], (*matData)[2][col][1], cache);
//         ATMUL[0][col] += cache;
//         ATMUL[1][col] -= cache;
//       }

//       #pragma unroll
//       for (uint32_t col=0; col < 4; ++col)
//       {
//         int32_t cache = 0;
//         cache = __dp4a((*matKernel)[3][col][0], (*matData)[3][col][0], cache);
//         cache = __dp4a((*matKernel)[3][col][1], (*matData)[3][col][1], cache);
//         ATMUL[1][col] -= cache;
        
//       }
//     }

//     // // #pragma unroll
//     // for (uint32_t col=0; col < 4; ++col)
//     // {
//     //   int8_t rmatData[4][16];
//     //   int8_t rmatKernel[4][16];
    
//     //   #pragma unroll
//     //   for (uint32_t chn=0; chn < 16; ++chn)
//     //   {
//     //     char4 d = dataWinograd[chn][threadIdx.x][col];
//     //     char4 k = (*weights)[chn][col];
//     //     rmatData[0][chn] = d.x;
//     //     rmatData[1][chn] = d.y;
//     //     rmatData[2][chn] = d.z;
//     //     rmatData[3][chn] = d.w;

//     //     rmatKernel[0][chn] = k.x;
//     //     rmatKernel[1][chn] = k.y;
//     //     rmatKernel[2][chn] = k.z;
//     //     rmatKernel[3][chn] = k.w;
//     //   }
//     //   char4 (*matData)[4][4] = (char4 (*)[4][4])(&rmatData);
//     //   char4 (*matKernel)[4][4] = (char4 (*)[4][4])(&rmatKernel);

//     //   #pragma unroll
//     //   for (uint32_t chn=0; chn < 4; ++chn)
//     //     ATMUL[0][col] = __dp4a((*matKernel)[0][chn], (*matData)[0][chn], ATMUL[0][col]);

//     //   int32_t cache = 0;
//     //   #pragma unroll
//     //   for (uint32_t chn=0; chn < 4; ++chn)
//     //     cache = __dp4a((*matKernel)[1][chn], (*matData)[1][chn], cache);

//     //   ATMUL[0][col] += cache;
//     //   ATMUL[1][col] += cache;
//     //   cache = 0;

//     //   #pragma unroll
//     //   for (uint32_t chn=0; chn < 4; ++chn)
//     //     cache = __dp4a((*matKernel)[2][chn], (*matData)[2][chn], cache);
      
//     //   ATMUL[0][col] += cache;
//     //   ATMUL[1][col] -= cache;
//     //   cache = 0;

//     //   #pragma unroll
//     //   for (uint32_t chn=0; chn < 4; ++chn)
//     //     cache = __dp4a((*matKernel)[3][chn], (*matData)[3][chn], cache);
//     //   ATMUL[1][col] -= cache;

//     // }

//     result[0][0] += float(ATMUL[0][0] + ATMUL[0][1] + ATMUL[0][2]) * scale;
//     result[0][1] += float(ATMUL[0][1] - ATMUL[0][2] - ATMUL[0][3]) * scale;

//     result[1][0] += float(ATMUL[1][0] + ATMUL[1][1] + ATMUL[1][2]) * scale;
//     result[1][1] += float(ATMUL[1][1] - ATMUL[1][2] - ATMUL[1][3]) * scale;
//   }

//   uint32_t y = threadIdx.x / 4;
//   uint32_t x = threadIdx.x - y * 4;
//   y *= 2;
//   x *= 2;
//   (*output)[outputYOffset + y + 0][outputXOffset + x + 0] = result[0][0];
//   (*output)[outputYOffset + y + 0][outputXOffset + x + 1] = result[0][1];
//   (*output)[outputYOffset + y + 1][outputXOffset + x + 0] = result[1][0];
//   (*output)[outputYOffset + y + 1][outputXOffset + x + 1] = result[1][1];
// }


template<
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW
>
__global__ void 
__launch_bounds__(128, 7)
CUDAConv2DForward3x3WinogradFused(const float* __restrict__ ridata,
                                  const float* __restrict__ ridataInvScale,   // Batch x inC / 4
                                  const float* __restrict__ rkernel,
                                  const float* __restrict__ rkernelInvScale,  // outC x inC / 4
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

  const float (*kernel)[outC][inC][3][3] = (const float (*)[outC][inC][3][3])(rkernel);
  const float (*kernelInvScale)[16][inC / 4] = (const float (*)[16][inC /4])(&(rkernelInvScale[blockIdx.z * uint32_t(16) * inC / 4]));
  
  __shared__ char4 dataWinograd[4][resultTileH / 2][resultTileW / 2][17];
  __shared__ char4 kernelWinograd[4][16][17];

  float result[2][2][2] = {
    {{0.f, 0.f}, {0.f, 0.f}},
    {{0.f, 0.f}, {0.f, 0.f}}
  };

  // nvcc bug, unrolling cycle leads to the wrong results
  // #pragma unroll
  for (uint32_t tInC = 0; tInC < inC; tInC += 16)   
  {
    // populate data to shared memory. 
    uint32_t ind = threadIdx.x + threadIdx.y * 8;    
    if (ind < 64) // first 2 warps to transform input data
    {
      uint32_t chn = ind / 16;  // 4 channels target group
      uint32_t y = (ind - chn * 16) / 4;
      uint32_t x = ind - chn * 16 - y * 4;
      // const float invScale = (*idataInvScale)[tInC / 4 + chn];
      const float invScale = 4.f;

      // printf("Input data: chn %d y %d x %d\n", chn, y, x);
      // printf("Inv scale: %f\n\n", invScale);

      int8_t winograd[4][4][4]; // y-x-channel
      float dataf[4][4];
      // #pragma unroll
      for (ind=0; ind < 4; ++ind)  // load data by filters and transform
      {
        // load image tile for single filter
        // #pragma unroll
        // for (tmp=0; tmp < 4; ++tmp)
        //   *reinterpret_cast<float4*>(&(dataf[tmp])) = 
        //     *reinterpret_cast<const float4*>(&((*idata)[tInC + chn*4 + ind][outputYOffset + y * 2 + tmp][outputXOffset + x * 2]));

        #pragma unroll
        for (uint32_t ty = 0; ty < 4; ++ty)
          #pragma unroll
          for (uint32_t tx = 0; tx < 4; ++tx)
            dataf[ty][tx] = (*idata)[tInC + chn*4 + ind][outputYOffset + y * 2 + ty][outputXOffset + x * 2 + tx];

        // transform by BT @ data @ B
        winograd[0][0][ind] = quantize(dataf[0][0] - dataf[0][2] - dataf[2][0] + dataf[2][2], invScale);
        winograd[0][1][ind] = quantize(dataf[0][1] + dataf[0][2] - dataf[2][1] - dataf[2][2], invScale);
        winograd[0][2][ind] = quantize(-dataf[0][1] + dataf[0][2] + dataf[2][1] - dataf[2][2], invScale);
        winograd[0][3][ind] = quantize(dataf[0][1] - dataf[0][3] - dataf[2][1] + dataf[2][3], invScale);

        winograd[1][0][ind] = quantize(dataf[1][0] - dataf[1][2] + dataf[2][0] - dataf[2][2], invScale);
        winograd[1][1][ind] = quantize(dataf[1][1] + dataf[1][2] + dataf[2][1] + dataf[2][2], invScale);
        winograd[1][2][ind] = quantize(-dataf[1][1] + dataf[1][2] - dataf[2][1] + dataf[2][2], invScale);
        winograd[1][3][ind] = quantize(dataf[1][1] - dataf[1][3] + dataf[2][1] - dataf[2][3], invScale);

        winograd[2][0][ind] = quantize(-dataf[1][0] + dataf[1][2] + dataf[2][0] - dataf[2][2], invScale);
        winograd[2][1][ind] = quantize(-dataf[1][1] - dataf[1][2] + dataf[2][1] + dataf[2][2], invScale);
        winograd[2][2][ind] = quantize(dataf[1][1] - dataf[1][2] - dataf[2][1] + dataf[2][2], invScale);
        winograd[2][3][ind] = quantize(-dataf[1][1] + dataf[1][3] + dataf[2][1] - dataf[2][3], invScale);

        winograd[3][0][ind] = quantize(dataf[1][0] - dataf[1][2] - dataf[3][0] + dataf[3][2], invScale);
        winograd[3][1][ind] = quantize(dataf[1][1] + dataf[1][2] - dataf[3][1] - dataf[3][2], invScale);
        winograd[3][2][ind] = quantize(-dataf[1][1] + dataf[1][2] + dataf[3][1] - dataf[3][2], invScale);
        winograd[3][3][ind] = quantize(dataf[1][1] - dataf[1][3] - dataf[3][1] + dataf[3][3], invScale);
      }
      // to shm
      #pragma unroll
      for (ind=0; ind < 4; ++ind)
        #pragma unroll 
        for (tmp=0; tmp < 4; ++tmp)
          dataWinograd[chn][y][x][ind*4 + tmp] = *reinterpret_cast<const char4*>(&(winograd[ind][tmp]));
    }
    else // second 2 warps to load transform weights
    {
      ind -= 64;
      uint32_t chn = ind / 16;  // 4 channels target group
      uint32_t filter = (ind - chn * 16);
      const float invScale = (*kernelInvScale)[filter][tInC / 4 + chn];
      // const float invScale = 4.f;

      int8_t winograd[4][4][4]; // y-x-channel
      float filterf[4][3][3];   // channel-y-x

      // printf("Input kernel: chn %d filter %d\n", chn, filter);
      // printf("Inv scale: %f\n\n", invScale);

      // load filter tile
      const float4 (*fkernel)[3 * 3] = (const float4 (*)[3 * 3])(&((*kernel)[blockIdx.z * uint32_t(16) + filter][tInC + chn*4]));
      float4 (*w4)[9] = (float4 (*)[9])(&filterf);
      #pragma unroll
      for (tmp=0; tmp < 9; ++tmp)
        (*w4)[tmp] = (*fkernel)[tmp];

      const float halfInvScale = invScale / 2;
      const float quarterInvScale = invScale / 4;
      // transform by G @ filter @ GT
      // #pragma unroll
      for (ind=0; ind < 4; ++ind)
      {
        winograd[0][0][ind] = quantize(filterf[ind][0][0], invScale);
        winograd[0][1][ind] = quantize(filterf[ind][0][0] + filterf[ind][0][1] + filterf[ind][0][2], halfInvScale);
        winograd[0][2][ind] = quantize(filterf[ind][0][0] - filterf[ind][0][1] + filterf[ind][0][2], halfInvScale);
        winograd[0][3][ind] = quantize(filterf[ind][0][2], invScale);

        winograd[1][0][ind] = quantize(filterf[ind][0][0] + filterf[ind][1][0] + filterf[ind][2][0], halfInvScale);
        winograd[1][1][ind] = quantize(filterf[ind][0][0] + filterf[ind][0][1] + filterf[ind][0][2] + 
                                       filterf[ind][1][0] + filterf[ind][1][1] + filterf[ind][1][2] + 
                                       filterf[ind][2][0] + filterf[ind][2][1] + filterf[ind][2][2], quarterInvScale);
        winograd[1][2][ind] = quantize(filterf[ind][0][0] - filterf[ind][0][1] + filterf[ind][0][2] + 
                                       filterf[ind][1][0] - filterf[ind][1][1] + filterf[ind][1][2] + 
                                       filterf[ind][2][0] - filterf[ind][2][1] + filterf[ind][2][2], quarterInvScale);
        winograd[1][3][ind] = quantize(filterf[ind][0][2] + filterf[ind][1][2] + filterf[ind][2][2], halfInvScale);

        winograd[2][0][ind] = quantize(filterf[ind][0][0] - filterf[ind][1][0] + filterf[ind][2][0], halfInvScale);
        winograd[2][1][ind] = quantize(filterf[ind][0][0] + filterf[ind][0][1] + filterf[ind][0][2] -
                                       filterf[ind][1][0] - filterf[ind][1][1] - filterf[ind][1][2] + 
                                       filterf[ind][2][0] + filterf[ind][2][1] + filterf[ind][2][2], quarterInvScale);
        winograd[2][2][ind] = quantize(filterf[ind][0][0] - filterf[ind][0][1] + filterf[ind][0][2] - 
                                       filterf[ind][1][0] + filterf[ind][1][1] - filterf[ind][1][2] + 
                                       filterf[ind][2][0] - filterf[ind][2][1] + filterf[ind][2][2], quarterInvScale);
        winograd[2][3][ind] = quantize(filterf[ind][0][2] - filterf[ind][1][2] + filterf[ind][2][2], halfInvScale);

        winograd[3][0][ind] = quantize(filterf[ind][2][0], invScale);
        winograd[3][1][ind] = quantize(filterf[ind][2][0] + filterf[ind][2][1] + filterf[ind][2][2], halfInvScale);
        winograd[3][2][ind] = quantize(filterf[ind][2][0] - filterf[ind][2][1] + filterf[ind][2][2], halfInvScale);
        winograd[3][3][ind] = quantize(filterf[ind][2][2], invScale);
      }

      // to shm
      #pragma unroll
      for (ind=0; ind < 4; ++ind)
        #pragma unroll 
        for (tmp=0; tmp < 4; ++tmp)
          kernelWinograd[chn][filter][ind*4 + tmp] = *reinterpret_cast<const char4*>(&(winograd[ind][tmp]));
    }
    __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0)
    // {
    // //   for (uint32_t filter=0; filter<16; ++filter)
    // //   {
    // //     printf("Kernel\n");
    // //     for (uint32_t asd=0; asd<4; ++asd)
    // //     {
    // //       for (uint32_t wy=0; wy<4; ++wy)
    // //       {
    // //         for (uint32_t wx=0; wx<4; ++wx)
    // //           printf("[%d, %d, %d, %d] ", kernelWinograd[filter][wy][wx][asd].x, kernelWinograd[filter][wy][wx][asd].y, 
    // //           kernelWinograd[filter][wy][wx][asd].z, kernelWinograd[filter][wy][wx][asd].w);
    // //         printf("\n");
    // //       }
    // //       printf("!\n");
    // //     }
    // //   }

    //   for (uint32_t dx=0; dx<4; ++dx)
    //   {
    //     printf("Data\n");
    //     for (uint32_t dy=0; dy<4; ++dy)
    //       for (uint32_t asd=0; asd<4; ++asd)
    //       {
    //         for (uint32_t wy=0; wy<4; ++wy)
    //         {
    //           for (uint32_t wx=0; wx<4; ++wx)
    //             printf("[%d, %d, %d, %d] ", dataWinograd[dy][dx][wy][wx][asd].x, dataWinograd[dy][dx][wy][wx][asd].y, 
    //                                         dataWinograd[dy][dx][wy][wx][asd].z, dataWinograd[dy][dx][wy][wx][asd].w);
    //           printf("\n");
    //         }
    //         printf("!\n");
    //       }
    //   }
    // }

    // printf("X %d\n", x);
    // char4 (*data)[2][4][4][4] = (char4 (*)[2][4][4][4])(&((*dataWinograd)[y][x*2]));  // dataWinograd[resultTileH / 2][resultTileW / 2][4][4][4];
    char4 (*data)[4][16][17] = (char4 (*)[4][16][17])(&dataWinograd);
    // do convolve
    // #pragma unroll
    for (ind = 0; ind < 4; ++ind)
    {
      const float scale = 1.f / (*kernelInvScale)[threadIdx.y][tInC / 4 + ind] * 
                          1.f / (*idataInvScale)[tInC / 4 + ind];
      // const float scale = 0.0625f;
      char4 weights[4][4];
      uint32_t y, x;

      // load weights
      #pragma unroll
      for (y=0; y < 4; ++y)
        #pragma unroll
        for (x=0; x < 4; ++x)
          weights[y][x] = kernelWinograd[ind][threadIdx.y][y*4 + x];
      

      // if (threadIdx.x == 1 && threadIdx.y == 0)
      // {
      //   printf("Weights\n");
      //   for (uint32_t wy=0; wy<4; ++wy){
      //     for (uint32_t wx=0; wx<4; ++wx)
      //       printf("[%d, %d, %d, %d] ", weights[wy][wx].x, weights[wy][wx].y, weights[wy][wx].z, weights[wy][wx].w);
      //     printf("\n");
      //   }
      // }


      // winograd convolution for 2 tiles
      // #pragma unroll
      for (tmp = 0; tmp < 2; ++tmp)
      {
        uint32_t block = threadIdx.x * 2 + tmp;
        // AT (result)
        int32_t ATMUL[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

        // if (threadIdx.x == 1 && threadIdx.y == 0)
        // {
        //   printf("Data\n");
        //   for (uint32_t wy=0; wy<4; ++wy)
        //   {
        //     for (uint32_t wx=0; wx<4; ++wx)
        //       printf("[%d, %d, %d, %d] ", (*data)[block][wy][wx][ind].x, (*data)[block][wy][wx][ind].y, 
        //                                   (*data)[block][wy][wx][ind].z, (*data)[block][wy][wx][ind].w);
        //     printf("\n");
        //   }
        // }

        // #pragma unroll
        for (x=0; x<4; ++x)
        {
          int32_t cache = 0;
          ATMUL[0][x] = __dp4a(weights[0][x], (*data)[ind][block][0*4 + x], ATMUL[0][x]);
          
          cache = __dp4a(weights[1][x], (*data)[ind][block][1*4 + x], int32_t(0));
          ATMUL[0][x] += cache;
          ATMUL[1][x] += cache;

          cache = __dp4a(weights[2][x], (*data)[ind][block][2*4 + x], int32_t(0));
          ATMUL[0][x] += cache;
          ATMUL[1][x] -= cache;

          ATMUL[1][x] -= __dp4a(weights[3][x], (*data)[ind][block][3*4 + x], int32_t(0));
        }

        // if (threadIdx.x == 1 && threadIdx.y == 0)
        // {
        //   printf("ATMUL\n");
        //   for (uint32_t wy=0; wy<2; ++wy){
        //     for (uint32_t wx=0; wx<4; ++wx)
        //       printf("%d ", ATMUL[wy][wx]);
        //     printf("\n");
        //   }
        // }

        result[tmp][0][0] += float(ATMUL[0][0] + ATMUL[0][1] + ATMUL[0][2]) * scale;
        result[tmp][0][1] += float(ATMUL[0][1] - ATMUL[0][2] - ATMUL[0][3]) * scale;

        result[tmp][1][0] += float(ATMUL[1][0] + ATMUL[1][1] + ATMUL[1][2]) * scale;
        result[tmp][1][1] += float(ATMUL[1][1] - ATMUL[1][2] - ATMUL[1][3]) * scale;
      }
    }
  }

  // if (threadIdx.x == 1 && threadIdx.y == 0)
  // {
  //   printf("Result\n");
  //   for (uint32_t wy=0; wy<4; ++wy){
  //     for (uint32_t wx=0; wx<4; ++wx)
  //       printf("%f ", result[0][wy][wx]);
  //     printf("\n");
  //   }
  // }
  // if (threadIdx.x == 1 && threadIdx.y == 0)
  // {
  //   printf("Result1\n");
  //   for (uint32_t wy=0; wy<4; ++wy){
  //     for (uint32_t wx=0; wx<4; ++wx)
  //       printf("%f ", result[1][wy][wx]);
  //     printf("\n");
  //   }
  // }

  uint32_t y = threadIdx.x / 2;
  uint32_t x = threadIdx.x - y * 2;
  y *= 2;
  x *= 4;
  // printf("Result x: %d y: %d\n", x, y);
  #pragma unroll
  for (tmp = 0; tmp < 2; ++tmp)
  {
    // if (threadIdx.y == 0) 
    // {
    //   if (result[tmp][0][0] != 144)
    //     printf("!%d -> y %d x %d [0,0] = %f\n", threadIdx.x, outputYOffset + y + 0, outputXOffset + tmp * 2 + x + 0, result[tmp][0][0]);
    //   if (result[tmp][0][1] != 144)
    //     printf("!%d -> y %d x %d [0,1] = %f\n", threadIdx.x, outputYOffset + y + 0, outputXOffset + tmp * 2 + x + 1, result[tmp][0][1]);
    //   if (result[tmp][1][0] != 144)
    //     printf("!%d -> y %d x %d [1,0] = %f\n", threadIdx.x, outputYOffset + y + 1, outputXOffset + tmp * 2 + x + 0, result[tmp][1][0]);
    //   if (result[tmp][1][1] != 144)
    //     printf("!%d -> y %d x %d [1,1] = %f\n", threadIdx.x, outputYOffset + y + 1, outputXOffset + tmp * 2 + x + 1, result[tmp][1][1]);
      
    // }
    (*output)[outputYOffset + y + 0][outputXOffset + tmp * 2 + x + 0] = result[tmp][0][0];
    (*output)[outputYOffset + y + 0][outputXOffset + tmp * 2 + x + 1] = result[tmp][0][1];
    (*output)[outputYOffset + y + 1][outputXOffset + tmp * 2 + x + 0] = result[tmp][1][0];
    (*output)[outputYOffset + y + 1][outputXOffset + tmp * 2 + x + 1] = result[tmp][1][1];
  }
}


template<
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t padH, const uint32_t padW
>
std::tuple<Tensor, float> conv2DForward3x3WinogradFused(const Tensor& rinput, const Tensor& rkernel) 
{
  constexpr uint32_t resultTileW = 8;
  constexpr uint32_t resultTileH = 8;

  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  
  auto output = at::empty({input.size(0), outC, outH, outW},
                          at::TensorOptions().dtype(torch::kFloat32)
                                             .device(input.device())
                                             .is_variable(true));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0 && outC % 16 == 0);

  // dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC / 16); // output tiles, batch, outC
  // dim3 block_size(16, 16, 1); // 8 threads per filter, 16 filters, 

  AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0 && outC % 16 == 0);

  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC / 16); // output tiles, batch, outC
  dim3 block_size(8, 16, 1); // 8 threads per filter, 16 filters, 

  // cudaFuncSetCacheConfig(CUDAConv2DForward3x3CudaV5<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferL1);

  if (true)
  {
    std::cout << input.sizes() << std::endl;
    std::cout << kernel.sizes() << std::endl;
    std::cout << output.sizes() << std::endl;
    std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
    std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;
  }
  // compute zero point and scales
  auto kernelInvScale = quantize_int8(kernel);
  auto inputInvScale = quantize_int8(input);

  inputInvScale.fill_(4);
  kernelInvScale.fill_(4);

  // std::cout << "INPUT" << std::endl;
  // std::cout << inputZero << std::endl;
  // std::cout << inputInvScale << std::endl;

  // std::cout << "KERNEL" << std::endl;
  // std::cout << kernelZero << std::endl;
  // std::cout << kernelInvScale << std::endl;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  CUDAConv2DForward3x3WinogradFused<inC, outC, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (float*)input.data_ptr(), (float*)inputInvScale.data_ptr(),
    (float*)kernel.data_ptr(), (float*)kernelInvScale.data_ptr(),
    (float*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  return {output, elapsedTime};
}	

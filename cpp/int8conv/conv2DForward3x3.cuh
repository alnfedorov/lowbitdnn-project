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
#include <mma.h>

// template<
//   const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW,
//   const uint32_t resultTileH, const uint32_t resultTileW
// >
// __global__ void 
// __launch_bounds__(256)
// CUDAConv2DForward3x3CudaV5(const int8_t* __restrict__ ridata, 
//                        const int8_t*  __restrict__ rkernel, 
//                        int32_t* __restrict__ routput)
// {
//   // grid
//   //        x - resultTile                    |2^31 - 1| Everything must be nice here
//   //        y - batch                         |2^16 - 1| Simple and straight forward
//   //        z - outC                          |2^16 - 1| Output channels, VECT_C included

//   // block
//   //        x - 8 threads to convolve 8 input lines                  |1024| 
//   //        y - channel offset for 32 output channels group          |1024|  

//   // output offsets
//   uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
//   const uint32_t outputYOffset = tmp * resultTileH; 
//   const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;
//   const uint32_t tOutCVectCOffset = threadIdx.y;     
//   const uint32_t tOutC = blockIdx.z;                // VECT_C included
  
//   int32_t (*output)[outH][outW][VECT_C] = 
//                 (int32_t (*)[outH][outW][VECT_C])(&(routput[blockIdx.y * outC * outH * outW * VECT_C + tOutC * outH * outW * VECT_C]));
  
//   // Our results
//   // int32_t cache[8];
//   // #pragma unroll
//   // for (tmp = 0; tmp < 8; ++tmp)
//   //   cache[tmp] = 0;
//   int32_t cache0 = 0;
//   int32_t cache1 = 0;
//   int32_t cache2 = 0;
//   int32_t cache3 = 0;
//   int32_t cache4 = 0;
//   int32_t cache5 = 0;
//   int32_t cache6 = 0;
//   int32_t cache7 = 0;

//   __shared__ int data[resultTileH + 2][resultTileW + 2][11];
  
//   // nvcc bug, unrolling cycle leads to the wrong results
//   // #pragma unroll
//   for (uint32_t tInC = 0; tInC < inC; ++tInC)   
//   {
//     // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.z == 0)
//     //   printf("tInC %d\n", tInC);
//     const int8_t (*idata)[inH][inW][VECT_C] = 
//                 (const int8_t (*)[inH][inW][VECT_C])(&(ridata[blockIdx.y*inC*inH*inW*VECT_C + tInC*inH*inW*VECT_C]));
//     const int8_t (*kernel)[kH][kW][VECT_C] = 
//                 (const int8_t (*)[kH][kW][VECT_C])(&(rkernel[(tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C]));

//     // populate data to shared memory. 
//     uint32_t y, x, chn;
//     #pragma unroll
//     for (uint32_t ind = 0; ind < 3; ++ind)
//     {
//       tmp = threadIdx.x + threadIdx.y * 8 + 256 * ind;
//       y = tmp / 80; 
//       x = (tmp - y * 80) / 8;
//       chn = tmp - y * 80 - x * 8;
//       // data[y][x][chn] = *reinterpret_cast<const int32_t*>(&ridata[uint32_t(blockIdx.y)*inC*inH*inW*VECT_C + 
//       //                                                             tInC*inH*inW*VECT_C + 
//       //                                                             (outputYOffset+y)*inW*VECT_C + 
//       //                                                             (outputXOffset+x)*VECT_C + 
//       //                                                             chn*4]);
//       data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
//     }

//     if (threadIdx.y < 4)
//     {
//       tmp = threadIdx.x + threadIdx.y * 8 + 256 * 3;
//       y = tmp / 80; 
//       x = (tmp - y * 80) / 8;
//       chn = tmp - y * 80 - x * 8;
//       data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
//       // data[y][x][chn] = *reinterpret_cast<const int32_t*>(&ridata[uint32_t(blockIdx.y)*inC*inH*inW*VECT_C + 
//       //                                                       tInC*inH*inW*VECT_C + 
//       //                                                       (outputYOffset+y)*inW*VECT_C + 
//       //                                                       (outputXOffset+x)*VECT_C + 
//       //                                                       chn*4]);
//     }
//     __syncthreads();

//     int32_t d0, d1, d2, d3, d4, d5, d6, d7;
//     int4 w1, w2;

//     // Load kernel weights. Each thread load the whole kernel
//     #pragma unroll
//     for (y = 0; y < 3; ++y)
//     {
//       int4 threadWeights[kW][2];
//       #pragma unroll
//       for (x = 0; x < kW; ++x)
//       {
//         // threadWeights[x][0] = *reinterpret_cast<const int4*>(&rkernel[(tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + 
//         //                                                               tInC*kH*kW*VECT_C + 
//         //                                                               y*kW*VECT_C + 
//         //                                                               x*VECT_C]);
//         // threadWeights[x][1] = *reinterpret_cast<const int4*>(&rkernel[(tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + 
//         //                                                               tInC*kH*kW*VECT_C + 
//         //                                                               y*kW*VECT_C + 
//         //                                                               x*VECT_C + 
//         //                                                               16]);
//         threadWeights[x][0] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][0]));
//         threadWeights[x][1] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][16]));
//       }

//       // #pragma unroll
//       // for (tmp = 0; tmp < 8; ++tmp)
//       //   #pragma unroll
//       //   for (x = 0; x < kW; ++x)
//       //   {
//       //     d0 = data[y + threadIdx.x][tmp + x][0];
//       //     d1 = data[y + threadIdx.x][tmp + x][1];
//       //     d2 = data[y + threadIdx.x][tmp + x][2];
//       //     d3 = data[y + threadIdx.x][tmp + x][3];
//       //     d4 = data[y + threadIdx.x][tmp + x][4];
//       //     d5 = data[y + threadIdx.x][tmp + x][5];
//       //     d6 = data[y + threadIdx.x][tmp + x][6];
//       //     d7 = data[y + threadIdx.x][tmp + x][7];
//       //     w1 = threadWeights[x][0];
//       //     w2 = threadWeights[x][1];
//       //     cache[tmp] = __dp4a(d0, w1.x, cache[tmp]);
//       //     cache[tmp] = __dp4a(d1, w1.y, cache[tmp]);
//       //     cache[tmp] = __dp4a(d2, w1.z, cache[tmp]);
//       //     cache[tmp] = __dp4a(d3, w1.w, cache[tmp]);
//       //     cache[tmp] = __dp4a(d4, w2.x, cache[tmp]);
//       //     cache[tmp] = __dp4a(d5, w2.y, cache[tmp]);
//       //     cache[tmp] = __dp4a(d6, w2.z, cache[tmp]);
//       //     cache[tmp] = __dp4a(d7, w2.w, cache[tmp]);
//       //   }
      
//       d0 = data[y + threadIdx.x][0][0];
//       d1 = data[y + threadIdx.x][0][1];
//       d2 = data[y + threadIdx.x][0][2];
//       d3 = data[y + threadIdx.x][0][3];
//       d4 = data[y + threadIdx.x][0][4];
//       d5 = data[y + threadIdx.x][0][5];
//       d6 = data[y + threadIdx.x][0][6];
//       d7 = data[y + threadIdx.x][0][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache0 = __dp4a(d0, w1.x, cache0);
//       cache0 = __dp4a(d1, w1.y, cache0);
//       cache0 = __dp4a(d2, w1.z, cache0);
//       cache0 = __dp4a(d3, w1.w, cache0);
//       cache0 = __dp4a(d4, w2.x, cache0);
//       cache0 = __dp4a(d5, w2.y, cache0);
//       cache0 = __dp4a(d6, w2.z, cache0);
//       cache0 = __dp4a(d7, w2.w, cache0);
//       d0 = data[y + threadIdx.x][1][0];
//       d1 = data[y + threadIdx.x][1][1];
//       d2 = data[y + threadIdx.x][1][2];
//       d3 = data[y + threadIdx.x][1][3];
//       d4 = data[y + threadIdx.x][1][4];
//       d5 = data[y + threadIdx.x][1][5];
//       d6 = data[y + threadIdx.x][1][6];
//       d7 = data[y + threadIdx.x][1][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache1 = __dp4a(d0, w1.x, cache1);
//       cache1 = __dp4a(d1, w1.y, cache1);
//       cache1 = __dp4a(d2, w1.z, cache1);
//       cache1 = __dp4a(d3, w1.w, cache1);
//       cache1 = __dp4a(d4, w2.x, cache1);
//       cache1 = __dp4a(d5, w2.y, cache1);
//       cache1 = __dp4a(d6, w2.z, cache1);
//       cache1 = __dp4a(d7, w2.w, cache1);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache0 = __dp4a(d0, w1.x, cache0);
//       cache0 = __dp4a(d1, w1.y, cache0);
//       cache0 = __dp4a(d2, w1.z, cache0);
//       cache0 = __dp4a(d3, w1.w, cache0);
//       cache0 = __dp4a(d4, w2.x, cache0);
//       cache0 = __dp4a(d5, w2.y, cache0);
//       cache0 = __dp4a(d6, w2.z, cache0);
//       cache0 = __dp4a(d7, w2.w, cache0);
//       d0 = data[y + threadIdx.x][2][0];
//       d1 = data[y + threadIdx.x][2][1];
//       d2 = data[y + threadIdx.x][2][2];
//       d3 = data[y + threadIdx.x][2][3];
//       d4 = data[y + threadIdx.x][2][4];
//       d5 = data[y + threadIdx.x][2][5];
//       d6 = data[y + threadIdx.x][2][6];
//       d7 = data[y + threadIdx.x][2][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache2 = __dp4a(d0, w1.x, cache2);
//       cache2 = __dp4a(d1, w1.y, cache2);
//       cache2 = __dp4a(d2, w1.z, cache2);
//       cache2 = __dp4a(d3, w1.w, cache2);
//       cache2 = __dp4a(d4, w2.x, cache2);
//       cache2 = __dp4a(d5, w2.y, cache2);
//       cache2 = __dp4a(d6, w2.z, cache2);
//       cache2 = __dp4a(d7, w2.w, cache2);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache1 = __dp4a(d0, w1.x, cache1);
//       cache1 = __dp4a(d1, w1.y, cache1);
//       cache1 = __dp4a(d2, w1.z, cache1);
//       cache1 = __dp4a(d3, w1.w, cache1);
//       cache1 = __dp4a(d4, w2.x, cache1);
//       cache1 = __dp4a(d5, w2.y, cache1);
//       cache1 = __dp4a(d6, w2.z, cache1);
//       cache1 = __dp4a(d7, w2.w, cache1);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache0 = __dp4a(d0, w1.x, cache0);
//       cache0 = __dp4a(d1, w1.y, cache0);
//       cache0 = __dp4a(d2, w1.z, cache0);
//       cache0 = __dp4a(d3, w1.w, cache0);
//       cache0 = __dp4a(d4, w2.x, cache0);
//       cache0 = __dp4a(d5, w2.y, cache0);
//       cache0 = __dp4a(d6, w2.z, cache0);
//       cache0 = __dp4a(d7, w2.w, cache0);
//       d0 = data[y + threadIdx.x][3][0];
//       d1 = data[y + threadIdx.x][3][1];
//       d2 = data[y + threadIdx.x][3][2];
//       d3 = data[y + threadIdx.x][3][3];
//       d4 = data[y + threadIdx.x][3][4];
//       d5 = data[y + threadIdx.x][3][5];
//       d6 = data[y + threadIdx.x][3][6];
//       d7 = data[y + threadIdx.x][3][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache3 = __dp4a(d0, w1.x, cache3);
//       cache3 = __dp4a(d1, w1.y, cache3);
//       cache3 = __dp4a(d2, w1.z, cache3);
//       cache3 = __dp4a(d3, w1.w, cache3);
//       cache3 = __dp4a(d4, w2.x, cache3);
//       cache3 = __dp4a(d5, w2.y, cache3);
//       cache3 = __dp4a(d6, w2.z, cache3);
//       cache3 = __dp4a(d7, w2.w, cache3);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache2 = __dp4a(d0, w1.x, cache2);
//       cache2 = __dp4a(d1, w1.y, cache2);
//       cache2 = __dp4a(d2, w1.z, cache2);
//       cache2 = __dp4a(d3, w1.w, cache2);
//       cache2 = __dp4a(d4, w2.x, cache2);
//       cache2 = __dp4a(d5, w2.y, cache2);
//       cache2 = __dp4a(d6, w2.z, cache2);
//       cache2 = __dp4a(d7, w2.w, cache2);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache1 = __dp4a(d0, w1.x, cache1);
//       cache1 = __dp4a(d1, w1.y, cache1);
//       cache1 = __dp4a(d2, w1.z, cache1);
//       cache1 = __dp4a(d3, w1.w, cache1);
//       cache1 = __dp4a(d4, w2.x, cache1);
//       cache1 = __dp4a(d5, w2.y, cache1);
//       cache1 = __dp4a(d6, w2.z, cache1);
//       cache1 = __dp4a(d7, w2.w, cache1);
//       d0 = data[y + threadIdx.x][4][0];
//       d1 = data[y + threadIdx.x][4][1];
//       d2 = data[y + threadIdx.x][4][2];
//       d3 = data[y + threadIdx.x][4][3];
//       d4 = data[y + threadIdx.x][4][4];
//       d5 = data[y + threadIdx.x][4][5];
//       d6 = data[y + threadIdx.x][4][6];
//       d7 = data[y + threadIdx.x][4][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache4 = __dp4a(d0, w1.x, cache4);
//       cache4 = __dp4a(d1, w1.y, cache4);
//       cache4 = __dp4a(d2, w1.z, cache4);
//       cache4 = __dp4a(d3, w1.w, cache4);
//       cache4 = __dp4a(d4, w2.x, cache4);
//       cache4 = __dp4a(d5, w2.y, cache4);
//       cache4 = __dp4a(d6, w2.z, cache4);
//       cache4 = __dp4a(d7, w2.w, cache4);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache3 = __dp4a(d0, w1.x, cache3);
//       cache3 = __dp4a(d1, w1.y, cache3);
//       cache3 = __dp4a(d2, w1.z, cache3);
//       cache3 = __dp4a(d3, w1.w, cache3);
//       cache3 = __dp4a(d4, w2.x, cache3);
//       cache3 = __dp4a(d5, w2.y, cache3);
//       cache3 = __dp4a(d6, w2.z, cache3);
//       cache3 = __dp4a(d7, w2.w, cache3);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache2 = __dp4a(d0, w1.x, cache2);
//       cache2 = __dp4a(d1, w1.y, cache2);
//       cache2 = __dp4a(d2, w1.z, cache2);
//       cache2 = __dp4a(d3, w1.w, cache2);
//       cache2 = __dp4a(d4, w2.x, cache2);
//       cache2 = __dp4a(d5, w2.y, cache2);
//       cache2 = __dp4a(d6, w2.z, cache2);
//       cache2 = __dp4a(d7, w2.w, cache2);
//       d0 = data[y + threadIdx.x][5][0];
//       d1 = data[y + threadIdx.x][5][1];
//       d2 = data[y + threadIdx.x][5][2];
//       d3 = data[y + threadIdx.x][5][3];
//       d4 = data[y + threadIdx.x][5][4];
//       d5 = data[y + threadIdx.x][5][5];
//       d6 = data[y + threadIdx.x][5][6];
//       d7 = data[y + threadIdx.x][5][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache5 = __dp4a(d0, w1.x, cache5);
//       cache5 = __dp4a(d1, w1.y, cache5);
//       cache5 = __dp4a(d2, w1.z, cache5);
//       cache5 = __dp4a(d3, w1.w, cache5);
//       cache5 = __dp4a(d4, w2.x, cache5);
//       cache5 = __dp4a(d5, w2.y, cache5);
//       cache5 = __dp4a(d6, w2.z, cache5);
//       cache5 = __dp4a(d7, w2.w, cache5);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache4 = __dp4a(d0, w1.x, cache4);
//       cache4 = __dp4a(d1, w1.y, cache4);
//       cache4 = __dp4a(d2, w1.z, cache4);
//       cache4 = __dp4a(d3, w1.w, cache4);
//       cache4 = __dp4a(d4, w2.x, cache4);
//       cache4 = __dp4a(d5, w2.y, cache4);
//       cache4 = __dp4a(d6, w2.z, cache4);
//       cache4 = __dp4a(d7, w2.w, cache4);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache3 = __dp4a(d0, w1.x, cache3);
//       cache3 = __dp4a(d1, w1.y, cache3);
//       cache3 = __dp4a(d2, w1.z, cache3);
//       cache3 = __dp4a(d3, w1.w, cache3);
//       cache3 = __dp4a(d4, w2.x, cache3);
//       cache3 = __dp4a(d5, w2.y, cache3);
//       cache3 = __dp4a(d6, w2.z, cache3);
//       cache3 = __dp4a(d7, w2.w, cache3);
//       d0 = data[y + threadIdx.x][6][0];
//       d1 = data[y + threadIdx.x][6][1];
//       d2 = data[y + threadIdx.x][6][2];
//       d3 = data[y + threadIdx.x][6][3];
//       d4 = data[y + threadIdx.x][6][4];
//       d5 = data[y + threadIdx.x][6][5];
//       d6 = data[y + threadIdx.x][6][6];
//       d7 = data[y + threadIdx.x][6][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache6 = __dp4a(d0, w1.x, cache6);
//       cache6 = __dp4a(d1, w1.y, cache6);
//       cache6 = __dp4a(d2, w1.z, cache6);
//       cache6 = __dp4a(d3, w1.w, cache6);
//       cache6 = __dp4a(d4, w2.x, cache6);
//       cache6 = __dp4a(d5, w2.y, cache6);
//       cache6 = __dp4a(d6, w2.z, cache6);
//       cache6 = __dp4a(d7, w2.w, cache6);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache5 = __dp4a(d0, w1.x, cache5);
//       cache5 = __dp4a(d1, w1.y, cache5);
//       cache5 = __dp4a(d2, w1.z, cache5);
//       cache5 = __dp4a(d3, w1.w, cache5);
//       cache5 = __dp4a(d4, w2.x, cache5);
//       cache5 = __dp4a(d5, w2.y, cache5);
//       cache5 = __dp4a(d6, w2.z, cache5);
//       cache5 = __dp4a(d7, w2.w, cache5);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache4 = __dp4a(d0, w1.x, cache4);
//       cache4 = __dp4a(d1, w1.y, cache4);
//       cache4 = __dp4a(d2, w1.z, cache4);
//       cache4 = __dp4a(d3, w1.w, cache4);
//       cache4 = __dp4a(d4, w2.x, cache4);
//       cache4 = __dp4a(d5, w2.y, cache4);
//       cache4 = __dp4a(d6, w2.z, cache4);
//       cache4 = __dp4a(d7, w2.w, cache4);
//       d0 = data[y + threadIdx.x][7][0];
//       d1 = data[y + threadIdx.x][7][1];
//       d2 = data[y + threadIdx.x][7][2];
//       d3 = data[y + threadIdx.x][7][3];
//       d4 = data[y + threadIdx.x][7][4];
//       d5 = data[y + threadIdx.x][7][5];
//       d6 = data[y + threadIdx.x][7][6];
//       d7 = data[y + threadIdx.x][7][7];
//       w1 = threadWeights[0][0];
//       w2 = threadWeights[0][1];
//       cache7 = __dp4a(d0, w1.x, cache7);
//       cache7 = __dp4a(d1, w1.y, cache7);
//       cache7 = __dp4a(d2, w1.z, cache7);
//       cache7 = __dp4a(d3, w1.w, cache7);
//       cache7 = __dp4a(d4, w2.x, cache7);
//       cache7 = __dp4a(d5, w2.y, cache7);
//       cache7 = __dp4a(d6, w2.z, cache7);
//       cache7 = __dp4a(d7, w2.w, cache7);
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache6 = __dp4a(d0, w1.x, cache6);
//       cache6 = __dp4a(d1, w1.y, cache6);
//       cache6 = __dp4a(d2, w1.z, cache6);
//       cache6 = __dp4a(d3, w1.w, cache6);
//       cache6 = __dp4a(d4, w2.x, cache6);
//       cache6 = __dp4a(d5, w2.y, cache6);
//       cache6 = __dp4a(d6, w2.z, cache6);
//       cache6 = __dp4a(d7, w2.w, cache6);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache5 = __dp4a(d0, w1.x, cache5);
//       cache5 = __dp4a(d1, w1.y, cache5);
//       cache5 = __dp4a(d2, w1.z, cache5);
//       cache5 = __dp4a(d3, w1.w, cache5);
//       cache5 = __dp4a(d4, w2.x, cache5);
//       cache5 = __dp4a(d5, w2.y, cache5);
//       cache5 = __dp4a(d6, w2.z, cache5);
//       cache5 = __dp4a(d7, w2.w, cache5);
//       d0 = data[y + threadIdx.x][8][0];
//       d1 = data[y + threadIdx.x][8][1];
//       d2 = data[y + threadIdx.x][8][2];
//       d3 = data[y + threadIdx.x][8][3];
//       d4 = data[y + threadIdx.x][8][4];
//       d5 = data[y + threadIdx.x][8][5];
//       d6 = data[y + threadIdx.x][8][6];
//       d7 = data[y + threadIdx.x][8][7];
//       w1 = threadWeights[1][0];
//       w2 = threadWeights[1][1];
//       cache7 = __dp4a(d0, w1.x, cache7);
//       cache7 = __dp4a(d1, w1.y, cache7);
//       cache7 = __dp4a(d2, w1.z, cache7);
//       cache7 = __dp4a(d3, w1.w, cache7);
//       cache7 = __dp4a(d4, w2.x, cache7);
//       cache7 = __dp4a(d5, w2.y, cache7);
//       cache7 = __dp4a(d6, w2.z, cache7);
//       cache7 = __dp4a(d7, w2.w, cache7);
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache6 = __dp4a(d0, w1.x, cache6);
//       cache6 = __dp4a(d1, w1.y, cache6);
//       cache6 = __dp4a(d2, w1.z, cache6);
//       cache6 = __dp4a(d3, w1.w, cache6);
//       cache6 = __dp4a(d4, w2.x, cache6);
//       cache6 = __dp4a(d5, w2.y, cache6);
//       cache6 = __dp4a(d6, w2.z, cache6);
//       cache6 = __dp4a(d7, w2.w, cache6);
//       d0 = data[y + threadIdx.x][9][0];
//       d1 = data[y + threadIdx.x][9][1];
//       d2 = data[y + threadIdx.x][9][2];
//       d3 = data[y + threadIdx.x][9][3];
//       d4 = data[y + threadIdx.x][9][4];
//       d5 = data[y + threadIdx.x][9][5];
//       d6 = data[y + threadIdx.x][9][6];
//       d7 = data[y + threadIdx.x][9][7];
//       w1 = threadWeights[2][0];
//       w2 = threadWeights[2][1];
//       cache7 = __dp4a(d0, w1.x, cache7);
//       cache7 = __dp4a(d1, w1.y, cache7);
//       cache7 = __dp4a(d2, w1.z, cache7);
//       cache7 = __dp4a(d3, w1.w, cache7);
//       cache7 = __dp4a(d4, w2.x, cache7);
//       cache7 = __dp4a(d5, w2.y, cache7);
//       cache7 = __dp4a(d6, w2.z, cache7);
//       cache7 = __dp4a(d7, w2.w, cache7);
//     }
//   }
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 0)*VECT_C + 
//   //         tOutCVectCOffset] = cache0;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 1)*VECT_C + 
//   //         tOutCVectCOffset] = cache1;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 2)*VECT_C + 
//   //         tOutCVectCOffset] = cache2;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 3)*VECT_C + 
//   //         tOutCVectCOffset] = cache3;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 4)*VECT_C + 
//   //         tOutCVectCOffset] = cache4;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 5)*VECT_C + 
//   //         tOutCVectCOffset] = cache5;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 6)*VECT_C + 
//   //         tOutCVectCOffset] = cache6;
//   // routput[uint32_t(blockIdx.y)*outC*outH*outW*VECT_C + 
//   //         tOutC*outH*outW*VECT_C + 
//   //         (outputYOffset + uint32_t(threadIdx.x))*outW*VECT_C + 
//   //         (outputXOffset + 7)*VECT_C + 
//   //         tOutCVectCOffset] = cache7;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 0][tOutCVectCOffset] = cache0;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 1][tOutCVectCOffset] = cache1;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 2][tOutCVectCOffset] = cache2;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 3][tOutCVectCOffset] = cache3;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 4][tOutCVectCOffset] = cache4;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 5][tOutCVectCOffset] = cache5;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 6][tOutCVectCOffset] = cache6;
//   (*output)[outputYOffset + threadIdx.x][outputXOffset + 7][tOutCVectCOffset] = cache7;
//   // #pragma unroll
//   // for (tmp = 0; tmp < 8; ++tmp)
//   //   (*output)[outputYOffset + threadIdx.x][outputXOffset + tmp][tOutCVectCOffset] = cache[tmp];
// }


// template<
//   const uint32_t inC, const uint32_t outC,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW
// >
// std::tuple<Tensor, float> conv2DForward3x3(const Tensor& rinput, const Tensor& rkernel) 
// {
//   constexpr uint32_t resultTileH = 8;
//   constexpr uint32_t resultTileW = 8;
//   constexpr uint32_t kH = 3;
//   constexpr uint32_t kW = 3;

//   // TODO add asserts

//   auto dtype = torch::kInt32;
//   auto input = rinput.contiguous();
//   auto kernel = rkernel.contiguous();
  
//   auto output = at::empty({input.size(0), outC, outH, outW, VECT_C},
//                     at::TensorOptions().dtype(torch::kInt32)
//                                        .device(input.device())
//                                        .is_variable(false));
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0);

//   dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC); // output tiles, batch, outC
//   dim3 block_size(8, VECT_C, 1); // 8 threads per filter, 32 filters, 
  
//   // if (true)
//   // {
//   //   std::cout << input.sizes() << std::endl;
//   //   std::cout << kernel.sizes() << std::endl;
//   //   std::cout << output.sizes() << std::endl;
//   //   std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
//   //   std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;
//   // }

//   // cudaFuncSetCacheConfig(CUDAConv2DForward3x3CudaV5<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferL1);

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   cudaEventRecord(start);
//   CUDAConv2DForward3x3CudaV5<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
//     (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
//   );
  
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);

//   float elapsedTime;
//   cudaEventElapsedTime(&elapsedTime, start, stop);  

//   return {output, elapsedTime};
// }	

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


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW, // == 2x2
  const uint32_t padH, const uint32_t padW
>
__global__ void 
__launch_bounds__(128)
CUDAConv2DForward3x3CudaV1(const int8_t* __restrict__ ridata, 
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
  // const uint32_t& tBatch = blockIdx.y;
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
  for (uint32_t y = 0; y < kH; ++y)
    #pragma unroll
      for (uint32_t x = 0; x < kW; ++x)
            threadWeights[y][x] = *reinterpret_cast<const int2*>(&((*kernel)[y][x][8*tmp]));

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
std::tuple<Tensor, float> conv2DForward3x3(const Tensor& rinput, const Tensor& rkernel)
{
  constexpr uint32_t resultTileH = 8;
  constexpr uint32_t resultTileW = 8;
  constexpr uint32_t kH = 3;
  constexpr uint32_t kW = 3;  
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), outC, outH, outW, VECT_C},
                    at::TensorOptions().dtype(torch::kInt32)
                                       .device(input.device())
                                       .is_variable(false));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0);
  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  // std::cout << output.sizes() << std::endl;
  // std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  // std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  CUDAConv2DForward3x3CudaV1<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW, padH, padW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  // cudaDeviceSetSharedMemConfig(conf);
  return {output, elapsedTime};
}	

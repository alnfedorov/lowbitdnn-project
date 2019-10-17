#pragma once

#include "utils.cuh"
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <mma.h>

#include <device_launch_parameters.h>

// constexpr uint32_t VECT_C = 32;
// constexpr uint32_t KERNELS_PER_BLOCK = 32;
namespace wmma = nvcuda::wmma;

// constexpr uint32_t WMMA_M = 8;
// constexpr uint32_t WMMA_N = 32;
// constexpr uint32_t WMMA_K = 16;

constexpr uint32_t WMMA_M = 32;
constexpr uint32_t WMMA_N = 8;
constexpr uint32_t WMMA_K = 16;

// template<
//   const uint32_t batch,
//   const uint32_t inCGroups, const uint32_t outCGroups, 
//   const uint32_t kH, const uint32_t kW,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW,
//   const uint32_t outTileH, const uint32_t outTileW
// >
// __global__ void 
// __launch_bounds__(256)
// CUDAConv2DForward3x3TensorCoures(
//   const int8_t* __restrict__ rdata,  const int8_t*  __restrict__ rkernel, int32_t* __restrict__ rresult
// )
// {
//   // blockIdx
//   //        x - outputTile                    |2^31 - 1| 
//   //        y - batch                         |2^16 - 1|
//   //        z - outCGroup                     |2^16 - 1|

//   // threadIdx
//   //        x - 32                            |1024|
//   //        y - 8 warps for 8 lines           |1024| 

//   // output offsets
//   constexpr uint32_t hTiles = outH / outTileH;
//   constexpr uint32_t wTiles = outW / outTileW;

//   // blockIdx.x = [hTile][wTile]
//   uint32_t tOutputYOffset = blockIdx.x / wTiles; 
//   uint32_t tOutputXOffset = blockIdx.x - tOutputYOffset * wTiles;

//   tOutputYOffset *= outTileH;
//   tOutputXOffset *= outTileW;

//   // if (blockIdx.y == 0 && threadIdx.x == 0)
//   // {
//   //   printf("Y offset %d, X offset %d, blockIdx.x %d\n", tOutputYOffset, tOutputXOffset, blockIdx.x);
//   //   if (blockIdx.x == 0 && threadIdx.y == 0)
//   //     printf("hTiles %d, wTiles %d\n", hTiles, wTiles);
//   // }

//   const uint32_t& tBatch = blockIdx.y;
//   const uint32_t& tOutCGroup = blockIdx.z;
  
//   // recast input parameters as arrays
//   const int8_t (*arrdata)[batch][inCGroups][inH][inW][32] = (const int8_t (*)[batch][inCGroups][inH][inW][32])(rdata);
//   const int8_t (*arrkernel)[outCGroups*32][inCGroups][kH][kW][32] = (const int8_t (*)[outCGroups*32][inCGroups][kH][kW][32])(rkernel);
//   int32_t (*arrresult)[batch][outCGroups][outH][outW][32] = (int32_t (*)[batch][outCGroups][outH][outW][32])(rresult);

//   // pick only relevant slices, recast if needed
//   const int4 (*idata)[inCGroups][inH][inW][2] = (const int4 (*)[inCGroups][inH][inW][2])(&((*arrdata)[tBatch]));
//   const int4 (*kernel)[32][inCGroups][kH][kW][2] = (const int4 (*)[32][inCGroups][kH][kW][2])(&((*arrkernel)[32*tOutCGroup]));
//   int32_t (*result)[outH][outW][32] = (int32_t (*)[outH][outW][32])(&((*arrresult)[tBatch][tOutCGroup]));

//   __shared__ alignas(32) int4 weights[2][kH][kW][32];
//   __shared__ alignas(32) int4 data[2][10][10];  // 2xtileHxtileW
  
//   int4 (*flatWeights)[2][kH*kW][32] = (int4 (*)[2][kH*kW][32])(&weights);
//   const int4 (*flatKernel)[32][inCGroups][kH*kW][2] = (const int4 (*)[32][inCGroups][kH*kW][2])(kernel);

//   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> dataMat;
//   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> kernelMat;

//   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> resultMat;
//   wmma::fill_fragment(resultMat, 0);

//   // #pragma unroll
//   for (uint32_t tInCGroup=0; tInCGroup < inCGroups; ++tInCGroup)
//   {
//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     //   printf("Before%d!\n", tInCGroup);
//     __syncthreads();
//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     //   printf("%d!\n", tInCGroup);
//     // load weights
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     #pragma unroll
//     for (uint32_t i=0; i < 2; ++i)
//     {
//       const uint32_t ind = i * 256 + threadIdx.y * 32 + threadIdx.x; // tKernel * kH * kW * 2
//       const uint32_t tKernel = ind / (kH * kW * 2);
//       const uint32_t tPosition = (ind - tKernel * (kH * kW * 2)) / 2; 
//       const uint32_t tVECT_C = ind % 2;
//       // if (threadIdx.x == 0 && threadIdx.y == 0)
//       // {
//       //   printf("ind %d \n", ind);
//       //   printf("flatKernel[%d][%d][%d][%d]\n", tKernel, tInCGroup, tPosition, tVECT_C);
//       // }
//       // printf("%d\n", (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C].x);
//       (*flatWeights)[tVECT_C][tPosition][tKernel] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     }
//     // fetch last elements
//     if (threadIdx.y > 5)
//     {
//       // printf("FETCHING LAST KERNEL THREADIDX.Y %d", threadIdx.y);
//       const uint32_t ind = 2 * 256 + (threadIdx.y - 6)* 32 + threadIdx.x; // tKernel * kH * kW * 2
//       const uint32_t tKernel = ind / (kH * kW * 2);
//       const uint32_t tPosition = (ind - tKernel * (kH * kW * 2)) / 2; 
//       const uint32_t tVECT_C = ind % 2;
//       // printf("%d\n", (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C].x);
//       (*flatWeights)[tVECT_C][tPosition][tKernel] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////

//     // load data
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     {
//       const uint32_t ind = threadIdx.y * 32 + threadIdx.x;
//       if (ind < 200)
//       {
//         const uint32_t tHeight = ind / (10 * 2);
//         const uint32_t tWidth = (ind - tHeight * (10 * 2)) / 2; 
//         const uint32_t tVECT_C = ind % 2;
//         data[tVECT_C][tHeight][tWidth] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth][tVECT_C];
//         // printf("[%d][%d][%d] = [%d][%d][%d][%d]\n", tVECT_C, tHeight, tWidth, tInCGroup, tOutputYOffset + tHeight, tOutputXOffset + tWidth, tVECT_C);
//       }
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     __syncthreads();
//     // {
//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     // {
//     //   printf("Data\n");
//     //   char4 (*tmp)[2][10][10][4] = (char4 (*)[2][10][10][4])(&data);
//     //   // const int4 data[2][10][10];
//     //   for(uint32_t y = 0; y < 10; ++y)
//     //     for(uint32_t x = 0; x<10; ++x)
//     //     {
//     //       printf("[");
//     //       for(uint32_t c=0; c<4; ++c)
//     //         printf("%d, %d, %d, %d,", (*tmp)[0][y][x][c].x, (*tmp)[0][y][x][c].y, (*tmp)[0][y][x][c].z, (*tmp)[0][y][x][c].w);
//     //       for(uint32_t c=0; c<4; ++c)
//     //         printf("%d, %d, %d, %d,", (*tmp)[1][y][x][c].x, (*tmp)[1][y][x][c].y, (*tmp)[1][y][x][c].z, (*tmp)[1][y][x][c].w);
//     //       printf("]\n");
//     //     }
//     // }

//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     // {
//     //   printf("Kernel\n");
//     //   char4 (*tmp)[2][kH][kW][32][4] = (char4 (*)[2][kH][kW][32][4])(&weights);
//     //   for(uint32_t k=0; k<32; ++k)
//     //     for(uint32_t ky = 0; ky < 3; ++ky)
//     //       for(uint32_t kx = 0; kx < 3; ++kx)
//     //       {
//     //         printf("[");
//     //         for(uint32_t c=0; c<4; ++c)
//     //           printf("%d, %d, %d, %d,", (*tmp)[0][ky][kx][k][c].x, (*tmp)[0][ky][kx][k][c].y, (*tmp)[0][ky][kx][k][c].z, (*tmp)[0][ky][kx][k][c].w);
//     //         for(uint32_t c=0; c<4; ++c)
//     //           printf("%d, %d, %d, %d,", (*tmp)[1][ky][kx][k][c].x, (*tmp)[1][ky][kx][k][c].y, (*tmp)[1][ky][kx][k][c].z, (*tmp)[1][ky][kx][k][c].w);
//     //         printf("]\n");
//     //       }
//     // }

//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     // {
//     //   char4 (*tmp)[2][10][10][4] = (char4 (*)[2][10][10][4])(&data);
//     //   // const int4 data[2][10][10];
//     //   for(uint32_t y = 0; y < 10; ++y)
//     //     for(uint32_t x = 0; x<10; ++x)
//     //     {
//     //       for(uint32_t c=0; c<4; ++c)
//     //         if ((*tmp)[0][y][x][c].x == 0 || (*tmp)[0][y][x][c].y == 0 || (*tmp)[0][y][x][c].z == 0 || (*tmp)[0][y][x][c].w == 0)
//     //           {
//     //             printf("blockIdx.x %d, blockIdx.y %d\n", blockIdx.x, blockIdx.y);
//     //             printf("%d, %d, %d, %d\n", 0, y, x, c);
//     //             printf("%d, %d, %d, %d\n", (*tmp)[0][y][x][c].x, (*tmp)[0][y][x][c].y, (*tmp)[0][y][x][c].z, (*tmp)[0][y][x][c].w);
//     //           }
//     //       for(uint32_t c=0; c<4; ++c)
//     //         if ((*tmp)[1][y][x][c].x == 0 || (*tmp)[1][y][x][c].y == 0 || (*tmp)[1][y][x][c].z == 0 || (*tmp)[1][y][x][c].w == 0)
//     //           {
//     //             printf("blockIdx.x %d, blockIdx.y %d\n", blockIdx.x, blockIdx.y);
//     //             printf("%d, %d, %d, %d\n", 0, y, x, c);
//     //             printf("%d, %d, %d, %d\n", (*tmp)[1][y][x][c].x, (*tmp)[1][y][x][c].y, (*tmp)[1][y][x][c].z, (*tmp)[1][y][x][c].w);
//     //           }
//     //     }
//     // }

//     // if (threadIdx.x == 0 && threadIdx.y == 0)
//     // {
//     //   char4 (*tmp)[2][kH][kW][32][4] = (char4 (*)[2][kH][kW][32][4])(&weights);
//     //   for(uint32_t k=0; k<32; ++k)
//     //     for(uint32_t ky = 0; ky < 3; ++ky)
//     //       for(uint32_t kx = 0; kx < 3; ++kx)
//     //       {
//     //         for(uint32_t c=0; c<4; ++c)
//     //           if ((*tmp)[0][ky][kx][k][c].x != 1 || (*tmp)[0][ky][kx][k][c].y != 1 || (*tmp)[0][ky][kx][k][c].z != 1 || (*tmp)[0][ky][kx][k][c].w != 1)
//     //             {
//     //               printf("blockIdx.x %d, blockIdx.y %d\n", blockIdx.x, blockIdx.y);
//     //               printf("%d, %d, %d, %d, %d\n", 0, ky, kx, k, c);
//     //               printf("%d, %d, %d, %d\n", (*tmp)[0][ky][kx][k][c].x, (*tmp)[0][ky][kx][k][c].y, (*tmp)[0][ky][kx][k][c].z, (*tmp)[0][ky][kx][k][c].w);
//     //             }
//     //         for(uint32_t c=0; c<4; ++c)
//     //           if ((*tmp)[1][ky][kx][k][c].x != 1 || (*tmp)[1][ky][kx][k][c].y != 1 || (*tmp)[1][ky][kx][k][c].z != 1 || (*tmp)[1][ky][kx][k][c].w != 1)
//     //             {
//     //               printf("blockIdx.x %d, blockIdx.y %d\n", blockIdx.x, blockIdx.y);
//     //               printf("%d, %d, %d, %d, %d\n", 1, ky, kx, k, c);
//     //               printf("%d, %d, %d, %d\n", (*tmp)[1][ky][kx][k][c].x, (*tmp)[1][ky][kx][k][c].y, (*tmp)[1][ky][kx][k][c].z, (*tmp)[1][ky][kx][k][c].w);
//     //             }
//     //       }
//     // }
//     // }

//     // load matrices
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     // #pragma unroll
//     for (uint32_t cGroup = 0; cGroup < 2; ++cGroup)
//       // #pragma unroll
//       for (uint32_t ky = 0; ky < kH; ++ky)
//         // #pragma unroll
//         for (uint32_t kx = 0; kx < kW; ++kx)
//         {
//           // int4 weights[2][kH][kW][32];
//           // int4 data[2][10][10];  // 2xtileHxtileW
//           wmma::load_matrix_sync(dataMat, (int8_t*)&(data[cGroup][threadIdx.y + ky][kx]), 16); // 8x16, row-major
          
//           // if (threadIdx.x == 0 && threadIdx.y == 0)
//           // {
//             // printf("Data\n");
//             // for(int i=0; i < dataMat.num_elements; i++) { 
//             //   int32_t t=static_cast<int32_t>(dataMat.x[i]); 
//             //   printf("%d\n",t);
//             // }
//           // }
//           wmma::load_matrix_sync(kernelMat, (int8_t*)&(weights[cGroup][ky][kx]), 16); // 16x32, col major

//           // if (threadIdx.x == 0 && threadIdx.y == 0)
//           // {
//           //   printf("Kernel\n");
//           //   for(int i=0; i < kernelMat.num_elements; i++) { 
//           //     int32_t t=static_cast<int32_t>(kernelMat.x[i]); 
//           //     printf("%d\n",t);
//           //   }
//           // }
//           wmma::mma_sync(resultMat, dataMat, kernelMat, resultMat, false);
//         }
//   }
//   // if (threadIdx.y == 3 && blockIdx.x == 1 && blockIdx.y == 0)
//   // {
//   //   if (threadIdx.x == 0)
//   //     printf("ResultRESULTRESULTRESULT\n");
//   //   for(int i=0; i < resultMat.num_elements; i++) { 
//   //     int32_t t=static_cast<int32_t>(resultMat.x[i]); 
//   //     printf("%d\n", t);
//   //   }
//   //   if (threadIdx.x == 0)
//   //     printf("Output [%d][%d][%d][%d]\n", tBatch, tOutCGroup, tOutputYOffset + threadIdx.y, tOutputXOffset);
//   // }

//   wmma::store_matrix_sync((int32_t*)&((*result)[tOutputYOffset + threadIdx.y][tOutputXOffset]), 
//                           resultMat, 32, wmma::mem_row_major); // 8 x 32
// }

// template<
//   const uint32_t batch,
//   const uint32_t inCGroups, const uint32_t outCGroups, 
//   const uint32_t kH, const uint32_t kW,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW,
//   const uint32_t outTileH, const uint32_t outTileW
// >
// __global__ void 
// __launch_bounds__(512, 2)
// CUDAConv2DForward3x3TensorCoures(
//   const int8_t* __restrict__ rdata,  const int8_t*  __restrict__ rkernel, int32_t* __restrict__ rresult
// )
// {
//   // blockIdx
//   //        x - outputTile                    |2^31 - 1| 
//   //        y - batch                         |2^16 - 1|
//   //        z - outCGroup                     |2^16 - 1|

//   // threadIdx
//   //        x - 32                              |1024|
//   //        y - 16 warps for 16 lines           |1024| 

//   // output offsets
//   constexpr uint32_t hTiles = outH / outTileH;
//   constexpr uint32_t wTiles = outW / outTileW;

//   // blockIdx.x = [hTile][wTile]
//   uint32_t tOutputYOffset = blockIdx.x / wTiles; 
//   uint32_t tOutputXOffset = blockIdx.x - tOutputYOffset * wTiles;

//   tOutputYOffset *= outTileH;
//   tOutputXOffset *= outTileW;

//   const uint32_t& tBatch = blockIdx.y;
//   const uint32_t& tOutCGroup = blockIdx.z;
  
//   // recast input parameters as arrays
//   const int8_t (*arrdata)[batch][inCGroups][inH][inW][16] = (const int8_t (*)[batch][inCGroups][inH][inW][16])(rdata);
//   const int8_t (*arrkernel)[outCGroups*16][inCGroups][kH][kW][16] = (const int8_t (*)[outCGroups*16][inCGroups][kH][kW][16])(rkernel);
//   int32_t (*arrresult)[batch][outCGroups][outH][outW][16] = (int32_t (*)[batch][outCGroups][outH][outW][16])(rresult);

//   // pick only relevant slices, recast if needed
//   const int32_t (*idata)[inCGroups][inH][inW][4] = (const int32_t (*)[inCGroups][inH][inW][4])(&((*arrdata)[tBatch]));
//   const int32_t (*kernel)[16][inCGroups][kH][kW][4] = (const int32_t (*)[16][inCGroups][kH][kW][4])(&((*arrkernel)[16*tOutCGroup]));
//   int32_t (*result)[outH][outW][16] = (int32_t (*)[outH][outW][16])(&((*arrresult)[tBatch][tOutCGroup]));

//   __shared__ alignas(32) int32_t weights[16][kH][kW][4];
//   __shared__ alignas(32) int32_t data[18][18][4];
  
//   int32_t (*flatWeights)[16][kH*kW][4] = (int32_t (*)[16][kH*kW][4])(&weights);
//   const int32_t (*flatKernel)[16][inCGroups][kH*kW][4] = (const int32_t (*)[16][inCGroups][kH*kW][4])(kernel);

//   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> dataMat;
//   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> kernelMat;

//   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> resultMat;
//   wmma::fill_fragment(resultMat, 0);

//   // #pragma unroll
//   for (uint32_t tInCGroup=0; tInCGroup < inCGroups; ++tInCGroup)
//   {
//     __syncthreads();
//     // load weights
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     const uint32_t ind = threadIdx.y * 32 + threadIdx.x; // tKernel * kH * kW * 4
//     const uint32_t tKernel = ind / (kH * kW * 4);
//     const uint32_t tPosition = (ind - tKernel * (kH * kW * 4)) / 4; 
//     const uint32_t tVECT_C = ind % 4;
//     (*flatWeights)[tKernel][tPosition][tVECT_C] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     // fetch last elements, only 14 and 15 warps
//     if (threadIdx.y > 13)
//     {
//       const uint32_t ind = 512 + (threadIdx.y - 14) * 32 + threadIdx.x; // tKernel * kH * kW * 4
//       const uint32_t tKernel = ind / (kH * kW * 4);
//       const uint32_t tPosition = (ind - tKernel * (kH * kW * 4)) / 4; 
//       const uint32_t tVECT_C = ind % 4;
//       (*flatWeights)[tKernel][tPosition][tVECT_C] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////

//     // load data
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     {
//       #pragma unroll
//       for (uint32_t i=0; i < 2; ++i)
//       {
//         const uint32_t ind = i * 512 + threadIdx.y * 32 + threadIdx.x;
//         const uint32_t tHeight = ind / (18 * 4);
//         const uint32_t tWidth = (ind - tHeight * (18 * 4)) / 4; 
//         const uint32_t tVECT_C = ind % 4;
//         data[tHeight][tWidth][tVECT_C] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth][tVECT_C];
//       }
//       uint32_t ind = threadIdx.y * 32 + threadIdx.x;
//       if (ind < 272)
//       {
//         ind += 2 * 512;
//         const uint32_t tHeight = ind / (18 * 4);
//         const uint32_t tWidth = (ind - tHeight * (18 * 4)) / 4; 
//         const uint32_t tVECT_C = ind % 4;
//         data[tHeight][tWidth][tVECT_C] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth][tVECT_C];
//       }
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     __syncthreads();

//     // load matrices
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     #pragma unroll
//     for (uint32_t ky = 0; ky < kH; ++ky)
//       #pragma unroll
//       for (uint32_t kx = 0; kx < kW; ++kx)
//       {
//         // int8_t weights[16][kH][kW][16];
//         // int8_t data[18][18][16];
//         wmma::load_matrix_sync(dataMat, (int8_t*)&(data[threadIdx.y + ky][kx]), 16); // 16x16, row-major
//         wmma::load_matrix_sync(kernelMat, (int8_t*)&(weights[0][ky][kx]), kH*kW*16); // 16x16, col-major
//         wmma::mma_sync(resultMat, dataMat, kernelMat, resultMat, false);
//       }
//   }

//   // scale the matrix here?

//   wmma::store_matrix_sync((int32_t*)&((*result)[tOutputYOffset + threadIdx.y][tOutputXOffset]), 
//                           resultMat, 16, wmma::mem_row_major); // 16x16
// }


// template<
//   const uint32_t batch,
//   const uint32_t inCGroups, const uint32_t outCGroups, 
//   const uint32_t kH, const uint32_t kW,
//   const uint32_t inH, const uint32_t inW, 
//   const uint32_t outH, const uint32_t outW,
//   const uint32_t outTileH, const uint32_t outTileW
// >
// __global__ void 
// __launch_bounds__(512, 2)
// CUDAConv2DForward3x3TensorCoures(
//   const int8_t* __restrict__ rdata,  const int8_t*  __restrict__ rkernel, int32_t* __restrict__ rresult
// )
// {
//   // blockIdx
//   //        x - outputTile                    |2^31 - 1| 
//   //        y - batch                         |2^16 - 1|
//   //        z - outCGroup                     |2^16 - 1|

//   // threadIdx
//   //        x - 32                              |1024|
//   //        y - 16 warps for 16 lines           |1024| 

//   // output offsets
//   constexpr uint32_t hTiles = outH / outTileH;
//   constexpr uint32_t wTiles = outW / outTileW;

//   // blockIdx.x = [hTile][wTile]
//   uint32_t tOutputYOffset = blockIdx.x / wTiles; 
//   uint32_t tOutputXOffset = blockIdx.x - tOutputYOffset * wTiles;

//   tOutputYOffset *= outTileH;
//   tOutputXOffset *= outTileW;

//   const uint32_t& tBatch = blockIdx.y;
//   const uint32_t& tOutCGroup = blockIdx.z;
  
//   // recast input parameters as arrays
//   const int8_t (*arrdata)[batch][inCGroups][inH][inW][16] = (const int8_t (*)[batch][inCGroups][inH][inW][16])(rdata);
//   const int8_t (*arrkernel)[outCGroups*16][inCGroups][kH][kW][16] = (const int8_t (*)[outCGroups*16][inCGroups][kH][kW][16])(rkernel);
//   int32_t (*arrresult)[batch][outCGroups][outH][outW][16] = (int32_t (*)[batch][outCGroups][outH][outW][16])(rresult);

//   // pick only relevant slices, recast if needed
//   const int32_t (*idata)[inCGroups][inH][inW][4] = (const int32_t (*)[inCGroups][inH][inW][4])(&((*arrdata)[tBatch]));
//   const int32_t (*kernel)[16][inCGroups][kH][kW][4] = (const int32_t (*)[16][inCGroups][kH][kW][4])(&((*arrkernel)[16*tOutCGroup]));
//   int32_t (*result)[outH][outW][16] = (int32_t (*)[outH][outW][16])(&((*arrresult)[tBatch][tOutCGroup]));

//   __shared__ alignas(32) int32_t weights[16][kH][kW][4];
//   __shared__ alignas(32) int32_t data[18][18][4];
  
//   int32_t (*flatWeights)[16][kH*kW][4] = (int32_t (*)[16][kH*kW][4])(&weights);
//   const int32_t (*flatKernel)[16][inCGroups][kH*kW][4] = (const int32_t (*)[16][inCGroups][kH*kW][4])(kernel);

//   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> dataMat;
//   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> kernelMat;

//   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> resultMat;
//   wmma::fill_fragment(resultMat, 0);

//   // #pragma unroll
//   for (uint32_t tInCGroup=0; tInCGroup < inCGroups; ++tInCGroup)
//   {
//     __syncthreads();
//     // load weights
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     const uint32_t ind = threadIdx.y * 32 + threadIdx.x; // tKernel * kH * kW * 4
//     const uint32_t tKernel = ind / (kH * kW * 4);
//     const uint32_t tPosition = (ind - tKernel * (kH * kW * 4)) / 4; 
//     const uint32_t tVECT_C = ind % 4;
//     (*flatWeights)[tKernel][tPosition][tVECT_C] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     // fetch last elements, only 14 and 15 warps
//     if (threadIdx.y > 13)
//     {
//       const uint32_t ind = 512 + (threadIdx.y - 14) * 32 + threadIdx.x; // tKernel * kH * kW * 4
//       const uint32_t tKernel = ind / (kH * kW * 4);
//       const uint32_t tPosition = (ind - tKernel * (kH * kW * 4)) / 4; 
//       const uint32_t tVECT_C = ind % 4;
//       (*flatWeights)[tKernel][tPosition][tVECT_C] = (*flatKernel)[tKernel][tInCGroup][tPosition][tVECT_C];
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////

//     // load data
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     {
//       #pragma unroll
//       for (uint32_t i=0; i < 2; ++i)
//       {
//         const uint32_t ind = i * 512 + threadIdx.y * 32 + threadIdx.x;
//         const uint32_t tHeight = ind / (18 * 4);
//         const uint32_t tWidth = (ind - tHeight * (18 * 4)) / 4; 
//         const uint32_t tVECT_C = ind % 4;
//         data[tHeight][tWidth][tVECT_C] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth][tVECT_C];
//       }
//       uint32_t ind = threadIdx.y * 32 + threadIdx.x;
//       if (ind < 272)
//       {
//         ind += 2 * 512;
//         const uint32_t tHeight = ind / (18 * 4);
//         const uint32_t tWidth = (ind - tHeight * (18 * 4)) / 4; 
//         const uint32_t tVECT_C = ind % 4;
//         data[tHeight][tWidth][tVECT_C] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth][tVECT_C];
//       }
//     }
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     __syncthreads();

//     // load matrices
//     ////////////////////////////////////////////////////////////////////////////////////////////////////////
//     #pragma unroll
//     for (uint32_t ky = 0; ky < kH; ++ky)
//       #pragma unroll
//       for (uint32_t kx = 0; kx < kW; ++kx)
//       {
//         // int8_t weights[16][kH][kW][16];
//         // int8_t data[18][18][16];
//         // wmma::load_matrix_sync(dataMat, (int8_t*)&(data[threadIdx.y + ky][kx]), 16); // 16x16, row-major
//         // wmma::load_matrix_sync(kernelMat, (int8_t*)&(weights[0][ky][kx]), kH*kW*16); // 16x16, col-major
//         wmma::mma_sync(resultMat, dataMat, kernelMat, resultMat, false);
//       }
//   }

//   // scale the matrix here?

//   wmma::store_matrix_sync((int32_t*)&((*result)[tOutputYOffset + threadIdx.y][tOutputXOffset]), 
//                           resultMat, 16, wmma::mem_row_major); // 16x16
// }


template<
  const uint32_t batch,
  const uint32_t inCGroups, const uint32_t outCGroups, 
  const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t outTileH, const uint32_t outTileW
>
__global__ void 
__launch_bounds__(512, 2)
CUDAConv2DForward3x3TensorCoures(
  const int8_t* __restrict__ rdata,  const int8_t*  __restrict__ rkernel, int32_t* __restrict__ rresult
)
{
  // blockIdx
  //        x - outputTile                    |2^31 - 1| 
  //        y - batch                         |2^16 - 1|
  //        z - outCGroup                     |2^16 - 1|

  // threadIdx
  //        x - 32                               |1024|
  //        y - 16 warps for 32 lines            |1024| 

  // output offsets
  constexpr uint32_t hTiles = outH / outTileH;
  constexpr uint32_t wTiles = outW / outTileW;

  // blockIdx.x = [hTile][wTile]
  uint32_t tOutputYOffset = blockIdx.x / wTiles; 
  uint32_t tOutputXOffset = blockIdx.x - tOutputYOffset * wTiles;

  tOutputYOffset *= outTileH;
  tOutputXOffset *= outTileW;

  const uint32_t& tBatch = blockIdx.y;
  const uint32_t& tOutCGroup = blockIdx.z;
  
  // recast input parameters as arrays
  const int8_t (*arrdata)[batch][inCGroups][inH][inW][16] = (const int8_t (*)[batch][inCGroups][inH][inW][16])(rdata);
  const int8_t (*arrkernel)[outCGroups*16][inCGroups][kH][kW][16] = (const int8_t (*)[outCGroups*16][inCGroups][kH][kW][16])(rkernel);
  int32_t (*arrresult)[batch][outCGroups][outH][outW][16] = (int32_t (*)[batch][outCGroups][outH][outW][16])(rresult);

  // pick only relevant slices, recast if needed
  const int4 (*idata)[inCGroups][inH][inW] = (const int4 (*)[inCGroups][inH][inW])(&((*arrdata)[tBatch]));
  const int4 (*kernel)[16][inCGroups][kH][kW] = (const int4 (*)[16][inCGroups][kH][kW])(&((*arrkernel)[16*tOutCGroup]));
  int32_t (*result)[outH][outW][16] = (int32_t (*)[outH][outW][16])(&((*arrresult)[tBatch][tOutCGroup]));

  __shared__ alignas(32) int4 weights[16][kH][kW];
  __shared__ alignas(32) int4 data[34][34];
  
  int4 (*flatWeights)[16][kH*kW] = (int4 (*)[16][kH*kW])(&weights);
  const int4 (*flatKernel)[16][inCGroups][kH*kW] = (const int4 (*)[16][inCGroups][kH*kW])(kernel);

  const uint32_t verticalOffset = threadIdx.y * 2;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> resultMat[2][2]; // 2 lines, 16 channels (each mat = 1 line, 8 channels) 
  #pragma unroll
  for (uint32_t line = 0; line < 2; ++line)
    #pragma unroll
    for (uint32_t chn = 0; chn < 2; ++chn)
      wmma::fill_fragment(resultMat[line][chn], 0);

  // #pragma unroll
  for (uint32_t tInCGroup=0; tInCGroup < inCGroups; ++tInCGroup)
  {
    __syncthreads();
    // load the data, first and second 256 elements
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
    {
      uint32_t ind = 512*i + threadIdx.y * 32 + threadIdx.x;
      const uint32_t tHeight = ind / 34;
      const uint32_t tWidth = ind - tHeight * 34;
      data[tHeight][tWidth] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth];
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    // load the rest of the data and weights. 12 threads are wasted
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
      uint32_t ind = threadIdx.y * 32 + threadIdx.x;
      if (ind < 144)
      {
        uint32_t indt = ind;
        const uint32_t tKernel = indt / (kH * kW);
        const uint32_t tPosition = indt - tKernel * (kH * kW);
        (*flatWeights)[tKernel][tPosition] = (*flatKernel)[tKernel][tInCGroup][tPosition];
      }
      if (ind > 379)
      {
        uint32_t indt = (ind - 380) + 512 * 2;
        const uint32_t tHeight = indt / 34;
        const uint32_t tWidth = indt - tHeight * 34;
        data[tHeight][tWidth] = (*idata)[tInCGroup][tOutputYOffset + tHeight][tOutputXOffset + tWidth];
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    __syncthreads();

    // load matrices, and perform mma.
    //////////////////////////////////////////////////////////////////////////////////////////////////////// 
    // int8_t weights[16][kH][kW][16]; - 16 kernels x kH x kW x 16 bytes.
    // int8_t data[34][34][16];
    // dataMat - 32x16, row-major
    // kernelMat - 16x8, col-major
    {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> dataMat;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> kernelMat[2];

      // by vertical lines
      #pragma unroll
      for (uint32_t kx = 0; kx < kW; ++kx)
      {
        // first line
        wmma::load_matrix_sync(kernelMat[0], (int8_t*)&(weights[0][0][kx]), kH*kW*16);
        wmma::load_matrix_sync(kernelMat[1], (int8_t*)&(weights[8][0][kx]), kH*kW*16);

        wmma::load_matrix_sync(dataMat, (int8_t*)&(data[verticalOffset + 0][kx]), 16); 
        wmma::mma_sync(resultMat[0][0], dataMat, kernelMat[0], resultMat[0][0], false);
        wmma::mma_sync(resultMat[0][1], dataMat, kernelMat[1], resultMat[0][1], false);
      
        // second line
        wmma::load_matrix_sync(dataMat, (int8_t*)&(data[verticalOffset + 1][kx]), 16); 
        wmma::mma_sync(resultMat[1][0], dataMat, kernelMat[0], resultMat[1][0], false);
        wmma::mma_sync(resultMat[1][1], dataMat, kernelMat[1], resultMat[1][1], false);

        wmma::load_matrix_sync(kernelMat[0], (int8_t*)&(weights[0][1][kx]), kH*kW*16);
        wmma::load_matrix_sync(kernelMat[1], (int8_t*)&(weights[8][1][kx]), kH*kW*16);
        wmma::mma_sync(resultMat[0][0], dataMat, kernelMat[0], resultMat[0][0], false);
        wmma::mma_sync(resultMat[0][1], dataMat, kernelMat[1], resultMat[0][1], false);

        //third line
        wmma::load_matrix_sync(dataMat, (int8_t*)&(data[verticalOffset + 2][kx]), 16); 
        wmma::mma_sync(resultMat[1][0], dataMat, kernelMat[0], resultMat[1][0], false);
        wmma::mma_sync(resultMat[1][1], dataMat, kernelMat[1], resultMat[1][1], false);

        wmma::load_matrix_sync(kernelMat[0], (int8_t*)&(weights[0][2][kx]), kH*kW*16);
        wmma::load_matrix_sync(kernelMat[1], (int8_t*)&(weights[8][2][kx]), kH*kW*16);
        wmma::mma_sync(resultMat[0][0], dataMat, kernelMat[0], resultMat[0][0], false);
        wmma::mma_sync(resultMat[0][1], dataMat, kernelMat[1], resultMat[0][1], false);

        // fourth line
        wmma::load_matrix_sync(dataMat, (int8_t*)&(data[verticalOffset + 3][kx]), 16);  
        wmma::mma_sync(resultMat[1][0], dataMat, kernelMat[0], resultMat[1][0], false);
        wmma::mma_sync(resultMat[1][1], dataMat, kernelMat[1], resultMat[1][1], false);
      }
    }
  }

  // scale the matrix here?
  #pragma unroll
  for (uint32_t line = 0; line < 2; ++line)
    #pragma unroll
    for (uint32_t chn = 0; chn < 2; ++chn)
      wmma::store_matrix_sync((int32_t*)&((*result)[tOutputYOffset + verticalOffset + line][tOutputXOffset][chn*8]), 
                              resultMat[line][chn], 16, wmma::mem_row_major); // 16x16
}

template<
  const uint32_t batch,
  const uint32_t inC, const uint32_t outC,
  const uint32_t inH, const uint32_t inW,
  const uint32_t outH, const uint32_t outW
>
std::tuple<Tensor, float> conv2DForward3x3(const Tensor& rinput, const Tensor& rkernel)
{
  constexpr uint32_t resultTileH = 32;
  constexpr uint32_t resultTileW = 32;
  constexpr uint32_t kH = 3;
  constexpr uint32_t kW = 3;  
  constexpr uint32_t inCGroups = inC / 16;
  constexpr uint32_t outCGroups = outC / 16;

  // std::cout << "Input " << rinput.sizes() << std::endl;
  // std::cout << "Kernel " << rkernel.sizes() << std::endl;

  // std::cout << "Input " << rinput << std::endl;

  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::empty({input.size(0), outCGroups, outH, outW, VECT_C},
                    at::TensorOptions().dtype(torch::kInt32)
                                       .device(input.device())
                                       .layout(input.layout())
                                       .is_variable(true));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_ASSERT(outH % resultTileH == 0 && outW % resultTileW == 0 && inC % 16 == 0 && outC % 16 == 0);
  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outCGroups);
  dim3 block_size(32, 16, 1);

  // std::cout << "Output " << output.sizes() << std::endl;
  // std::cout << "Grid " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  // std::cout << "Block " << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;

  // std::cout << "FUCK! " << kernel << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // std::cout << "Started " << std::endl;
  CUDAConv2DForward3x3TensorCoures<batch, inCGroups, outCGroups, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  // std::cout << "Finished " << std::endl;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  // cudaDeviceSetSharedMemConfig(conf);
  return {output, elapsedTime};
}

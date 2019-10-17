#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_61_intrinsics.h>
#include <vector_types.h>

#include <cooperative_groups.h>
#include <cfloat>


constexpr uint32_t VECT_C = 32;
constexpr uint32_t MAX_THREADS_PER_BLOCK = 128;


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


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(512)
Conv2DForward3x3CudaV6(const int8_t* __restrict__ ridata, 
                       const int8_t*  __restrict__ rkernel, 
                       int32_t* __restrict__ routput)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - outC                          |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  const uint32_t tOutCVectCOffset = threadIdx.y;     
  const uint32_t tOutC = blockIdx.z;                // VECT_C included
  
  int32_t (*output)[outH][outW][VECT_C] = (int32_t (*)[outH][outW][VECT_C])(&(routput[blockIdx.y * outC * outH * outW * VECT_C + tOutC * outH * outW * VECT_C]));
  
  // Our results
  int32_t cache0 = 0;
  int32_t cache1 = 0;
  int32_t cache2 = 0;
  int32_t cache3 = 0;

  __shared__ int data[resultTileH + 2][resultTileW + 2][11];

  const int8_t (*kernel)[outC * VECT_C][inC][kH][kW][VECT_C] = (const int8_t (*)[outC * VECT_C][inC][kH][kW][VECT_C])(rkernel);
  const uint32_t threadYOffset = threadIdx.x / 2;
  const uint32_t threadXOffset = (threadIdx.x % 2) * 4;

  #pragma unroll
  for (uint32_t tInC = 0; tInC < inC; ++tInC)   
  {
    const int8_t (*idata)[inH][inW][VECT_C] = (const int8_t (*)[inH][inW][VECT_C])(&(ridata[blockIdx.y * inC * inH * inW * VECT_C + tInC * inH * inW * VECT_C]));
  
    // populate data to shared memory. 
    uint32_t ind, y, x, chn;
    tmp = threadIdx.x + threadIdx.y * 16;
    y = tmp / 80; 
    x = (tmp - y * 80) / 8;
    chn = tmp - y * 80 - x * 8;
    data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));

    if (threadIdx.y < 18)
    {
      tmp = threadIdx.x + threadIdx.y * 16 + 512;
      y = tmp / 80; 
      x = (tmp - y * 80) / 8;
      chn = tmp - y * 80 - x * 8;
      data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
    }
    __syncthreads();


    int32_t d0, d1, d2, d3, d4, d5, d6, d7;
    int4 w1, w2;

    // Load kernel weights. Each thread load the whole kernel, but only 8 int8_t elements
    #pragma unroll
    for (y = 0; y < 2; ++y)
    {
      int4 threadWeights[kW][2];
      #pragma unroll
      for (x = 0; x < kW; ++x)
      {
        threadWeights[x][0] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][0]));
        threadWeights[x][1] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][16]));
      }
      
      d0 = data[y + threadYOffset][0 + threadXOffset][0];
      d1 = data[y + threadYOffset][0 + threadXOffset][1];
      d2 = data[y + threadYOffset][0 + threadXOffset][2];
      d3 = data[y + threadYOffset][0 + threadXOffset][3];
      d4 = data[y + threadYOffset][0 + threadXOffset][4];
      d5 = data[y + threadYOffset][0 + threadXOffset][5];
      d6 = data[y + threadYOffset][0 + threadXOffset][6];
      d7 = data[y + threadYOffset][0 + threadXOffset][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadYOffset][1 + threadXOffset][0];
      d1 = data[y + threadYOffset][1 + threadXOffset][1];
      d2 = data[y + threadYOffset][1 + threadXOffset][2];
      d3 = data[y + threadYOffset][1 + threadXOffset][3];
      d4 = data[y + threadYOffset][1 + threadXOffset][4];
      d5 = data[y + threadYOffset][1 + threadXOffset][5];
      d6 = data[y + threadYOffset][1 + threadXOffset][6];
      d7 = data[y + threadYOffset][1 + threadXOffset][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadYOffset][2 + threadXOffset][0];
      d1 = data[y + threadYOffset][2 + threadXOffset][1];
      d2 = data[y + threadYOffset][2 + threadXOffset][2];
      d3 = data[y + threadYOffset][2 + threadXOffset][3];
      d4 = data[y + threadYOffset][2 + threadXOffset][4];
      d5 = data[y + threadYOffset][2 + threadXOffset][5];
      d6 = data[y + threadYOffset][2 + threadXOffset][6];
      d7 = data[y + threadYOffset][2 + threadXOffset][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadYOffset][3 + threadXOffset][0];
      d1 = data[y + threadYOffset][3 + threadXOffset][1];
      d2 = data[y + threadYOffset][3 + threadXOffset][2];
      d3 = data[y + threadYOffset][3 + threadXOffset][3];
      d4 = data[y + threadYOffset][3 + threadXOffset][4];
      d5 = data[y + threadYOffset][3 + threadXOffset][5];
      d6 = data[y + threadYOffset][3 + threadXOffset][6];
      d7 = data[y + threadYOffset][3 + threadXOffset][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      d0 = data[y + threadYOffset][4 + threadXOffset][0];
      d1 = data[y + threadYOffset][4 + threadXOffset][1];
      d2 = data[y + threadYOffset][4 + threadXOffset][2];
      d3 = data[y + threadYOffset][4 + threadXOffset][3];
      d4 = data[y + threadYOffset][4 + threadXOffset][4];
      d5 = data[y + threadYOffset][4 + threadXOffset][5];
      d6 = data[y + threadYOffset][4 + threadXOffset][6];
      d7 = data[y + threadYOffset][4 + threadXOffset][7];
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      d0 = data[y + threadYOffset][5 + threadXOffset][0];
      d1 = data[y + threadYOffset][5 + threadXOffset][1];
      d2 = data[y + threadYOffset][5 + threadXOffset][2];
      d3 = data[y + threadYOffset][5 + threadXOffset][3];
      d4 = data[y + threadYOffset][5 + threadXOffset][4];
      d5 = data[y + threadYOffset][5 + threadXOffset][5];
      d6 = data[y + threadYOffset][5 + threadXOffset][6];
      d7 = data[y + threadYOffset][5 + threadXOffset][7];
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
    }
  }
  
  (*output)[outputYOffset + threadYOffset][outputXOffset + threadXOffset + 0][tOutCVectCOffset] = cache0;
  (*output)[outputYOffset + threadYOffset][outputXOffset + threadXOffset + 1][tOutCVectCOffset] = cache1;
  (*output)[outputYOffset + threadYOffset][outputXOffset + threadXOffset + 2][tOutCVectCOffset] = cache2;
  (*output)[outputYOffset + threadYOffset][outputXOffset + threadXOffset + 3][tOutCVectCOffset] = cache3;
}


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(256, 4)
Conv2DForward3x3CudaV5(const int8_t* __restrict__ ridata, 
                       const int8_t*  __restrict__ rkernel, 
                       int32_t* __restrict__ routput)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - outC                    |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  const uint32_t tOutCVectCOffset = threadIdx.y;     
  const uint32_t tOutC = blockIdx.z;                // VECT_C included
  
  int32_t (*output)[outH][outW][VECT_C] = (int32_t (*)[outH][outW][VECT_C])(&(routput[blockIdx.y * outC * outH * outW * VECT_C + tOutC * outH * outW * VECT_C]));
  
  // Our results
  int32_t cache0 = 0;
  int32_t cache1 = 0;
  int32_t cache2 = 0;
  int32_t cache3 = 0;
  int32_t cache4 = 0;
  int32_t cache5 = 0;
  int32_t cache6 = 0;
  int32_t cache7 = 0;
  __shared__ int data[resultTileH + 2][resultTileW + 2][11];

  #pragma unroll
  for (uint32_t tInC = 0; tInC < inC; ++tInC)   
  {
    const int8_t (*idata)[inH][inW][VECT_C] = (const int8_t (*)[inH][inW][VECT_C])(&(ridata[blockIdx.y * inC * inH * inW * VECT_C + tInC * inH * inW * VECT_C]));
    const int8_t (*kernel)[kH][kW][VECT_C] = (const int8_t (*)[kH][kW][VECT_C])(&(rkernel[(tOutC + tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C]));

    // populate data to shared memory. 
    uint32_t y, x, chn;
    #pragma unroll
    for (uint32_t ind = 0; ind < 3; ++ind)
    {
      tmp = threadIdx.x + threadIdx.y * 8 + 256 * ind;
      y = tmp / 80; 
      x = (tmp - y * 80) / 8;
      chn = tmp - y * 80 - x * 8;
      data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
    }

    if (threadIdx.y < 4)
    {
      tmp = threadIdx.x + threadIdx.y * 8 + 256 * 3;
      y = tmp / 80; 
      x = (tmp - y * 80) / 8;
      chn = tmp - y * 80 - x * 8;
      data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
    }
    __syncthreads();

    int32_t d0, d1, d2, d3, d4, d5, d6, d7;
    int4 w1, w2;

    // Load kernel weights. Each thread load the whole kernel, but only 8 int8_t elements
    #pragma unroll
    for (y = 0; y < 2; ++y)
    {
      int4 threadWeights[kW][2];
      #pragma unroll
      for (x = 0; x < kW; ++x)
      {
        threadWeights[x][0] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][0]));
        threadWeights[x][1] = *reinterpret_cast<const int4*>(&((*kernel)[y][x][16]));
      }
      
      d0 = data[y + threadIdx.x][0][0];
      d1 = data[y + threadIdx.x][0][1];
      d2 = data[y + threadIdx.x][0][2];
      d3 = data[y + threadIdx.x][0][3];
      d4 = data[y + threadIdx.x][0][4];
      d5 = data[y + threadIdx.x][0][5];
      d6 = data[y + threadIdx.x][0][6];
      d7 = data[y + threadIdx.x][0][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadIdx.x][1][0];
      d1 = data[y + threadIdx.x][1][1];
      d2 = data[y + threadIdx.x][1][2];
      d3 = data[y + threadIdx.x][1][3];
      d4 = data[y + threadIdx.x][1][4];
      d5 = data[y + threadIdx.x][1][5];
      d6 = data[y + threadIdx.x][1][6];
      d7 = data[y + threadIdx.x][1][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadIdx.x][2][0];
      d1 = data[y + threadIdx.x][2][1];
      d2 = data[y + threadIdx.x][2][2];
      d3 = data[y + threadIdx.x][2][3];
      d4 = data[y + threadIdx.x][2][4];
      d5 = data[y + threadIdx.x][2][5];
      d6 = data[y + threadIdx.x][2][6];
      d7 = data[y + threadIdx.x][2][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache0 = __dp4a(d0, w1.x, cache0);
      cache0 = __dp4a(d1, w1.y, cache0);
      cache0 = __dp4a(d2, w1.w, cache0);
      cache0 = __dp4a(d3, w1.z, cache0);
      cache0 = __dp4a(d4, w2.x, cache0);
      cache0 = __dp4a(d5, w2.y, cache0);
      cache0 = __dp4a(d6, w2.w, cache0);
      cache0 = __dp4a(d7, w2.z, cache0);
      d0 = data[y + threadIdx.x][3][0];
      d1 = data[y + threadIdx.x][3][1];
      d2 = data[y + threadIdx.x][3][2];
      d3 = data[y + threadIdx.x][3][3];
      d4 = data[y + threadIdx.x][3][4];
      d5 = data[y + threadIdx.x][3][5];
      d6 = data[y + threadIdx.x][3][6];
      d7 = data[y + threadIdx.x][3][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache1 = __dp4a(d0, w1.x, cache1);
      cache1 = __dp4a(d1, w1.y, cache1);
      cache1 = __dp4a(d2, w1.w, cache1);
      cache1 = __dp4a(d3, w1.z, cache1);
      cache1 = __dp4a(d4, w2.x, cache1);
      cache1 = __dp4a(d5, w2.y, cache1);
      cache1 = __dp4a(d6, w2.w, cache1);
      cache1 = __dp4a(d7, w2.z, cache1);
      d0 = data[y + threadIdx.x][4][0];
      d1 = data[y + threadIdx.x][4][1];
      d2 = data[y + threadIdx.x][4][2];
      d3 = data[y + threadIdx.x][4][3];
      d4 = data[y + threadIdx.x][4][4];
      d5 = data[y + threadIdx.x][4][5];
      d6 = data[y + threadIdx.x][4][6];
      d7 = data[y + threadIdx.x][4][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache4 = __dp4a(d0, w1.x, cache4);
      cache4 = __dp4a(d1, w1.y, cache4);
      cache4 = __dp4a(d2, w1.w, cache4);
      cache4 = __dp4a(d3, w1.z, cache4);
      cache4 = __dp4a(d4, w2.x, cache4);
      cache4 = __dp4a(d5, w2.y, cache4);
      cache4 = __dp4a(d6, w2.w, cache4);
      cache4 = __dp4a(d7, w2.z, cache4);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache2 = __dp4a(d0, w1.x, cache2);
      cache2 = __dp4a(d1, w1.y, cache2);
      cache2 = __dp4a(d2, w1.w, cache2);
      cache2 = __dp4a(d3, w1.z, cache2);
      cache2 = __dp4a(d4, w2.x, cache2);
      cache2 = __dp4a(d5, w2.y, cache2);
      cache2 = __dp4a(d6, w2.w, cache2);
      cache2 = __dp4a(d7, w2.z, cache2);
      d0 = data[y + threadIdx.x][5][0];
      d1 = data[y + threadIdx.x][5][1];
      d2 = data[y + threadIdx.x][5][2];
      d3 = data[y + threadIdx.x][5][3];
      d4 = data[y + threadIdx.x][5][4];
      d5 = data[y + threadIdx.x][5][5];
      d6 = data[y + threadIdx.x][5][6];
      d7 = data[y + threadIdx.x][5][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache5 = __dp4a(d0, w1.x, cache5);
      cache5 = __dp4a(d1, w1.y, cache5);
      cache5 = __dp4a(d2, w1.w, cache5);
      cache5 = __dp4a(d3, w1.z, cache5);
      cache5 = __dp4a(d4, w2.x, cache5);
      cache5 = __dp4a(d5, w2.y, cache5);
      cache5 = __dp4a(d6, w2.w, cache5);
      cache5 = __dp4a(d7, w2.z, cache5);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache4 = __dp4a(d0, w1.x, cache4);
      cache4 = __dp4a(d1, w1.y, cache4);
      cache4 = __dp4a(d2, w1.w, cache4);
      cache4 = __dp4a(d3, w1.z, cache4);
      cache4 = __dp4a(d4, w2.x, cache4);
      cache4 = __dp4a(d5, w2.y, cache4);
      cache4 = __dp4a(d6, w2.w, cache4);
      cache4 = __dp4a(d7, w2.z, cache4);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache3 = __dp4a(d0, w1.x, cache3);
      cache3 = __dp4a(d1, w1.y, cache3);
      cache3 = __dp4a(d2, w1.w, cache3);
      cache3 = __dp4a(d3, w1.z, cache3);
      cache3 = __dp4a(d4, w2.x, cache3);
      cache3 = __dp4a(d5, w2.y, cache3);
      cache3 = __dp4a(d6, w2.w, cache3);
      cache3 = __dp4a(d7, w2.z, cache3);
      d0 = data[y + threadIdx.x][6][0];
      d1 = data[y + threadIdx.x][6][1];
      d2 = data[y + threadIdx.x][6][2];
      d3 = data[y + threadIdx.x][6][3];
      d4 = data[y + threadIdx.x][6][4];
      d5 = data[y + threadIdx.x][6][5];
      d6 = data[y + threadIdx.x][6][6];
      d7 = data[y + threadIdx.x][6][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache6 = __dp4a(d0, w1.x, cache6);
      cache6 = __dp4a(d1, w1.y, cache6);
      cache6 = __dp4a(d2, w1.w, cache6);
      cache6 = __dp4a(d3, w1.z, cache6);
      cache6 = __dp4a(d4, w2.x, cache6);
      cache6 = __dp4a(d5, w2.y, cache6);
      cache6 = __dp4a(d6, w2.w, cache6);
      cache6 = __dp4a(d7, w2.z, cache6);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache5 = __dp4a(d0, w1.x, cache5);
      cache5 = __dp4a(d1, w1.y, cache5);
      cache5 = __dp4a(d2, w1.w, cache5);
      cache5 = __dp4a(d3, w1.z, cache5);
      cache5 = __dp4a(d4, w2.x, cache5);
      cache5 = __dp4a(d5, w2.y, cache5);
      cache5 = __dp4a(d6, w2.w, cache5);
      cache5 = __dp4a(d7, w2.z, cache5);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache4 = __dp4a(d0, w1.x, cache4);
      cache4 = __dp4a(d1, w1.y, cache4);
      cache4 = __dp4a(d2, w1.w, cache4);
      cache4 = __dp4a(d3, w1.z, cache4);
      cache4 = __dp4a(d4, w2.x, cache4);
      cache4 = __dp4a(d5, w2.y, cache4);
      cache4 = __dp4a(d6, w2.w, cache4);
      cache4 = __dp4a(d7, w2.z, cache4);
      d0 = data[y + threadIdx.x][7][0];
      d1 = data[y + threadIdx.x][7][1];
      d2 = data[y + threadIdx.x][7][2];
      d3 = data[y + threadIdx.x][7][3];
      d4 = data[y + threadIdx.x][7][4];
      d5 = data[y + threadIdx.x][7][5];
      d6 = data[y + threadIdx.x][7][6];
      d7 = data[y + threadIdx.x][7][7];
      w1 = threadWeights[0][0];
      w2 = threadWeights[0][1];
      cache7 = __dp4a(d0, w1.x, cache7);
      cache7 = __dp4a(d1, w1.y, cache7);
      cache7 = __dp4a(d2, w1.w, cache7);
      cache7 = __dp4a(d3, w1.z, cache7);
      cache7 = __dp4a(d4, w2.x, cache7);
      cache7 = __dp4a(d5, w2.y, cache7);
      cache7 = __dp4a(d6, w2.w, cache7);
      cache7 = __dp4a(d7, w2.z, cache7);
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache6 = __dp4a(d0, w1.x, cache6);
      cache6 = __dp4a(d1, w1.y, cache6);
      cache6 = __dp4a(d2, w1.w, cache6);
      cache6 = __dp4a(d3, w1.z, cache6);
      cache6 = __dp4a(d4, w2.x, cache6);
      cache6 = __dp4a(d5, w2.y, cache6);
      cache6 = __dp4a(d6, w2.w, cache6);
      cache6 = __dp4a(d7, w2.z, cache6);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache5 = __dp4a(d0, w1.x, cache5);
      cache5 = __dp4a(d1, w1.y, cache5);
      cache5 = __dp4a(d2, w1.w, cache5);
      cache5 = __dp4a(d3, w1.z, cache5);
      cache5 = __dp4a(d4, w2.x, cache5);
      cache5 = __dp4a(d5, w2.y, cache5);
      cache5 = __dp4a(d6, w2.w, cache5);
      cache5 = __dp4a(d7, w2.z, cache5);
      d0 = data[y + threadIdx.x][8][0];
      d1 = data[y + threadIdx.x][8][1];
      d2 = data[y + threadIdx.x][8][2];
      d3 = data[y + threadIdx.x][8][3];
      d4 = data[y + threadIdx.x][8][4];
      d5 = data[y + threadIdx.x][8][5];
      d6 = data[y + threadIdx.x][8][6];
      d7 = data[y + threadIdx.x][8][7];
      w1 = threadWeights[1][0];
      w2 = threadWeights[1][1];
      cache7 = __dp4a(d0, w1.x, cache7);
      cache7 = __dp4a(d1, w1.y, cache7);
      cache7 = __dp4a(d2, w1.w, cache7);
      cache7 = __dp4a(d3, w1.z, cache7);
      cache7 = __dp4a(d4, w2.x, cache7);
      cache7 = __dp4a(d5, w2.y, cache7);
      cache7 = __dp4a(d6, w2.w, cache7);
      cache7 = __dp4a(d7, w2.z, cache7);
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache6 = __dp4a(d0, w1.x, cache6);
      cache6 = __dp4a(d1, w1.y, cache6);
      cache6 = __dp4a(d2, w1.w, cache6);
      cache6 = __dp4a(d3, w1.z, cache6);
      cache6 = __dp4a(d4, w2.x, cache6);
      cache6 = __dp4a(d5, w2.y, cache6);
      cache6 = __dp4a(d6, w2.w, cache6);
      cache6 = __dp4a(d7, w2.z, cache6);
      d0 = data[y + threadIdx.x][9][0];
      d1 = data[y + threadIdx.x][9][1];
      d2 = data[y + threadIdx.x][9][2];
      d3 = data[y + threadIdx.x][9][3];
      d4 = data[y + threadIdx.x][9][4];
      d5 = data[y + threadIdx.x][9][5];
      d6 = data[y + threadIdx.x][9][6];
      d7 = data[y + threadIdx.x][9][7];
      w1 = threadWeights[2][0];
      w2 = threadWeights[2][1];
      cache7 = __dp4a(d0, w1.x, cache7);
      cache7 = __dp4a(d1, w1.y, cache7);
      cache7 = __dp4a(d2, w1.w, cache7);
      cache7 = __dp4a(d3, w1.z, cache7);
      cache7 = __dp4a(d4, w2.x, cache7);
      cache7 = __dp4a(d5, w2.y, cache7);
      cache7 = __dp4a(d6, w2.w, cache7);
      cache7 = __dp4a(d7, w2.z, cache7);
    }
  }

  (*output)[outputYOffset + threadIdx.x][outputXOffset + 0][tOutCVectCOffset] = cache0;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 1][tOutCVectCOffset] = cache1;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 2][tOutCVectCOffset] = cache2;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 3][tOutCVectCOffset] = cache3;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 4][tOutCVectCOffset] = cache4;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 5][tOutCVectCOffset] = cache5;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 6][tOutCVectCOffset] = cache6;
  (*output)[outputYOffset + threadIdx.x][outputXOffset + 7][tOutCVectCOffset] = cache7;
}


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK, 8)
Conv2DForward3x3CudaV4(const int8_t* __restrict__ ridata, 
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

  const uint32_t tOutCVectCOffset = threadIdx.y;          

  // channels
  const uint32_t tInC = blockIdx.z / outC;          // VECT_C included
  const uint32_t tOutC = blockIdx.z - tInC * outC;  // VECT_C included

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;


  // recast for convinience. Take into consideration target batch dimension and channel dimension
  const int8_t (*idata)[inH][inW][VECT_C] = (const int8_t (*)[inH][inW][VECT_C])(&(ridata[blockIdx.y * inC * inH * inW * VECT_C + tInC * inH * inW * VECT_C]));
  int32_t (*output)[outH][outW][VECT_C] = (int32_t (*)[outH][outW][VECT_C])(&(routput[blockIdx.y * outC * outH * outW * VECT_C + tOutC * outH * outW * VECT_C]));
  const int8_t (*kernel)[kH][kW][VECT_C] = (const int8_t (*)[kH][kW][VECT_C])(&(rkernel[(tOutC + tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C]));

  // 10x10
  // __shared__ int data[resultTileH + 2][resultTileW + 2][11];  // 11, only 4 collisions per warp
  __shared__ int data[resultTileH + 2][resultTileW + 2][9];
  // populate data to shared memory. 

  uint32_t y, x, chn;
  tmp = threadIdx.x + threadIdx.y * 4;
  if (tmp < 100)
  {

    int4 dl1, dl2;
    y = tmp / 10; 
    x = (tmp - y * 10);

    dl1 = *reinterpret_cast<const int4*>(&((*idata)[outputYOffset+y][outputXOffset+x][0]));
    dl2 = *reinterpret_cast<const int4*>(&((*idata)[outputYOffset+y][outputXOffset+x][16]));
    data[y][x][0] = dl1.x;
    data[y][x][1] = dl1.y;
    data[y][x][2] = dl1.w;
    data[y][x][3] = dl1.z;
    data[y][x][4] = dl2.x;
    data[y][x][5] = dl2.y;
    data[y][x][6] = dl2.w;
    data[y][x][7] = dl2.z;
  }
  
  // uint32_t y, x, chn;
  // #pragma unroll
  // for (uint32_t ind = 0; ind < 6; ++ind)
  // {
  //   tmp = threadIdx.x + threadIdx.y * 4 + 128 * ind;
  //   y = tmp / 80; 
  //   x = (tmp - y * 80) / 8;
  //   chn = tmp - y * 80 - x * 8;
  //   data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
  // }

  // if (threadIdx.y < 8)
  // {
  //   tmp = threadIdx.x + threadIdx.y * 4 + 128 * 6;
  //   y = tmp / 80; 
  //   x = (tmp - y * 80) / 8;
  //   chn = tmp - y * 80 - x * 8;
  //   data[y][x][chn] = *reinterpret_cast<const int32_t*>(&((*idata)[outputYOffset+y][outputXOffset+x][4 * chn]));
  // }

  // int4 d;
  // tmp = threadIdx.x + threadIdx.y * 4;
  // uint32_t y = tmp / 20; 
  // uint32_t x = (tmp - y * 20) / 2;
  // uint32_t chn = tmp - y * 20 - x * 2;

  // d = *reinterpret_cast<const int4*>(&((*idata)[outputYOffset+y][outputXOffset+x][16 * chn]));
  // data[y][x][4*chn] = d.x;
  // data[y][x][4*chn + 1] = d.y;
  // data[y][x][4*chn + 2] = d.w;
  // data[y][x][4*chn + 3] = d.z;

  // if (tmp < 72)
  // {
  //   tmp += 128;
  //   y = tmp / 20; 
  //   x = (tmp - y * 20) / 2;
  //   chn = tmp - y * 20 - x * 2;

  //   d = *reinterpret_cast<const int4*>(&((*idata)[outputYOffset+y][outputXOffset+x][16 * chn]));
  //   data[y][x][4*chn] = d.x;
  //   data[y][x][4*chn + 1] = d.y;
  //   data[y][x][4*chn + 2] = d.w;
  //   data[y][x][4*chn + 3] = d.z;
  // }
  // data loaded, sync threads
  __syncthreads();

  // Load kernel weights. Each thread load the whole kernel, but only 8 int8_t elements
  int2 threadWeights[kH][kW];

  #pragma unroll
  for (y = 0; y < kH; ++y)
    #pragma unroll
      for (x = 0; x < kW; ++x)
        threadWeights[y][x] = *reinterpret_cast<const int2*>(&((*kernel)[y][x][8 * threadIdx.x]));

  int d1, d2;
  int2 w;

  #pragma unroll
  for (y = 0; y < 8; ++y)
  {
    int32_t cache0 = 0;
    int32_t cache1 = 0;
    int32_t cache2 = 0;
    int32_t cache3 = 0;
    int32_t cache4 = 0;
    int32_t cache5 = 0;
    int32_t cache6 = 0;
    int32_t cache7 = 0;

    d1 = data[0 + y][0][2 * threadIdx.x];
    d2 = data[0 + y][0][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[0 + y][1][2 * threadIdx.x];
    d2 = data[0 + y][1][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[0][1];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[0 + y][2][2 * threadIdx.x];
    d2 = data[0 + y][2][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[0][1];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[0][2];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[0 + y][3][2 * threadIdx.x];
    d2 = data[0 + y][3][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[0][1];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[0][2];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    d1 = data[0 + y][4][2 * threadIdx.x];
    d2 = data[0 + y][4][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[0][1];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[0][2];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    d1 = data[0 + y][5][2 * threadIdx.x];
    d2 = data[0 + y][5][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[0][1];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[0][2];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    d1 = data[0 + y][6][2 * threadIdx.x];
    d2 = data[0 + y][6][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[0][1];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[0][2];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    d1 = data[0 + y][7][2 * threadIdx.x];
    d2 = data[0 + y][7][2 * threadIdx.x + 1];
    w = threadWeights[0][0];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[0][1];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[0][2];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    d1 = data[0 + y][8][2 * threadIdx.x];
    d2 = data[0 + y][8][2 * threadIdx.x + 1];
    w = threadWeights[0][1];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[0][2];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    d1 = data[0 + y][9][2 * threadIdx.x];
    d2 = data[0 + y][9][2 * threadIdx.x + 1];
    w = threadWeights[0][2];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    d1 = data[1 + y][0][2 * threadIdx.x];
    d2 = data[1 + y][0][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[1 + y][1][2 * threadIdx.x];
    d2 = data[1 + y][1][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[1][1];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[1 + y][2][2 * threadIdx.x];
    d2 = data[1 + y][2][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[1][1];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[1][2];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[1 + y][3][2 * threadIdx.x];
    d2 = data[1 + y][3][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[1][1];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[1][2];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    d1 = data[1 + y][4][2 * threadIdx.x];
    d2 = data[1 + y][4][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[1][1];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[1][2];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    d1 = data[1 + y][5][2 * threadIdx.x];
    d2 = data[1 + y][5][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[1][1];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[1][2];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    d1 = data[1 + y][6][2 * threadIdx.x];
    d2 = data[1 + y][6][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[1][1];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[1][2];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    d1 = data[1 + y][7][2 * threadIdx.x];
    d2 = data[1 + y][7][2 * threadIdx.x + 1];
    w = threadWeights[1][0];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[1][1];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[1][2];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    d1 = data[1 + y][8][2 * threadIdx.x];
    d2 = data[1 + y][8][2 * threadIdx.x + 1];
    w = threadWeights[1][1];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[1][2];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    d1 = data[1 + y][9][2 * threadIdx.x];
    d2 = data[1 + y][9][2 * threadIdx.x + 1];
    w = threadWeights[1][2];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    d1 = data[2 + y][0][2 * threadIdx.x];
    d2 = data[2 + y][0][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[2 + y][1][2 * threadIdx.x];
    d2 = data[2 + y][1][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[2][1];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[2 + y][2][2 * threadIdx.x];
    d2 = data[2 + y][2][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[2][1];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    w = threadWeights[2][2];
    cache0 = __dp4a(d1, w.x, cache0);
    cache0 = __dp4a(d2, w.y, cache0);
    d1 = data[2 + y][3][2 * threadIdx.x];
    d2 = data[2 + y][3][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[2][1];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    w = threadWeights[2][2];
    cache1 = __dp4a(d1, w.x, cache1);
    cache1 = __dp4a(d2, w.y, cache1);
    d1 = data[2 + y][4][2 * threadIdx.x];
    d2 = data[2 + y][4][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[2][1];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    w = threadWeights[2][2];
    cache2 = __dp4a(d1, w.x, cache2);
    cache2 = __dp4a(d2, w.y, cache2);
    d1 = data[2 + y][5][2 * threadIdx.x];
    d2 = data[2 + y][5][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[2][1];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    w = threadWeights[2][2];
    cache3 = __dp4a(d1, w.x, cache3);
    cache3 = __dp4a(d2, w.y, cache3);
    d1 = data[2 + y][6][2 * threadIdx.x];
    d2 = data[2 + y][6][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[2][1];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    w = threadWeights[2][2];
    cache4 = __dp4a(d1, w.x, cache4);
    cache4 = __dp4a(d2, w.y, cache4);
    d1 = data[2 + y][7][2 * threadIdx.x];
    d2 = data[2 + y][7][2 * threadIdx.x + 1];
    w = threadWeights[2][0];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[2][1];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    w = threadWeights[2][2];
    cache5 = __dp4a(d1, w.x, cache5);
    cache5 = __dp4a(d2, w.y, cache5);
    d1 = data[2 + y][8][2 * threadIdx.x];
    d2 = data[2 + y][8][2 * threadIdx.x + 1];
    w = threadWeights[2][1];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);
    w = threadWeights[2][2];
    cache6 = __dp4a(d1, w.x, cache6);
    cache6 = __dp4a(d2, w.y, cache6);
    d1 = data[2 + y][9][2 * threadIdx.x];
    d2 = data[2 + y][9][2 * threadIdx.x + 1];
    w = threadWeights[2][2];
    cache7 = __dp4a(d1, w.x, cache7);
    cache7 = __dp4a(d2, w.y, cache7);

    // atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 0][tOutCVectCOffset]), cache0+cache1+cache2+cache3+cache4+cache5+cache6+cache7);

    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 0][tOutCVectCOffset]), cache0);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 1][tOutCVectCOffset]), cache1);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 2][tOutCVectCOffset]), cache2);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 3][tOutCVectCOffset]), cache3);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 4][tOutCVectCOffset]), cache4);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 5][tOutCVectCOffset]), cache5);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 6][tOutCVectCOffset]), cache6);
    atomicAdd(&((*output)[outputYOffset + y][outputXOffset + 7][tOutCVectCOffset]), cache7);
  }
}


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK)
Conv2DForward3x3CudaV3(const int8_t* __restrict__ idata, 
                       const int8_t*  __restrict__ kernel, 
                       int32_t* __restrict__ output)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - inC * outC                    |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  
  
  const uint32_t tOutCVectCOffset = threadIdx.y;          

  // channels
  const uint32_t tInC = blockIdx.z / outC;          // VECT_C included
  const uint32_t tOutC = blockIdx.z - tInC * outC;  // VECT_C included

  // output offsets
  const uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  // Load kernel weights. Each thread load the whole kernel
  int4 threadWeights[kH][kW][2];
  #pragma unroll
  for (uint32_t y = 0; y < kH; ++y)
    #pragma unroll
      for (uint32_t x = 0; x < kW; ++x)
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++i)
          threadWeights[y][x][i] = 
                  *reinterpret_cast<const int4*>(&kernel[KERNEL_INDEX(inC, kH, kW, tOutC*VECT_C+tOutCVectCOffset, tInC, y, x) + 16*i]);

  // 8x8
  __shared__ int2 data[resultTileH + 2][resultTileW + 2][4];
  
  // populate data to shared memory. First 4 lines with 8 elements
  uint32_t y = threadIdx.y / (resultTileH + 2); 
  uint32_t x = threadIdx.y - y * (resultTileH + 2);

  data[y][x][threadIdx.x] = *reinterpret_cast<const int2*>(&idata[
    DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*threadIdx.x
  ]);

  // second 4 lines
  y += 4;
  data[y][x][threadIdx.x] = *reinterpret_cast<const int2*>(&idata[
    DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*threadIdx.x
  ]);

  // data loaded, sync threads
  __syncthreads();

  #pragma unroll
  for (uint32_t ind = threadIdx.x; ind < resultTileH * resultTileW; ind += 4)
  {
    y = ind / resultTileH;
    x = ind - y * resultTileH;
    
    int32_t cache = 0;
    #pragma unroll
      for (uint32_t ky = 0; ky < kH; ++ky)
        #pragma unroll
        for (uint32_t kx = 0; kx < kW; ++kx)
          {
            auto w = threadWeights[ky][kx][0];
            auto d11 = data[y+ky][x+kx][0];
            auto d12 = data[y+ky][x+kx][1];

            cache = __dp4a(d11.x, w.x, cache);
            cache = __dp4a(d11.y, w.y, cache);
            cache = __dp4a(d12.x, w.z, cache);
            cache = __dp4a(d12.y, w.w, cache);

            w = threadWeights[ky][kx][1];
            auto d21 = data[y+ky][x+kx][2];
            auto d22 = data[y+ky][x+kx][3];

            cache = __dp4a(d21.x, w.x, cache);
            cache = __dp4a(d21.y, w.y, cache);
            cache = __dp4a(d22.x, w.z, cache);
            cache = __dp4a(d22.y, w.w, cache);
          }

    atomicAdd(&output[DATA_INDEX(outC, outH, outW, blockIdx.y, tOutC, outputYOffset+y, outputXOffset+x) + tOutCVectCOffset], cache);
  }
}



template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK)
Conv2DForward3x3CudaV2(const int8_t* __restrict__ idata, 
                       const int8_t*  __restrict__ kernel, 
                       int32_t* __restrict__ output)
{
  // grid
  //        x - resultTile                    |2^31 - 1| Everything must be nice here
  //        y - batch                         |2^16 - 1| Simple and straight forward
  //        z - inC * outC                    |2^16 - 1| Input and output channels, VECT_C included

  // block
  //        x - 4 threads to simultaniously convolve 2 patches       |1024| 
  //        y - channel offset for 32 output channels group          |1024|  
  
  const uint32_t tOutCVectCOffset = threadIdx.y;          

  // channels
  const uint32_t tInC = blockIdx.z / outC;          // VECT_C included
  const uint32_t tOutC = blockIdx.z - tInC * outC;  // VECT_C included

  // output offsets
  uint32_t tmp = blockIdx.x / (outH / resultTileH); // Tile vertical number
  const uint32_t outputYOffset = tmp * resultTileH; 
  const uint32_t outputXOffset = (blockIdx.x - tmp * (outH / resultTileH)) * resultTileW;

  tmp = threadIdx.x; // 0, 1, 2, 3 to decide where to look for the kernel
                     // Each 8 bytes are loaded only once

  // printf("GRID DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // printf("BLOCK DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // printf("COMPUTED %d %d %d %d %d %d\n", tBatch, tOutCVectCOffset, tInC, tOutC, outputYOffset, outputXOffset);

  // Load kernel weights
  // uint32_t offset = (tOutC*VECT_C+tOutCVectCOffset)*inC*kH*kW*VECT_C + tInC*kH*kW*VECT_C;
  int2 threadWeights[kH][kW];
  #pragma unroll
  for (uint32_t y = 0; y < kH; ++y)
    #pragma unroll
      for (uint32_t x = 0; x < kW; ++x)
        threadWeights[y][x] = *reinterpret_cast<const int2*>(&kernel[KERNEL_INDEX(inC, kH, kW, tOutC*VECT_C+tOutCVectCOffset, tInC, y, x) + 8*tmp]);
        // threadWeights[y][x] = *reinterpret_cast<const int2*>(&kernel[offset  + y*kW*VECT_C + x*VECT_C + 8*tmp]);
            // *((int2 *)&) = ;

  // 8x8
  __shared__ int2 data[resultTileH + 2][resultTileW + 2][4];
  __shared__ int32_t result[resultTileH][resultTileW][VECT_C];
  
  #pragma unroll
  for (uint32_t i=threadIdx.x; i<resultTileH*resultTileW; i+=4)
  {
    const uint32_t rY = i / 6;
    const uint32_t rX = i - rY * 6;
    result[rY][rX][threadIdx.y] = 0;
  }
  
  // populate data to shared memory. First 4 lines with 8 elements
  uint32_t y = threadIdx.y / (resultTileH + 2); 
  uint32_t x = threadIdx.y - y * (resultTileH + 2);

  data[y][x][tmp] = *reinterpret_cast<const int2*>(&idata[
    DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*tmp
  ]);

  y += 4;
  data[y][x][tmp] = *reinterpret_cast<const int2*>(&idata[
    DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*tmp
  ]);
  // auto d = *reinterpret_cast<int2*>(&idata[
  //   DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*tmp
  // ]);
  // data[y][x][tmp].x = d.x;
  // data[y][x][tmp].y = d.y;

  // offset = blockIdx.y*inC*inH*inW*VECT_C + tInC*inH*inW*VECT_C;// + tH*W*VECT_C + tW*VECT_C;;
  // data[y][x][tmp] = *(reinterpret_cast<const int2*>(&idata[offset + (outputYOffset+y)*inW*VECT_C + (outputXOffset+x)*VECT_C + 8*tmp]));

  // data[y][x][tmp] = *reinterpret_cast<const int2*>(&idata[
  //   DATA_INDEX(inC, inH, inW,  blockIdx.y, tInC, outputYOffset+y, outputXOffset+x) + 8*tmp
  // ]);
  // second 4 lines
  // data[y][x][tmp] = *(reinterpret_cast<const int2*>(&idata[offset + (outputYOffset+y)*inW*VECT_C + (outputXOffset+x)*VECT_C + 8*tmp]));

  __syncthreads();


  // offset = blockIdx.y*outC*outH*outW*VECT_C + tOutC*outH*outW*VECT_C;// + tH*W*VECT_C + tW*VECT_C;;
  #pragma unroll
  for (y = 0; y < resultTileH; ++y)
    #pragma unroll
    for (x = 0; x < resultTileW; ++x)
    {
      int32_t cache = 0;
      #pragma unroll
        for (uint32_t ky = 0; ky < kH; ++ky)
          #pragma unroll
          for (uint32_t kx = 0; kx < kW; ++kx)
            {
              // const int8_t d0 = *((int8_t*)&data[y+ky][x+kx][tmp]);
              // const int8_t d1 = *((int8_t*)&data[y+ky][x+kx][tmp] + 1);
              // const int8_t d2 = *((int8_t*)&data[y+ky][x+kx][tmp] + 2);
              // const int8_t d3 = *((int8_t*)&data[y+ky][x+kx][tmp] + 3);

              // const int8_t w0 = *((int8_t*)&threadWeights[ky][kx]);
              // const int8_t w1 = *((int8_t*)&threadWeights[ky][kx] + 1);
              // const int8_t w2 = *((int8_t*)&threadWeights[ky][kx] + 2);
              // const int8_t w3 = *((int8_t*)&threadWeights[ky][kx] + 3);

              // printf("Loaded from shm data %d %d %d %d\n", d0, d1, d2, d3);

              // auto w = *reinterpret_cast<const int2*>(&kernel[KERNEL_INDEX(inC, kH, kW, tOutC*VECT_C+tOutCVectCOffset, tInC, ky, kx) + 8*tmp]);
              auto w = threadWeights[ky][kx];
              auto d = data[y+ky][x+kx][tmp];
              
              cache = __dp4a(d.x, w.x, cache);
              cache = __dp4a(d.y, w.y, cache);
            }
      // atomicAdd(&output[DATA_INDEX(outC, outH, outW, blockIdx.y, tOutC, outputYOffset+y, outputXOffset+x) + tOutCVectCOffset], cache);
      atomicAdd(&result[y][x][threadIdx.y], cache);
    }

  __syncthreads();
  if (threadIdx.x == 0)
  {
    #pragma unroll
    for (y = 0; y < resultTileH; ++y)
      #pragma unroll
      for (x = 0; x < resultTileW; ++x)
        atomicAdd(&output[DATA_INDEX(outC, outH, outW, blockIdx.y, tOutC, outputYOffset+y, outputXOffset+x) + tOutCVectCOffset], result[y][x][threadIdx.y]);
  }
}


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK)
Conv2DForward3x3CudaV1(const int8_t* __restrict__ idata, 
                       const int8_t*  __restrict__ kernel, 
                       int32_t* __restrict__ output)
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

  tmp = threadIdx.x % 4; // 0, 1, 2, 3 to decide where to look for the kernel
                         // Each 8 bytes are loaded only once

  // if (threadIdx.y == 0)
  // {
  // printf("GRID DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // printf("BLOCK DIMS %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  // }
  // printf("COMPUTED %d %d %d %d %d %d\n", tBatch, tOutCVectCOffset, tInC, tOutC, outputYOffset, outputXOffset);

  // std::cout << tBatch << " " << tOutCVectCOffset << " " << tInC << " " << tOutC << " " << outputYOffset << " " << outputXOffset << std::endl;

  // Load kernel weights
  int2 threadWeights[kH][kW];
  #pragma unroll
  for (uint32_t y = 0; y < kH; ++y)
    #pragma unroll
      for (uint32_t x = 0; x < kW; ++x)
            threadWeights[y][x] = *reinterpret_cast<const int2*>(&kernel[KERNEL_INDEX(inC, kH, kW, tOutC*VECT_C+tOutCVectCOffset, tInC, y, x) + 8*tmp]);

  // convolve
  // const uint32_t columnShift = threadIdx.x / resultTileW; // 0 or 1, for 1 or 2 column to process
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
            {
              data = *reinterpret_cast<const int2*>(&idata[DATA_INDEX(inC, inH, inW, tBatch, tInC, outputYOffset+y+ky, outputXOffset+x+kx) + 8*tmp]);
              result = __dp4a(data.x, threadWeights[ky][kx].x, result);
              result = __dp4a(data.y, threadWeights[ky][kx].y, result);
            }
              
      atomicAdd(&output[DATA_INDEX(outC, outH, outW, tBatch, tOutC, outputYOffset+y, outputXOffset+x) + tOutCVectCOffset], result);
    }
}


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv6(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC); // output tiles, batch, outC
  dim3 block_size(16, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaSharedMemConfig conf;
  cudaDeviceGetSharedMemConfig(&conf);
  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaFuncSetCacheConfig(Сonv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferShared);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // cudaFuncSetAttribute(&Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, 
  //                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV6<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  cudaDeviceSetSharedMemConfig(conf);

  return {output, elapsedTime};
}	


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv5(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), outC); // output tiles, batch, outC
  dim3 block_size(8, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaSharedMemConfig conf;
  cudaDeviceGetSharedMemConfig(&conf);
  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaFuncSetCacheConfig(Сonv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferShared);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // cudaFuncSetAttribute(&Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, 
  //                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV5<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  cudaDeviceSetSharedMemConfig(conf);

  return {output, elapsedTime};
}	

template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv4(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaSharedMemConfig conf;
  cudaDeviceGetSharedMemConfig(&conf);
  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaFuncSetCacheConfig(Сonv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferShared);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // cudaFuncSetAttribute(&Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, 
  //                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV4<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  cudaDeviceSetSharedMemConfig(conf);

  return {output, elapsedTime};
}	

template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv3(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaSharedMemConfig conf;
  cudaDeviceGetSharedMemConfig(&conf);
  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaFuncSetCacheConfig(Сonv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferShared);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  // cudaFuncSetAttribute(&Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, 
  //                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV3<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  cudaDeviceSetSharedMemConfig(conf);

  return {output, elapsedTime};
}	


template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv2(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  cudaSharedMemConfig conf;
  cudaDeviceGetSharedMemConfig(&conf);
  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // cudaFuncSetCacheConfig(Сonv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, cudaFuncCachePreferShared);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  // cudaFuncSetAttribute(&Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW>, 
  //                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV2<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  cudaDeviceSetSharedMemConfig(conf);

  return {output, elapsedTime};
}	

template<
  const uint32_t inC, const uint32_t outC, const uint32_t kH, const uint32_t kW,
  const uint32_t inH, const uint32_t inW, 
  const uint32_t outH, const uint32_t outW,
  const uint32_t resultTileH, const uint32_t resultTileW // == 2x2
>
std::tuple<Tensor, float> customConv2Dv1(const Tensor& rinput, const Tensor& rkernel) {
  auto dtype = torch::kInt32;
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::zeros({input.size(0), kernel.size(0) / VECT_C, outH, outW, VECT_C}, input.options().dtype(dtype));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(outH / resultTileH * outW / resultTileW, input.size(0), inC*outC); // output tiles, batch, inC * outC
  dim3 block_size(4, VECT_C, 1); // 4 threads per filter, 32 filters, 2 batch elements per block
  
  std::cout << output.sizes() << std::endl;
  std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  std::cout << block_size.x << " " << block_size.y << " " << block_size.z << std::endl;


  // cudaSharedMemConfig conf;
  // cudaDeviceGetSharedMemConfig(&conf);
  
  // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  Conv2DForward3x3CudaV1<inC, outC, kH, kW, inH, inW, outH, outW, resultTileH, resultTileW><<<grid, block_size, 0, stream>>>(
    (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  

  // cudaDeviceSetSharedMemConfig(conf);
  return {output, elapsedTime};
}	


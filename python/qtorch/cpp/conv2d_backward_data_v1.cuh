#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_61_intrinsics.hpp>
#include <vector_types.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

constexpr int APPROX_THREADS_PER_BLOCK = 256;
constexpr uint32_t VECT_C = 32;

using c10::IntList;
using at::Tensor;
using std::string;

// const __restrict__
// shared memory
// vector load
// dp4a

__device__ __inline__ uint32_t VEC_C_INDEX(const uint32_t& target_batch, const uint32_t& target_C, const uint32_t& target_H, const uint32_t& target_W, 
                                           const uint32_t& C, const uint32_t& H, const uint32_t& W, const uint32_t& vect_c)
{
  return target_batch*C*H*W*vect_c + target_C*H*W*vect_c + target_H*W*vect_c + target_W*vect_c;
}


// Thread ograds must be accessible, no trade-offs
// uint32_t is used intentionally. Overflow might occur otherwise.
__global__ void Conv2DBackwardFilterCuda(int8_t* idata, int8_t* ograds, float* output, 
                                         const uint32_t B, const uint32_t kH, const uint32_t kW,
                                         const uint32_t inC, const uint32_t inH, const uint32_t inW,
                                         const uint32_t outC, const uint32_t outH, const uint32_t outW,
                                         const uint32_t padH, const uint32_t padW, const uint32_t stride)
// idata is B, C-in, H-in, W-in, VECT_C
// ograds is B, C-out, H-out, W-out, VECT_C
// output is C-out, C-in, kH, kW, VECT_C

// Each block processes a single C-out accross ALL C-in and B
// I.e. each block updates grads for a blockIdx.z == C-out
{
  // blockDim, gridDim
  // blockIdx, threadIdx

  // the simplest case, one block per a W line
  const uint32_t targetOutC = blockIdx.z;
  const uint32_t targetOutH = blockIdx.y;
  int32_t wgrad;

  // target idata lines for the block
  uint32_t startInH = (uint32_t)std::max(-int(padH) + int(targetOutH), int(0));
  uint32_t endInH = (uint32_t)std::min(-int(padH) + int(targetOutH) + int(kH), int(inH));
  endInH = std::max(endH, uint32_t(0));

  // loop over batches
  for (uint32_t targetB = 0; targetB < B; ++targetB)
  {
    // For this batch for this thread 
    int8_t* ograd = &ograds[VEC_C_INDEX(targetB, targetOutC, targetOutH, threadIdx.x, outC, outH, outW, VECT_C)];
      
    // loop over input channels 
    for (uint32_t targetInC = 0; targetInC < inC; ++targetInC)
    {
      // loop over idata lines
      for (uint32_t stepH = startH; stepH < endH; ++stepH)
      {
        // loop over idata columns
        for (uint stepW = 0; stepW < kW; ++stepW)
        {
          uint32_t index = stepW + threadIdx.x; // thread column
          if (index < padW || (index - padW) > inH) // index is inside the range
          {
              int8_t* idata_vect_c = &idata[VEC_C_INDEX(targetB, targetInC, stepH, index - padW, inC, inH, inW, VECT_C)]  // thread column data vect_c

          }
        }
      }
    }
  }
}

Tensor Conv2DBackwardFilter(const Tensor& input, const Tensor& grads, const double& scale, const IntList& wsizes,
                            const IntList& stride, const IntList& padding, const IntList& dilation, const int& groups) {
  auto dtype = torch::kFloat32;
  auto weights_grads = at::empty(wsizes, at::TensorOptions().dtype(dtype)
                                                            .device(input.device())
                                                            .layout(input.layout())
                                                            .is_variable(false));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Around 256 threads
  dim3 block_size;
  auto W = grads.size(3); // B, C, H, W, 32
  auto H = grads.size(2);
  float blocks_per_line = 1;
  // Just to give it a shot later. If needed.
  // Relevant part is strided batched gemm
  // cublasGemmEx()
  // 
  // if (APPROX_THREADS_PER_BLOCK == W)
  // {
  //   block_size.x = W;
  // }
  // else
  // {
  //   blocks_per_line = W / (float)APPROX_THREADS_PER_BLOCK;
  //   if (blocks_per_line < 1)  // multiple lines per block
  //   {
  //     auto columns_in_block = (int)std::round(1 / blocks_per_line);   // Not that simple. We must ensure that grid is exactly divisible
  //     if (columns_in_block > H)
  //     {
  //       columns_in_block = H;
  //       LOG(10) << "Consider multiple channels per block. Not efficient right now, because single thread use all H and W. REFACTOR IT.";
  //     }
  //     if (H % columns_in_block != 0)
  //     {
  //       // Find all divisors of the H
  //       // Pick closest to the columns in block
  //       AT_ASSERT(H % 2 == 0);
  //       columns_in_block = 2;
  //       LOG(10) << "Multiple columns per a single block. IT IS NOT EFFICIENT RIGHT NOW. REFACTOR IT.";
  //     }
  //     block_size.x = W;
  //     block_size.y = columns_in_block;
  //   }
  //   else                      // multiple blocks per line
  //   {
  //     // blocks_per_line = std::round(blocks_per_line);
  //     AT_ASSERT(W % 2 == 0);
  //     blocks_per_line = 2;
  //     LOG(10) << "Multiple blocks per a single line. IT IS NOT EFFICIENT RIGHT NOW. REFACTOR IT.";
  //     block_size.x = W / int(blocks_per_line);      
  //   }
  // }
  
  AT_ASSERT((H % block_size.y == 0) && (W % block_size.x == 0));
  dim3 grid;
  grid.x = W / block_size.x;
  grid.y = H / block_size.y;
  grid.z = grads.size(1); // Each block takes care of a single output channel

  AT_DISPATCH_FLOATING_TYPES(logits.type(), "SigmoidFocalLoss_forward", [&] {
    SigmoidFocalLossForward<scalar_t><<<grid, block, 0, stream>>>(
         losses_size,
         logits.contiguous().data<scalar_t>(),
	 targets.contiguous().data<int>(),
         num_classes,
	 gamma,
	 alpha,
	 num_samples,
         losses.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return weights_grads;   
}	
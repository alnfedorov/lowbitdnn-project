#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <torch/extension.h>
#include <iostream>
#include <omp.h>


using torch::Tensor;
using c10::IntArrayRef;


template<
    const uint32_t batch, const uint32_t inC, const uint32_t inH, const uint32_t inW,
    const uint32_t outC, const uint32_t outH, const uint32_t outW,
    const uint32_t kH, const uint32_t kW>
void refConv2DForwardImpl(
    const __restrict__ int8_t* rinput, const Tensor kernel, const Tensor output
)
{  
    // const int8_t (*kernel)[outC][inC][kH][kW] = (const int8_t (*)[outC][inC][kH][kW])(&rkernel);
    for (uint32_t b=0; b < batch; ++b)
    {
        // int32_t (*output)[outC][outH][outW] = (int32_t (*)[outC][outH][outW])(&(routput[b*outC*outH*outW]));
        const int8_t (*input)[inC][inH][inW] = (const int8_t (*)[inC][inH][inW])(&(rinput[b*inC*inH*inW]));

        std::cout << "\tBatch " << b << std::endl;
        #pragma omp parallel for
        for (uint32_t outChn=0; outChn < outC; ++outChn)    // kernel
            for (uint32_t outY=0; outY < outH; ++outY)      // location Y
                for (uint32_t outX=0; outX < outW; ++outX)  // location X
                {
                    int32_t result = 0;
                    for (uint32_t inChn=0; inChn < inC; ++inChn)    // input channel
                        #pragma unroll
                        for (uint32_t ky=0; ky < kH; ++ky)          // kernel Y
                            #pragma unroll
                            for (uint32_t kx=0; kx < kW; ++kx)      // kernel X
                            {
                                // printf("kernel[%d][%d][%d][%d] * input[%d][%d][%d]\n", outChn, inChn, ky, kx, inChn, outY+ky, outX+kx);
                                // result += int32_t((*kernel)[outChn][inChn][ky][kx]) * int32_t((*input)[inChn][outY+ky][outX+kx]);
                                // result += kernel[outChn][inChn][ky][kx].item<int32_t>() * input[b][inChn][outY+ky][outX+kx].item<int32_t>();
                                result += kernel[outChn][inChn][ky][kx].item<int32_t>() * int32_t((*input)[inChn][outY+ky][outX+kx]);
                            }
                    // printf("output[%d][%d][%d] = result\n", outChn, outY, outX);
                    // (*output)[outChn][outY][outX] = result;
                    // output[b][outChn][outY][outX]<int32_t>() = result;
                    *((int32_t *)(output[b][outChn][outY][outX].data_ptr())) = result;
                }       
    }
}


template<
    const uint32_t batch, const uint32_t inC, const uint32_t inH, const uint32_t inW,
    const uint32_t outC, const uint32_t outH, const uint32_t outW,
    const uint32_t kH, const uint32_t kW>
Tensor refConv2DForward(const Tensor& rinput, const Tensor& rkernel)
{
  auto input = rinput.contiguous();
  auto kernel = rkernel.contiguous();
  auto output = at::empty({batch, outC, outH, outW},
                    at::TensorOptions().dtype(torch::kInt32)
                                       .device(input.device())
                                       .is_variable(true));
  
  AT_ASSERT((input.sizes() == IntArrayRef({batch, inC, inH, inW})) && (kernel.sizes() == IntArrayRef({outC, inC, kH, kW})));
  AT_ASSERT((input.dtype() == torch::kInt8) && (kernel.dtype() == torch::kInt8) && (output.dtype() == torch::kInt32));
  AT_ASSERT((input.device() == kernel.device()) && (kernel.device() == output.device()) && (output.device() == torch::kCPU));

//   std::cout << "Begin cpu" << std::endl;
  refConv2DForwardImpl<batch, inC, inH, inW, outC, outH, outW, kH, kW>(
    //   (int8_t*)input.data_ptr(), (int8_t*)kernel.data_ptr(), (int32_t*)output.data_ptr()
    (int8_t*)input.data_ptr(), kernel, output
  );
//   std::cout << "End cpu" << std::endl;
  return output;
}	

#include "conv2d.cuh"
#include "conv2d_backward_data.cuh"
#include "conv2d_forward.cuh"
#include "pool2d.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "conv2d int8 forward");
  m.def("conv2d_backward_data", &conv2d_backward_data, "conv2d float backward");
  m.def("conv2d_forwardv1", &Conv2DCustomForwardV1, "conv2d int8 forward v1");
  m.def("max_pool2d", &max_pool2d, "max_pool2d int8 forward");
}
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>

#include <iostream>
#include <string>
#include <vector>
#include <iostream>

#include "cudnn2DConvolution.cuh"
#include "custom2DConvolution.cuh"

#include <cudnn_v7.h>
#include <vector>
#include <numeric>
#include <map>

using c10::IntList;
using torch::Tensor;

constexpr uint32_t REPEATS = 1000;
constexpr uint32_t WARMUP = 100;

constexpr uint32_t BATCH = 32;
constexpr uint32_t IN_CHANNELS = 256;
constexpr uint32_t IN_H = 122;
constexpr uint32_t IN_W = 122;

constexpr uint32_t OUT_CHANNELS = 256;
constexpr uint32_t OUT_H = 120;
constexpr uint32_t OUT_W = 120;

constexpr uint32_t KH = 3;
constexpr uint32_t KW = 3;

// constexpr uint32_t BATCH_TILE = 2;
constexpr uint32_t RESULT_TILE_W = 6;
constexpr uint32_t RESULT_TILE_H = 6;

constexpr IntList PADDING = {0, 0};
constexpr IntList STRIDE = {1, 1};
constexpr IntList DILATION = {1, 1};
constexpr int GROUPS = 1;

Tensor to_vect_c(Tensor tensor)
{
    auto shape = tensor.sizes();
    tensor = tensor.reshape({shape[0], shape[1] / 32, 32, shape[2], shape[3]})
                   .permute({0, 1, 3, 4, 2}).contiguous();
    return tensor;
}

int main()
{
    std::vector<float> cudnn_timing = {};
    std::vector<float> custom_timing = {};
    
    auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                         .mul_(4.0f).sub_(2.0f).round_();
    auto kernel_int8x32 = to_vect_c(kernel.to(torch::kInt8));

    auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                           .mul_(4.0f).sub_(2.0f).round_();
    auto data_int8x32 = to_vect_c(data.to(torch::kInt8));
    
    // auto qresv6 = customConv2Dv6<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, 8, 8>(data_int8x32, kernel_int8x32);
    // auto qresv5 = customConv2Dv5<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, 8, 8>(data_int8x32, kernel_int8x32);
    // auto qresv4 = customConv2Dv4<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, 8, 8>(data_int8x32, kernel_int8x32);
    // auto qresv3 = customConv2Dv3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
    // auto qresv2 = customConv2Dv2<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
    // auto qresv1 = customConv2Dv1<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
    // auto cfres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    // auto cqres = cudnnConv2DInt8X32(data_int8x32, kernel_int8x32, STRIDE, PADDING, DILATION, GROUPS);

    // auto diff = (to_vect_c(std::get<0>(cfres)) - std::get<0>(qres).to(torch::kFloat32)).abs();
    // AT_ASSERT((diff.max() == 0).all().item<bool>());

    // diff = (to_vect_c(std::get<0>(cfres)) - std::get<0>(cqres).to(torch::kFloat32)).abs();
    // AT_ASSERT((diff.max() == 0).all().item<bool>());

    // std::cout << "WARMUP" << std::endl;
    // for (size_t i = 0; i < WARMUP; ++i)
    // {   
    //     std::cout << i << std::endl;
    //     auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                        .mul_(4.0f).sub_(2.0f).round_();
    //     auto data_int8x32 = to_vect_c(data.to(torch::kInt8));

    //     auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    //     // auto qres = customConv2Dv2<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
    //     auto qres = customConv2Dv1<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
        
    //     auto diff = (to_vect_c(std::get<0>(fres)) - std::get<0>(qres).to(torch::kFloat32)).abs();
    //     std::cout << "Diff " << diff.max() << std::endl;

    //     AT_ASSERT((diff.max() == 0).all().item<bool>());
    // }

    std::map<std::string, std::vector<float>> timings;
    timings["float"] = {};
    timings["v6"] = {};
    
    // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                     .mul_(4.0f).sub_(2.0f).round_();
    // auto data_int8x32 = to_vect_c(data.to(torch::kInt8));
    

    std::cout << "BENCHMARKING" << std::endl;
    for (size_t i = 0; i < REPEATS; ++i)
    {   
        std::cout << i << std::endl;
        auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
        // auto qres = customConv2Dv2<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
        auto qresv5 = customConv2Dv5<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, 8, 8>(data_int8x32, kernel_int8x32);
    
        timings["float"].push_back(std::get<1>(fres));
        timings["v6"].push_back(std::get<1>(qresv5));
    }

    for (const auto& kv : timings)
    {
        double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
        std::cout << kv.first << " : " << mean << std::endl;
    }

    return 0;
}

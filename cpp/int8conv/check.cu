#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>

#include <iostream>
#include <string>
#include <vector>
#include <iostream>

#include "utils.cuh"
// #include "conv2DForward3x3.cuh"
#include "conv2DForward3x3TensorCores.cuh"
#include "refConv2DForward.hpp"
// #include "conv2DForward3x3Fused.cuh"
// #include "conv2DForward3x3WinogradFused.cuh"
// #include "conv2DBackwardData3x3.cuh"
// #include "conv2DBackwardWeights3x3.cuh"
#include "cudnn2DConvolution.cuh"

#include <cudnn_v7.h>
#include <vector>
#include <numeric>
#include <map>

constexpr uint32_t REPEATS = 1000;
constexpr uint32_t WARMUP = 1000;

constexpr uint32_t BATCH = 16;
constexpr uint32_t IN_CHANNELS = 128;
constexpr uint32_t IN_H = 130;
constexpr uint32_t IN_W = 130;

constexpr uint32_t OUT_CHANNELS = 128;
constexpr uint32_t OUT_H = 128;
constexpr uint32_t OUT_W = 128;

constexpr uint32_t KH = 3;
constexpr uint32_t KW = 3;

constexpr float RMUL = 1;
constexpr float RSUB = 0;


constexpr IntList PADDING = {0, 0};
constexpr IntList STRIDE = {1, 1};
constexpr IntList DILATION = {1, 1};
constexpr int GROUPS = 1;

inline Tensor randomTensor(IntList shape)
{
    return torch::rand(shape, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                                                         .mul_(RMUL)
                                                         .sub_(RSUB)
                                                         .round_()
                                                         .to(torch::kInt8)
                                                         .to(torch::kFloat32);
}

void checkForward3x3()
{
    std::cout << "FORWARD 3x3 " << std::endl;

    std::vector<float> cudnn_timing = {};
    std::vector<float> custom_timing = {};

    auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                            .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto kernel = torch::ones({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).to(torch::kInt8).to(torch::kFloat32);
    auto kernel_int8x32 = to_vect_c(kernel.to(torch::kInt8));

    auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                            .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto data = torch::ones({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).to(torch::kInt8).to(torch::kFloat32);
    auto data_int8x32 = to_vect_c(data.to(torch::kInt8));

    std::cout << "WARMUP" << std::endl;
    for (size_t i = 0; i < WARMUP; ++i)
    {   
        std::cout << i << "\r";
        auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                            .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
        // auto data = torch::ones({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).to(torch::kInt8).to(torch::kFloat32);
        auto data_int8x32 = to_vect_c(data.to(torch::kInt8));

        auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
                        .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
        // auto kernel = torch::ones({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).to(torch::kInt8).to(torch::kFloat32);
        auto kernel_int8x32 = to_vect_c(kernel.to(torch::kInt8));

        // auto data = torch::ones({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).to(torch::kInt8).to(torch::kFloat32);
        // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCPU).dtype(torch::kFloat32))
        //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8);
        // auto data_int8x32 = to_vect_c(data).to(torch::kCUDA);

        // auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCPU).dtype(torch::kFloat32))
        //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8);
        // auto kernel_int8x32 = to_vect_c(kernel).to(torch::kCUDA);
        

        // std::cout << "Float conv" << std::endl;
        // auto qres = conv2DForward3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);
        // std::cout << "Qconv " << std::endl;
        auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
        auto qresTuple = conv2DForward3x3<BATCH, IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W>(data_int8x32, kernel_int8x32);
        // auto qres = std::get<0>(qresTuple).abs().to(torch::kCPU);

        // auto refQRes = refConv2DForward<BATCH, IN_CHANNELS, IN_H, IN_W, OUT_CHANNELS, OUT_H, OUT_W, KH, KW>(data, kernel);
        // refQRes = to_vect_c(refQRes);
        
        // std::cout << "Comparison" << std::endl;
        auto diff = (to_vect_c(std::get<0>(fres).round_()) - std::get<0>(qresTuple).to(torch::kFloat32)).abs();
        // auto diff = refQRes - qres;

        if (!(diff.max() == 0).all().item<bool>())
        {
            // std::cout << "Q " << std::get<0>(qresTuple).to(torch::kFloat32).narrow(0, 0, 1).squeeze(0).narrow(0, 0, 1).squeeze(0).narrow(0, 0, 1).squeeze(0) << std::endl;
            // std::cout << "F " << to_vect_c(std::get<0>(fres).round_()).narrow(0, 0, 1).squeeze(0).narrow(0, 0, 1).squeeze(0).narrow(0, 0, 1).squeeze(0) << std::endl;
            // std::cout << diff.narrow(0, 0, 1).squeeze(0) << std::endl;
            std::cout << i << std::endl;
            auto tmp = diff;
            std::cout << tmp.sizes() << std::endl;
            std::cout << tmp << std::endl;
            std::cout << diff.max_values(0, false).max_values(0, false).max_values(0, false) << std::endl;
        }

        AT_ASSERT((diff.max() == 0).all().item<bool>());
    }
    std::cout << std::endl;

    std::map<std::string, std::vector<float>> timings;
    timings["float"] = {};
    timings["int8"] = {};

    std::cout << "BENCHMARKING" << std::endl;
    for (size_t i = 0; i < REPEATS; ++i)
    {   
        std::cout << i << "\r";
        auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
        // auto qres = conv2DForward3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);
        auto qres = conv2DForward3x3<BATCH, IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W>(data_int8x32, kernel_int8x32);

        timings["float"].push_back(std::get<1>(fres));
        timings["int8"].push_back(std::get<1>(qres));
    }
    std::cout << std::endl;

    for (const auto& kv : timings)
    {
        double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
        std::cout << kv.first << " : " << mean << std::endl;
    }
}

// void checkBackwardData3x3()
// {
//     std::cout << "BACKWARD DATA 3x3 " << std::endl;

//     std::vector<float> cudnn_timing = {};
//     std::vector<float> custom_timing = {};
    
//     auto kernel = randomTensor({OUT_CHANNELS, IN_CHANNELS, KH, KW});
//     auto kernel_int8x32 = to_vect_c(kernel.permute({1,0,2,3}).to(torch::kInt8));    // IMPORTANT

//     auto data = randomTensor({BATCH, IN_CHANNELS, IN_H, IN_W});
//     auto data_int8x32 = to_vect_c(data.to(torch::kInt8));
    
//     std::cout << "WARMUP" << std::endl;
//     for (size_t i = 0; i < WARMUP; ++i)
//     {   
//         std::cout << i << "\r";
//         auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
//                            .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
//         auto data_int8x32 = to_vect_c(data.to(torch::kInt8));

//         auto fres = cudnnConv2DBackwardData(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
//         auto qres = conv2DBackwardData3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);
        
//         auto diff = (to_vect_c(std::get<0>(fres).round_()) - std::get<0>(qres).to(torch::kFloat32)).abs();

//         AT_ASSERT((diff.max() == 0).all().item<bool>());

//         std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     std::map<std::string, std::vector<float>> timings;
//     timings["float"] = {};
//     timings["int8"] = {};

//     std::cout << "BENCHMARKING" << std::endl;
//     for (size_t i = 0; i < REPEATS; ++i)
//     {   
//         std::cout << i << "\r";
//         auto fres = cudnnConv2DBackwardData(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
//         auto qres = conv2DBackwardData3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);
        
//         timings["float"].push_back(std::get<1>(fres));
//         timings["int8"].push_back(std::get<1>(qres));
//     }
//     std::cout << std::endl;

//     for (const auto& kv : timings)
//     {
//         double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
//         std::cout << kv.first << " : " << mean << std::endl;
//     }
// }

// void checkBackwardWeights3x3()
// {
//     std::cout << "BACKWARD WEIGHTS 3x3 " << std::endl;

//     std::vector<float> cudnn_timing = {};
//     std::vector<float> custom_timing = {};
    
//     auto ograds = randomTensor({BATCH, OUT_CHANNELS, OUT_H, OUT_W});
//     auto ograds_int8x32 = to_vect_c(ograds).to(torch::kInt8);

//     auto data = randomTensor({BATCH, IN_CHANNELS, IN_H, IN_W});
//     auto data_int8x32 = to_vect_c(data).to(torch::kInt8);
    
//     std::cout << "WARMUP" << std::endl;
//     for (size_t i = 0; i < WARMUP; ++i)
//     {   
//         std::cout << i << "\r";
//         auto data = randomTensor({BATCH, IN_CHANNELS, IN_H, IN_W});
//         auto data_int8x32 = to_vect_c(data).to(torch::kInt8);

//         auto fres = cudnnConv2DBackwardFilter(data, ograds, STRIDE, PADDING, DILATION, GROUPS, 3, 3);
//         auto qres = conv2DBackwardWeights3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, ograds_int8x32);
        
//         auto diff = (to_vect_c(std::get<0>(fres).round_()) - std::get<0>(qres).to(torch::kFloat32)).abs();

//         AT_ASSERT((diff.max() == 0).all().item<bool>());
//     }
//     std::cout << std::endl;

//     std::map<std::string, std::vector<float>> timings;
//     timings["float"] = {};
//     timings["int8"] = {};

//     std::cout << "BENCHMARKING" << std::endl;
//     for (size_t i = 0; i < REPEATS; ++i)
//     {   
//         std::cout << i << "\r";
//         auto fres = cudnnConv2DBackwardFilter(data, ograds, STRIDE, PADDING, DILATION, GROUPS, 3, 3);
//         auto qres = conv2DBackwardWeights3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, ograds_int8x32);
        
//         timings["float"].push_back(std::get<1>(fres));
//         timings["int8"].push_back(std::get<1>(qres));
//     }
//     std::cout << std::endl;

//     for (const auto& kv : timings)
//     {
//         double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
//         std::cout << kv.first << " : " << mean << std::endl;
//     }
// }


// void checkForward3x3Fused()
// {
//     std::cout << "FORWARD 3x3 FUSED" << std::endl;
//     auto guard = torch::NoGradGuard();
//     std::vector<float> cudnn_timing = {};
//     std::vector<float> custom_timing = {};
//     // auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
//     //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kFloat32);
//     auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
//                             .mul_(RMUL).sub_(RSUB).round_().to(torch::kFloat32);

//     auto cpuConv = torch::nn::Conv2d(torch::nn::Conv2dOptions(IN_CHANNELS, OUT_CHANNELS, 3).padding(PADDING[0]).with_bias(false));
//     cpuConv->eval();
//     cpuConv->weight.detach_();
//     cpuConv->to(torch::kCPU);
//     cpuConv->weight.uniform_().mul_(RMUL).sub_(RSUB).round_();
//     // cpuConv->weight.fill_(1.0f);

//     auto kernel = cpuConv->weight.clone().detach().cuda();
//     std::cout << "WARMUP" << std::endl;
//     for (size_t i = 0; i < WARMUP; ++i)
//     {   
//         std::cout << i << "\n";
//         // std::cout << "ASDADA123123" << std::endl;
//         data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
//                             .mul_(RMUL).sub_(RSUB).round_();
//         // data = torch::ones({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
//         auto cpudata = data.clone().to(torch::kCPU);
//         auto fres = cpuConv->forward(cpudata);
//         fres = fres.to(torch::kCUDA);
//         // auto qres = conv2DForward3x3Fused<IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data, kernel);
//         auto qres = conv2DForward3x3WinogradFused<IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data, kernel);
        
        
//         // auto diff = (std::get<0>(fres) - std::get<0>(qres)).abs();
//         auto diff = (fres - std::get<0>(qres)).abs();
//         // std::cout << "ASDADA";
//         if (!((diff.max() == 0).all().item<bool>()))
//         {
//             // auto fres2 = std::get<0>(fres).view({-1});
//             auto fres2 = fres.view({-1});
//             Tensor qres2 = std::get<0>(qres).view({-1});
//             Tensor t = std::get<1>(torch::sort(diff.view({-1}), -1L, true));
//             Tensor v = std::get<0>(torch::sort(diff.view({-1}), -1L, true));
//             t = t.masked_select(v > 0);
//             std::cout << "SIZES " << t.sizes() << std::endl;
//             // // std::cout << t << std::endl;
//             std::cout << "MAX " << diff.max() << std::endl;
//             t = t.narrow(0, 0, std::min(int64_t(19), t.size(0)));
//             std::cout << "F " << fres2.index(t).view({1, -1}) << std::endl;
//             std::cout << "Q " << qres2.index(t).view({1, -1}) << std::endl;
//             std::cout << "TIME " << std::get<1>(qres) << std::endl;
//         }
//         // std::cout << "ASDADA";
//         AT_ASSERT((diff.max() == 0).all().item<bool>());
//     }
//     // std::cout << std::endl;

//     std::map<std::string, std::vector<float>> timings;
//     timings["float"] = {};
//     timings["uint8"] = {};

//     std::cout << "BENCHMARKING" << std::endl;
//     for (size_t i = 0; i < REPEATS; ++i)
//     {   
//         std::cout << i << "\r";
//         auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
//         // auto qres = conv2DForward3x3Fused<IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data, kernel);
//         auto qres = conv2DForward3x3WinogradFused<IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data, kernel);
        
//         timings["float"].push_back(std::get<1>(fres));
//         timings["uint8"].push_back(std::get<1>(qres));
//     }
//     std::cout << std::endl;

//     for (const auto& kv : timings)
//     {
//         double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
//         std::cout << kv.first << " : " << mean << std::endl;
//     }
// }

int main()
{
    checkForward3x3();
    // checkBackwardData3x3();
    // checkBackwardWeights3x3();

    // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto data_int8x16 = to_vect_c(data.to(torch::kInt8));

    // auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                 .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto kernel_int8x16 = to_vect_c(kernel.to(torch::kInt8));
    // auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    // auto qres = conv2DForward3x3<BATCH, IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W>(data_int8x16, kernel_int8x16);


    // auto data_int8x32 = to_vect_c(data.to(torch::kInt8), 32);
    // auto kernel_int8x32 = to_vect_c(kernel.to(torch::kInt8), 32);
    // auto cqres = cudnnConv2DInt8X32(data_int8x32, kernel_int8x32, STRIDE, PADDING, DILATION, GROUPS);


    // auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kFloat32);
    // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                         .mul_(RMUL).sub_(RSUB).round_().to(torch::kFloat32);

    // auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    // auto qres = conv2DForward3x3Fused<IN_CHANNELS, OUT_CHANNELS, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data, kernel);
        
    // checkForward3x3Fused();
    // auto g = torch::NoGradGuard();
    // torch::manual_seed(123);
    // std::vector<float> cudnn_timing = {};
    // std::vector<float> custom_timing = {};
    
    // auto kernel = torch::rand({OUT_CHANNELS, IN_CHANNELS, KH, KW}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                      .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto kernel_int8x32 = to_vect_c(kernel.to(torch::kInt8));

    // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                        .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    // auto data_int8x32 = to_vect_c(data.to(torch::kInt8));
    
    // // auto qres = conv2DForward3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W>(data_int8x32, kernel_int8x32);
    // // auto cfres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    // // auto cqres = cudnnConv2DInt8X32(data_int8x32, kernel_int8x32, STRIDE, PADDING, DILATION, GROUPS);

    // // auto diff = (to_vect_c(std::get<0>(cfres)) - std::get<0>(qres).to(torch::kFloat32)).abs();
    // // AT_ASSERT((diff.max() == 0).all().item<bool>());

    // // diff = (to_vect_c(std::get<0>(cfres)) - std::get<0>(cqres).to(torch::kFloat32)).abs();
    // // AT_ASSERT((diff.max() == 0).all().item<bool>());
    // std::cout << "WARMUP" << std::endl;
    // for (size_t i = 0; i < WARMUP; ++i)
    // {   
    //     std::cout << i << "\r";
    //     // std::cout << i << "\n";
    //     auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    //                        .mul_(RMUL).sub_(RSUB).round_().to(torch::kInt8).to(torch::kFloat32);
    //     // auto data = torch::ones({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32)).round_();
    //     auto data_int8x32 = to_vect_c(data.to(torch::kInt8));

    //     // std::cout << "DATA MAX MIN: " << data.max() << " " << data.min() << std::endl;
    //     // std::cout << "KERNEL MAX MIN: " << data.max() << " " << kernel.min() << std::endl;

    //     auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    //     auto qres = conv2DForward3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);
        
    //     auto diff = (to_vect_c(std::get<0>(fres).round_()) - std::get<0>(qres).to(torch::kFloat32)).abs();
    //     // if (!((diff.max() == 0).all().item<bool>()))
    //     // {
    //     //     auto fres2 = to_vect_c(std::get<0>(fres)).view({-1});
    //     //     Tensor qres2 = std::get<0>(qres).view({-1});
    //     //     Tensor t = std::get<1>(torch::sort(diff.view({-1}), -1L, true));
    //     //     Tensor v = std::get<0>(torch::sort(diff.view({-1}), -1L, true));
    //     //     t = t.masked_select(v > 0);
    //     //     std::cout << "SIZES " << t.sizes() << std::endl;
    //     //     std::cout << t << std::endl;
    //     //     t = t.narrow(0, 0, std::min(int64_t(19), t.size(0)));
    //     //     std::cout << "F " << fres2.index(t).view({1, -1}) << std::endl;
    //     //     std::cout << "Q " << qres2.index(t).view({1, -1}) << std::endl;
    //     // }

    //     // // std::cout << "Diff " << torch::nonzero(diff) << std::endl;
    //     // auto diff1 = diff.view({-1});
    //     // // diff = diff.index(torch::nonzero(diff > 0));
    //     // diff1 = diff1.masked_select(diff1 > 0);
    //     // // std::cout << "Diff " << diff1 << std::endl;
    //     // std::cout << data_int8x32 << std::endl;
    //     // std::cout << kernel_int8x32 << std::endl;
    //     // std::cout << to_vect_c(std::get<0>(fres).round_()) << std::endl;
    //     // std::cout << std::get<0>(qres).to(torch::kFloat32) << std::endl;
    //     AT_ASSERT((diff.max() == 0).all().item<bool>());
    // }
    // std::cout << std::endl;

    // std::map<std::string, std::vector<float>> timings;
    // timings["float"] = {};
    // timings["int8"] = {};
    
    // // auto data = torch::rand({BATCH, IN_CHANNELS, IN_H, IN_W}, torch::device(torch::kCUDA).dtype(torch::kFloat32))
    // //                     .mul_(4.0f).sub_(2.0f).round_();
    // // auto data_int8x32 = to_vect_c(data.to(torch::kInt8));
    

    // std::cout << "BENCHMARKING" << std::endl;
    // for (size_t i = 0; i < REPEATS; ++i)
    // {   
    //     std::cout << i << "\r";
    //     auto fres = cudnnConv2DFloat(data, kernel, STRIDE, PADDING, DILATION, GROUPS);
    //     // auto qres = customConv2Dv2<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, KH, KW, IN_H, IN_W, OUT_H, OUT_W, RESULT_TILE_H, RESULT_TILE_W>(data_int8x32, kernel_int8x32);
    //     auto qres = conv2DForward3x3<IN_CHANNELS / VECT_C, OUT_CHANNELS / VECT_C, IN_H, IN_W, OUT_H, OUT_W, PADDING[0], PADDING[1]>(data_int8x32, kernel_int8x32);

    //     timings["float"].push_back(std::get<1>(fres));
    //     timings["int8"].push_back(std::get<1>(qres));
    // }
    // std::cout << std::endl;

    // for (const auto& kv : timings)
    // {
    //     double mean = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / kv.second.size();
    //     std::cout << kv.first << " : " << mean << std::endl;
    // }

    return 0;
}

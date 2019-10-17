#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "benchmark.cuh"

namespace pt=boost::property_tree;


auto dispatch(cudnnDataType_t inputDataType, cudnnDataType_t filterDataType, cudnnDataType_t outDataType)
{
    // FLOAT CONFIG
    if (inputDataType == CUDNN_DATA_FLOAT && filterDataType == CUDNN_DATA_FLOAT && outDataType == CUDNN_DATA_FLOAT)
        return benchmark_convolution<float, float, float>;

    // TRUE HALF CONFIG
    else if (inputDataType == CUDNN_DATA_HALF && filterDataType == CUDNN_DATA_HALF && outDataType == CUDNN_DATA_HALF)
        return benchmark_convolution<half, half, half>;

    // DOUBLE CONFIG
    else if (inputDataType == CUDNN_DATA_DOUBLE && filterDataType == CUDNN_DATA_DOUBLE && outDataType == CUDNN_DATA_DOUBLE)
        return benchmark_convolution<double, double, double>;

    // INT8* CONFIG
    else if ((inputDataType == CUDNN_DATA_INT8  || inputDataType == CUDNN_DATA_INT8x4  || inputDataType == CUDNN_DATA_INT8x32) && 
             (filterDataType == CUDNN_DATA_INT8 || filterDataType == CUDNN_DATA_INT8x4 || filterDataType == CUDNN_DATA_INT8x32))
    {
        if (outDataType == CUDNN_DATA_INT8 || outDataType == CUDNN_DATA_INT8x4 || outDataType == CUDNN_DATA_INT8x32)
            return benchmark_convolution<int8_t, int8_t, int8_t>;
        else if (outDataType == CUDNN_DATA_FLOAT)
            return benchmark_convolution<int8_t, int8_t, float>;    // EXT config

    }
    // UINT8* CONFIG
    else if ((inputDataType == CUDNN_DATA_UINT8 || inputDataType == CUDNN_DATA_UINT8x4) && 
             (filterDataType == CUDNN_DATA_INT8 || filterDataType == CUDNN_DATA_INT8x4 || filterDataType == CUDNN_DATA_INT8x32))
    {
        if (outDataType == CUDNN_DATA_INT8 || outDataType == CUDNN_DATA_INT8x4 || outDataType == CUDNN_DATA_INT8x32)
            return benchmark_convolution<uint8_t, int8_t, int8_t>;
        else if (outDataType == CUDNN_DATA_FLOAT)
            return benchmark_convolution<uint8_t, int8_t, float>;    // EXT config
    }
    throw std::runtime_error("Not found cudnn convolution for specified data types.");
}

float benchmark(size_t B, size_t C, size_t H, size_t W, size_t numFilters, size_t filterH, size_t filterW, 
                const pt::ptree& configuration, uint16_t repeats, int verbose)
{
    std::vector<size_t> timings = {};
    
    auto inputTensorFormat = tensorFormatMapping.at(configuration.get<std::string>("input_tensor_format"));
    auto filterTensorFormat = tensorFormatMapping.at(configuration.get<std::string>("filters_tensor_format"));
    auto outputTensorFormat = tensorFormatMapping.at(configuration.get<std::string>("output_tensor_format"));

    auto inputDataType = dataTypeMapping.at(configuration.get<std::string>("input_data_type"));
    auto filterDataType = dataTypeMapping.at(configuration.get<std::string>("filters_data_type"));
    auto convDataType = dataTypeMapping.at(configuration.get<std::string>("accumulator_data_type"));
    auto outputDataType = dataTypeMapping.at(configuration.get<std::string>("output_data_type"));

    auto function = dispatch(inputDataType, filterDataType, outputDataType);
    for (size_t i=0; i < repeats; ++i)
    {
        auto elapsed = function(B, C, H, W, 
                                numFilters, filterH, filterW,
                                1, 1, 1, 1, 1, 1,
                                inputTensorFormat, filterTensorFormat, outputTensorFormat,
                                inputDataType, filterDataType, convDataType, outputDataType, int(verbose >= 2 && i == 0)
        );
        timings.push_back(elapsed.count());
    }
    float mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    return mean;
}

inline std::string composeExperimentShortName(size_t B, size_t C, size_t H, size_t W, 
                                              size_t numFilters, size_t filterH, size_t filterW, std::string configName)
{

    auto stream = std::stringstream();
    stream << B << 'x' << C << 'x' << H << 'x' << W << " * " 
            << numFilters << 'x' << filterH << 'x' << filterW << ", " << configName;
    return stream.str();
}

pt::ptree composeExperimentData(size_t B, size_t C, size_t H, size_t W, 
                                       size_t numFilters, size_t filterH, size_t filterW, std::string configName)
{
    pt::ptree data;
    data.put("B", B);
    data.put("C", C);
    data.put("H", H);
    data.put("W", W);
    data.put("filters", numFilters);
    data.put("filter_width", filterW);
    data.put("filter_height", filterH);
    data.put("config", configName);
    data.put("name", composeExperimentShortName(B, C, H, W, numFilters, filterH, filterW, configName));
    return data;
}

int main() {
    pt::ptree root, configs, benchmarks;
    uint16_t repeats;
    try
    {
        pt::read_json("config.json", root);
        repeats = root.get<uint16_t>("repeats");
        configs = root.get_child("configs");
    }
    catch(const std::exception& e)
    {
        std::cerr << "Config file config.json is not found or is not correct: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    for(const auto& exp : root.get_child("experiments"))
    {
        auto H = exp.second.get<size_t>("height");
        auto W = exp.second.get<size_t>("width");
        auto verbose = exp.second.get<int>("verbose");

        auto filterW = exp.second.get<size_t>("filter_width");
        auto filterH = exp.second.get<size_t>("filter_height");

        for (const auto& B : exp.second.get_child("batch"))
            for (const auto& C : exp.second.get_child("channels"))
                for (const auto& numFilters : exp.second.get_child("filters"))
                {
                    auto _B = B.second.get<size_t>("");
                    auto _C = C.second.get<size_t>("");
                    auto _numFilters = numFilters.second.get<size_t>("");
                    if (_numFilters < _C)
                        continue;
                    for (auto& configName : exp.second.get_child("configs"))
                        try 
                        {
                            auto _configName = configName.second.get<std::string>("");
                            auto config = configs.get_child(_configName);
                            auto data = composeExperimentData(_B, _C, H, W, _numFilters, filterH, filterW, _configName);

                            std::clog << data.get<std::string>("name") << "..." << std::endl;
                            auto timing = benchmark(_B, _C, H, W, _numFilters, filterH, filterW, config, repeats, verbose);
                            data.put("timing", timing);
                            benchmarks.push_back(std::make_pair("", data));
                            std::clog << data.get<std::string>("name") << " timing is " << data.get<std::string>("timing") << std::endl;
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << "Failed to perform experiment: " << e.what() << '\n';
                        }
                }
    }   

    pt::ptree output;
    output.add("repeats", repeats);
    output.add_child("configs", configs);
    output.add_child("benchmarks", benchmarks);

    pt::write_json("output.json", output);
    std::clog << "Finished." << std::endl;
    return EXIT_SUCCESS;
}
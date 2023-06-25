#include <iostream>
#include <mutex>
#include <vector>

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_weight.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int ReadInputIds(std::vector<int>* input_ids_lengths,
                 std::vector<int>* input_ids,
                 int*              max_input_ids_length,
                 const int         pad_token_id)
{
    std::vector<std::vector<int>> tmp_input_ids;
    std::vector<int>              tmp_input_ids_lengths;

    std::string   file_name = "./examples/cpp/nllb_moe/input_ids.txt";
    std::ifstream input_ids_file(file_name, std::ios::in);
    ft::FT_CHECK(input_ids_file.is_open());
    {
        std::string line;
        while (std::getline(input_ids_file, line)) {
            std::stringstream line_stream(line);
            std::string       vals;
            int               i1 = 0;
            std::vector<int>  tmp_vec;
            while (std::getline(line_stream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_input_ids.push_back(tmp_vec);
            tmp_input_ids_lengths.push_back(i1);
        }
    }

    *max_input_ids_length = tmp_input_ids_lengths[0];
    for (uint i = 1; i < (uint)tmp_input_ids_lengths.size(); i++) {
        *max_input_ids_length = std::max(*max_input_ids_length, tmp_input_ids_lengths[i]);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_input_ids.size(); i++) {
        for (int j = (int)tmp_input_ids[i].size(); j < *max_input_ids_length; j++) {
            tmp_input_ids[i].push_back(pad_token_id);
        }
    }

    for (int i = 0; i < (int)tmp_input_ids.size(); i++) {
        input_ids->insert(input_ids->end(), tmp_input_ids[i].begin(), tmp_input_ids[i].end());
        input_ids_lengths->push_back(tmp_input_ids_lengths[i]);
    }

    return 0;
}

template<typename T>
void NllbMoeExample(const INIReader& reader)
{
    std::string model_dir           = reader.Get("ft_instance_hyperparameter", "model_dir");
    INIReader   model_config_reader = INIReader(model_dir + "/config.ini");
    ft::FT_CHECK(model_config_reader.ParseError() == 0);
    int pad_token_id = std::stoi(model_config_reader.Get("nllb_moe", "pad_token_id"));

    std::vector<int> input_ids_lengths;
    std::vector<int> input_ids;
    int              max_input_ids_length;
    ReadInputIds(&input_ids_lengths, &input_ids, &max_input_ids_length, pad_token_id);
    int batch_size = input_ids.size() / max_input_ids_length;

    int* d_input_ids_lengths;
    int* d_input_ids;
    ft::deviceMalloc(&d_input_ids, input_ids.size(), false);
    ft::deviceMalloc(&d_input_ids_lengths, input_ids_lengths.size(), false);
    ft::cudaH2Dcpy(d_input_ids, input_ids.data(), input_ids.size());
    ft::cudaH2Dcpy(d_input_ids_lengths, input_ids_lengths.data(), input_ids_lengths.size());

    ft::NllbMoeWeight<T> nllb_moe_weight(model_dir);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ft::Allocator<ft::AllocatorType::CUDA> allocator(ft::getDevice());

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    ft::cublasAlgoMap*  cublas_algo_map      = new ft::cublasAlgoMap(GEMM_CONFIG);
    std::mutex*         cublas_wrapper_mutex = new std::mutex();
    ft::cublasMMWrapper cublas_wrapper =
        ft::cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    ft::NllbMoe<T> nllb_moe(model_config_reader, stream, &cublas_wrapper, &allocator);

    std::unordered_map<std::string, ft::Tensor> input_tensors = {
        {"input_ids",
         {ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size, max_input_ids_length}, d_input_ids}},
        {"input_ids_lengths", {ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, d_input_ids_lengths}},
    };
    std::unordered_map<std::string, ft::Tensor> output_tensors = {};
    nllb_moe.Forward(&output_tensors, &input_tensors, &nllb_moe_weight);
}

int main(int argc, char** argv)
{
    std::string ini_path;
    if (argc == 2) {
        ini_path = std::string(argv[1]);
    }
    else {
        ini_path = "./examples/cpp/nllb_moe/nllb_moe_config.ini";
    }

    INIReader reader = INIReader(ini_path);
    ft::FT_CHECK(reader.ParseError() == 0);

    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        NllbMoeExample<float>(reader);
    }
    else if (data_type == "fp16") {
        NllbMoeExample<half>(reader);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        NllbMoeExample<__nv_bfloat16>(reader);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    return 0;
}
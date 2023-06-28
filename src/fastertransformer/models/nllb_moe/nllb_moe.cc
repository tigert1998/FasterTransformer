#include "src/fastertransformer/models/nllb_moe/nllb_moe.h"

#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include <iostream>
#include <vector>

namespace fastertransformer {

template<typename T>
NllbMoe<T>::NllbMoe(const INIReader& reader,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator)
{
    stream_         = stream;
    cublas_wrapper_ = cublas_wrapper;
    allocator_      = allocator;

    d_model_ = reader.GetInteger("nllb_moe", "d_model");

    encoder_ = std::make_unique<NllbMoeEncoder<T>>(reader, stream, cublas_wrapper, allocator);
}

template<typename T>
NllbMoe<T>::~NllbMoe()
{
    FreeBuffer();
}

template<typename T>
void NllbMoe<T>::Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         const NllbMoeWeight<T>*                        nllb_moe_weight)
{
    DataType data_type = getTensorType<T>();

    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    AllocateBuffer(batch_size, max_input_ids_length);

    {
        std::unordered_map<std::string, Tensor> output_tensors_for_encoder = {
            {"last_hidden_state",
             {MEMORY_GPU,
              data_type,
              std::vector<size_t>{batch_size, max_input_ids_length, d_model_},
              encoder_hidden_states_}},
        };
        std::unordered_map<std::string, Tensor> input_tensors_for_encoder = {
            {"input_ids", input_tensors->at("input_ids")},
            {"input_ids_lengths", input_tensors->at("input_ids_lengths")},
        };
        encoder_->Forward(&output_tensors_for_encoder, &input_tensors_for_encoder, nllb_moe_weight->encoder.get());
    }

    xiaohu_dbg::PrintGPUArray(encoder_hidden_states_, batch_size * max_input_ids_length * d_model_);
}

template<typename T>
void NllbMoe<T>::AllocateBuffer(uint64_t batch_size, uint64_t max_input_ids_length)
{
    encoder_hidden_states_ = (T*)allocator_->reMalloc(
        encoder_hidden_states_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
}

template<typename T>
void NllbMoe<T>::FreeBuffer()
{
    allocator_->free((void**)(&encoder_hidden_states_));
}

template struct NllbMoe<float>;
template struct NllbMoe<half>;
#ifdef ENABLE_BF16
template struct NllbMoe<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
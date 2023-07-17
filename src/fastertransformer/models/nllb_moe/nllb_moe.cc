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

    encoder_ = std::make_unique<NllbMoeEncoder<T>>(reader, stream_, cublas_wrapper_, allocator_);
    decoder_ = std::make_unique<NllbMoeDecoder<T>>(reader, stream_, cublas_wrapper_, allocator_);
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
             {MEMORY_GPU, data_type, {batch_size, max_input_ids_length, d_model_}, encoder_hidden_states_}},
        };
        std::unordered_map<std::string, Tensor> input_tensors_for_encoder = {
            {"input_ids", input_tensors->at("input_ids")},
            {"input_ids_lengths", input_tensors->at("input_ids_lengths")},
        };
        encoder_->Forward(&output_tensors_for_encoder, &input_tensors_for_encoder, nllb_moe_weight->encoder.get());
    }

    {
        std::vector<int> input_ids = {2, 2};
        int*             d_input_ids;
        deviceMalloc(&d_input_ids, input_ids.size(), false);
        cudaH2Dcpy(d_input_ids, input_ids.data(), input_ids.size());

        int32_t step = 1;

        std::vector<int> output_ids_lengths = {0, 0};
        int*             d_output_ids_lengths;
        deviceMalloc(&d_output_ids_lengths, output_ids_lengths.size(), false);
        cudaH2Dcpy(d_output_ids_lengths, output_ids_lengths.data(), output_ids_lengths.size());

        uint64_t decoder_layers          = 4;
        uint64_t max_output_ids_length   = 16;
        uint64_t decoder_attention_heads = 4;

        T* last_hidden_state;
        T* d_key_cache;
        T* d_value_cache;
        T* d_cross_attention_key_cache;
        T* d_cross_attention_value_cache;
        deviceMalloc(&last_hidden_state, batch_size * d_model_, false);
        deviceMalloc(&d_key_cache, decoder_layers * batch_size * d_model_ * max_output_ids_length, false);
        deviceMalloc(&d_value_cache, decoder_layers * batch_size * d_model_ * max_output_ids_length, false);
        deviceMalloc(
            &d_cross_attention_key_cache, decoder_layers * batch_size * d_model_ * max_input_ids_length, false);
        deviceMalloc(
            &d_cross_attention_value_cache, decoder_layers * batch_size * d_model_ * max_input_ids_length, false);

        std::unordered_map<std::string, Tensor> input_tensors_for_decoder = {
            {"input_ids", {MEMORY_GPU, TYPE_INT32, {2, 1}, d_input_ids}},
            {"step", {MEMORY_CPU, TYPE_INT32, {1}, &step}},
            {"output_ids_lengths", {MEMORY_GPU, TYPE_INT32, {2}, d_output_ids_lengths}},
            {"encoder_hidden_states",
             {MEMORY_GPU, data_type, {batch_size, max_input_ids_length, d_model_}, encoder_hidden_states_}},
            {"encoder_input_ids_lengths", input_tensors->at("input_ids_lengths")},
        };

        std::unordered_map<std::string, Tensor> output_tensors_for_decoder = {
            {"last_hidden_state", {MEMORY_GPU, data_type, {batch_size, 1, d_model_}, last_hidden_state}},
            {"key_cache",
             {MEMORY_GPU,
              data_type,
              {
                  decoder_layers,
                  batch_size,
                  decoder_attention_heads,
                  d_model_ / decoder_attention_heads / (16 / sizeof(T)),
                  max_output_ids_length,
                  16 / sizeof(T),
              },
              d_key_cache}},
            {"value_cache",
             {MEMORY_GPU,
              data_type,
              {
                  decoder_layers,
                  batch_size,
                  decoder_attention_heads,
                  max_output_ids_length,
                  d_model_ / decoder_attention_heads,
              },
              d_value_cache}},
            {"cross_attention_key_cache",
             {MEMORY_GPU,
              data_type,
              {
                  decoder_layers,
                  batch_size,
                  decoder_attention_heads,
                  d_model_ / decoder_attention_heads / (16 / sizeof(T)),
                  max_input_ids_length,
                  16 / sizeof(T),
              },
              d_cross_attention_key_cache}},
            {"cross_attention_value_cache",
             {MEMORY_GPU,
              data_type,
              {
                  decoder_layers,
                  batch_size,
                  decoder_attention_heads,
                  max_input_ids_length,
                  d_model_ / decoder_attention_heads,
              },
              d_cross_attention_value_cache}},
        };

        decoder_->Forward(&output_tensors_for_decoder, &input_tensors_for_decoder, nllb_moe_weight->decoder.get());
    }
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
#include "src/fastertransformer/models/nllb_moe/nllb_moe_decoder.h"
#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeDecoder<T>::NllbMoeDecoder(const INIReader& reader,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator)
{
    stream_         = stream;
    cublas_wrapper_ = cublas_wrapper;
    allocator_      = allocator;

    pad_token_id_            = reader.GetInteger("nllb_moe", "pad_token_id");
    d_model_                 = reader.GetInteger("nllb_moe", "d_model");
    decoder_layers_          = reader.GetInteger("nllb_moe", "decoder_layers");
    decoder_attention_heads_ = reader.GetInteger("nllb_moe", "decoder_attention_heads");

    self_attn_ = std::make_unique<DecoderSelfAttentionLayer<T>>(0,
                                                                decoder_attention_heads_,
                                                                d_model_ / decoder_attention_heads_,
                                                                stream_,
                                                                cublas_wrapper_,
                                                                allocator_,
                                                                false,
                                                                false,
                                                                0);
}

template<typename T>
NllbMoeDecoder<T>::~NllbMoeDecoder()
{
    FreeBuffer();
}

template<typename T>
void NllbMoeDecoder<T>::Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                const std::unordered_map<std::string, Tensor>* input_tensors,
                                const NllbMoeDecoderWeight<T>*                 nllb_moe_decoder_weight)
{
    DataType data_type = getTensorType<T>();

    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    int*     input_ids            = input_tensors->at("input_ids").getPtr<int>();

    // FIXME
    uint64_t past_key_values_length = input_tensors->at("step").getVal<int>() - 1;

    FT_CHECK(max_input_ids_length == 1);

    uint64_t embedding_lookup_temp_storage_size = 0;

    NllbMoeEmbeddingLookup<T>(nullptr,
                              pad_token_id_,
                              nullptr,
                              nullptr,
                              nullptr,
                              batch_size,
                              max_input_ids_length,
                              d_model_,
                              true,
                              past_key_values_length,
                              &embedding_lookup_temp_storage_size,
                              nullptr,
                              stream_);
    AllocateBuffer(batch_size, max_input_ids_length, embedding_lookup_temp_storage_size);

    NllbMoeEmbeddingLookup<T>(input_ids,
                              pad_token_id_,
                              nllb_moe_decoder_weight->shared,
                              nllb_moe_decoder_weight->sinusoidal_positional_embedding->weight,
                              hidden_states_,
                              batch_size,
                              max_input_ids_length,
                              d_model_,
                              true,
                              past_key_values_length,
                              &embedding_lookup_temp_storage_size,
                              embedding_lookup_temp_storage_,
                              stream_);

    for (int i = 0; i < decoder_layers_; i++) {
        invokeGeneralLayerNorm<T>(self_attn_input_,
                                  hidden_states_,
                                  nllb_moe_decoder_weight->layers[i]->self_attn_layer_norm.gamma,
                                  nllb_moe_decoder_weight->layers[i]->self_attn_layer_norm.beta,
                                  1e-5,
                                  batch_size * max_input_ids_length,
                                  d_model_,
                                  nullptr,
                                  0,
                                  stream_);
        {
            std::vector<size_t> self_attn_key_cache_shape = output_tensors->at("key_cache").shape;
            self_attn_key_cache_shape.erase(self_attn_key_cache_shape.begin());
            uint64_t self_attn_key_cache_size = std::accumulate(
                self_attn_key_cache_shape.begin(), self_attn_key_cache_shape.end(), 1, std::multiplies<size_t>{});
            T* self_attn_key_cache = output_tensors->at("key_cache").getPtr<T>() + i * self_attn_key_cache_size;

            std::vector<size_t> self_attn_value_cache_shape = output_tensors->at("value_cache").shape;
            self_attn_value_cache_shape.erase(self_attn_value_cache_shape.begin());
            uint64_t self_attn_value_cache_size = std::accumulate(
                self_attn_value_cache_shape.begin(), self_attn_value_cache_shape.end(), 1, std::multiplies<size_t>{});
            T* self_attn_value_cache = output_tensors->at("value_cache").getPtr<T>() + i * self_attn_value_cache_size;

            TensorMap self_attn_input_tensors = {
                {"input_query", {MEMORY_GPU, data_type, {batch_size, d_model_}, self_attn_input_}},
                {"sequence_lengths", input_tensors->at("output_ids_lengths")},
                {"step", input_tensors->at("step")},
            };
            TensorMap self_attn_output_tensors = {
                {"hidden_features", {MEMORY_GPU, data_type, {batch_size, d_model_}, self_attn_output_}},
                {"key_cache", {MEMORY_GPU, data_type, self_attn_key_cache_shape, self_attn_key_cache}},
                {"value_cache", {MEMORY_GPU, data_type, self_attn_value_cache_shape, self_attn_value_cache}},
            };
            self_attn_->forward(
                &self_attn_output_tensors, &self_attn_input_tensors, &nllb_moe_decoder_weight->layers[i]->self_attn);
        }

        invokeGeneralAddBiasResidualPreLayerNorm(
            residual_,
            cross_attention_input_,
            self_attn_output_,
            hidden_states_,
            nllb_moe_decoder_weight->layers[i]->cross_attention_layer_norm.gamma,
            nllb_moe_decoder_weight->layers[i]->cross_attention_layer_norm.beta,
            nllb_moe_decoder_weight->layers[i]->self_attn.attention_output_weight.bias,
            1e-5,
            batch_size * max_input_ids_length,
            d_model_,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            stream_);

        break;
    }

    xiaohu_dbg::PrintGPUArray(residual_, batch_size * max_input_ids_length * d_model_);
    xiaohu_dbg::PrintGPUArray(cross_attention_input_, batch_size * max_input_ids_length * d_model_);
}

template<typename T>
void NllbMoeDecoder<T>::AllocateBuffer(uint64_t batch_size,
                                       uint64_t max_input_ids_length,
                                       uint64_t embedding_lookup_temp_storage_size)
{
    embedding_lookup_temp_storage_ =
        (void*)allocator_->reMalloc(embedding_lookup_temp_storage_, embedding_lookup_temp_storage_size, false);
    hidden_states_ =
        (T*)allocator_->reMalloc(hidden_states_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    self_attn_input_ =
        (T*)allocator_->reMalloc(self_attn_input_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    self_attn_output_ =
        (T*)allocator_->reMalloc(self_attn_output_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    residual_ = (T*)allocator_->reMalloc(residual_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    cross_attention_input_ = (T*)allocator_->reMalloc(
        cross_attention_input_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
}

template<typename T>
void NllbMoeDecoder<T>::FreeBuffer()
{
    allocator_->free((void**)(&embedding_lookup_temp_storage_));
    allocator_->free((void**)(&hidden_states_));
    allocator_->free((void**)(&self_attn_input_));
    allocator_->free((void**)(&self_attn_output_));
    allocator_->free((void**)(&residual_));
    allocator_->free((void**)(&cross_attention_input_));
}

template class NllbMoeDecoder<float>;
template class NllbMoeDecoder<half>;
#ifdef ENABLE_BF16
template class NllbMoeDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
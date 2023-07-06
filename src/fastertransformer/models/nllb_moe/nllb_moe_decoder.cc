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

    pad_token_id_   = reader.GetInteger("nllb_moe", "pad_token_id");
    d_model_        = reader.GetInteger("nllb_moe", "d_model");
    decoder_layers_ = reader.GetInteger("nllb_moe", "decoder_layers");
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
    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    int*     input_ids            = input_tensors->at("input_ids").getPtr<int>();

    // TODO
    uint64_t past_key_values_length = 2;

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
        break;
    }

    xiaohu_dbg::PrintGPUArray(self_attn_input_, batch_size * max_input_ids_length * d_model_);
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
}

template<typename T>
void NllbMoeDecoder<T>::FreeBuffer()
{
    allocator_->free((void**)(&embedding_lookup_temp_storage_));
    allocator_->free((void**)(&hidden_states_));
    allocator_->free((void**)(&self_attn_input_));
}

template class NllbMoeDecoder<float>;
template class NllbMoeDecoder<half>;
#ifdef ENABLE_BF16
template class NllbMoeDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
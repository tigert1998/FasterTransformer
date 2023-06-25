#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder.h"

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeEncoder<T>::NllbMoeEncoder(const INIReader& reader,
                                  cudaStream_t     stream,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator*      allocator)
{
    pad_token_id_        = reader.GetInteger("nllb_moe", "pad_token_id");
    d_model_             = reader.GetInteger("nllb_moe", "d_model");
    encoder_sparse_step_ = reader.GetInteger("nllb_moe", "encoder_sparse_step");
    encoder_layers_      = reader.GetInteger("nllb_moe", "encoder_layers");

    stream_         = stream;
    cublas_wrapper_ = cublas_wrapper;
    allocator_      = allocator;

    uint64_t encoder_attention_heads = reader.GetInteger("nllb_moe", "encoder_attention_heads");

    self_attn_ = std::make_unique<UnfusedAttentionLayer<T>>(0,
                                                            0,
                                                            encoder_attention_heads,
                                                            d_model_ / encoder_attention_heads,
                                                            1,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            false);
}

template<typename T>
NllbMoeEncoder<T>::~NllbMoeEncoder()
{
    FreeBuffer();
}

template<typename T>
void NllbMoeEncoder<T>::Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                                const std::unordered_map<std::string, Tensor>* input_tensors,
                                const NllbMoeEncoderWeight<T>*                 nllb_moe_encoder_weight)
{
    DataType data_type = getTensorType<T>();

    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    int*     input_ids            = input_tensors->at("input_ids").getPtr<int>();
    int*     input_ids_lengths    = input_tensors->at("input_ids_lengths").getPtr<int>();

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
                              &embedding_lookup_temp_storage_size,
                              nullptr,
                              stream_);
    AllocateBuffer(batch_size, max_input_ids_length, embedding_lookup_temp_storage_size);

    NllbMoeEmbeddingLookup<T>(input_ids,
                              pad_token_id_,
                              nllb_moe_encoder_weight->shared,
                              nllb_moe_encoder_weight->sinusoidal_positional_embedding->weight,
                              hidden_states_,
                              batch_size,
                              max_input_ids_length,
                              d_model_,
                              true,
                              &embedding_lookup_temp_storage_size,
                              embedding_lookup_temp_storage_,
                              stream_);

    // note that this attention mask is different from the huggingface attention mask
    invokeBuildEncoderAttentionMask<T>(attention_mask_, input_ids_lengths, batch_size, max_input_ids_length, stream_);

    for (int i = 0; i < encoder_layers_; i++) {
        invokeGeneralLayerNorm<T>(self_attn_input_,
                                  hidden_states_,
                                  nllb_moe_encoder_weight->layers[i]->self_attn_layer_norm.gamma,
                                  nllb_moe_encoder_weight->layers[i]->self_attn_layer_norm.beta,
                                  1e-5,
                                  batch_size * max_input_ids_length,
                                  d_model_,
                                  nullptr,
                                  0,
                                  stream_);

        TensorMap self_attn_input_tensors = {
            {"input_query",
             {MEMORY_GPU,
              data_type,
              std::vector<size_t>{batch_size * max_input_ids_length, d_model_},
              self_attn_input_}},
            {"attention_mask",
             {MEMORY_GPU,
              data_type,
              std::vector<size_t>{batch_size, 1, max_input_ids_length, max_input_ids_length},
              attention_mask_}},
        };
        TensorMap self_attn_output_tensors = {
            {"hidden_features",
             {MEMORY_GPU,
              data_type,
              std::vector<size_t>{batch_size * max_input_ids_length, d_model_},
              self_attn_output_}},
        };
        self_attn_->forward(
            &self_attn_output_tensors, &self_attn_input_tensors, &nllb_moe_encoder_weight->layers[i]->self_attn);
        break;
    }
    cudaStreamSynchronize(stream_);
    xiaohu_dbg::PrintGPUArray(self_attn_output_, batch_size * max_input_ids_length * d_model_);
}

template<typename T>
void NllbMoeEncoder<T>::AllocateBuffer(uint64_t batch_size,
                                       uint64_t max_input_ids_length,
                                       uint64_t embedding_lookup_temp_storage_size)
{
    embedding_lookup_temp_storage_ =
        (void*)allocator_->reMalloc(embedding_lookup_temp_storage_, embedding_lookup_temp_storage_size, false);
    hidden_states_ =
        (T*)allocator_->reMalloc(hidden_states_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    self_attn_input_ =
        (T*)allocator_->reMalloc(self_attn_input_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    attention_mask_ = (T*)allocator_->reMalloc(
        attention_mask_, batch_size * max_input_ids_length * max_input_ids_length * sizeof(T), false);
    self_attn_output_ =
        (T*)allocator_->reMalloc(self_attn_output_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
}

template<typename T>
void NllbMoeEncoder<T>::FreeBuffer()
{
    allocator_->free((void**)(&embedding_lookup_temp_storage_));
    allocator_->free((void**)(&hidden_states_));
    allocator_->free((void**)(&self_attn_input_));
    allocator_->free((void**)(&attention_mask_));
    allocator_->free((void**)(&self_attn_output_));
}

template class NllbMoeEncoder<float>;
template class NllbMoeEncoder<half>;
#ifdef ENABLE_BF16
template class NllbMoeEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
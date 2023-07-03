#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder.h"

#include "src/fastertransformer/kernels/add_residual_kernels.h"
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
    num_experts_         = reader.GetInteger("nllb_moe", "num_experts");
    encoder_ffn_dim_     = reader.GetInteger("nllb_moe", "encoder_ffn_dim");
    moe_token_dropout_   = reader.GetFloat("nllb_moe", "moe_token_dropout");

    float moe_eval_capacity_token_fraction = reader.GetFloat("nllb_moe", "moe_eval_capacity_token_fraction");
    FT_CHECK(moe_eval_capacity_token_fraction >= 1 - 1e-5);

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

    FT_CHECK(reader.Get("nllb_moe", "activation_function") == "relu");
    ffn_ = std::make_unique<ReluFfnLayer<T>>(0,
                                             0,
                                             encoder_attention_heads,
                                             d_model_ / encoder_attention_heads,
                                             num_experts_,
                                             encoder_ffn_dim_,
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

    T* last_hidden_state = output_tensors->at("last_hidden_state").getPtr<T>();

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
                              0,
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
                              0,
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

        invokeGeneralAddBiasResidualPreLayerNorm<T>(
            residual_,
            ffn_input_,
            self_attn_output_,
            hidden_states_,
            nllb_moe_encoder_weight->layers[i]->ff_layer_norm.gamma,
            nllb_moe_encoder_weight->layers[i]->ff_layer_norm.beta,
            nllb_moe_encoder_weight->layers[i]->self_attn.attention_output_weight.bias,
            1e-5,
            batch_size * max_input_ids_length,
            d_model_,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            stream_);

        if (nllb_moe_encoder_weight->layers[i]->is_sparse()) {
            uint64_t  moe_k             = 2;
            TensorMap ffn_input_tensors = {
                {"ffn_input",
                 {MEMORY_GPU, data_type, std::vector<size_t>{batch_size * max_input_ids_length, d_model_}, ffn_input_}},
                {"moe_k", {MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &moe_k}}};
            TensorMap ffn_output_tensors = {
                {"ffn_output",
                 {MEMORY_GPU,
                  data_type,
                  std::vector<size_t>{moe_k * batch_size * max_input_ids_length, d_model_},
                  ffn_output_}},
                {"expert_scales",
                 {MEMORY_GPU,
                  data_type,
                  std::vector<size_t>{batch_size * max_input_ids_length, moe_k},
                  expert_scales_}},
                {"expanded_source_row_to_expanded_dest_row",
                 {MEMORY_GPU,
                  TYPE_INT32,
                  std::vector<size_t>{batch_size * max_input_ids_length, moe_k},
                  expanded_source_row_to_expanded_dest_row_}},
                {"expert_for_source_row",
                 {MEMORY_GPU,
                  TYPE_INT32,
                  std::vector<size_t>{batch_size * max_input_ids_length, moe_k},
                  expert_for_source_row_}},
            };
            ffn_->forward(&ffn_output_tensors, &ffn_input_tensors, &nllb_moe_encoder_weight->layers[i]->ffn);

            NllbMoeNormalizeRouterProbabilities<T>(
                expert_scales_, moe_token_dropout_, batch_size * max_input_ids_length, stream_);

            finalize_moe_routing_kernelLauncher<T>(ffn_output_,
                                                   hidden_states_,
                                                   residual_,
                                                   nllb_moe_encoder_weight->layers[i]->ffn.output_weight.bias,
                                                   expert_scales_,
                                                   expanded_source_row_to_expanded_dest_row_,
                                                   expert_for_source_row_,
                                                   batch_size * max_input_ids_length,
                                                   d_model_,
                                                   moe_k,
                                                   stream_);
        }
        else {
            TensorMap ffn_input_tensors = {
                {"ffn_input",
                 {MEMORY_GPU, data_type, std::vector<size_t>{batch_size * max_input_ids_length, d_model_}, ffn_input_}},
            };
            TensorMap ffn_output_tensors = {
                {"ffn_output",
                 {MEMORY_GPU,
                  data_type,
                  std::vector<size_t>{batch_size * max_input_ids_length, d_model_},
                  ffn_output_}},
            };
            ffn_->forward(&ffn_output_tensors, &ffn_input_tensors, &nllb_moe_encoder_weight->layers[i]->ffn);

            invokeAddBiasResidual<T>(hidden_states_,
                                     ffn_output_,
                                     residual_,
                                     nullptr,
                                     nllb_moe_encoder_weight->layers[i]->ffn.output_weight.bias,
                                     nullptr,
                                     nullptr,
                                     batch_size * max_input_ids_length,
                                     d_model_,
                                     stream_);
        }
    }

    invokeGeneralLayerNorm<T>(last_hidden_state,
                              hidden_states_,
                              nllb_moe_encoder_weight->layer_norm.gamma,
                              nllb_moe_encoder_weight->layer_norm.beta,
                              1e-5,
                              batch_size * max_input_ids_length,
                              d_model_,
                              nullptr,
                              0,
                              stream_);
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
    residual_  = (T*)allocator_->reMalloc(residual_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    ffn_input_ = (T*)allocator_->reMalloc(ffn_input_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    ffn_output_ =
        (T*)allocator_->reMalloc(ffn_output_, 2 * batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    expert_scales_ = (T*)allocator_->reMalloc(expert_scales_, batch_size * max_input_ids_length * 2 * sizeof(T), false);
    expanded_source_row_to_expanded_dest_row_ = (int*)allocator_->reMalloc(
        expanded_source_row_to_expanded_dest_row_, batch_size * max_input_ids_length * 2 * sizeof(int), false);
    expert_for_source_row_ =
        (int*)allocator_->reMalloc(expert_for_source_row_, batch_size * max_input_ids_length * 2 * sizeof(int), false);
}

template<typename T>
void NllbMoeEncoder<T>::FreeBuffer()
{
    allocator_->free((void**)(&embedding_lookup_temp_storage_));
    allocator_->free((void**)(&hidden_states_));
    allocator_->free((void**)(&self_attn_input_));
    allocator_->free((void**)(&attention_mask_));
    allocator_->free((void**)(&self_attn_output_));
    allocator_->free((void**)(&residual_));
    allocator_->free((void**)(&ffn_input_));
    allocator_->free((void**)(&ffn_output_));
    allocator_->free((void**)(&expert_scales_));
    allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
    allocator_->free((void**)(&expert_for_source_row_));
}

template class NllbMoeEncoder<float>;
template class NllbMoeEncoder<half>;
#ifdef ENABLE_BF16
template class NllbMoeEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
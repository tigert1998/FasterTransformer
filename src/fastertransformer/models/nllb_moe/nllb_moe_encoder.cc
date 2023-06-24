#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder.h"

#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeEncoder<T>::NllbMoeEncoder(const INIReader& reader, cudaStream_t stream, IAllocator* allocator)
{
    pad_token_id_ = reader.GetInteger("nllb_moe", "pad_token_id");
    d_model_      = reader.GetInteger("nllb_moe", "d_model");

    stream_    = stream;
    allocator_ = allocator;
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
    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    int*     input_ids            = input_tensors->at("input_ids").getPtr<int>();

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

    xiaohu_dbg::PrintGPUArray<T>(hidden_states_, batch_size * max_input_ids_length * d_model_);
}

template<typename T>
void NllbMoeEncoder<T>::AllocateBuffer(uint64_t batch_size,
                                       uint64_t max_input_ids_length,
                                       uint64_t embedding_lookup_temp_storage_size)
{
    hidden_states_ =
        (T*)allocator_->reMalloc(hidden_states_, batch_size * max_input_ids_length * d_model_ * sizeof(T), false);
    embedding_lookup_temp_storage_ =
        (void*)allocator_->reMalloc(embedding_lookup_temp_storage_, embedding_lookup_temp_storage_size, false);
}

template<typename T>
void NllbMoeEncoder<T>::FreeBuffer()
{
    allocator_->free((void**)(&hidden_states_));
    allocator_->free((void**)(&embedding_lookup_temp_storage_));
}

template class NllbMoeEncoder<float>;
template class NllbMoeEncoder<half>;
#ifdef ENABLE_BF16
template class NllbMoeEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
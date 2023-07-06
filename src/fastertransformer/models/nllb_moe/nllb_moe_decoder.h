#pragma once

#include <memory>

#include "3rdparty/INIReader.h"

#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_decoder_weight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

template<typename T>
class NllbMoeDecoder {
public:
    inline NllbMoeDecoder() = default;
    NllbMoeDecoder(const INIReader& reader,
                   cudaStream_t     stream,
                   cublasMMWrapper* cublas_wrapper,
                   IAllocator*      allocator);
    ~NllbMoeDecoder();

    void Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const NllbMoeDecoderWeight<T>*                 nllb_moe_decoder_weight);

private:
    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;

    uint64_t pad_token_id_, d_model_, decoder_layers_, decoder_attention_heads_;

    void* embedding_lookup_temp_storage_ = nullptr;
    T*    hidden_states_                 = nullptr;
    T*    self_attn_input_               = nullptr;

    std::unique_ptr<DecoderSelfAttentionLayer<T>> self_attn_;

    void
    AllocateBuffer(uint64_t batch_size, uint64_t max_input_ids_length, uint64_t embedding_lookup_temp_storage_size);
    void FreeBuffer();
};

}  // namespace fastertransformer
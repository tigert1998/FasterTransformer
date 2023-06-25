#pragma once

#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder_weight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

#include "3rdparty/INIReader.h"

#include <stdint.h>
#include <string>
#include <unordered_map>

namespace fastertransformer {

template<typename T>
class NllbMoeEncoder {
public:
    inline NllbMoeEncoder() = default;
    NllbMoeEncoder(const INIReader& reader, cudaStream_t stream, IAllocator* allocator);
    ~NllbMoeEncoder();

    void Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const NllbMoeEncoderWeight<T>*                 nllb_moe_encoder_weight);

private:
    cudaStream_t stream_;
    IAllocator*  allocator_;

    uint64_t d_model_, pad_token_id_, encoder_sparse_step_, encoder_layers_;

    void* embedding_lookup_temp_storage_ = nullptr;
    T*    hidden_states_                 = nullptr;
    T*    self_attn_input_               = nullptr;

    void
    AllocateBuffer(uint64_t batch_size, uint64_t max_input_ids_length, uint64_t embedding_lookup_temp_storage_size);
    void FreeBuffer();
};

}  // namespace fastertransformer
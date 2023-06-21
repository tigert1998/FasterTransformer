#include "src/fastertransformer/models/nllb_moe/nllb_moe.h"

#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include <iostream>
#include <vector>

namespace fastertransformer {

template<typename T>
NllbMoe<T>::NllbMoe(const INIReader& reader, cudaStream_t stream)
{
    stream_ = stream;

    d_model_      = reader.GetInteger("nllb_moe", "d_model");
    pad_token_id_ = reader.GetInteger("nllb_moe", "pad_token_id");
}

template<typename T>
void NllbMoe<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         const NllbMoeWeight<T>*                        nllb_moe_weight)
{
    uint64_t batch_size           = input_tensors->at("input_ids").shape[0];
    uint64_t max_input_ids_length = input_tensors->at("input_ids").shape[1];
    int*     input_ids            = input_tensors->at("input_ids").getPtr<int>();

    T* hidden_states;
    deviceMalloc(&hidden_states, batch_size * max_input_ids_length * d_model_, false);
    NllbMoeEmbeddingLookup(input_ids,
                           pad_token_id_,
                           nllb_moe_weight->shared,
                           nllb_moe_weight->sinusoidal_positional_embedding->weight,
                           hidden_states,
                           batch_size,
                           max_input_ids_length,
                           d_model_,
                           true,
                           stream_);
}

template struct NllbMoe<float>;
template struct NllbMoe<half>;
#ifdef ENABLE_BF16
template struct NllbMoe<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
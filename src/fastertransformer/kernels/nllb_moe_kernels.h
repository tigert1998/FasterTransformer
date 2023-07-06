#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace fastertransformer {

template<typename T>
void NllbMoeEmbeddingLookup(const int*   input_ids,
                            int          pad_token_id,
                            const T*     embedding,
                            const T*     positional_embedding,
                            T*           output,
                            uint64_t     batch_size,
                            uint64_t     max_input_ids_length,
                            uint64_t     d_model,
                            bool         scale_embedding,
                            uint64_t     past_key_values_length,
                            uint64_t*    temp_storage_size,
                            void*        temp_storage,
                            cudaStream_t stream);

template<typename T>
void NllbMoeNormalizeRouterProbabilities(T*           expert_scales,
                                         const int*   input_ids_lengths,
                                         float        moe_token_dropout,
                                         uint64_t     batch_size,
                                         uint64_t     max_input_ids_length,
                                         cudaStream_t stream);

}  // namespace fastertransformer
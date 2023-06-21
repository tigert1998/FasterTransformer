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
                            cudaStream_t stream);

}  // namespace fastertransformer
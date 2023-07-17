#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

namespace fastertransformer {

namespace {
__global__ void PrefixSum(const int* input, int* output, uint64_t n, uint64_t m)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    // to be optimized
    output[tid * m] = input[tid * m];
    for (int i = 1; i < m; i++) {
        output[tid * m + i] = output[tid * m + i - 1] + input[tid * m + i];
    }
}

template<typename T>
__global__ void EmbeddingLookup(const int* input_ids,
                                const T*   embedding,
                                T*         output,
                                uint64_t   batch_size,
                                uint64_t   max_input_ids_length,
                                uint64_t   d_model)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * max_input_ids_length * d_model)
        return;
    int input_id = input_ids[tid / d_model];
    output[tid]  = embedding[input_id * d_model + (tid % d_model)];
}
}  // namespace

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
                            cudaStream_t stream)
{
    int *mask_begin, *mask_end;
    int *prefix_sum_begin, *prefix_sum_end;
    int *position_ids_begin, *position_ids_end;
    T *  embed_pos_begin, *embed_pos_end;
    T *  inputs_embeds_begin, *inputs_embeds_end;

    if (temp_storage == nullptr) {
        *temp_storage_size = batch_size * max_input_ids_length * 3 * sizeof(int)
                             + batch_size * max_input_ids_length * d_model * 2 * sizeof(T);
        return;
    }
    else {
        char* temp_storage_ptr = (char*)temp_storage;
        mask_begin             = (int*)temp_storage_ptr;
        temp_storage_ptr += batch_size * max_input_ids_length * sizeof(int);
        mask_end = prefix_sum_begin = (int*)temp_storage_ptr;
        temp_storage_ptr += batch_size * max_input_ids_length * sizeof(int);
        prefix_sum_end = position_ids_begin = (int*)temp_storage_ptr;
        temp_storage_ptr += batch_size * max_input_ids_length * sizeof(int);
        position_ids_end = (int*)temp_storage_ptr;
        embed_pos_begin  = (T*)temp_storage_ptr;
        temp_storage_ptr += batch_size * max_input_ids_length * d_model * sizeof(T);
        embed_pos_end = inputs_embeds_begin = (T*)temp_storage_ptr;
        temp_storage_ptr += batch_size * max_input_ids_length * d_model * sizeof(T);
        inputs_embeds_end = (T*)temp_storage_ptr;
    }

    thrust::transform(thrust::cuda::par.on(stream),
                      input_ids,
                      input_ids + batch_size * max_input_ids_length,
                      mask_begin,
                      [pad_token_id] __device__(int input_id) { return (int)(input_id != pad_token_id); });
    PrefixSum<<<(batch_size + 7) / 8, 8, 0, stream>>>(mask_begin, prefix_sum_begin, batch_size, max_input_ids_length);
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_zip_iterator(thrust::make_tuple(prefix_sum_begin, mask_begin)),
                      thrust::make_zip_iterator(thrust::make_tuple(prefix_sum_end, mask_end)),
                      position_ids_begin,
                      [pad_token_id, past_key_values_length] __device__(auto x) {
                          return (thrust::get<0>(x) + past_key_values_length) * thrust::get<1>(x) + pad_token_id;
                      });

    EmbeddingLookup<T><<<batch_size * max_input_ids_length, d_model, 0, stream>>>(
        position_ids_begin, positional_embedding, embed_pos_begin, batch_size, max_input_ids_length, d_model);
    EmbeddingLookup<T><<<batch_size * max_input_ids_length, d_model, 0, stream>>>(
        input_ids, embedding, inputs_embeds_begin, batch_size, max_input_ids_length, d_model);

    T embed_scale = scale_embedding ? std::sqrt(d_model) : 1;
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_zip_iterator(thrust::make_tuple(inputs_embeds_begin, embed_pos_begin)),
                      thrust::make_zip_iterator(thrust::make_tuple(inputs_embeds_end, embed_pos_end)),
                      output,
                      [embed_scale] __device__(auto x) { return thrust::get<0>(x) * embed_scale + thrust::get<1>(x); });
}

template void NllbMoeEmbeddingLookup(const int*   input_ids,
                                     int          pad_token_id,
                                     const float* embedding,
                                     const float* positional_embedding,
                                     float*       output,
                                     uint64_t     batch_size,
                                     uint64_t     max_input_ids_length,
                                     uint64_t     d_model,
                                     bool         scale_embedding,
                                     uint64_t     past_key_values_length,
                                     uint64_t*    temp_storage_size,
                                     void*        temp_storage,
                                     cudaStream_t stream);

template void NllbMoeEmbeddingLookup(const int*   input_ids,
                                     int          pad_token_id,
                                     const half*  embedding,
                                     const half*  positional_embedding,
                                     half*        output,
                                     uint64_t     batch_size,
                                     uint64_t     max_input_ids_length,
                                     uint64_t     d_model,
                                     bool         scale_embedding,
                                     uint64_t     past_key_values_length,
                                     uint64_t*    temp_storage_size,
                                     void*        temp_storage,
                                     cudaStream_t stream);

#ifdef ENABLE_BF16
template void NllbMoeEmbeddingLookup(const int*           input_ids,
                                     int                  pad_token_id,
                                     const __nv_bfloat16* embedding,
                                     const __nv_bfloat16* positional_embedding,
                                     __nv_bfloat16*       output,
                                     uint64_t             batch_size,
                                     uint64_t             max_input_ids_length,
                                     uint64_t             d_model,
                                     bool                 scale_embedding,
                                     uint64_t             past_key_values_length,
                                     uint64_t*            temp_storage_size,
                                     void*                temp_storage,
                                     cudaStream_t         stream);
#endif

namespace {
template<typename T>
__global__ void NormalizeRouterProbabilities(T*         expert_scales,
                                             const int* input_ids_lengths,
                                             T          moe_token_dropout,
                                             uint64_t   batch_size,
                                             uint64_t   max_input_ids_length)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * max_input_ids_length)
        return;

    T denom                    = expert_scales[tid * 2] + expert_scales[tid * 2 + 1];
    expert_scales[tid * 2]     = expert_scales[tid * 2] * (T)(((T)1.0f - moe_token_dropout) / denom);
    expert_scales[tid * 2 + 1] = expert_scales[tid * 2 + 1] * (T)(((T)1.0f - moe_token_dropout) / denom);
    if (input_ids_lengths != nullptr && tid % max_input_ids_length >= input_ids_lengths[tid / max_input_ids_length]) {
        expert_scales[tid * 2] = expert_scales[tid * 2 + 1] = (T)0.0f;
    }
}
}  // namespace

template<typename T>
void NllbMoeNormalizeRouterProbabilities(T*           expert_scales,
                                         const int*   input_ids_lengths,
                                         float        moe_token_dropout,
                                         uint64_t     batch_size,
                                         uint64_t     max_input_ids_length,
                                         cudaStream_t stream)
{
    uint64_t num_tokens = batch_size * max_input_ids_length;
    NormalizeRouterProbabilities<T><<<(num_tokens + 127) / 128, 128, 0, stream>>>(
        expert_scales, input_ids_lengths, moe_token_dropout, batch_size, max_input_ids_length);
}

template void NllbMoeNormalizeRouterProbabilities(float*       expert_scales,
                                                  const int*   input_ids_lengths,
                                                  float        moe_token_dropout,
                                                  uint64_t     batch_size,
                                                  uint64_t     max_input_ids_length,
                                                  cudaStream_t stream);

template void NllbMoeNormalizeRouterProbabilities(half*        expert_scales,
                                                  const int*   input_ids_lengths,
                                                  float        moe_token_dropout,
                                                  uint64_t     batch_size,
                                                  uint64_t     max_input_ids_length,
                                                  cudaStream_t stream);

#ifdef ENABLE_BF16
template void NllbMoeNormalizeRouterProbabilities(__nv_bfloat16* expert_scales,
                                                  const int*     input_ids_lengths,
                                                  float          moe_token_dropout,
                                                  uint64_t       batch_size,
                                                  uint64_t       max_input_ids_length,
                                                  cudaStream_t   stream);
#endif

}  // namespace fastertransformer
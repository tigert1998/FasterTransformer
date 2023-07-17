#pragma once

#include <memory>
#include <string>
#include <vector>

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"
#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

template<typename T>
struct NllbMoeDecoderLayerWeight {
public:
    inline NllbMoeDecoderLayerWeight() = default;
    NllbMoeDecoderLayerWeight(const std::string& dir_path, uint64_t layer_index);
    ~NllbMoeDecoderLayerWeight();

    NllbMoeDecoderLayerWeight<T> operator=(const NllbMoeDecoderLayerWeight<T>&) = delete;

    LayerNormWeight<T> self_attn_layer_norm;
    AttentionWeight<T> self_attn;
    LayerNormWeight<T> cross_attention_layer_norm;
    AttentionWeight<T> cross_attention;
    LayerNormWeight<T> ff_layer_norm;
    FfnWeight<T>       ffn;

    inline bool is_sparse()
    {
        return is_sparse_;
    }

private:
    uint64_t d_model_, is_sparse_, router_bias_, decoder_ffn_dim_, num_experts_;

    void MallocWeights();
    void LoadModel(const std::string& dir_path, uint64_t layer_index);
};

template<typename T>
struct NllbMoeDecoderWeight {
public:
    inline NllbMoeDecoderWeight() = default;
    NllbMoeDecoderWeight(const std::string& dir_path, T* shared);
    ~NllbMoeDecoderWeight();

    NllbMoeDecoderWeight<T> operator=(const NllbMoeDecoderWeight<T>&) = delete;

    T* shared = nullptr;
    std::unique_ptr<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>
                                                               sinusoidal_positional_embedding;  // positional embedding
    std::vector<std::unique_ptr<NllbMoeDecoderLayerWeight<T>>> layers;
    LayerNormWeight<T>                                         layer_norm;

private:
    uint64_t decoder_layers_, d_model_;

    void MallocWeights();
    void LoadModel(const std::string& dir_path);
};

}  // namespace fastertransformer
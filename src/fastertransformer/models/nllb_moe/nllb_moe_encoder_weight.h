#pragma once

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"

#include <memory>
#include <vector>

namespace fastertransformer {

template<typename T>
struct NllbMoeEncoderLayerWeight {
public:
    inline NllbMoeEncoderLayerWeight() = default;
    NllbMoeEncoderLayerWeight(const std::string& dir_path, uint64_t layer_index);
    ~NllbMoeEncoderLayerWeight();

    NllbMoeEncoderLayerWeight<T> operator=(const NllbMoeEncoderLayerWeight<T>&) = delete;

    LayerNormWeight<T> self_attn_layer_norm;
    AttentionWeight<T> self_attn;
    LayerNormWeight<T> ff_layer_norm;
    FfnWeight<T>       ffn;

    inline bool is_sparse()
    {
        return is_sparse_;
    }

private:
    uint64_t d_model_;
    bool     is_sparse_;  // use MoE or not
    uint64_t num_experts_, encoder_ffn_dim_;
    bool     router_bias_;

    void MallocWeights();
    void LoadModel(const std::string& dir_path, uint64_t layer_index);
};

template<typename T>
struct NllbMoeEncoderWeight {
public:
    inline NllbMoeEncoderWeight() = default;
    NllbMoeEncoderWeight(const std::string& dir_path, T* shared);
    ~NllbMoeEncoderWeight();

    NllbMoeEncoderWeight<T> operator=(const NllbMoeEncoderWeight<T>&) = delete;

    T* shared = nullptr;
    std::unique_ptr<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>
                                                               sinusoidal_positional_embedding;  // positional embedding
    std::vector<std::unique_ptr<NllbMoeEncoderLayerWeight<T>>> layers;
    LayerNormWeight<T>                                         layer_norm;

private:
    uint64_t encoder_layers_, d_model_;

    void MallocWeights();
    void LoadModel(const std::string& dir_path);
};

}  // namespace fastertransformer
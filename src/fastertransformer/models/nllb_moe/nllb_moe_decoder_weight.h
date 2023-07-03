#pragma once

#include <memory>
#include <string>

#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"
#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

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

private:
    uint64_t decoder_layers_, d_model_;

    void MallocWeights();
    void LoadModel(const std::string& dir_path);
};

}  // namespace fastertransformer
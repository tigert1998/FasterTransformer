#pragma once

#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"

#include <memory>

namespace fastertransformer {

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

private:
};

}  // namespace fastertransformer
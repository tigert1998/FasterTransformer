#pragma once

#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"

#include <string>

#include <memory>
#include <stdint.h>

namespace fastertransformer {

template<typename T>
struct NllbMoeWeight {
public:
    inline NllbMoeWeight() = default;

    NllbMoeWeight(const std::string& dir_path);
    ~NllbMoeWeight();

    NllbMoeWeight<T> operator=(const NllbMoeWeight<T>&) = delete;

    T* shared = nullptr;                  // embedding table
    std::unique_ptr<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>
        sinusoidal_positional_embedding;  // positional embedding

private:
    uint64_t d_model_, vocab_size_;

    void LoadModel(const std::string& dir_path);
    void MallocWeights();
};

}  // namespace fastertransformer

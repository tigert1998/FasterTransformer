#pragma once

#include "3rdparty/INIReader.h"

namespace fastertransformer {

template<typename T>
struct NllbMoeSinusoidalPositionalEmbeddingWeight {
public:
    inline NllbMoeSinusoidalPositionalEmbeddingWeight() = default;

    NllbMoeSinusoidalPositionalEmbeddingWeight(const INIReader& reader);
    ~NllbMoeSinusoidalPositionalEmbeddingWeight();

    NllbMoeSinusoidalPositionalEmbeddingWeight<T>
    operator=(const NllbMoeSinusoidalPositionalEmbeddingWeight<T>&) = delete;

    T* weight = nullptr;

private:
    uint64_t max_position_embeddings_, d_model_, pad_token_id_;

    inline uint64_t num_embeddings()
    {
        return max_position_embeddings_ + 2;
    }

    void CalcPositionalEmbedding();
};

}  // namespace fastertransformer
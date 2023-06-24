#pragma once

#include <string>

#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder_weight.h"

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

    T*                                       shared  = nullptr;  // embedding table
    std::unique_ptr<NllbMoeEncoderWeight<T>> encoder = nullptr;  // encoder

private:
    uint64_t d_model_, vocab_size_;

    void LoadModel(const std::string& dir_path);
    void MallocWeights();
};

}  // namespace fastertransformer

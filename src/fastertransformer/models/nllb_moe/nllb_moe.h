#pragma once

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_weight.h"
#include "src/fastertransformer/utils/Tensor.h"

#include <string>
#include <unordered_map>

namespace fastertransformer {

template<typename T>
class NllbMoe {
public:
    NllbMoe(const INIReader& reader, cudaStream_t stream);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const NllbMoeWeight<T>*                        nllb_moe_weight);

private:
    uint64_t     d_model_;
    int          pad_token_id_;
    cudaStream_t stream_;
};

}  // namespace fastertransformer
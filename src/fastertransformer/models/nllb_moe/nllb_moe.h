#pragma once

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_decoder.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder.h"
#include "src/fastertransformer/models/nllb_moe/nllb_moe_weight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

#include <stdint.h>
#include <string>
#include <unordered_map>

namespace fastertransformer {

template<typename T>
class NllbMoe {
public:
    NllbMoe(const INIReader& reader, cudaStream_t stream, cublasMMWrapper* cublas_wrapper, IAllocator* allocator);
    ~NllbMoe();

    void Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const NllbMoeWeight<T>*                        nllb_moe_weight);

private:
    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;

    uint64_t d_model_;

    std::unique_ptr<NllbMoeEncoder<T>> encoder_;
    std::unique_ptr<NllbMoeDecoder<T>> decoder_;

    T* encoder_hidden_states_ = nullptr;

    void AllocateBuffer(uint64_t batch_size, uint64_t max_input_ids_length);
    void FreeBuffer();
};

}  // namespace fastertransformer
#include "src/fastertransformer/models/nllb_moe/nllb_moe.h"

#include "src/fastertransformer/kernels/nllb_moe_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include <iostream>
#include <vector>

namespace fastertransformer {

template<typename T>
NllbMoe<T>::NllbMoe(const INIReader& reader, cudaStream_t stream, IAllocator* allocator)
{
    stream_    = stream;
    allocator_ = allocator;

    encoder_ = std::make_unique<NllbMoeEncoder<T>>(reader, stream, allocator);
}

template<typename T>
void NllbMoe<T>::Forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         const NllbMoeWeight<T>*                        nllb_moe_weight)
{
    {
        std::unordered_map<std::string, Tensor> output_tensors_for_encoder = {};
        std::unordered_map<std::string, Tensor> input_tensors_for_encoder  = {
            {"input_ids", input_tensors->at("input_ids")}};
        encoder_->Forward(&output_tensors_for_encoder, &input_tensors_for_encoder, nllb_moe_weight->encoder.get());
    }
}

template<typename T>
void NllbMoe<T>::AllocateBuffer(uint64_t batch_size, uint64_t max_input_ids_length, uint64_t d_model)
{
}

template struct NllbMoe<float>;
template struct NllbMoe<half>;
#ifdef ENABLE_BF16
template struct NllbMoe<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
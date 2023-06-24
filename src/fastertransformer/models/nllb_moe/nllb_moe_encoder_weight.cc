#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder_weight.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include "3rdparty/INIReader.h"

namespace fastertransformer {

template<typename T>
NllbMoeEncoderWeight<T>::NllbMoeEncoderWeight(const std::string& dir_path, T* shared): shared(shared)
{
    INIReader reader(dir_path + "/config.ini");
    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);
}

template<typename T>
NllbMoeEncoderWeight<T>::~NllbMoeEncoderWeight()
{
    // We don't free shared pointer here since it is managed by NllbMoeWeight
}

template struct NllbMoeEncoderWeight<float>;
template struct NllbMoeEncoderWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeEncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
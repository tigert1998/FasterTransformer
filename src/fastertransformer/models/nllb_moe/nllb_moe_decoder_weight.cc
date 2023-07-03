#include "src/fastertransformer/models/nllb_moe/nllb_moe_decoder_weight.h"
#include "3rdparty/INIReader.h"

namespace fastertransformer {

template<typename T>
NllbMoeDecoderWeight<T>::NllbMoeDecoderWeight(const std::string& dir_path, T* shared): shared(shared)
{
    INIReader reader(dir_path + "/config.ini");
    decoder_layers_ = reader.GetInteger("nllb_moe", "decoder_layers");
    d_model_        = reader.GetInteger("nllb_moe", "d_model");

    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);

    MallocWeights();
    LoadModel(dir_path);
}

template<typename T>
void NllbMoeDecoderWeight<T>::MallocWeights()
{
}

template<typename T>
void NllbMoeDecoderWeight<T>::LoadModel(const std::string& dir_path)
{
}

template<typename T>
NllbMoeDecoderWeight<T>::~NllbMoeDecoderWeight()
{
}

template struct NllbMoeDecoderWeight<float>;
template struct NllbMoeDecoderWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeDecoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
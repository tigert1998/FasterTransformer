#include "src/fastertransformer/models/nllb_moe/nllb_moe_weight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeWeight<T>::NllbMoeWeight(const std::string& dir_path)
{
    INIReader reader = INIReader(dir_path + "/config.ini");
    FT_CHECK(reader.ParseError() == 0);
    d_model_    = reader.GetInteger("nllb_moe", "d_model");
    vocab_size_ = reader.GetInteger("nllb_moe", "vocab_size");

    MallocWeights();

    LoadModel(dir_path);

    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);
}

template<typename T>
void NllbMoeWeight<T>::LoadModel(const std::string& dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "nllb_moe");
    loadWeightFromBin<T>(shared, {vocab_size_, d_model_}, dir_path + "/model.shared.weight", model_file_type);
}

template<typename T>
void NllbMoeWeight<T>::MallocWeights()
{
    deviceMalloc(&shared, vocab_size_ * d_model_);
}

template<typename T>
NllbMoeWeight<T>::~NllbMoeWeight()
{
    deviceFree(shared);
}

template struct NllbMoeWeight<float>;
template struct NllbMoeWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
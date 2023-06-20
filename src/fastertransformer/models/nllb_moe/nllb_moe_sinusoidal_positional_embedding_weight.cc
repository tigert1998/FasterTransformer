#include <cmath>
#include <vector>

#include "src/fastertransformer/models/nllb_moe/nllb_moe_sinusoidal_positional_embedding_weight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeSinusoidalPositionalEmbeddingWeight<T>::NllbMoeSinusoidalPositionalEmbeddingWeight(const INIReader& reader)
{
    max_position_embeddings_ = reader.GetInteger("nllb_moe", "max_position_embeddings");
    d_model_                 = reader.GetInteger("nllb_moe", "d_model");
    pad_token_id_            = reader.GetInteger("nllb_moe", "pad_token_id");

    CalcPositionalEmbedding();
}

template<typename T>
void NllbMoeSinusoidalPositionalEmbeddingWeight<T>::CalcPositionalEmbedding()
{
    std::vector<float> emb;

    for (int i = 0; i < num_embeddings(); i++) {
        for (int j = 0; j < d_model_; j++) {
            if (i == pad_token_id_ || j >= d_model_ / 2 * 2) {
                emb.push_back(0);
            }
            else {
                uint64_t half_dim = d_model_ / 2;

                float ret = i * std::exp((j % half_dim) * (-std::log(10000) / (half_dim - 1)));
                if (j < d_model_ / 2) {
                    ret = std::sin(ret);
                }
                else {
                    ret = std::cos(ret);
                }
                emb.push_back(ret);
            }
        }
    }

    deviceMalloc(&weight, emb.size(), false);
    if (std::is_same<T, float>::value) {
        cudaH2Dcpy(weight, (T*)emb.data(), emb.size());
    }
    else {
        float* ptr = nullptr;
        deviceMalloc(&ptr, emb.size(), false);
        cudaH2Dcpy(ptr, emb.data(), emb.size());
        invokeCudaD2DcpyConvert(weight, ptr, emb.size());
        deviceFree(ptr);
    }
}

template<typename T>
NllbMoeSinusoidalPositionalEmbeddingWeight<T>::~NllbMoeSinusoidalPositionalEmbeddingWeight()
{
    deviceFree(weight);
}

template struct NllbMoeSinusoidalPositionalEmbeddingWeight<float>;
template struct NllbMoeSinusoidalPositionalEmbeddingWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeSinusoidalPositionalEmbeddingWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder_weight.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include "3rdparty/INIReader.h"

namespace fastertransformer {

template<typename T>
NllbMoeEncoderLayerWeight<T>::NllbMoeEncoderLayerWeight(const std::string& dir_path, uint64_t layer_index)
{
    INIReader reader(dir_path + "/config.ini");
    d_model_ = reader.GetInteger("nllb_moe", "d_model");

    MallocWeights();
    LoadModel(dir_path, layer_index);
}

template<typename T>
NllbMoeEncoderLayerWeight<T>::~NllbMoeEncoderLayerWeight()
{
    deviceFree((T*&)self_attn_layer_norm.gamma);
    deviceFree((T*&)self_attn_layer_norm.beta);
    deviceFree((T*&)self_attn.query_weight.kernel);
    deviceFree((T*&)self_attn.query_weight.bias);
    deviceFree((T*&)self_attn.key_weight.kernel);
    deviceFree((T*&)self_attn.key_weight.bias);
    deviceFree((T*&)self_attn.value_weight.kernel);
    deviceFree((T*&)self_attn.value_weight.bias);
    deviceFree((T*&)self_attn.attention_output_weight.kernel);
    deviceFree((T*&)self_attn.attention_output_weight.bias);
}

template<typename T>
void NllbMoeEncoderLayerWeight<T>::MallocWeights()
{
    deviceMalloc((T**)&self_attn_layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&self_attn_layer_norm.beta, d_model_, false);
    deviceMalloc((T**)&self_attn.query_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&self_attn.query_weight.bias, d_model_, false);
    deviceMalloc((T**)&self_attn.key_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&self_attn.key_weight.bias, d_model_, false);
    deviceMalloc((T**)&self_attn.value_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&self_attn.value_weight.bias, d_model_, false);
    deviceMalloc((T**)&self_attn.attention_output_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&self_attn.attention_output_weight.bias, d_model_, false);
}

template<typename T>
void NllbMoeEncoderLayerWeight<T>::LoadModel(const std::string& dir_path, uint64_t layer_index)
{
    FtCudaDataType model_file_type  = getModelFileType(dir_path + "/config.ini", "nllb_moe");
    std::string    file_path_prefix = dir_path + "/model.encoder.layers." + std::to_string(layer_index);
    loadWeightFromBin<T>(
        (T*)self_attn_layer_norm.gamma, {d_model_}, file_path_prefix + ".self_attn_layer_norm.weight", model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn_layer_norm.beta, {d_model_}, file_path_prefix + ".self_attn_layer_norm.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.query_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".self_attn.q_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn.query_weight.bias, {d_model_}, file_path_prefix + ".self_attn.q_proj.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.key_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".self_attn.k_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn.key_weight.bias, {d_model_}, file_path_prefix + ".self_attn.k_proj.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.value_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".self_attn.v_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn.value_weight.bias, {d_model_}, file_path_prefix + ".self_attn.v_proj.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.attention_output_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".self_attn.out_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)self_attn.attention_output_weight.bias,
                         {d_model_},
                         file_path_prefix + ".self_attn.out_proj.bias",
                         model_file_type);
}

template struct NllbMoeEncoderLayerWeight<float>;
template struct NllbMoeEncoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeEncoderLayerWeight<__nv_bfloat16>;
#endif

template<typename T>
NllbMoeEncoderWeight<T>::NllbMoeEncoderWeight(const std::string& dir_path, T* shared): shared(shared)
{
    INIReader reader(dir_path + "/config.ini");
    encoder_layers_ = reader.GetInteger("nllb_moe", "encoder_layers");

    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);
    for (int i = 0; i < encoder_layers_; i++) {
        layers.emplace_back(std::move(std::make_unique<NllbMoeEncoderLayerWeight<T>>(dir_path, i)));
    }
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
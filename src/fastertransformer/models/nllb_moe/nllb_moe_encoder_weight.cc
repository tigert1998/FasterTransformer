#include "src/fastertransformer/models/nllb_moe/nllb_moe_encoder_weight.h"
#include "src/fastertransformer/utils/memory_utils.h"

#include "3rdparty/INIReader.h"

namespace fastertransformer {

template<typename T>
NllbMoeEncoderLayerWeight<T>::NllbMoeEncoderLayerWeight(const std::string& dir_path, uint64_t layer_index)
{
    INIReader reader(dir_path + "/config.ini");
    d_model_                     = reader.GetInteger("nllb_moe", "d_model");
    uint64_t encoder_sparse_step = reader.GetInteger("nllb_moe", "encoder_sparse_step");
    is_sparse_                   = encoder_sparse_step > 0 ? (layer_index + 1) % encoder_sparse_step == 0 : false;
    num_experts_                 = reader.GetInteger("nllb_moe", "num_experts");
    encoder_ffn_dim_             = reader.GetInteger("nllb_moe", "encoder_ffn_dim");
    router_bias_                 = reader.GetBoolean("nllb_moe", "router_bias", false);

    FT_CHECK(!router_bias_);

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
    deviceFree((T*&)ff_layer_norm.gamma);
    deviceFree((T*&)ff_layer_norm.beta);
    if (is_sparse_) {
        deviceFree((T*&)ffn.gating_weight.kernel);
        if (router_bias_) {
            deviceFree((T*&)ffn.gating_weight.bias);
        }
    }
    deviceFree((T*&)ffn.intermediate_weight.kernel);
    deviceFree((T*&)ffn.intermediate_weight.bias);
    deviceFree((T*&)ffn.output_weight.kernel);
    deviceFree((T*&)ffn.output_weight.bias);
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
    deviceMalloc((T**)&ff_layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&ff_layer_norm.beta, d_model_, false);
    if (is_sparse_) {
        deviceMalloc((T**)&ffn.gating_weight.kernel, d_model_ * num_experts_, false);
        if (router_bias_) {
            deviceMalloc((T**)&ffn.gating_weight.bias, num_experts_, false);
        }
        deviceMalloc((T**)&ffn.intermediate_weight.kernel, num_experts_ * d_model_ * encoder_ffn_dim_, false);
        deviceMalloc((T**)&ffn.intermediate_weight.bias, num_experts_ * encoder_ffn_dim_, false);
        deviceMalloc((T**)&ffn.output_weight.kernel, num_experts_ * encoder_ffn_dim_ * d_model_, false);
        deviceMalloc((T**)&ffn.output_weight.bias, num_experts_ * d_model_, false);
    }
    else {
        deviceMalloc((T**)&ffn.intermediate_weight.kernel, d_model_ * encoder_ffn_dim_, false);
        deviceMalloc((T**)&ffn.intermediate_weight.bias, encoder_ffn_dim_, false);
        deviceMalloc((T**)&ffn.output_weight.kernel, encoder_ffn_dim_ * d_model_, false);
        deviceMalloc((T**)&ffn.output_weight.bias, d_model_, false);
    }
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
    loadWeightFromBin<T>(
        (T*)ff_layer_norm.gamma, {d_model_}, file_path_prefix + ".ff_layer_norm.weight", model_file_type);
    loadWeightFromBin<T>((T*)ff_layer_norm.beta, {d_model_}, file_path_prefix + ".ff_layer_norm.bias", model_file_type);
    if (is_sparse_) {
        loadWeightFromBin<T>((T*)ffn.gating_weight.kernel,
                             {d_model_, num_experts_},
                             file_path_prefix + ".ffn.router.classifier.weight",
                             model_file_type);
        if (router_bias_) {
            loadWeightFromBin<T>((T*)ffn.gating_weight.bias,
                                 {num_experts_},
                                 file_path_prefix + ".ffn.router.classifier.bias",
                                 model_file_type);
        }
        loadWeightFromBin<T>((T*)ffn.intermediate_weight.kernel,
                             {num_experts_ * d_model_, encoder_ffn_dim_},
                             file_path_prefix + ".ffn.fc1.weight",
                             model_file_type);
        loadWeightFromBin<T>((T*)ffn.intermediate_weight.bias,
                             {num_experts_, encoder_ffn_dim_},
                             file_path_prefix + ".ffn.fc1.bias",
                             model_file_type);
        loadWeightFromBin<T>((T*)ffn.output_weight.kernel,
                             {num_experts_ * encoder_ffn_dim_, d_model_},
                             file_path_prefix + ".ffn.fc2.weight",
                             model_file_type);
        loadWeightFromBin<T>(
            (T*)ffn.output_weight.bias, {num_experts_, d_model_}, file_path_prefix + ".ffn.fc2.bias", model_file_type);
    }
    else {
        loadWeightFromBin<T>((T*)ffn.intermediate_weight.kernel,
                             {d_model_, encoder_ffn_dim_},
                             file_path_prefix + ".ffn.fc1.weight",
                             model_file_type);
        loadWeightFromBin<T>(
            (T*)ffn.intermediate_weight.bias, {encoder_ffn_dim_}, file_path_prefix + ".ffn.fc1.bias", model_file_type);
        loadWeightFromBin<T>((T*)ffn.output_weight.kernel,
                             {encoder_ffn_dim_, d_model_},
                             file_path_prefix + ".ffn.fc2.weight",
                             model_file_type);
        loadWeightFromBin<T>(
            (T*)ffn.output_weight.bias, {d_model_}, file_path_prefix + ".ffn.fc2.bias", model_file_type);
    }
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
    d_model_        = reader.GetInteger("nllb_moe", "d_model");

    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);
    for (int i = 0; i < encoder_layers_; i++) {
        layers.emplace_back(std::move(std::make_unique<NllbMoeEncoderLayerWeight<T>>(dir_path, i)));
    }

    MallocWeights();
    LoadModel(dir_path);
}

template<typename T>
NllbMoeEncoderWeight<T>::~NllbMoeEncoderWeight()
{
    // We don't free shared pointer here since it is managed by NllbMoeWeight
    deviceFree((T*&)layer_norm.gamma);
    deviceFree((T*&)layer_norm.beta);
}

template<typename T>
void NllbMoeEncoderWeight<T>::MallocWeights()
{
    deviceMalloc((T**)&layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&layer_norm.beta, d_model_, false);
}

template<typename T>
void NllbMoeEncoderWeight<T>::LoadModel(const std::string& dir_path)
{
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "nllb_moe");

    std::string file_path_prefix = dir_path + "/model.encoder";
    loadWeightFromBin<T>((T*)layer_norm.gamma, {d_model_}, file_path_prefix + ".layer_norm.weight", model_file_type);
    loadWeightFromBin<T>((T*)layer_norm.beta, {d_model_}, file_path_prefix + ".layer_norm.bias", model_file_type);
}

template struct NllbMoeEncoderWeight<float>;
template struct NllbMoeEncoderWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeEncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
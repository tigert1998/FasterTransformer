#include "3rdparty/INIReader.h"

#include "src/fastertransformer/models/nllb_moe/nllb_moe_decoder_weight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
NllbMoeDecoderLayerWeight<T>::NllbMoeDecoderLayerWeight(const std::string& dir_path, uint64_t layer_index)
{
    INIReader reader(dir_path + "/config.ini");
    d_model_                     = reader.GetInteger("nllb_moe", "d_model");
    uint64_t decoder_sparse_step = reader.GetInteger("nllb_moe", "decoder_sparse_step");
    is_sparse_                   = decoder_sparse_step > 0 ? (layer_index + 1) % decoder_sparse_step == 0 : false;

    MallocWeights();
    LoadModel(dir_path, layer_index);
}

template<typename T>
NllbMoeDecoderLayerWeight<T>::~NllbMoeDecoderLayerWeight()
{
    deviceFree((T*&)self_attn_layer_norm.gamma);
    deviceFree((T*&)self_attn_layer_norm.beta);
    deviceFree((T*&)self_attn.query_weight.kernel);
    deviceFree((T*&)self_attn.query_weight.bias);
    deviceFree((T*&)self_attn.attention_output_weight.kernel);
    deviceFree((T*&)self_attn.attention_output_weight.bias);
    deviceFree((T*&)cross_attention_layer_norm.gamma);
    deviceFree((T*&)cross_attention_layer_norm.beta);
    deviceFree((T*&)cross_attention.query_weight.kernel);
    deviceFree((T*&)cross_attention.query_weight.bias);
    deviceFree((T*&)cross_attention.key_weight.kernel);
    deviceFree((T*&)cross_attention.key_weight.bias);
    deviceFree((T*&)cross_attention.value_weight.kernel);
    deviceFree((T*&)cross_attention.value_weight.bias);
    deviceFree((T*&)cross_attention.attention_output_weight.kernel);
    deviceFree((T*&)cross_attention.attention_output_weight.bias);
    deviceFree((T*&)ff_layer_norm.gamma);
    deviceFree((T*&)ff_layer_norm.beta);
}

template<typename T>
void NllbMoeDecoderLayerWeight<T>::MallocWeights()
{
    deviceMalloc((T**)&self_attn_layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&self_attn_layer_norm.beta, d_model_, false);
    deviceMalloc((T**)&self_attn.query_weight.kernel, d_model_ * 3 * d_model_, false);
    deviceMalloc((T**)&self_attn.query_weight.bias, 3 * d_model_, false);
    deviceMalloc((T**)&self_attn.attention_output_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&self_attn.attention_output_weight.bias, d_model_, false);
    deviceMalloc((T**)&cross_attention_layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&cross_attention_layer_norm.beta, d_model_, false);
    deviceMalloc((T**)&cross_attention.query_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&cross_attention.query_weight.bias, d_model_, false);
    deviceMalloc((T**)&cross_attention.key_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&cross_attention.key_weight.bias, d_model_, false);
    deviceMalloc((T**)&cross_attention.value_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&cross_attention.value_weight.bias, d_model_, false);
    deviceMalloc((T**)&cross_attention.attention_output_weight.kernel, d_model_ * d_model_, false);
    deviceMalloc((T**)&cross_attention.attention_output_weight.bias, d_model_, false);
    deviceMalloc((T**)&ff_layer_norm.gamma, d_model_, false);
    deviceMalloc((T**)&ff_layer_norm.beta, d_model_, false);
}

template<typename T>
void NllbMoeDecoderLayerWeight<T>::LoadModel(const std::string& dir_path, uint64_t layer_index)
{
    FtCudaDataType model_file_type  = getModelFileType(dir_path + "/config.ini", "nllb_moe");
    std::string    file_path_prefix = dir_path + "/model.decoder.layers." + std::to_string(layer_index);
    loadWeightFromBin<T>(
        (T*)self_attn_layer_norm.gamma, {d_model_}, file_path_prefix + ".self_attn_layer_norm.weight", model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn_layer_norm.beta, {d_model_}, file_path_prefix + ".self_attn_layer_norm.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.query_weight.kernel,
                         {d_model_, 3 * d_model_},
                         file_path_prefix + ".self_attn.q_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>(
        (T*)self_attn.query_weight.bias, {3 * d_model_}, file_path_prefix + ".self_attn.q_proj.bias", model_file_type);
    loadWeightFromBin<T>((T*)self_attn.attention_output_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".self_attn.out_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)self_attn.attention_output_weight.bias,
                         {d_model_},
                         file_path_prefix + ".self_attn.out_proj.bias",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention_layer_norm.gamma,
                         {d_model_},
                         file_path_prefix + ".cross_attention_layer_norm.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention_layer_norm.beta,
                         {d_model_},
                         file_path_prefix + ".cross_attention_layer_norm.bias",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.query_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".cross_attention.q_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.query_weight.bias,
                         {d_model_},
                         file_path_prefix + ".cross_attention.q_proj.bias",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.key_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".cross_attention.k_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.key_weight.bias,
                         {d_model_},
                         file_path_prefix + ".cross_attention.k_proj.bias",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.value_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".cross_attention.v_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.value_weight.bias,
                         {d_model_},
                         file_path_prefix + ".cross_attention.v_proj.bias",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.attention_output_weight.kernel,
                         {d_model_, d_model_},
                         file_path_prefix + ".cross_attention.out_proj.weight",
                         model_file_type);
    loadWeightFromBin<T>((T*)cross_attention.attention_output_weight.bias,
                         {d_model_},
                         file_path_prefix + ".cross_attention.out_proj.bias",
                         model_file_type);
    loadWeightFromBin<T>(
        (T*)ff_layer_norm.gamma, {d_model_}, file_path_prefix + ".ff_layer_norm.weight", model_file_type);
    loadWeightFromBin<T>((T*)ff_layer_norm.beta, {d_model_}, file_path_prefix + ".ff_layer_norm.bias", model_file_type);
}

template struct NllbMoeDecoderLayerWeight<float>;
template struct NllbMoeDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct NllbMoeDecoderLayerWeight<__nv_bfloat16>;
#endif

template<typename T>
NllbMoeDecoderWeight<T>::NllbMoeDecoderWeight(const std::string& dir_path, T* shared): shared(shared)
{
    INIReader reader(dir_path + "/config.ini");
    decoder_layers_ = reader.GetInteger("nllb_moe", "decoder_layers");
    d_model_        = reader.GetInteger("nllb_moe", "d_model");

    sinusoidal_positional_embedding = std::make_unique<NllbMoeSinusoidalPositionalEmbeddingWeight<T>>(reader);
    for (int i = 0; i < decoder_layers_; i++) {
        layers.emplace_back(std::move(std::make_unique<NllbMoeDecoderLayerWeight<T>>(dir_path, i)));
    }

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
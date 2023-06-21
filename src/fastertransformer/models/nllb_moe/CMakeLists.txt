cmake_minimum_required(VERSION 3.8)

add_library(nllb_moe_weight STATIC nllb_moe_weight.cc nllb_moe_sinusoidal_positional_embedding_weight.cc)
set_property(TARGET nllb_moe_weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nllb_moe_weight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(nllb_moe_weight PUBLIC cuda_utils logger)

add_library(nllb_moe STATIC nllb_moe.cc)
set_property(TARGET nllb_moe PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET nllb_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(nllb_moe PUBLIC -lcudart decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias nllb_moe_weight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils nllb_moe_kernels)
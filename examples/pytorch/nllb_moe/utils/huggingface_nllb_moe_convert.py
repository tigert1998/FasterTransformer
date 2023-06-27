import os
import argparse
import configparser
import re

import numpy as np
import torch
import torch.nn as nn
from transformers import NllbMoeForConditionalGeneration


def fetch_module_by_name(module: nn.Module, name: str):
    if name == "":
        return module
    for name in name.split('.'):
        try:
            idx = int(name)
            module = module[idx]
        except:
            module = getattr(module, name)
    return module


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def convert(saved_dir, in_file, weight_data_type):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NllbMoeForConditionalGeneration.from_pretrained(in_file).to(torch_device)

    hf_config = vars(model.config)
    config = configparser.ConfigParser()
    config["nllb_moe"] = {}
    for name in [
        '_name_or_path',
        'activation_dropout',
        'activation_function',
        'attention_dropout',
        'batch_prioritized_routing',
        'bos_token_id',
        'd_model',
        'decoder_attention_heads',
        'decoder_ffn_dim',
        'decoder_layerdrop',
        'decoder_layers',
        'decoder_sparse_step',
        'decoder_start_token_id',
        'dropout',
        'encoder_attention_heads',
        'encoder_ffn_dim',
        'encoder_layerdrop',
        'encoder_layers',
        'encoder_sparse_step',
        'eos_token_id',
        'expert_capacity',
        'init_std',
        'is_encoder_decoder',
        'max_length',
        'max_position_embeddings',
        'moe_eval_capacity_token_fraction',
        'moe_token_dropout',
        'normalize_router_prob_before_dropping',
        'num_experts',
        'num_hidden_layers',
        'output_router_logits',
        'pad_token_id',
        'router_aux_loss_coef',
        'router_bias',
        'router_ignore_padding_tokens',
        'router_jitter_noise',
        'router_type',
        'router_z_loss_coef',
        'scale_embedding',
        'second_expert_policy',
        'use_cache',
        'vocab_size'
    ]:
        if type(hf_config[name]) in [float, int, bool]:
            config["nllb_moe"][name] = str(hf_config[name])
        else:
            config["nllb_moe"][name] = hf_config[name]
    config["nllb_moe"]["weight_data_type"] = weight_data_type
    with open(saved_dir + "/config.ini", 'w') as configfile:
        config.write(configfile)

    moe_params = {}

    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    for name, param in model.named_parameters():
        np_param: np.ndarray = param.detach().cpu().numpy().astype(np_weight_data_type)
        is_linear = isinstance(fetch_module_by_name(model, os.path.splitext(name)[0]), nn.Linear)
        if is_linear:
            np_param = np_param.T
        search_result = re.search(".experts.expert_(\d+).", name)
        if search_result is None:
            np_param.tofile(os.path.join(saved_dir, name))
        else:
            expert_idx = int(search_result.group(1))
            new_name = re.sub(".experts.expert_\d+", '', name)
            if moe_params.get(new_name, None) is None:
                moe_params[new_name] = []
            if expert_idx >= len(moe_params[new_name]):
                moe_params[new_name] = moe_params[new_name] + [None] * (expert_idx + 1 - len(moe_params[new_name]))
            moe_params[new_name][expert_idx] = np_param

    for name, param_list in moe_params.items():
        param = np.concatenate(
            [np.expand_dims(param, 0) for param in param_list],
            axis=0
        )
        param.tofile(os.path.join(saved_dir, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--saved-dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('--in-file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument("--weight-data-type", type=str, default="fp32", choices=["fp32", "fp16"])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    convert(args.saved_dir, args.in_file, args.weight_data_type)
"""Compare the output of Flax and PyTorch LLaMA models.

# disable cuda for precision
CUDA_VISIBLE_DEVICES='' python diff_m1_flax_pt_output.py

"""
import os.path
import pdb

import torch

torch.set_printoptions(precision=8)
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import flax
from addict import Dict
import json

## Meta LLaMA
from pathlib import Path
from typing import Optional
import fire
from termcolor import cprint
from models.llama3.reference_impl.generation import Llama
from models.llama3.reference_impl.model import Transformer
from models.llama3.api.args import ModelArgs
from models.llama3.api.tokenizer import Tokenizer

## EasyLM
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.models.model import ModelConfig, CausalLM
from ttt.infra.jax_utils import JaxRNG, next_rng, set_random_seed


flax_args = Dict()

BS = 1
flax_args.vocab_size = 128256  # llama3
flax_args.seq_length = 2048
flax_args.seed = 42

pt_args = Dict()

model_size = "1b-llama3.2"

flax_args.weight_path = "params::/nlp/scr/yusun/data/xinhao/retrofit/llama-models/converted_weights/llama_3_dot_2_1B_Base_jax_from_meta/streaming_params"
pt_args.weight_path = "/sphinx/u/ttt/weights/meta-llama/Llama3.2-1B"

flax_args.llama_config_update = dict(
    vocab_size=flax_args.vocab_size,
    max_sequence_length=flax_args.seq_length,
    tie_word_embeddings=True  # TODO: @xinhao: follow llama 3.2 1B
)


def forward_flax_token(input_tokens):
    sharded_rng = next_rng()

    llama_config = ModelConfig.load_config(model_size)
    update_dic = flax_args.llama_config_update
    update_keys = set(update_dic.keys())
    original_keys = set(llama_config.__dict__.keys())
    assert update_keys.issubset(
        original_keys
    ), f"Update keys {update_keys-original_keys} not in llama_config"
    llama_config.update(update_dic)

    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        flax_args.weight_path, disallow_trainstate=True
    )
    params = jax.tree_map(lambda x: x.astype(jnp.float32), params)

    flax_hf_model = CausalLM(llama_config)
    rng_generator = JaxRNG(sharded_rng)
    logits = flax_hf_model.apply(
        params,
        input_tokens,
        deterministic=True,
        rngs=rng_generator(llama_config.rng_keys()),
    ).logits
    logits = jax.device_get(logits)
    return logits


@torch.no_grad()
def forward_pt_token(input_tokens):
    print('forward_pt_token')
    ckpt_dir = pt_args.weight_path
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path="./models/llama3/api/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=4,
        model_parallel_size=None,
    )
    model = generator.model
    model = model.cpu()
    print('model', model)
    input_tokens = torch.from_numpy(input_tokens)
    logits = model(input_tokens, start_pos=0)
    return logits.detach().cpu().numpy()


def load_input_data():
    tokenized_dataset_path = '/nlp/scr/yusun/data/xinhao/retrofit/datasets/pile_dummy_llama3_tokenized_0.6B/' \
                             'tokenizer_name-meta-llama/Meta-Llama-3.1-8B-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/validation.npy'
    tokenized_dataset = np.load(tokenized_dataset_path, mmap_mode="r")
    total_len = len(tokenized_dataset)
    L = flax_args.seq_length
    input_tokens, target_tokens = [], []
    for i in range(BS):
        st = np.random.randint(low=0, high=total_len - L - 1)
        seq = tokenized_dataset[st:st + L + 1]
        input_tokens.append(seq[:L])
        target_tokens.append(seq[1:])
    input_tokens = np.stack(input_tokens, dtype=np.int64)  # [BS,L]
    target_tokens = np.stack(target_tokens, dtype=np.int64)  # [BS,L]
    return input_tokens, target_tokens


def get_matching_statistics(key1, key2, logits_dict, label):
    '''
    Args:
        logits_x: [BS,L,V]
        logits_y: [BS,L,V]
        label: [BS,L]

    Returns:
        Total variance distance
        Diffrent predicted tokens
        Perplexity difference
    '''
    logits_x = logits_dict[key1]
    logits_y = logits_dict[key2]

    prob_x = torch.nn.functional.softmax(logits_x, dim=-1)
    prob_y = torch.nn.functional.softmax(logits_y, dim=-1)

    pred_x = logits_x.argmax(dim=-1)
    pred_y = logits_y.argmax(dim=-1)

    total_var = torch.abs(prob_x - prob_y).max(dim=-1)[0].mean(dim=0).numpy()  # [B,L,V] -> [B,L] -> [L,]
    diff_pred_token = torch.mean((pred_x != pred_y).float(), dim=0).numpy()  # [L,]

    ppl_x = np.exp(torch.nn.functional.cross_entropy(logits_x.view(-1, flax_args.vocab_size), label.view(-1)).numpy())
    ppl_y = np.exp(torch.nn.functional.cross_entropy(logits_y.view(-1, flax_args.vocab_size), label.view(-1)).numpy())

    all_stats = {
        'total_var': total_var,
        'diff_pred_token': diff_pred_token,
        'ppl': {
            key1: ppl_x,
            key2: ppl_y
        },
        'max_logits_diff': torch.abs(logits_x - logits_y).max().numpy(),
        'ppl_diff': np.abs(ppl_x - ppl_y),
        'total_diff_pred_count': (pred_x != pred_y).sum().numpy(),
    }

    for key in all_stats.keys():
        print(f"{key}: {all_stats[key]}")

    return all_stats


if __name__ == "__main__":
    save_path = 'converted_weights/match_inference_jax_meta'
    os.makedirs(save_path, exist_ok=True)
    set_random_seed(flax_args.seed)

    input_tokens, target_tokens = load_input_data()  # np
    print('Input shape: ', input_tokens.shape)

    pt_logits = forward_pt_token(input_tokens)
    flax_logits = forward_flax_token(input_tokens)

    pt_logits = torch.tensor(pt_logits).to('cpu')
    flax_logits = torch.tensor(flax_logits).to('cpu')
    target_tokens = torch.tensor(target_tokens).to('cpu')

    logits_dict = {
        'jax': flax_logits,
        'pt': pt_logits,
    }
    all_stats = get_matching_statistics('jax', 'pt', logits_dict, target_tokens)
    torch.save(all_stats, os.path.join(save_path, 'all_stats.pth'))

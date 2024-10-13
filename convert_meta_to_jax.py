"""
Usage:
python convert_hf_to_easylm.py  \
       --hf_model     /path/hf_format_dir    \
       --output_file /path/easylm_format.easylm   \
       --llama.base_model llama_7b \
       --streaming
"""
import pdb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import time
from pathlib import Path
import torch
import numpy as np
import gc
import json
import math
import shutil
import copy

## JAX
import mlxu
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict
from ttt.models.model import CONFIGS, ModelConfig
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.infra.jax_utils import float_tensor_to_dtype, get_float_dtype_by_name

## Meta LLaMA
from pathlib import Path
from typing import Optional
import fire
from termcolor import cprint
from models.llama3.reference_impl.generation import Llama
from models.llama3.reference_impl.model import Transformer
from models.llama3.api.args import ModelArgs
from models.llama3.api.tokenizer import Tokenizer

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint="/sphinx/u/ttt/weights/meta-llama/Llama3.2-1B",
    model_size='1b-llama3.2',
    update_llama_config='',
    output_dir='./converted_weights/llama_3_dot_2_1B_Base_jax_from_meta',
    float_dtype="fp32",
    streaming=True,
)

def inverse_permute(w, n_heads, input_dim, output_dim):
    '''
    PyTorch weight shape: (Out_dim, In_dim)
    '''
    reshaped_w = w.reshape(n_heads, 2, output_dim // n_heads // 2, input_dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(output_dim, input_dim)
    return inverted_w


def main(argv):
    start = time.time()

    ckpt_dir = FLAGS.load_checkpoint
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    ckpt_path = checkpoints[0]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for k in ckpt.keys():
        ckpt[k] = ckpt[k].float()
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=2048,
        max_batch_size=4,
        **params,
    )
    # tokenizer = Tokenizer(model_path="./models/llama3/api/tokenizer.model")
    # assert model_args.vocab_size == tokenizer.n_words
    # model_meta = Transformer(model_args)
    # model_meta.load_state_dict(ckpt, strict=True)
    # meta_params = sum(p.numel() for p in model_meta.parameters())
    meta_params = sum(p.numel() for p in ckpt.values())
    print(f"Meta LLaMA: number of parameters: {meta_params}")

    ## JAX model def
    llama_config = ModelConfig.from_dict(CONFIGS[FLAGS.model_size])
    if FLAGS.update_llama_config != '':
        llama_config_update = FLAGS.update_llama_config
        update_dic = dict(eval(llama_config_update))
        update_keys = set(update_dic.keys())
        original_keys = set(llama_config.__dict__.keys())
        assert update_keys.issubset(original_keys), f"Update keys {update_keys - original_keys} not in llama_config"
        llama_config.update(update_dic)

    ## Convert Meta to JAX
    print(f"Start convert weight to easylm format...")
    jax_weights = {
        "model": {
            "wte": {"embedding": ckpt["tok_embeddings.weight"].numpy()},
            "ln_f": {"kernel": ckpt["norm.weight"].numpy()},
            "h": {
                "%d"
                % (layer): {
                    "seq_modeling_block": {
                        # "wq": {
                        #     "kernel": inverse_permute(
                        #         ckpt[f"layers.{layer}.self_attn.q_proj.weight"].numpy(),
                        #         llama_config.num_attention_heads,
                        #         llama_config.hidden_size,
                        #         llama_config.hidden_size,
                        #     ).transpose()
                        # },
                        # "wk": {
                        #     "kernel": inverse_permute(
                        #         ckpt[f"layers.{layer}.self_attn.k_proj.weight"].numpy(),
                        #         llama_config.num_key_value_heads,
                        #         llama_config.hidden_size,
                        #         llama_config.hidden_size // (
                        #             llama_config.num_attention_heads
                        #             // llama_config.num_key_value_heads
                        #         ),
                        #     ).transpose()
                        # },
                        # @xinhao: both Meta and EasyLM don't have the weird interleaved way of adding rope like HF
                        "wq": {
                            "kernel": ckpt[f"layers.{layer}.attention.wq.weight"]
                            .numpy().transpose()
                        },
                        "wk": {
                            "kernel": ckpt[f"layers.{layer}.attention.wk.weight"]
                            .numpy().transpose()
                        },
                        "wv": {
                            "kernel": ckpt[f"layers.{layer}.attention.wv.weight"]
                            .numpy().transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"layers.{layer}.attention.wo.weight"]
                            .numpy().transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"layers.{layer}.feed_forward.w1.weight"]
                            .numpy().transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"layers.{layer}.feed_forward.w2.weight"]
                            .numpy().transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"layers.{layer}.feed_forward.w3.weight"]
                            .numpy().transpose()
                        },
                    },
                    "seq_norm": {
                        "kernel": ckpt[f"layers.{layer}.attention_norm.weight"].numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"layers.{layer}.ffn_norm.weight"
                        ].numpy()
                    },
                }
                for layer in range(llama_config.num_hidden_layers)
            },
        },
        # "lm_head": {"kernel": ckpt["lm_head.weight"].numpy().transpose()},
    }
    if not llama_config.tie_word_embeddings:
        jax_weights["lm_head"] = {"kernel": ckpt["output.weight"].numpy().transpose()}
    print(f"Convert weight to easylm format finished...")
    jax_params = sum(x.size for x in jax.tree_util.tree_leaves(jax_weights))
    print(f"EasyLM LLaMA: number of parameters: {jax_params}")

    print(f"Start to save...")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    output_file = os.path.join(FLAGS.output_dir, 'streaming_params')
    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights,
            output_file,
            float_dtype=get_float_dtype_by_name(FLAGS.float_dtype),
        )
    else:
        with mlxu.open_file(output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {output_file}"
    )


if __name__ == "__main__":
    mlxu.run(main)

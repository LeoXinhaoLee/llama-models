"""
An implementation of Blockwise parallel transformer https://arxiv.org/abs/2305.19370
Also include a reference implementation of memory-efficient transformer https://arxiv.org/abs/2112.05682
"""

import functools
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange

"""
Computing ffn blockwise without materializing the large hidden tensor, training
4x longer sequences than the memory-efficient transformer.
Blockwise parallel transformer https://arxiv.org/abs/2305.19370 Liu et al. 2023
"""
def blockwise_ffn(remat_ffn, inputs, chunk_size=2048, deterministic=True):
    # remat_ffn: a rematerialized ffn with policy jax.checkpoint_policies.nothing_saveable()
    # inputs: (batch, seq_len, dim)
    # chunk_size: the chunk size to split the sequence
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_ffn(remat_ffn, carry, hidden_states):
        # outputs = remat_ffn(hidden_states, deterministic=deterministic)
        outputs = remat_ffn(hidden_states, deterministic)  # @xinhao: when mlp is rematted, should directly pass `deterministic` instead of using keyword. Otherwise, `deterministic` will be ignored.
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, res = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_ffn, None, inputs)
    res = rearrange(res, 'b c n d -> b (c n) d')
    return res

class Carry(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array

"""
Compute attention blockwise without materializing the full attention matrix,
initially proposed in memory-efficient transformer https://arxiv.org/abs/2112.05682 Rabe et al. 2021;
flash attention https://arxiv.org/abs/2205.14135 Dao et al. 2022 proposes a CUDA
efficient implementation; blockwise parallel transformer https://arxiv.org/abs/2305.19370
Liu et al. 2023 proposes blockwise computing both attention and FFN, enabling 4x
longer sequences than memory-efficient/flash-attention and fusion of attention and FFN.
"""
def blockwise_attn(
        query, key, value,
        bias=None,
        deterministic=True,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal=True,
        query_chunk_size=2048,
        key_chunk_size=2048,
        dtype=jnp.float32,
        policy=jax.checkpoint_policies.nothing_saveable(),
        precision=None,
        float32_logits=True,
        prevent_cse=True,
    ):
    # query, key, value: (batch, seq_len, num_heads, dim_per_head)
    # bias: (batch, seq_len) can be used to mask out attention (e.g. padding)
    # causal: whether to use causal mask
    # policy: one of jax.checkpoint_policies
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            # bias_dim is either 1 or exactly batch/num_heads/q_len/kv_len
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)  # [B,1,q_cs,k_cs] -> [B,q_cs,1,k_cs]
            attn_weights = attn_weights + bias_chunk     # [B,q_cs,h,k_cs] + [B,q_cs,1,k_cs]

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)  # [B,q_cs,h,1]
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)            # [B,q_cs,h,k_cs]
            exp_values = jnp.einsum(
                'bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision
            )  # [.,q_cs,.,kv_cs] @ [.,kv_cs,.,d] -> [.,q_cs,.,d]
            correction = jnp.exp(prev_max_score - max_score)  # [B,q_cs,h,1]
            numerator = numerator * correction + exp_values   # [B,q_cs,h,d] * [B,q_cs,h,1] + [B,q_cs,h,d] -> [B,q_cs,h,d]
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)  #  * [B,q_cs,h,1] + [B,q_cs,h,1] ->
            return Carry(numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),  # @xinhao: TODO: why denominator has d? Should be [BS,q_cs,h,1]?
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(
        lambda _, x: ((), scan_attention(x)),
        (), xs=(query, jnp.arange(0, num_q))
    )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res


def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, deterministic, attn_dropout, attn_pdrop, causal,
            dtype, query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )  # [B,1,1,T] or [B,1,T,T] -> [B,1,1,k_cs] or [B,1,q_cs,k_cs]

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)  # [q_cs, 1]
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)      # [1, k_cs]
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min  # [q_cs, k_cs]
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)  # [B,1, q_cs, k_cs]

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)

###
# xinhao: CDM
###
def blockwise_attn_cdm(
        query, key, value,
        doc_id=None,
        deterministic=True,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal=True,
        query_chunk_size=2048,
        key_chunk_size=2048,
        dtype=jnp.float32,
        policy=jax.checkpoint_policies.nothing_saveable(),
        precision=None,
        float32_logits=True,
        prevent_cse=True,
    ):
    # query, key, value: (batch, seq_len, num_heads, dim_per_head)
    # bias: (batch, seq_len) can be used to mask out attention (e.g. padding)
    # causal: whether to use causal mask
    # policy: one of jax.checkpoint_policies
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    assert doc_id.shape[0] == batch and doc_id.shape[1] == kv_len
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias_cdm,
        query_chunk_size, key_chunk_size, doc_id, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)  # [B,1,q_cs,k_cs] -> [B,q_cs,1,k_cs]
            attn_weights = attn_weights + bias_chunk     # [B,q_cs,h,k_cs] + [B,q_cs,1,k_cs]

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)  # [B,q_cs,h,1]
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)            # [B,q_cs,h,k_cs]
            exp_values = jnp.einsum(
                'bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision
            )  # [.,q_cs,.,kv_cs] @ [.,kv_cs,.,d] -> [.,q_cs,.,d]
            correction = jnp.exp(prev_max_score - max_score)  # [B,q_cs,h,1]
            numerator = numerator * correction + exp_values   # [B,q_cs,h,d] * [B,q_cs,h,1] + [B,q_cs,h,d] -> [B,q_cs,h,d]
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)  #  * [B,q_cs,h,1] + [B,q_cs,h,1] ->
            return Carry(numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),  # @xinhao: TODO: why denominator has d? Should be [BS,q_cs,h,1]?
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(
        lambda _, x: ((), scan_attention(x)),
        (), xs=(query, jnp.arange(0, num_q))
    )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res


def _chunk_attention_bias_cdm(
    query_chunk_size, key_chunk_size,
    doc_id, deterministic, attn_dropout, attn_pdrop, causal,
    dtype, query_chunk_idx, key_chunk_idx
):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)

    query_doc_id_chunk = lax.dynamic_slice(
        doc_id,
        start_indices=(0, query_offset),
        slice_sizes=(doc_id.shape[0], query_chunk_size)
    )
    key_doc_id_chunk = lax.dynamic_slice(
        doc_id,
        start_indices=(0, key_offset),
        slice_sizes=(doc_id.shape[0], key_chunk_size)
    )
    query_doc_id_chunk = query_doc_id_chunk.reshape(-1, 1, query_chunk_size, 1)  # [B,q_cs] -> [B,1,q_cs,1]
    key_doc_id_chunk = key_doc_id_chunk.reshape(-1, 1, 1, key_chunk_size)            # [B,k_cs] -> [B,1,1,k_cs]
    chunk_cdm_mask = (query_doc_id_chunk != key_doc_id_chunk) * jnp.finfo(dtype).min  # [B,1, q_cs,k_cs]
    chunk_bias += chunk_cdm_mask

    if causal:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)  # [q_cs, 1]
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)      # [1, k_cs]
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min  # [q_cs, k_cs]
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)  # [B,1, q_cs, k_cs]

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
    return chunk_bias.astype(dtype)


def blockwise_ttt(
    remat_ffn, inputs, input_ids, position_ids, cache,
    deterministic=True, output_ttt_stats=False, ttt_lr_mult=1.,
    chunk_size=2048
):
    inputs = rearrange(inputs, 'b (n c) d -> n b c d', c=chunk_size)
    position_ids = rearrange(position_ids, 'b (n c) -> n b c', c=chunk_size)

    def scan_ttt(remat_ffn, carry_cache, hidden_states, position_ids):
        outputs, stats, carry_cache = remat_ffn(
            hidden_states,
            input_ids,
            position_ids,
            carry_cache,
            deterministic,
            output_ttt_stats,
            ttt_lr_mult
        )
        return carry_cache, (outputs, stats)

    cache, res = nn.scan(
        scan_ttt,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=0,
        out_axes=0,
    )(
        remat_ffn,
        cache,  # initial value of carry
        inputs,
        position_ids
    )

    outputs, stats = res
    outputs = rearrange(outputs, 'n b c d -> b (n c) d')
    stats = jax.tree_util.tree_map(lambda x: rearrange(x, 'n c -> (n c)'), stats)
    return outputs, stats, cache


def blockwise_block_group(
    remat_ffn,
    hidden_states,
    input_ids,
    attention_mask,
    position_ids,
    cache_list,
    ttt_lr_mult=1.,
    deterministic=True,
    init_cache=False,
    output_ttt_stats=True,
    fcm_mask=None,
    target_tokens=None,
    loss_masks=None,
    word_embeddings=None,
    chunk_size=256,
    is_first=False,
):
    position_ids = rearrange(position_ids, 'b (n c) -> n b c', c=chunk_size)
    target_tokens = rearrange(target_tokens, 'b (n c) -> n b c', c=chunk_size)
    loss_masks = rearrange(loss_masks, 'b (n c) -> n b c', c=chunk_size)
    input_ids = rearrange(input_ids, 'b (n c) -> n b c', c=chunk_size)

    # NOTE: First block doesn't have any hidden states
    inputs = (input_ids, position_ids, target_tokens, loss_masks) if is_first else (
        rearrange(hidden_states, 'b (n c) d -> n b c d', c=chunk_size),
        input_ids,
        position_ids,
        target_tokens,
        loss_masks
    )

    def scan_body(remat_ffn, carry_list, inputs):
        hidden_states = inputs[0] if not is_first else None
        input_ids, position_ids, target_tokens, loss_masks = inputs[-4:]

        res_tuple, carry_list = remat_ffn(
            hidden_states,
            input_ids,
            attention_mask,
            position_ids,
            carry_list,
            ttt_lr_mult,
            deterministic,
            init_cache,
            output_ttt_stats,
            None,
            target_tokens,
            loss_masks,
            word_embeddings
        )
        return carry_list, res_tuple

    cache_list, res = nn.scan(
        scan_body,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=0,
        out_axes=0,
    )(remat_ffn, cache_list, inputs)

    outputs, all_ttt_stats = rearrange(res[0], 'n b c d -> b (n c) d'), res[1]

    if output_ttt_stats:
        all_ttt_stats = jax.tree_util.tree_map(lambda x: rearrange(x, 'n c -> (n c)'), all_ttt_stats)

    return (outputs, all_ttt_stats), cache_list


def blockwise_lm_head(remat_ffn, hidden_states, target_tokens, loss_masks, word_embeddings, chunk_size=2048):
    """
    Inputs:
        remat_ffn: rematted LMHead
        hidden_states: [B,T,D]
        target_tokens: [B,T]
        loss_masks: [B,T]
        word_embeddings: None or [V,D]
    """
    hidden_states = rearrange(hidden_states, 'b (n c) d -> n b c d', c=chunk_size)
    target_tokens = rearrange(target_tokens, 'b (n c) -> n b c', c=chunk_size)
    loss_masks = rearrange(loss_masks, 'b (n c) -> n b c', c=chunk_size)
    inputs = (hidden_states, target_tokens, loss_masks)

    def scan_ffn(remat_ffn, carry, inputs):
        hidden_states, target_tokens, loss_masks = inputs
        lm_loss_block_sum = remat_ffn(hidden_states, target_tokens, loss_masks, word_embeddings)
        return carry, lm_loss_block_sum

    _, lm_loss_block_sum = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=0,
        out_axes=0,
    )(remat_ffn, None, inputs)  # [num_block, B]

    lm_loss_sum = jnp.sum(lm_loss_block_sum, axis=0)  # [B,]

    return lm_loss_sum


if __name__ == '__main__':
    # test
    def reference_attn(query, key, value, causal, dtype):
        query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
        logits = jnp.einsum("bqhc,bkhc->bhqk", query, key)
        if causal:
            mask_value = jnp.finfo(logits.dtype).min
            _, q_seq_len, _, _ = query.shape
            _, kv_seq_len, _, _ = key.shape
            mask_shape = (q_seq_len, kv_seq_len)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            causal_mask = (row_ids < col_ids)[None, None, :, :]
            logits = logits + jnp.where(causal_mask, mask_value, 0.0)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhqk,bkhc->bqhc", weights, value)
        return out

    # random inputs
    shape = (1, 32, 8, 64)
    query = jax.random.normal(jax.random.PRNGKey(0), shape)
    key = jax.random.normal(jax.random.PRNGKey(1), shape)
    value = jax.random.normal(jax.random.PRNGKey(2), shape)

    causal = True
    chunk_size = 4
    policy = jax.checkpoint_policies.nothing_saveable()

    blockwise = blockwise_attn(query, key, value, None, False, None, 0.0, causal, chunk_size, chunk_size, jnp.float32, policy, 'float32', True, False)
    reference = reference_attn(query, key, value, causal, 'float32')

    assert jnp.allclose(reference, blockwise, atol=1e-6)

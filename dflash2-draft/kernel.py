"""The DFlash2 draft model, ported to this track's conventions.

WHAT THIS IS
------------
The drafter for the 50 tok/s path. It is a *block-diffusion* drafter: instead of
one token at a time, it proposes a whole block of tokens in a single pass and
keeps the top candidates at every position; a lightweight selector then traces
one coherent path through them. The target verifies the block in one pass (our
chunked path, `../multi-token-verification/`) and the standard speculative rule
accepts the longest matching prefix plus one.

WHY THIS DRAFTER
----------------
Its acceptance length for *this exact base architecture* is published
(sourced: `huggingface.co/z-lab/Qwen3.8-27B-DFlash2`, measured by them against
the bf16 Qwen3.8-27B on an H200):

| drafter | GSM8K | MATH-500 | MBPP | HumanEval | MT-Bench |
| --- | --- | --- | --- | --- | --- |
| **DFlash 2** | **5.46** | **5.28** | **4.79** | **4.39** | **4.10** |
| MTP | 5.02 | 4.72 | 3.99 | 3.91 | 3.74 |
| DSpark | 4.36 | 3.92 | 3.51 | 3.30 | 3.01 |

(acceptance length = completed tokens per verification step, with seven draft
tokens proposed per step.)

The draft is small: **five layers**, hidden 5120 -- the same width as our
target -- 32 query heads over 8 kv heads, head dim 128, sliding window 2048,
and the target's own vocabulary (248,320). It reuses the *target's* embedding
and the *target's* LM head, so it carries no embedding of its own.

WHAT IT NEEDS FROM OUR KERNEL
-----------------------------
The draft reads the target's hidden states at layers **5, 19, 33, 47 and 61**
(`dflash_config.target_layer_ids`), concatenated into a 5H feature row per
committed token. Our fused 64-layer step does not expose them yet; that is the
one change the kernel still needs.

PROVENANCE AND LICENCE
----------------------
Ported from `Inferact/tpu-megakernels`, `qwen/dflash.py` (Apache-2.0,
github.com/Inferact/tpu-megakernels), the reference implementation behind
"700 TPS on Kimi K3: A Case for TPU Megakernels". The structure -- the
two-tap dynamic convolutions, the bidirectional block attention, the
codebook-scored candidate selector -- is that work; the rewiring onto this
track's ternary target, its embedding and its LM head is ours.

    Copyright 2026 Recursive Superintelligence, Inc.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

The draft's own weights are Apache-2.0
(`huggingface.co/z-lab/Qwen3.8-27B-DFlash2`).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
class DraftConfig:
    """The DFlash2 draft's configuration, from its `config.json`."""

    def __init__(self, cfg):
        d = cfg["dflash_config"]
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.intermediate = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.sliding_window = cfg["sliding_window"]
        self.block_size = d["block_size"]
        self.conv_kernel = d["conv_kernel_size"]
        self.conv_group = d["conv_group_size"]
        self.mask_token_id = d["mask_token_id"]
        self.selector_rank = d["selector_rank"]
        self.selector_top_k = d["selector_top_k"]
        self.target_layer_ids = d["target_layer_ids"]
        # the feature row is the target's hidden states at those layers, joined
        self.feature_dim = self.hidden * len(self.target_layer_ids)


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------
def rms_norm(x, w, eps):
    """Qwen3RMSNorm: fp32 normalize, cast back, then scale (torch's order)."""
    v = jnp.mean(x.astype(F32) ** 2, axis=-1, keepdims=True)
    return (x.astype(F32) * jax.lax.rsqrt(v + eps)).astype(x.dtype) * w


def _convolve(hidden, dynamic, base, group_size):
    """hidden [L, H]; dynamic [L, K, G]; base [K, H] -> [L, H]."""
    L, H = hidden.shape
    G = H // group_size
    blocks = hidden.reshape(L, G, group_size)
    out = jnp.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else jnp.pad(blocks[:-offset],
                                                   ((offset, 0), (0, 0), (0, 0)))
        kernel = base[offset].reshape(1, G, group_size)
        out = out + kernel * values + dynamic[:, offset, :, None] * values
    return out.reshape(L, H)


def conv_prepare(hidden, cw, cfg):
    """The pre-attention / MLP conv; returns (convolved, dynamic_for_finish)."""
    groups = hidden.shape[-1] // cfg.conv_group
    K = cfg.conv_kernel
    dynamic = jnp.dot(hidden, cw["kernel_projection"].T, preferred_element_type=F32)
    dynamic = dynamic.reshape(*hidden.shape[:-1], 2, K, groups).astype(hidden.dtype)
    return (
        _convolve(hidden, dynamic[:, 0], cw["base_kernel"][0], cfg.conv_group),
        dynamic[:, 1],
    )


def conv_finish(hidden, dynamic, cw, cfg):
    return _convolve(hidden, dynamic, cw["base_kernel"][1], cfg.conv_group)


def rope_tables(theta, head_dim, positions):
    """positions [P] -> (cos, sin) [P, head_dim], bf16, split-half convention."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=F32) * 2.0 / head_dim))
    angles = positions.astype(F32)[:, None] * inv
    return tuple(
        jnp.concatenate((f(angles), f(angles)), axis=-1).astype(jnp.bfloat16)
        for f in (jnp.cos, jnp.sin)
    )


def apply_rope(vec, cos, sin):
    """vec [L, h, d]; cos/sin [L, d] -> the rotate_half convention."""
    half = cos.shape[-1] // 2
    a, b = vec[..., :half], vec[..., half:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return jnp.concatenate(
        (a * cos[..., :half] - b * sin[..., :half],
         b * cos[..., half:] + a * sin[..., half:]),
        axis=-1,
    ).astype(vec.dtype)


def draft_layer(h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all,
                kpos, qpos, kvalid):
    """One decoder layer. h [B, H] the block stream; ctx [C, H] the context.

    The attention is *bidirectional* over the context and the block: the block
    is a diffusion draft, not a causal continuation, so the block tokens see
    each other. The keys before `key_lo` are padding.
    """
    B = h.shape[0]
    residual = h
    hn = rms_norm(h, layer["input_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["attention_conv"], cfg)
    q = jnp.dot(hn, layer["q_proj"].T, preferred_element_type=F32).astype(h.dtype)
    q = rms_norm(q.reshape(B, cfg.heads, cfg.head_dim), layer["q_norm"], cfg.eps)
    k_ctx = jnp.dot(ctx, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype)
    k_noise = jnp.dot(hn, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype)
    k = jnp.concatenate([k_ctx, k_noise], axis=0)
    k = rms_norm(k.reshape(-1, cfg.kv_heads, cfg.head_dim), layer["k_norm"], cfg.eps)
    v = jnp.concatenate(
        [
            jnp.dot(ctx, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
            jnp.dot(hn, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
        ],
        axis=0,
    ).reshape(-1, cfg.kv_heads, cfg.head_dim)
    q = apply_rope(q, cos_blk, sin_blk)
    k = apply_rope(k, cos_all, sin_all)
    grp = cfg.heads // cfg.kv_heads
    qg = q.transpose(1, 0, 2).reshape(cfg.kv_heads, grp, B, cfg.head_dim)
    kt = k.transpose(1, 0, 2).astype(F32)
    scores = jnp.einsum("gibd,gtd->gibt", qg.astype(F32), kt) * cfg.head_dim ** -0.5
    dist = jnp.abs(qpos[:, None] - kpos[None, :])
    keep = (dist < cfg.sliding_window) & kvalid[None, :]
    scores = jnp.where(keep[None, None], scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("gibt,gtd->gibd", probs, v.transpose(1, 0, 2).astype(F32))
    out = out.reshape(cfg.heads, B, cfg.head_dim).transpose(1, 0, 2).reshape(B, -1)
    out = jnp.dot(out, layer["o_proj"].T, preferred_element_type=F32).astype(h.dtype)
    out = conv_finish(out, dyn, layer["attention_conv"], cfg)
    h = residual + out
    residual = h
    hn = rms_norm(h, layer["post_attention_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["mlp_conv"], cfg)
    gate = jax.nn.silu(jnp.dot(hn, layer["gate_proj"].T, preferred_element_type=F32))
    up = jnp.dot(hn, layer["up_proj"].T, preferred_element_type=F32)
    out = jnp.dot((gate * up).astype(h.dtype), layer["down_proj"].T,
                  preferred_element_type=F32)
    out = conv_finish(out.astype(h.dtype), dyn, layer["mlp_conv"], cfg)
    return h + out


def draft_forward(params, tgt, anchor_and_masks, feats, ctx_pos, ctx_valid,
                  blk_pos, cfg):
    """One draft round.

    anchor_and_masks: [B] the anchor token and the mask tokens.
    feats: [C, 5H] the target's features at the absolute positions ctx_pos.
    ctx_valid: [C] bool, the committed non-pad slots only.
    blk_pos: [B] the absolute positions of the block.

    Returns the draft's hidden states at the block positions [B, H] (post final
    norm, pre the LM head).
    """
    noise = tgt["emb"][anchor_and_masks]              # [B, H]
    ctx = jnp.dot(feats, params["fc"].T, preferred_element_type=F32)
    ctx = rms_norm(ctx.astype(noise.dtype), params["hidden_norm"], cfg.eps)
    all_pos = jnp.concatenate([ctx_pos, blk_pos])
    kvalid = jnp.concatenate([ctx_valid, jnp.ones(len(blk_pos), bool)])
    cos_all, sin_all = rope_tables(cfg.theta, cfg.head_dim, all_pos)
    cos_blk, sin_blk = cos_all[-len(blk_pos):], sin_all[-len(blk_pos):]
    h = noise
    for layer in params["layers"]:
        h = draft_layer(h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all,
                        all_pos.astype(F32), blk_pos.astype(F32), kvalid)
    return rms_norm(h, params["norm"], cfg.eps)


def draft_logits(hidden, tgt):
    """The LM head of the TARGET model, applied to the draft's hidden states."""
    return jnp.dot(hidden, tgt["lmw"], preferred_element_type=F32)


def select_path(hidden, logits, anchor_id, selector):
    """Greedy candidate-path selection (the torch CandidateSelector at T=0).

    hidden [B, H]; logits [B, V]; anchor_id int. Returns (path [B], cand [B, K],
    unary [B, K]): the path, the candidates per position, and their unary
    scores.
    """
    K = selector["top_k"]
    unary, cand = jax.lax.top_k(logits, K)       # sorted desc
    hproj = jnp.dot(hidden, selector["hidden_projection"].T,
                    preferred_element_type=F32)
    pred = anchor_id
    path = []
    for i in range(hidden.shape[0]):
        cond = selector["predecessor_codebook"][pred] * hproj[i]      # [R]
        scores = unary[i] + selector["successor_codebook"][cand[i]] @ cond
        idx = jnp.argmax(scores)
        pred = cand[i, idx]
        path.append(pred)
    return jnp.stack(path).astype(jnp.int32), cand, unary


def load_draft(path):
    """Load the DFlash2 draft checkpoint into stacked numpy arrays."""
    from safetensors import safe_open

    path = str(path)
    with open(Path(path) / "config.json") as f:
        cfg = DraftConfig(json.load(f))
    raw = {}
    with safe_open(Path(path) / "model.safetensors", framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    params = {
        "fc": raw["fc.weight"],
        "hidden_norm": raw["hidden_norm.weight"],
        "norm": raw["norm.weight"],
        "selector": {
            "hidden_projection": raw["candidate_selector.hidden_projection.weight"],
            "predecessor_codebook": raw["candidate_selector.predecessor_codebook"],
            "successor_codebook": raw["candidate_selector.successor_codebook"],
            "top_k": cfg.selector_top_k,
        },
    }
    layers = {}
    for name in ("input_layernorm", "post_attention_layernorm"):
        layers[name] = np.stack(
            [raw[f"layers.{i}.{name}.weight"] for i in range(cfg.layers)])
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        layers[name] = np.stack(
            [raw[f"layers.{i}.self_attn.{name}.weight"] for i in range(cfg.layers)])
    for name in ("q_norm", "k_norm"):
        layers[name] = np.stack(
            [raw[f"layers.{i}.self_attn.{name}.weight"] for i in range(cfg.layers)])
    for name in ("gate_proj", "up_proj", "down_proj"):
        layers[name] = np.stack(
            [raw[f"layers.{i}.mlp.{name}.weight"] for i in range(cfg.layers)])
    for conv in ("attention_conv", "mlp_conv"):
        layers[f"{conv}.base_kernel"] = np.stack(
            [raw[f"layers.{i}.{conv}.base_kernel"] for i in range(cfg.layers)])
        layers[f"{conv}.kernel_projection"] = np.stack(
            [raw[f"layers.{i}.{conv}.kernel_projection.weight"]
             for i in range(cfg.layers)])
    params["layers"] = layers
    return params, cfg


def pack_draft(params, cfg):
    """Restructure the stacked layer arrays into the per-layer dict form the
    forward pass expects."""
    layers = []
    for i in range(cfg.layers):
        layers.append({
            "input_layernorm": params["layers"]["input_layernorm"][i],
            "post_attention_layernorm": params["layers"]["post_attention_layernorm"][i],
            "q_proj": params["layers"]["q_proj"][i],
            "k_proj": params["layers"]["k_proj"][i],
            "v_proj": params["layers"]["v_proj"][i],
            "o_proj": params["layers"]["o_proj"][i],
            "q_norm": params["layers"]["q_norm"][i],
            "k_norm": params["layers"]["k_norm"][i],
            "gate_proj": params["layers"]["gate_proj"][i],
            "up_proj": params["layers"]["up_proj"][i],
            "down_proj": params["layers"]["down_proj"][i],
            "attention_conv": {
                "base_kernel": params["layers"]["attention_conv.base_kernel"][i],
                "kernel_projection":
                    params["layers"]["attention_conv.kernel_projection"][i],
            },
            "mlp_conv": {
                "base_kernel": params["layers"]["mlp_conv.base_kernel"][i],
                "kernel_projection":
                    params["layers"]["mlp_conv.kernel_projection"][i],
            },
        })
    out = dict(params)
    out["layers"] = layers
    return out


# --------------------------------------------------------------------------
# the contract: shapes, the selector, and the rope convention
# --------------------------------------------------------------------------
def _fake_cfg():
    """The real DFlash2 configuration, from its published config.json."""
    return DraftConfig({
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000.0},
        "sliding_window": 2048,
        "dflash_config": {
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
    })


def _fake_params(cfg, rng, vocab=None):
    """Random parameters with the published shapes."""
    H, L, HD = cfg.hidden, cfg.layers, cfg.head_dim
    vocab = vocab or 512
    p = {
        "fc": rng.standard_normal((H, cfg.feature_dim)).astype(np.float32) * 0.02,
        "hidden_norm": np.ones(H, np.float32),
        "norm": np.ones(H, np.float32),
        "selector": {
            # applied as dot(hidden, w.T), so the stored shape is (rank, hidden)
            "hidden_projection":
                rng.standard_normal((cfg.selector_rank, H)).astype(np.float32) * 0.02,
            "predecessor_codebook":
                rng.standard_normal((vocab, cfg.selector_rank)).astype(np.float32) * 0.02,
            "successor_codebook":
                rng.standard_normal((vocab, cfg.selector_rank)).astype(np.float32) * 0.02,
            "top_k": cfg.selector_top_k,
        },
        "layers": {},
    }
    for name in ("input_layernorm", "post_attention_layernorm"):
        p["layers"][name] = np.ones((L, H), np.float32)
    for name, out, inn in (("q_proj", cfg.heads * cfg.head_dim, cfg.hidden),
                           ("k_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden),
                           ("v_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden),
                           ("o_proj", cfg.hidden, cfg.heads * cfg.head_dim)):
        p["layers"][name] = (rng.standard_normal((L, out, inn)).astype(np.float32) * 0.02)
    for name in ("q_norm", "k_norm"):
        p["layers"][name] = np.ones((L, HD), np.float32)
    for name in ("gate_proj", "up_proj"):
        p["layers"][name] = (rng.standard_normal((L, cfg.intermediate, H))
                             .astype(np.float32) * 0.02)
    p["layers"]["down_proj"] = (rng.standard_normal((L, H, cfg.intermediate))
                                .astype(np.float32) * 0.02)
    for conv in ("attention_conv", "mlp_conv"):
        # base_kernel holds TWO taps per layer: [0] is the pre-attention/MLP
        # conv and [1] the one that finishes it, each (K, H).
        p["layers"][f"{conv}.base_kernel"] = (
            rng.standard_normal((L, 2, cfg.conv_kernel, H)).astype(np.float32) * 0.1)
        # a torch Linear's weight is (out, in), and the forward pass does
        # `dot(hidden, w.T)` -- so the projection is (out, H), not (H, out).
        # The output is two taps x kernel x groups (conv_prepare reshapes it
        # to (..., 2, K, groups)).
        p["layers"][f"{conv}.kernel_projection"] = (
            rng.standard_normal(
                (L, 2 * cfg.conv_kernel * (H // cfg.conv_group), H))
            .astype(np.float32) * 0.02)
    return pack_draft(p, cfg)


def test_port(verbose=True):
    """The port must run, with the published shapes, and pick its path from
    its own candidates."""
    rng = np.random.default_rng(20261022)
    cfg = _fake_cfg()
    # the test runs on a small fake vocabulary, so the mask token has to fit
    # inside it; the id itself is only an index into the embedding
    cfg.mask_token_id = 500
    params = _fake_params(cfg, rng)
    B = cfg.block_size
    C = 12                                  # a short context
    vocab = 512
    hidden = rng.standard_normal((B, cfg.hidden)).astype(np.float32)
    tgt = {
        "emb": rng.standard_normal((vocab, cfg.hidden)).astype(np.float32),
        # the LM head is applied untransposed, so it is (hidden, vocab)
        "lmw": rng.standard_normal((cfg.hidden, vocab)).astype(np.float32) * 0.02,
    }
    anchor = np.int32(7)
    anchor_and_masks = jnp.concatenate(
        [jnp.asarray([anchor]), jnp.full(B - 1, cfg.mask_token_id, jnp.int32)])
    feats = rng.standard_normal((C, cfg.feature_dim)).astype(np.float32)
    ctx_pos = jnp.arange(C, dtype=jnp.int32)
    ctx_valid = jnp.ones(C, bool)
    blk_pos = jnp.arange(C, C + B, dtype=jnp.int32)

    out = draft_forward(params, tgt, anchor_and_masks, feats, ctx_pos,
                        ctx_valid, blk_pos, cfg)
    assert out.shape == (B, cfg.hidden), out.shape
    assert np.isfinite(np.asarray(out)).all(), "the draft produced a non-finite value"

    logits = draft_logits(out, tgt)
    assert logits.shape == (B, vocab), logits.shape
    path, cand, unary = select_path(out, logits, anchor, params["selector"])
    assert path.shape == (B,), path.shape
    assert cand.shape == (B, cfg.selector_top_k), cand.shape
    # the path must be drawn from the candidates at each position
    for i in range(B):
        assert int(path[i]) in set(int(c) for c in np.asarray(cand[i])), \
            f"position {i}: the path left the candidate set"

    # the rope: the draft's rotate_half must equal the target's half-split
    # convention (kernel.py's rope_neox), because the draft was trained
    # against that target
    pos = jnp.arange(4, dtype=jnp.int32)
    cos, sin = rope_tables(cfg.theta, cfg.head_dim, pos)
    vec = rng.standard_normal((4, cfg.heads, cfg.head_dim)).astype(np.float32)
    got = np.asarray(apply_rope(vec, cos, sin))
    half = cfg.head_dim // 2
    inv = 1.0 / (cfg.theta ** (np.arange(half) * 2.0 / cfg.head_dim))
    ang = pos.astype(np.float32)[:, None] * inv
    c, s = np.cos(ang), np.sin(ang)
    lo, hi = vec[..., :half], vec[..., half:2 * half]
    want = np.concatenate([lo * c[:, None, :] - hi * s[:, None, :],
                           lo * s[:, None, :] + hi * c[:, None, :]], -1)
    rope_err = float(np.max(np.abs(got - want)))

    if verbose:
        print(f"draft forward : {out.shape}  finite={bool(np.isfinite(np.asarray(out)).all())}")
        print(f"draft logits  : {logits.shape}")
        print(f"selector      : path {np.asarray(path).tolist()}")
        print(f"                 candidates per position {cfg.selector_top_k}")
        print(f"rope vs the target's half-split convention: max |diff| {rope_err:.3e}")
        print("PASS" if rope_err < 2e-2 else "FAIL")
    return rope_err < 2e-2


if __name__ == "__main__":
    ok = test_port()
    raise SystemExit(0 if ok else 1)

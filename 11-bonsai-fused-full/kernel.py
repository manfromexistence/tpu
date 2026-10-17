# %% cell [markdown]
"""
# Bonsai 2 27B on TPU with the fused ternary matmul (notebook 11)

Runs the **actual model** end to end, with every ternary projection served by
the fused PTQ1_0 Pallas kernel from notebook 10. No micro-benchmarks: the
first thing this script times is real generation.

Why this shape:

- Notebooks 06/08/09 spent 27.5-108 GB/step decoding weights to fp32 in HBM
  and were capped at 3.5-4.0 tok/s by that traffic. The fused kernel removes
  it -- the packed bytes are consumed in place.
- `10-bonsai-fused` measured the kernel at 4.1 ms/52.4 M params, **flat in
  batch**, so batching is free and everything rides on the batch size.
- The measured-fast layout used 128-byte rows and cannot hold the model
  (27.0 GB vs 15.75 GiB). This script uses **28-byte rows, the raw GGUF block
  layout**, which is the only layout that fits without a host repack:
  5.90 GB of tiles + 0.84 GB of scales.
- One `pallas_call` per projection, 2 jitted layer functions reused across
  all 64 layers, so a token is ~64 dispatches instead of ~300.

Run it and read the tok/s. Every number printed is measured.
"""

# %% cell
import functools
import gc
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print(f"jax {jax.__version__} | backend {jax.default_backend()}")

PROMPT = os.environ.get("PROMPT", "Hello")
N_GEN = int(os.environ.get("N_TOKENS", "32"))
BATCH = int(os.environ.get("BATCH", "16"))
MAX_T = int(os.environ.get("MAX_T", "256"))

# %% cell [markdown]
"""
## Base: GGUF reader, proven decoder, tokenizer, config

Everything up to the state initialisation, taken from notebook 04 unchanged.
`LAYER_RAW` (5.9 GB of packed bytes on device) is the input to the tile build
below and is freed layer by layer as the tiles replace it.
"""
# %% cell
_src = open(os.environ.get("FWD_SRC", "/content/04.py"), encoding="utf-8").read()
fw = {"__name__": "fwd"}
exec(compile(_src.split("st = {")[0], "fwd.py", "exec"), fw)
jax.config.update("jax_default_matmul_precision", "default")
print("base loaded | matmul precision:",
      jax.config.jax_default_matmul_precision)

cfg, tok = fw["cfg"], fw["tok"]
NL, IS_RECR = cfg["n_layer"], list(cfg["is_recr"])
NEMB, NFF, EPS, VOCAB = cfg["n_embd"], cfg["n_ff"], cfg["eps"], cfg["vocab"]
H, HKV, HD = cfg["n_head"], cfg["n_head_kv"], cfg["head_dim"]
ST, NV, NK, REP = cfg["state"], cfg["n_v"], cfg["n_k"], cfg["rep"]
KEY_DIM, CONV, CONV_DIM = cfg["key_dim"], cfg["conv"], cfg["conv_dim"]
TC, BLK, QK = 128, 28, 128

fold, l2n, rms, rope = fw["fold"], fw["l2n"], fw["rms"], fw["rope"]
SIGNS, R, BLOCK = fw["SIGNS"], fw["R"], fw["BLOCK"]
ROW = SIGNS[NEMB]
SMALLW, LAYER_MAP = fw["SMALLW"], fw["LAYER_MAP"]
LAYER_RAW = fw["LAYER_RAW"]
OUT_RAW, EMB_RAW, O_NORM = fw["OUT_RAW"], fw["EMB_RAW"], fw["O_NORM"]
decode_ptq1_dev = fw["decode_ptq1_dev"]
embed = fw["embed"]

GIDX = [i for i in range(NL) if IS_RECR[i]]
AIDX = [i for i in range(NL) if not IS_RECR[i]]
print(f"{NL} layers: {len(GIDX)} gdn + {len(AIDX)} attn | "
      f"n_embd {NEMB} n_ff {NFF} vocab {VOCAB}")

# %% cell [markdown]
"""
## The fused projection

`pl.BlockSpec` index maps return **block** indices, not element offsets
(FINDINGS §4.1). `rep` is laid out so tile row `(m*BPC + k)*TC + c` is out
column `m*TC + c` of in-block `k`, which is exactly the model's weight matrix
(proven maxdiff 0.0). Only the row width changed from notebook 10: 28 bytes
instead of 128, because 128-byte rows would need 27.0 GB.
"""
# %% cell
def _t(v):
    v = v & 0xFF
    return ((((v + (v >> 1)) >> 1) >> 6))


POW3 = (1, 3, 9, 27, 81)
QH3 = (1, 3, 9, 27)


def _kernel(xt_ref, rep_ref, sc_ref, z_ref, acc_ref, *, nsteps):
    b = rep_ref[...]
    qs = b[:, :24].astype(jnp.int32)
    qh = b[:, 24:26].astype(jnp.int32)
    t80 = jnp.stack([_t(qs[:, :16] * p) for p in POW3], 1).reshape(TC, 80)
    t40 = jnp.stack([_t(qs[:, 16:] * p) for p in POW3], 1).reshape(TC, 40)
    h = jnp.concatenate([
        jnp.concatenate([_t(qh[:, :1] * p), _t(qh[:, 1:] * p)], -1)
        for p in QH3], -1).reshape(TC, 8)
    w = (jnp.concatenate([t80, t40, h], -1).astype(jnp.float32) - 1.0) \
        .astype(jnp.bfloat16) * sc_ref[...].reshape(TC, 1)
    @pl.when(pl.program_id(2) == 0)
    def _z():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jax.lax.dot_general(
        w, xt_ref[...], (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _s():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def _pallas(geom, B):
    IN, OUT, BPC, TOUT = geom
    return pl.pallas_call(
        functools.partial(_kernel, nsteps=max(BPC, 1)),
        out_shape=jax.ShapeDtypeStruct((OUT, B), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((TC, B), lambda m, n, k: (k, 0)),
                pl.BlockSpec((TC, BLK), lambda m, n, k: (m * BPC + k, 0)),
                pl.BlockSpec((TC,), lambda m, n, k: (m * BPC + k,)),
            ],
            out_specs=pl.BlockSpec((TC, B), lambda m, n, k: (m, 0)),
            scratch_shapes=[pltpu.VMEM((TC, B), jnp.float32)],
            grid=(TOUT, 1, max(BPC, 1))),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )


GEOM = {}


def proj(x, tile, name):
    """x: (B, IN) fp32 -> (B, OUT) fp32, weights consumed as packed bytes."""
    rep, sc = tile
    IN, OUT, BPC, TOUT = GEOM[name]
    B = x.shape[0]
    y = _pallas((IN, OUT, BPC, TOUT), B)(
        x.T.astype(jnp.bfloat16), rep, sc)
    return y.T.astype(jnp.float32)


# %% cell [markdown]
"""
## Tile build: raw GGUF bytes -> tile order, no trit decode

Only a byte reorder: block (out `o`, in-block `k`) lives at flat index
`o*BPC + k` and must land at tile row `(m*BPC + k)*TC + c` with `o = m*TC + c`.
That is a `reshape` + `transpose`, no arithmetic, so it costs seconds per
layer rather than the minutes a trit decode would.
"""
# %% cell
def build_layer(il):
    """LAYER_RAW[il] -> {name: (rep[ROWS,28] u8, sc[ROWS] f32)}."""
    blob = np.asarray(LAYER_RAW[il])
    LAYER_RAW[il] = None                       # free the device copy
    out = {}
    for short, (off, n, dims) in LAYER_MAP[il].items():
        b0 = (off // QK) * BLK
        b1 = ((off + n) // QK) * BLK
        IN, OUT = int(dims[0]), int(dims[1])
        assert IN % TC == 0 and OUT % TC == 0, f"{short} {dims} not TC-aligned"
        BPC, TOUT = IN // TC, OUT // TC
        r8 = blob[b0:b1]
        assert len(r8) == OUT * BPC * BLK, f"{short} {len(r8)} != {OUT*BPC*BLK}"
        rv = r8.reshape(OUT, BPC, BLK).reshape(TOUT, TC, BPC, BLK) \
               .transpose(0, 2, 1, 3)                     # [m,k,c,28]
        sc = (rv[..., 26:28].copy().reshape(-1, 2).view(np.uint16).reshape(-1)
              .view(np.float16).astype(np.float32)
              .reshape(TOUT, BPC, TC).reshape(-1))
        out[short] = (jnp.asarray(rv.reshape(-1, BLK)), jnp.asarray(sc))
        g = (IN, OUT, BPC, TOUT)
        if short in GEOM:
            assert GEOM[short] == g, f"{short} geometry differs across layers"
        GEOM[short] = g
    return out


t0 = time.perf_counter()
TILES = []
for il in range(NL):
    TILES.append(build_layer(il))
    if il % 16 == 15:
        print(f"  tiles {il+1}/{NL} ({time.perf_counter()-t0:.0f}s)")
gc.collect()
print(f"tile build: {time.perf_counter()-t0:.1f}s | "
      f"{sum(r.nbytes + s.nbytes for t in TILES for r, s in t.values())/2**30:.2f} GiB on device")
for k in sorted(GEOM):
    print(f"  {k:24s} IN {GEOM[k][0]:6d} OUT {GEOM[k][1]:6d} "
          f"BPC {GEOM[k][2]:3d} TOUT {GEOM[k][3]:5d}")

# %% cell [markdown]
"""
## Batched units

Notebook 08's units are per-token and vmap'd. The fused kernel takes the
whole batch as `(IN, B)`, so batching has to be inside each unit instead --
otherwise vmap would fire one kernel launch per sequence and throw the
amortisation away.
"""
# %% cell
def fold_b(x, sign):
    return ((x * sign).reshape(x.shape[0], -1, BLOCK) @ R.T) \
        .reshape(x.shape[0], -1)


def l2n_b(x, eps):
    r = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    return x / jnp.maximum(r, eps)


_IIDX = jnp.arange(NV) % NK
_HK = jnp.arange(NK)[:, None]
_RR = jnp.arange(REP)[None, :]
_PERM = _HK + _RR * NK
_KIDX = jnp.arange(H) // (H // HKV)
_ISS = 1.0 / jnp.sqrt(jnp.float32(HD))
_SSS = 1.0 / jnp.sqrt(jnp.float32(ST))


def gdn_core(h, T, SW, S, hist):
    x, B = h, h.shape[0]
    hf = fold_b(h, ROW)
    qkv = proj(hf, T["attn_qkv.weight"], "attn_qkv.weight")
    z = proj(hf, T["attn_gate.weight"], "attn_gate.weight")
    beta = jax.nn.sigmoid(h @ SW["ssm_beta.weight"])
    alpha = h @ SW["ssm_alpha.weight"]
    gate = SW["ssm_a"] * jnp.logaddexp(alpha + SW["ssm_dt.bias"], 0.0)

    win = jnp.concatenate([hist, qkv[:, None, :]], 1)      # (B, CONV, CONV_DIM)
    y = jnp.sum(win * SW["ssm_conv1d.weight"][None], 1)
    y = y * jax.nn.sigmoid(y)

    qk = l2n_b(y[:, :2 * KEY_DIM].reshape(B, 2 * NK, ST).transpose(0, 2, 1), EPS)
    q, k = qk[:, :, :NK], qk[:, :, NK:]
    v = y[:, 2 * KEY_DIM:].reshape(B, NV, ST).transpose(0, 2, 1)
    q, k = q[:, :, _IIDX], k[:, :, _IIDX]

    Sg = S * jnp.exp(gate)[:, None, None, :]
    sk = jnp.einsum("bijv,biv->bjv", Sg, k)
    dv = (v - sk) * beta[:, None, :]
    Sg = Sg + jnp.einsum("biv,bjv->bijv", k, dv)
    o = jnp.einsum("bijv,biv->bjv", Sg, q) * _SSS

    zz = z.reshape(B, NV, ST).transpose(0, 2, 1)
    o = o * jax.lax.rsqrt(jnp.mean(o * o, axis=1, keepdims=True) + EPS)
    o = o * SW["ssm_norm.weight"][None, :, None]
    o = jax.nn.sigmoid(zz) * zz * o
    Q = jnp.transpose(o[:, :, _PERM], (0, 2, 3, 1))        # (B, NK, REP, ST)
    flat = Q.reshape(B, -1)
    return proj(fold_b(flat, SIGNS[ST * NV]), T["ssm_out.weight"],
                "ssm_out.weight"), Sg, win[:, 1:, :]


def attn_core(h, T, SW, K, V, pos):
    B = h.shape[0]
    hf = fold_b(h, ROW)
    qg = proj(hf, T["attn_q.weight"], "attn_q.weight").reshape(B, H, 2, HD)
    q, g = qg[:, :, 0, :], qg[:, :, 1, :]
    q = rms(q, SW["attn_q_norm.weight"], EPS)
    k = rms(proj(hf, T["attn_k.weight"], "attn_k.weight").reshape(B, HKV, HD),
            SW["attn_k_norm.weight"], EPS)
    v = proj(hf, T["attn_v.weight"], "attn_v.weight").reshape(B, HKV, HD)
    q, k = rope(q, pos), rope(k, pos)
    K = jax.lax.dynamic_update_slice(K, k[:, None], (0, pos, 0, 0))
    V = jax.lax.dynamic_update_slice(V, v[:, None], (0, pos, 0, 0))
    scores = jnp.einsum("bhd,bthd->bht", q, K[:, :, _KIDX, :]) * _ISS
    allow = jnp.arange(K.shape[1]) <= pos
    scores = jnp.where(allow[None, None, :], scores, -1e30)
    p = jax.nn.softmax(scores, -1)
    o = jnp.einsum("bht,bthd->bhd", p, V[:, :, _KIDX, :])
    o = o * jax.nn.sigmoid(g)
    return proj(fold_b(o.reshape(B, -1), SIGNS[H * HD]), T["attn_output.weight"],
                "attn_output.weight"), K, V


def ffn_core(h, T):
    hf = fold_b(h, ROW)
    g = proj(hf, T["ffn_gate.weight"], "ffn_gate.weight")
    u = proj(hf, T["ffn_up.weight"], "ffn_up.weight")
    return proj(fold_b(jax.nn.silu(g) * u, SIGNS[NFF]), T["ffn_down.weight"],
                "ffn_down.weight")


@jax.jit
def gdn_step(x, T, S, hist, an, pan, beta_w, alpha_w, aa, dtb, cv1, sn, norm_w):
    h = rms(x, an, EPS)
    a, S, hist = gdn_core(h, T, {"ssm_beta.weight": beta_w, "ssm_alpha.weight": alpha_w,
                                 "ssm_a": aa, "ssm_dt.bias": dtb,
                                 "ssm_conv1d.weight": cv1, "ssm_norm.weight": sn},
                          S, hist)
    f = ffn_core(rms(x + a, pan, EPS), T)
    return x + a + f, S, hist


@jax.jit
def attn_step(x, T, K, V, pos, an, pan, qn, kn):
    h = rms(x, an, EPS)
    a, K, V = attn_core(h, T, {"attn_q_norm.weight": qn,
                               "attn_k_norm.weight": kn}, K, V, pos)
    f = ffn_core(rms(x + a, pan, EPS), T)
    return x + a + f, K, V


def logits_b(h):
    xf = fold_b(rms(h, O_NORM, EPS), ROW).T                # (NEMB, B)
    nb = NEMB // QK
    per = nb * BLK
    parts = []
    for t0 in range(0, VOCAB, 4096):
        t1 = min(t0 + 4096, VOCAB)
        W = decode_ptq1_dev(OUT_RAW[t0 * per:t1 * per]).reshape(t1 - t0, -1)
        parts.append(W @ xf)
    return jnp.concatenate(parts, 0)                       # (VOCAB, B)


# %% cell [markdown]
"""
## Real model, real generation

The gate is informational: notebook 08 asserts the first token is `,` (11)
with fp32 weights, and the fused kernel runs the weights at bf16, so a
mismatch is recorded rather than fatal. The run continues either way -- the
number we came for is the throughput.
"""
# %% cell
B, T_ = BATCH, MAX_T
ids = tok.encode(PROMPT)
print(f"prompt {PROMPT!r} -> {len(ids)} tokens | batch {B} | max_t {T_} | "
      f"gen {N_GEN}")

xg = jnp.broadcast_to(embed(int(ids[0])), (B, NEMB))
S = [jnp.zeros((B, ST, ST, NV), jnp.float32) for _ in GIDX]
C = [jnp.zeros((B, CONV - 1, CONV_DIM), jnp.float32) for _ in GIDX]
K = [jnp.zeros((B, T_, HKV, HD), jnp.float32) for _ in AIDX]
V = [jnp.zeros((B, T_, HKV, HD), jnp.float32) for _ in AIDX]


def step(x, pos, S, C, K, V):
    gi = ai = 0
    for il in range(NL):
        T, SW = TILES[il], SMALLW[il]
        if IS_RECR[il]:
            x, S[gi], C[gi] = gdn_step(
                x, T, S[gi], C[gi], SW["attn_norm.weight"],
                SW["post_attention_norm.weight"], SW["ssm_beta.weight"],
                SW["ssm_alpha.weight"], SW["ssm_a"], SW["ssm_dt.bias"],
                SW["ssm_conv1d.weight"], SW["ssm_norm.weight"])
            gi += 1
        else:
            x, K[ai], V[ai] = attn_step(
                x, T, K[ai], V[ai], pos, SW["attn_norm.weight"],
                SW["post_attention_norm.weight"], SW["attn_q_norm.weight"],
                SW["attn_k_norm.weight"])
            ai += 1
    return x


t0 = time.perf_counter()
x = step(xg, jnp.int32(0), S, C, K, V)
lg = logits_b(x)
cur = np.asarray(jnp.argmax(lg, axis=0))
print(f"prefill (compile + 1 step): {time.perf_counter()-t0:.1f}s")
print(f"gate: first token {int(cur[0])} {tok.decode([int(cur[0])])!r} "
      f"(notebook 08 expects 11 ',')")

gen = []
t0 = time.perf_counter()
for i_step in range(N_GEN):
    for b in range(B):
        gen.append(int(cur[b]))
    x = jnp.broadcast_to(embed(int(cur[0])), (B, NEMB))
    x = step(x, jnp.int32(1 + i_step), S, C, K, V)
    cur = np.asarray(jnp.argmax(logits_b(x), axis=0))
t_gen = time.perf_counter() - t0
agg = B * N_GEN / t_gen

print("\n=== notebook 11 receipt ===")
print(f"batch            : {B}")
print(f"decode wall time : {t_gen:.2f}s for {N_GEN} steps")
print(f"aggregate tok/s  : {agg:.2f}")
print(f"per-step         : {t_gen/N_GEN*1000:.0f} ms -> "
      f"{t_gen/N_GEN*1000/B:.1f} ms/token")
print(f"baseline nb 04   : 0.27 tok/s | nb 06: 3.5 | nb 08: 1.5 | nb 09: 4.0")
print(f"sample           : {tok.decode(gen[:48])!r}")

json.dump({
    "notebook": "11-bonsai-fused-full",
    "backend": str(jax.default_backend()),
    "jax": jax.__version__,
    "batch": B, "n_gen": N_GEN, "prompt": PROMPT,
    "weights": "PTQ1_0 consumed as packed bytes, in-kernel unpack",
    "layout": "28-byte rows (raw GGUF block layout)",
    "decode_seconds": round(t_gen, 3),
    "aggregate_tok_per_s": round(agg, 2),
    "ms_per_step": round(t_gen / N_GEN * 1000, 1),
    "first_token": int(cur[0]),
    "claim": "[exec] measured on Colab TPU v5e-1",
}, open(os.environ.get("RECEIPT", "/content/receipt-colab-tpu-11.json"), "w"),
    indent=2)
print("receipt ->", os.environ.get("RECEIPT", "/content/receipt-colab-tpu-11.json"))

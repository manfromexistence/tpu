# %% cell [markdown]
"""
# Bonsai 2 27B on TPU, lane-aligned ternary decode (notebook 14)

Runs the **actual model** end to end, with the same fused PTQ1_0 kernel as
notebook 11 but with the weight tiles **transposed**: the packed bytes are
stored as `(32, rows)` and the projection is computed as `x @ w`, so the
out-columns sit on the 128-lane axis instead of the 2/8/16/28-wide byte axis.

Why this is the biggest lever left:

- JAX's TPU guide, *Writing TPU kernels with Pallas*: "Pallas will only ever
  consider mapping the **last two dimensions** of intermediate arrays to the
  8x128 vector register dimensions (sublanes and lanes respectively)" and "it
  is generally wasteful to have singleton dimensions in the last two axes".
- Mosaic's layout solver (`getTilingFactor`, `infer(MatmulOp)`) tiles the last
  two dims as `(sublane, 128)` and puts the **RHS operand's free axis (N) on
  the lanes** -- i.e. a matmul's weight out-columns are the lane axis.
- Notebook 11's decode put the *in-dims* on the lane axis, so every decode op
  ran on a 16-lane (group A), 8-lane (group B) or 2-lane (group C) tile of the
  128 available. Notebook 13 measured the same effect independently: a
  `(TO,16,5)` broadcast spelling (5 lanes) cost 4.2x a `(TO,16)` spelling
  (16 lanes) at identical op count.

Byte rows are padded 28 -> 32 because the transposed block `(32, TC)` keeps
**byte index on the sublanes**: Pallas requires block dims to be divisible by 8
(or equal to the array dim), and 28 is not. 32 is also the natural sublane
tiling (4 sublanes), so the byte axis is fully dense. The cost is 32/28 = 14%
more weight bytes, measured below.

Nothing else changes: the same `_d` decode, the same in-dim order (the
concatenate only moves to the other axis), so the decoded weights are
bit-identical to notebook 11's -- proven in `check14.py` with no TPU.

Every number printed is measured.
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
jax.config.update("jax_compilation_cache_dir", "/content/jaxcache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

PROMPT = os.environ.get("PROMPT", "Hello")
N_GEN = int(os.environ.get("N_TOKENS", "32"))
MAX_T = int(os.environ.get("MAX_T", "64"))
OUTDIR = os.environ.get("OUTDIR", "/content")
# `batch:state_dtype` pairs. Half-precision state is *storage only*: the
# recurrence still runs in fp32 and the state is cast back on the way out.
RUNS = [r for r in os.environ.get(
    "RUNS", "32:float32").split(",") if r]

# `TC` is the **lane extent** of the tile: the out-tile columns run along the
# 128 lanes, so a `(32, TC)` block is contiguous and TC may exceed 128.
# Notebook 13 measured 128 -> 512 at 1.22x on the projection (per-program
# overhead), so 512 is the default here too.
TC, BLK, RW, QK = int(os.environ.get("TC", "512")), 28, 32, 128

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

cfg, tok = fw["cfg"], fw["tok"]
NL, IS_RECR = cfg["n_layer"], list(cfg["is_recr"])
NEMB, NFF, EPS, VOCAB = cfg["n_embd"], cfg["n_ff"], cfg["eps"], cfg["vocab"]
H, HKV, HD = cfg["n_head"], cfg["n_head_kv"], cfg["head_dim"]
ST, NV, NK, REP = cfg["state"], cfg["n_v"], cfg["n_k"], cfg["rep"]
KEY_DIM, CONV, CONV_DIM = cfg["key_dim"], cfg["conv"], cfg["conv_dim"]

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
      f"n_embd {NEMB} n_ff {NFF} vocab {VOCAB} | out-tile TC {TC}")

# %% cell [markdown]
"""
## The fused projection, transposed

`rep` is `(rows, TC)` with row index `(m*BPC + k)*RW + j`: byte `j` of the
PTQ1_0 block for out column `m*TC + c` and in-block `k` sits at column `c` of
that row, and rows `j >= 28` are zero padding. The `(RW, TC)` block the kernel
loads is therefore a single contiguous run whose **lanes are the out-columns**
and whose sublanes are the packed byte index -- and every decode op runs on a
full-width tile.

The decode is notebook 11's measured-fast spelling (`_d`, int16 lanes, one
`concatenate` per group) with the axis flipped. Power `p` of byte `b` lands in
row `16p + b` (group A), `80 + 8p + (b-16)` (group B), `120 + 2p + i` (group C),
which is the same in-dim order notebook 11 built along the columns, so `x`
needs no permutation.
"""
# %% cell
def _t(v):
    v = v & 0xFF
    return ((((v + (v >> 1)) >> 1) >> 6))


POW3 = (1, 3, 9, 27, 81)
QH3 = (1, 3, 9, 27)


# `_t(v)` is exactly `(3*v) >> 8`:
#   floor(floor(floor(v + floor(v/2))/2)/64) = floor(0.75v/64) = floor(1.5v/128)
#   floor(3v/256)                                                   = floor(1.5v/128)
# Same integers, so the 5-op shift chain collapses to mul+shift, and for p=1
# the `& 0xFF` is dead (q < 256 already).
def _d(q, p):
    if p == 1:
        return (q * 3) >> 8
    return (((q * p) & 0xFF) * 3) >> 8


def _kernel(x_ref, rep_ref, sc_ref, z_ref, acc_ref, *, nsteps):
    b = rep_ref[...]                     # (32, TC) uint8, lanes = out-columns
    # int16, not int32: max `q*p` is 255*81 = 20655 and `& 255` -> `*3` -> `>> 8`
    # only decreases it. Notebook 13: 1.866 ms int32 against 1.393 ms int16.
    qs = b[:24].astype(jnp.int16)        # (24, TC)
    qh = b[24:26].astype(jnp.int16)      # (2, TC)
    # NOT `jnp.stack(...)`: notebook 13 measured stack+reshape at 3.129 ms
    # against concatenate at 1.872 ms for identical trits (a layout conversion).
    # The fourteen power groups, kept separate so that the concatenations can be
    # skipped: 5 x (16, TC) from qs[:16], 5 x (8, TC) from qs[16:], 4 x (2, TC)
    # from qh.  The two concatenations are full read+write passes over the
    # unpacked weights -- the largest single term in the projection's memory
    # traffic -- and at batch 1 the MXU is ~99 % idle, so skipping them costs no
    # measurable arithmetic.  It is a batch-1 optimisation: at large batch the
    # 8x padding of the K dimension is real work.
    g80 = [_d(qs[:16], p) for p in POW3]                           # (16, TC) x 5
    g40 = [_d(qs[16:], p) for p in POW3]                           # (8, TC) x 5
    gh = [jnp.concatenate([_d(qh[:1], p), _d(qh[1:], p)], 0)
          for p in QH3]                                            # (2, TC) x 4
    @pl.when(pl.program_id(2) == 0)
    def _z():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    sc = sc_ref[...][None, :]                                      # (1, TC)
    # (B, QK) @ (QK, TC) -> (B, TC): the orientation the Pallas matmul tutorial
    # uses and the one Mosaic's RHS-on-lanes rule packs for.
    if NOCONCAT:
        # The 128 in-dims of a block are split across the groups in the file's
        # order: 5 x 16 (from qs[:16]), 5 x 8 (from qs[16:]), 4 x 2 (from qh).
        # Each group's rows ARE in-dims, so it contracts against the matching
        # slice of x -- 16, 8 and 2 in-dims at a time -- instead of one
        # concatenated (128, TC) block.
        xr = x_ref[...]
        for p in range(5):
            a = p * 16
            wg = (g80[p] - 1).astype(jnp.bfloat16) * sc              # (16, TC)
            acc_ref[...] += jax.lax.dot_general(
                xr[:, a:a + 16], wg, (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32)
        for p in range(5):
            a = 80 + p * 8
            wg = (g40[p] - 1).astype(jnp.bfloat16) * sc              # (8, TC)
            acc_ref[...] += jax.lax.dot_general(
                xr[:, a:a + 8], wg, (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32)
        for p in range(4):
            a = 120 + p * 2
            wg = (gh[p] - 1).astype(jnp.bfloat16) * sc               # (2, TC)
            acc_ref[...] += jax.lax.dot_general(
                xr[:, a:a + 2], wg, (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32)
    else:
        t80 = jnp.concatenate(g80, 0)                              # (80, TC)
        t40 = jnp.concatenate(g40, 0)                              # (40, TC)
        h8 = jnp.concatenate(gh, 0)                                # (8, TC)
        w = (jnp.concatenate([t80, t40, h8], 0) - 1).astype(jnp.bfloat16) \
            * sc                                                  # (128, TC)
        acc_ref[...] += jax.lax.dot_general(
            x_ref[...], w, (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _s():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


PREFETCH = int(os.environ.get("PREFETCH", "0"))
NOCONCAT = os.environ.get("NOCONCAT", "0") == "1"
# GQA=1 computes the attention without materialising the expanded K and V. The
# shipped path builds K[:, :, _KIDX, :] -- (B, T, 24, 256) -- which at an 8k
# window is 201 MB per layer per step, 3.2 GB per step, and that expansion is
# the bulk of the 29 ms the 8k window costs at batch 1. The grouped form
# contracts the 24 query heads against the 4 kv heads directly, so the KV tiles
# are read once instead of six times.
GQA = os.environ.get("GQA", "0") == "1"


def _pallas(geom, B):
    IN, OUT, BPC, TOUT = geom
    return pl.pallas_call(
        functools.partial(_kernel, nsteps=max(BPC, 1)),
        out_shape=jax.ShapeDtypeStruct((B, OUT), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=PREFETCH,
            in_specs=[
                pl.BlockSpec((B, QK), lambda m, n, k: (0, k)),
                pl.BlockSpec((RW, TC), lambda m, n, k: (m * BPC + k, 0)),
                pl.BlockSpec((TC,), lambda m, n, k: (m * BPC + k,)),
            ],
            out_specs=pl.BlockSpec((B, TC), lambda m, n, k: (0, m)),
            scratch_shapes=[pltpu.VMEM((B, TC), jnp.float32)],
            grid=(TOUT, 1, max(BPC, 1))),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )


# `proj` is called ~400x per step. `_pallas(...)` builds a fresh `jax.jit`
# wrapper every time, which means every call re-traces and re-lowers the
# kernel on the host -- the device never sees a warm cache. CACHE=1 keeps one
# callable per (geometry, batch), so the dispatched executable is reused.
CACHE = os.environ.get("CACHE", "1") == "1"
_pallas_cached = functools.lru_cache(maxsize=None)(_pallas) if CACHE else _pallas


GEOM = {}

# `FUSE=1` merges the projections that share one input into a single kernel
# launch. All three fusable groups have IN=5120, and the tile layout is
# block-major in the out-column, so concatenating the member tiles along the
# row axis IS the combined tile: block (m, k) of the combined projection sits
# at the same (m*BPC + k)*RW offset, and the scales concatenate the same way.
# The pallas output block is (B, TC) at column block m, so the combined output
# is [member a | member b] in that order.
FUSE = os.environ.get("FUSE", "1") == "1"
FUSE_GROUPS = {
    "gdn": (("attn_qkv_gate.weight",
             ("attn_qkv.weight", "attn_gate.weight")),
            ("ffn_gate_up.weight",
             ("ffn_gate.weight", "ffn_up.weight"))),
    "attn": (("attn_qkv_fused.weight",
              ("attn_q.weight", "attn_k.weight", "attn_v.weight")),
             ("ffn_gate_up.weight",
              ("ffn_gate.weight", "ffn_up.weight"))),
}


def proj(x, tile, name, B=None):
    """x: (B, IN) fp32 -> (B, OUT) fp32, weights consumed as packed bytes."""
    if B is None:
        B = x.shape[0]
    rep, sc = tile
    IN, OUT, BPC, TOUT = GEOM[name]
    return _pallas_cached((IN, OUT, BPC, TOUT), B)(
        x.astype(jnp.bfloat16), rep, sc).astype(jnp.float32)


# %% cell [markdown]
"""
## Tile build: raw GGUF bytes -> transposed tile order

Only a byte reorder, no trit decode, so seconds per layer instead of minutes.
Block `(out o, in-block k)` lives at flat index `o*BPC + k` with 28 bytes; it
must land at row `(m*BPC + k)*RW + j`, column `c` with `o = m*TC + c`, and the
four padding bytes at `j = 28..31` stay zero.
"""
# %% cell
def build_layer(il):
    """LAYER_RAW[il] -> {name: (rep[rows,RW] u8, sc[rows] f32)}."""
    blob = np.asarray(LAYER_RAW[il])
    LAYER_RAW[il] = None                       # free the device copy
    out = {}
    for short, (off, n, dims) in LAYER_MAP[il].items():
        b0 = (off // QK) * BLK
        b1 = ((off + n) // QK) * BLK
        IN, OUT = int(dims[0]), int(dims[1])
        assert IN % QK == 0 and OUT % TC == 0, f"{short} {dims} not aligned"
        BPC, TOUT = IN // QK, OUT // TC
        r8 = blob[b0:b1]
        assert len(r8) == OUT * BPC * BLK, f"{short} {len(r8)} != {OUT*BPC*BLK}"
        # [m, k, c, 28]: out column m*TC+c, in-block k, byte j
        rv = r8.reshape(OUT, BPC, BLK).reshape(TOUT, TC, BPC, BLK) \
               .transpose(0, 2, 1, 3)
        sc = (rv[..., 26:28].copy().reshape(-1, 2).view(np.uint16).reshape(-1)
              .view(np.float16).astype(np.float32)
              .reshape(TOUT, BPC, TC).reshape(-1))
        # [m, k, j, c] -> (rows, TC), 28 real bytes then 4 zero pad bytes
        tp = np.zeros((TOUT, BPC, RW, TC), np.uint8)
        tp[:, :, :BLK, :] = rv.transpose(0, 1, 3, 2)
        rep = np.ascontiguousarray(tp).reshape(TOUT * BPC * RW, TC)
        out[short] = (jnp.asarray(rep), jnp.asarray(sc))
        g = (IN, OUT, BPC, TOUT)
        if short in GEOM:
            assert GEOM[short] == g, f"{short} geometry differs across layers"
        GEOM[short] = g
    if FUSE:
        kind = "gdn" if IS_RECR[il] else "attn"
        for cname, members in FUSE_GROUPS[kind]:
            if not all(m in out for m in members):
                continue
            ins = {GEOM[m][0] for m in members}
            bpcs = {GEOM[m][2] for m in members}
            assert len(ins) == 1 and len(bpcs) == 1, f"{cname} members disagree"
            rep_c = jnp.concatenate([out[m][0] for m in members], axis=0)
            sc_c = jnp.concatenate([out[m][1] for m in members], axis=0)
            out[cname] = (rep_c, sc_c)
            # The members are dead once the combined tile exists: keeping them
            # would hold a second copy of ~300 MB per GDN layer and blow the
            # 15.75 GiB budget. The fused path never reads them.
            for m in members:
                out.pop(m, None)
            IN, BPC = ins.pop(), bpcs.pop()
            OUT = sum(GEOM[m][1] for m in members)
            assert OUT % TC == 0, f"{cname} OUT {OUT} not a multiple of TC"
            GEOM[cname] = (IN, OUT, BPC, OUT // TC)
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

Notebook 08's units are per-token and vmap'd; the fused kernel takes the whole
batch as its out-tile, so batching lives inside each unit. Same code as
notebook 11 except that `proj` no longer transposes (`(B, IN)` in, `(B, OUT)`
out) and `gdn_core` may store the state in half precision.

`CACHE=1` (default) keeps one Pallas callable per (geometry, batch) instead of
building a fresh `jax.jit` wrapper on each of the ~400 calls a step makes;
`CACHE=0` reproduces the per-call rebuild to price that overhead.
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


def gdn_core(h, T, SW, S, hist, B=None):
    if B is None:
        B = h.shape[0]
    sdt = S.dtype
    S = S.astype(jnp.float32)              # fp16 storage, fp32 recurrence
    x = h
    hf = fold_b(h, ROW)
    if FUSE and "attn_qkv_gate.weight" in T:
        qg = proj(hf, T["attn_qkv_gate.weight"], "attn_qkv_gate.weight", B)
        nq = GEOM["attn_qkv.weight"][1]
        qkv, z = qg[:, :nq], qg[:, nq:]
    else:
        qkv = proj(hf, T["attn_qkv.weight"], "attn_qkv.weight", B)
        z = proj(hf, T["attn_gate.weight"], "attn_gate.weight", B)
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
                "ssm_out.weight", B), Sg.astype(sdt), win[:, 1:, :]


def attn_core(h, T, SW, K, V, pos, B=None):
    if B is None:
        B = h.shape[0]
    hf = fold_b(h, ROW)
    if FUSE and "attn_qkv_fused.weight" in T:
        qkv = proj(hf, T["attn_qkv_fused.weight"], "attn_qkv_fused.weight", B)
        nq = GEOM["attn_q.weight"][1]
        nk = GEOM["attn_k.weight"][1]
        qg = qkv[:, :nq].reshape(B, H, 2, HD)
        kk = rms(qkv[:, nq:nq + nk].reshape(B, HKV, HD),
                 SW["attn_k_norm.weight"], EPS)
        v = qkv[:, nq + nk:].reshape(B, HKV, HD)
    else:
        qg = proj(hf, T["attn_q.weight"], "attn_q.weight", B).reshape(B, H, 2, HD)
        kk = rms(proj(hf, T["attn_k.weight"], "attn_k.weight", B).reshape(B, HKV, HD),
                 SW["attn_k_norm.weight"], EPS)
        v = proj(hf, T["attn_v.weight"], "attn_v.weight", B).reshape(B, HKV, HD)
    q, g = qg[:, :, 0, :], qg[:, :, 1, :]
    q = rms(q, SW["attn_q_norm.weight"], EPS)
    k = kk
    q, k = rope(q, pos), rope(k, pos)
    K = jax.lax.dynamic_update_slice(K, k[:, None], (0, pos, 0, 0))
    V = jax.lax.dynamic_update_slice(V, v[:, None], (0, pos, 0, 0))
    if GQA:
        # (B, HKV, G, HD) x (B, T, HKV, HD) -> (B, HKV, G, T): the 6 query
        # heads that share a kv head contract against it directly, so the KV
        # tiles are read once instead of being expanded to 24 heads first.
        G = H // HKV
        qr = q.reshape(B, HKV, G, HD)
        # 'h' is the kv head in BOTH operands: K and V are (B, T, HKV, HD),
        # and the query's kv head is the middle axis of (B, HKV, G, HD). The
        # query head index is h*G + g, which is the order the expanded
        # _KIDX path uses, so the two are equivalent.
        scores = jnp.einsum("bhgd,bthd->bhgt", qr, K).reshape(B, H, K.shape[1]) \
            * _ISS
        allow = jnp.arange(K.shape[1]) <= pos
        scores = jnp.where(allow[None, None, :], scores, -1e30)
        p = jax.nn.softmax(scores, -1).reshape(B, HKV, G, K.shape[1])
        o = jnp.einsum("bhgt,bthd->bhgd", p, V).reshape(B, H, HD)
    else:
        scores = jnp.einsum("bhd,bthd->bht", q, K[:, :, _KIDX, :]) * _ISS
        allow = jnp.arange(K.shape[1]) <= pos
        scores = jnp.where(allow[None, None, :], scores, -1e30)
        p = jax.nn.softmax(scores, -1)
        o = jnp.einsum("bht,bthd->bhd", p, V[:, :, _KIDX, :])
    o = o * jax.nn.sigmoid(g)
    return proj(fold_b(o.reshape(B, -1), SIGNS[H * HD]), T["attn_output.weight"],
                "attn_output.weight", B), K, V


def ffn_core(h, T, B=None):
    if B is None:
        B = h.shape[0]
    hf = fold_b(h, ROW)
    if FUSE and "ffn_gate_up.weight" in T:
        gu = proj(hf, T["ffn_gate_up.weight"], "ffn_gate_up.weight", B)
        ng = GEOM["ffn_gate.weight"][1]
        g, u = gu[:, :ng], gu[:, ng:]
    else:
        g = proj(hf, T["ffn_gate.weight"], "ffn_gate.weight", B)
        u = proj(hf, T["ffn_up.weight"], "ffn_up.weight", B)
    return proj(fold_b(jax.nn.silu(g) * u, SIGNS[NFF]), T["ffn_down.weight"],
                "ffn_down.weight", B)


def gdn_step(x, T, S, hist, an, pan, beta_w, alpha_w, aa, dtb, cv1, sn, B=None):
    if B is None:
        B = x.shape[0]
    h = rms(x, an, EPS)
    a, S, hist = gdn_core(h, T, {"ssm_beta.weight": beta_w, "ssm_alpha.weight": alpha_w,
                                 "ssm_a": aa, "ssm_dt.bias": dtb,
                                 "ssm_conv1d.weight": cv1, "ssm_norm.weight": sn},
                          S, hist, B)
    f = ffn_core(rms(x + a, pan, EPS), T, B)
    return x + a + f, S, hist


def attn_step(x, T, K, V, pos, an, pan, qn, kn, B=None):
    if B is None:
        B = x.shape[0]
    h = rms(x, an, EPS)
    a, K, V = attn_core(h, T, {"attn_q_norm.weight": qn,
                               "attn_k_norm.weight": kn}, K, V, pos, B)
    f = ffn_core(rms(x + a, pan, EPS), T, B)
    return x + a + f, K, V

# %% cell
# Fused single-step: all 64 layers in one jitted program.
# Eliminates 63 of the 64 per-layer host dispatches that dominate the 216 ms/step.
def full_step(x, pos, S_in, C_in, K_in, V_in, W, B):
    S = list(S_in)
    C = list(C_in)
    K = list(K_in)
    V = list(V_in)
    gi = ai = 0
    for il in range(NL):
        T, SW = W[2 * il], W[2 * il + 1]
        if IS_RECR[il]:
            x, S[gi], C[gi] = gdn_step(
                x, T, S[gi], C[gi], SW["attn_norm.weight"],
                SW["post_attention_norm.weight"], SW["ssm_beta.weight"],
                SW["ssm_alpha.weight"], SW["ssm_a"], SW["ssm_dt.bias"],
                SW["ssm_conv1d.weight"], SW["ssm_norm.weight"], B)
            gi += 1
        else:
            x, K[ai], V[ai] = attn_step(
                x, T, K[ai], V[ai], pos, SW["attn_norm.weight"],
                SW["post_attention_norm.weight"], SW["attn_q_norm.weight"],
                SW["attn_k_norm.weight"], B)
            ai += 1
    return x, tuple(S), tuple(C), tuple(K), tuple(V)

jitted_step = jax.jit(full_step, static_argnums=(7,))



def _logits_raw(h, out_raw):
    xf = fold_b(rms(h, O_NORM, EPS), ROW).T                # (NEMB, B)
    nb = NEMB // QK
    per = nb * BLK
    parts = []
    for t0 in range(0, VOCAB, 4096):
        t1 = min(t0 + 4096, VOCAB)
        W = decode_ptq1_dev(out_raw[t0 * per:t1 * per]).reshape(t1 - t0, -1)
        parts.append(W @ xf)
    return jnp.concatenate(parts, 0)                       # (VOCAB, B)


# The 61-chunk vocab loop above was ~120 host dispatches per token when run
# eagerly. One jit fuses it into a single dispatched program.
logits_b = jax.jit(_logits_raw)


# %% cell [markdown]
"""
## Real model, real generation

The gate is informational: notebook 08 asserts the first token is `,` (11) with
fp32 weights, and the fused kernel runs the weights at bf16, so a mismatch is
recorded rather than fatal. The run continues either way -- the number we came
for is the throughput. Every batch size in this session runs the same 64-layer
model end to end and writes its own receipt.
"""
# %% cell
def run_batch(B, state_dt):
    ids = tok.encode(PROMPT)
    T_ = max(MAX_T, len(ids) + N_GEN + 2)
    st = jax.devices()[0].memory_stats()
    tag = state_dt.replace("float", "f")
    print(f"\n=== batch {B} | state {state_dt} | TC {TC} | max_t {T_} ===")
    print(f"memory before: {st['bytes_in_use']/2**30:.2f} GiB in use, "
          f"peak {st['peak_bytes_in_use']/2**30:.2f} GiB, "
          f"limit {st['bytes_limit']/2**30:.2f} GiB")
    S = [jnp.zeros((B, ST, ST, NV), np.dtype(state_dt)) for _ in GIDX]
    C = [jnp.zeros((B, CONV - 1, CONV_DIM), jnp.float32) for _ in GIDX]
    K = [jnp.zeros((B, T_, HKV, HD), jnp.float32) for _ in AIDX]
    V = [jnp.zeros((B, T_, HKV, HD), jnp.float32) for _ in AIDX]
    xg = jnp.broadcast_to(embed(int(ids[0])), (B, NEMB))

    t0 = time.perf_counter()
    x = xg
    S_t, C_t, K_t, V_t = tuple(S), tuple(C), tuple(K), tuple(V)
    W_t = tuple(a for il in range(NL) for a in (TILES[il], SMALLW[il]))
    import threading as _th
    _done = _th.Event()

    def _wd():
        n = 0
        while not _done.wait(60):
            n += 60
            print(f"    [watchdog] first fused_step still compiling/running ({n}s)...", flush=True)
    _th.Thread(target=_wd, daemon=True).start()
    print(f"prefill: {len(ids)} step(s); first call compiles the fused 64-layer step")
    for j, tid in enumerate(ids):              # full prompt prefill, one token per step
        x = jnp.broadcast_to(embed(int(tid)), (B, NEMB))
        x, S_t, C_t, K_t, V_t = jitted_step(
            x, jnp.int32(j), S_t, C_t, K_t, V_t, W_t, B)
    _done.set()
    lg = logits_b(x, OUT_RAW)
    cur = np.asarray(jnp.argmax(lg, axis=0))
    first_tok = int(cur[0])                    # captured here: `cur` is reassigned below
    print(f"prefill (compile + {len(ids)} step(s)): {time.perf_counter()-t0:.1f}s")
    print(f"gate: first token {first_tok} {tok.decode([first_tok])!r} "
          f"(notebook 08 expects 11 ',')")
    st = jax.devices()[0].memory_stats()
    print(f"memory after prefill: {st['bytes_in_use']/2**30:.2f} GiB in use, "
          f"peak {st['peak_bytes_in_use']/2**30:.2f} GiB")

    gen = []
    t0 = time.perf_counter()
    for i_step in range(N_GEN):
        for b in range(B):
            gen.append(int(cur[b]))
        x = jnp.broadcast_to(embed(int(cur[0])), (B, NEMB))
        x, S_t, C_t, K_t, V_t = jitted_step(
            x, jnp.int32(len(ids) + i_step), S_t, C_t, K_t, V_t, W_t, B)
        cur = np.asarray(jnp.argmax(logits_b(x, OUT_RAW), axis=0))
    t_gen = time.perf_counter() - t0
    agg = B * N_GEN / t_gen
    sample = tok.decode(gen[::B][:N_GEN])

    print(f"decode wall time : {t_gen:.2f}s for {N_GEN} steps")
    print(f"aggregate tok/s  : {agg:.2f}")
    print(f"per-step         : {t_gen/N_GEN*1000:.0f} ms -> "
          f"{t_gen/N_GEN*1000/B:.1f} ms/token")
    print(f"sample           : {sample!r}")

    rec = {
        "notebook": "14-bonsai-transposed",
        "backend": str(jax.default_backend()),
        "jax": jax.__version__,
        "batch": B, "n_gen": N_GEN, "prompt": PROMPT, "prompt_tokens": len(ids),
        "max_t": T_,
        "state_dtype": state_dt, "out_tile_tc": TC,
        "cache_pallas_callable": CACHE,
        "proj_fusion": "on" if FUSE else "off",
        "num_scalar_prefetch": PREFETCH,
        "no_concat": "on" if NOCONCAT else "off",
        "gqa_attention": "on" if GQA else "off",
        "projections_per_step": (4 * NL) if FUSE else (6 * len(GIDX) + 7 * len(AIDX)),
        "weights": "PTQ1_0 consumed as packed bytes, in-kernel unpack",
        "layout": f"transposed ({RW}, rows); lanes = out-columns; x @ w",
        "padding": f"{BLK} real bytes + {RW-BLK} pad in each {RW}-byte row",
        "decode": "concatenate along axis 0 (in-dims on sublanes), int16 lanes",
        "decode_seconds": round(t_gen, 3),
        "aggregate_tok_per_s": round(agg, 2),
        "ms_per_step": round(t_gen / N_GEN * 1000, 1),
        "first_token": first_tok,
        "sample": sample,
        "claim": "[exec] measured on Colab TPU v5e-1",
    }
    path = f"{OUTDIR}/receipt-14-b{B}-{tag}-tc{TC}.json"
    json.dump(rec, open(path, "w"), indent=2)
    print("receipt ->", path)
    del S, C, K, V, S_t, C_t, K_t, V_t, x, xg, lg, cur
    gc.collect()
    return rec


RES = []
for spec in RUNS:
    b_s, _, d_s = spec.partition(":")
    try:
        RES.append(run_batch(int(b_s), d_s or "float32"))
    except Exception as e:                                  # noqa: BLE001
        print(f"!!! run {spec} failed: {type(e).__name__}: {e}")
        gc.collect()

print("\n=== summary ===")
for r in RES:
    print(f"  batch {r['batch']:3d} state {r['state_dtype']:8s} "
          f"{r['aggregate_tok_per_s']:7.2f} tok/s  "
          f"{r['ms_per_step']:7.1f} ms/step  first {r['first_token']}")

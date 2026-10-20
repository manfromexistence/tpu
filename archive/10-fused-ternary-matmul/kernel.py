"""Fused PTQ1_0 ternary matmul in Pallas on Colab TPU v5e-1.

One program per (out-tile, in-block): unpack the 28-byte PTQ1_0 GGUF blocks
in VMEM, scale, and MXU-accumulate over the in-block dimension with the
weight as the MXU LHS.  No bf16/fp32 weight ever reaches HBM.

Validated on layer 0 `blk.0.attn_qkv.weight`: the tiled reconstruction equals
the model's own weight matrix exactly (maxdiff 0.0) and the fused matmul
matches to bf16 rounding (1.65e-02 on ref|max| 5.46 at B=128).

Three traps, every one of which cost a debugging cycle (see
`tpu/07-pallas-matmul-findings/FINDINGS.md` for the full write-up):

1. `pl.BlockSpec` index maps return BLOCK indices, not element offsets.
   Pallas multiplies them by the block shape.  Passing `(m*BPC + k)*TC`
   walks 128x outside the array and halts the core with
   `BoundsCheck ... dma.hbm_to_vmem`, wedging the runtime until
   `colab restart-kernel`.
2. notebook 04 sets `jax_default_matmul_precision="highest"` at import, and
   that alone makes Mosaic reject the bf16 MXU op with `Bad rhs type`.
   Reset it after the exec and pin Precision.DEFAULT on the dot_general.
3. `dec.reshape((IN, OUT), order="F")` is NOT
   `dec.reshape(-1).reshape((IN, OUT), order="F")`.  On layer-0 qkv they
   disagree in 89% of entries, and the model uses the second form
   (`tpu/04-reference-forward-pass/kernel.py:280-281`).

Needs jax 0.11.2 + libtpu 0.0.50 (FINDINGS.md §1) and the GGUF on the host.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

WANT = "blk.0.attn_qkv.weight"
TC = 128                       # trits per PTQ1_0 block, and the MXU tile
BLK = 28                       # bytes per PTQ1_0 block
POW3 = (1, 3, 9, 27, 81)       # base-3 digit multipliers inside `qs`
QH3 = (1, 3, 9, 27)            # ... and inside `qh`

print("jax:", jax.__version__, "backend:", jax.default_backend())

# ---- proven GGUF reader + decoder, taken from notebook 04 ---------------
# Exec only the head of it: everything after `n_layers = ws.meta` is the
# forward pass, which this script does not need.
src = open("/content/04.py", encoding="utf-8").read()
ns = {"__name__": "nb04"}
exec(compile(src.split("n_layers = ws.meta")[0], "04m.py", "exec"), ns)
jax.config.update("jax_default_matmul_precision", "default")   # trap 2
print("matmul_precision reset to:", jax.config.jax_default_matmul_precision)

raw, dims = ns["ws"].read(WANT)[:2]
IN, OUT = int(dims[0]), int(dims[1])
BPC, TOUT = IN // TC, OUT // TC
ROWS = TOUT * BPC * TC
r8 = np.frombuffer(raw, np.uint8)
print(f"{WANT}: dims {dims} IN {IN} OUT {OUT} BPC {BPC} TOUT {TOUT} "
      f"ROWS {ROWS} raw {len(raw)} B")

# ---- reference: the model's own convention (trap 3, form B) -------------
dec = np.asarray(ns["decode_ptq1_dev"](jnp.asarray(r8)), np.float32)
Wmodel = dec.reshape(-1).reshape((IN, OUT), order="F")
Wother = dec.reshape((IN, OUT), order="F")
print("the two F-reshape forms differ by:",
      float(np.abs(Wmodel - Wother).max()))


def _trit_host(q, p):
    v = (q.astype(np.uint32) * np.uint32(p)) % np.uint32(256)
    return (((v + (v >> 1)) >> 1) >> 6).astype(np.int8) - 1


def trits_of_block(b28):
    """(N, 28) uint8 -> (N, 128) int8 trits, in the proven power-major order.

    Indices 0..79   qs bytes 0..15, power-major     (i*16 + byte)
    Indices 80..119 qs bytes 16..23, power-major    (80 + i*8 + byte)
    Indices 120..127 qh, INTERLEAVED per power      (120 + 2i + {0,1})
    """
    qs, qh = b28[:, :24], b28[:, 24:26]
    o = np.empty((b28.shape[0], TC), np.int8)
    for i, p in enumerate(POW3):
        o[:, i * 16:(i + 1) * 16] = _trit_host(qs[:, :16], p)
        o[:, 80 + i * 8:80 + (i + 1) * 8] = _trit_host(qs[:, 16:24], p)
    for i, p in enumerate(QH3):
        o[:, 120 + 2 * i] = _trit_host(qh[:, 0], p)
        o[:, 121 + 2 * i] = _trit_host(qh[:, 1], p)
    return o


# ---- tiled host layout --------------------------------------------------
# Block (out o, in-block k) lives at flat block index o*BPC + k, because
# `layer_weights` F-reshapes the flat decode: n = i + o*IN, so block =
# n // TC = o*BPC + i//TC and position = n % TC = i % TC.  Tile row
# (m*BPC + k)*TC + c is therefore out column m*TC + c, in-block k.
cols = r8.reshape(OUT, BPC, BLK)
repv = cols.reshape(TOUT, TC, BPC, BLK).transpose(0, 2, 1, 3)      # [m,k,c,28]
scv = (repv[..., 26:28].copy().reshape(-1, 2).view(np.uint16).reshape(-1)
       .view(np.float16).astype(np.float32).reshape(TOUT, BPC, TC))

rep2 = np.zeros((ROWS, 128), np.uint8)
rep2[:, :BLK] = repv.reshape(ROWS, BLK)
S = jnp.asarray(scv.reshape(ROWS))
R = jnp.asarray(rep2)

# reconstruction check: kernel tile indexing == the model's weight matrix
Wm = trits_of_block(rep2[:, :BLK]).reshape(TOUT, BPC, TC, TC)
Wm = Wm.transpose(0, 2, 1, 3).reshape(OUT, BPC * TC).astype(np.float32)
Wm = Wm * np.repeat(scv.transpose(0, 2, 1).reshape(OUT, BPC), TC, axis=1)
print("tile reconstruction vs model convention:",
      float(np.abs(Wm - Wmodel.T).max()))


# ---- the kernel ---------------------------------------------------------
def _t(v):
    v = v & 0xFF
    return ((((v + (v >> 1)) >> 1) >> 6))


def kernel(xt_ref, rep_ref, sc_ref, z_ref, acc_ref, *, nsteps):
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
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jax.lax.dot_general(
        w, xt_ref[...], (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,          # trap 2
        preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _store():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def build(B):
    return pl.pallas_call(
        functools.partial(kernel, nsteps=BPC),
        out_shape=jax.ShapeDtypeStruct((OUT, B), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((TC, B), lambda m, n, k: (k, 0)),            # trap 1
                pl.BlockSpec((TC, 128), lambda m, n, k: (m * BPC + k, 0)),
                pl.BlockSpec((TC,), lambda m, n, k: (m * BPC + k,)),
            ],
            out_specs=pl.BlockSpec((TC, B), lambda m, n, k: (m, 0)),
            scratch_shapes=[pltpu.VMEM((TC, B), jnp.float32)],
            grid=(TOUT, 1, BPC)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )


FLOP = 2.0 * OUT * IN
for B in (128, 32, 1):
    xt = jnp.asarray(np.random.randn(IN, B).astype(np.float32) * 0.5,
                     jnp.bfloat16)
    ref = Wmodel.T @ np.asarray(xt, np.float32)
    f = build(B)
    got = np.asarray(f(xt, R, S), np.float32)
    err = float(np.abs(ref - got).max())
    t0 = time.perf_counter()
    for _ in range(20):
        jax.block_until_ready(f(xt, R, S))
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"B={B:3d} maxdiff {err:.3e} ref|max| {np.abs(ref).max():5.2f}  "
          f"{ms:6.2f}ms  {FLOP/(ms/1e3)/1e12:5.2f} TFLOP/s")

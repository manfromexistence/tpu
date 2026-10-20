"""Fused ternary matmul in the DOCS matmul orientation.

out[b,o] = sum_k x[b,k] * W[k,o];  LHS = x block, RHS = unpacked W block
(transposed in VMEM), exactly like the canonical Pallas matmul whose only
change is that the RHS is computed rather than loaded.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("jax:", jax.__version__)
_src = open("/content/04.py", encoding="utf-8").read()
ns = {"__name__": "x"}
exec(compile(_src.split("n_layers = ws.meta")[0], "04m.py", "exec"), ns)
g = ns
raw, dims, dt = g["ws"].read("blk.0.attn_qkv.weight")
Wq = g["decode_ptq1_dev"](jnp.asarray(np.frombuffer(raw, np.uint8))) \
    .reshape(dims, order="F")                     # [IN, OUT]
IN, OUT = int(dims[0]), int(dims[1])
TC = 128
BPC, TOUT = IN // TC, OUT // TC
B = 128
print(f"IN {IN} OUT {OUT} BPC {BPC} TOUT {TOUT}")

r8 = np.frombuffer(raw, np.uint8)
cols = r8.reshape(OUT, BPC, 28)
rep = cols.reshape(TOUT, TC, BPC, 28).transpose(0, 2, 1, 3)   # [TOUT,BPC,TC,28]
sc = (rep[..., 26:28].copy().reshape(-1, 2).view(np.uint16).reshape(-1)
      .view(np.float16).astype(np.float32).reshape(TOUT, BPC, TC))
rep2 = np.zeros((TOUT * BPC * TC, 128), np.uint8)
rep2[:, :28] = rep.reshape(TOUT * BPC * TC, 28)
R = jnp.asarray(rep2)
S = jnp.asarray(sc.reshape(TOUT * BPC * TC))
X = jax.random.normal(jax.random.key(0), (B, IN), jnp.bfloat16)


def _t(v):
    v = v & 0xFF
    return ((((v + (v >> 1)) >> 1) >> 6))


def unpack_T(b):
    """b [TC,128] u8 -> W^T [o, k] f32 for this tile (128 out, 128 in)."""
    qs = b[:, :24].astype(jnp.int32)
    qh = b[:, 24:26].astype(jnp.int32)
    a = []
    for p in (1, 3, 9, 27, 81):
        a.append(_t(qs[:, :16] * p))
    for p in (1, 3, 9, 27, 81):
        a.append(_t(qs[:, 16:] * p))
    t80 = jnp.stack(a[:5], -1).reshape(TC, 80)
    t40 = jnp.stack(a[5:], -1).reshape(TC, 40)
    h = jnp.concatenate([
        jnp.stack([_t(qh[:, :1] * p) for p in (1, 3, 9, 27)], -1),
        jnp.stack([_t(qh[:, 1:] * p) for p in (1, 3, 9, 27)], -1)],
        -1).reshape(TC, 8)
    return jnp.concatenate([t80, t40, h], -1).astype(jnp.float32) - 1.0


def kernel(x_ref, rep_ref, sc_ref, o_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(1) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    wt = unpack_T(rep_ref[...][:, :28])           # [o, k]
    wt = wt * sc_ref[...].reshape(TC, 1)          # scale per out-row
    acc_ref[...] += jnp.dot(x_ref[...], wt.T,
                            preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(1) == nsteps - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


@jax.jit
def fused(x, R, S):
    return pl.pallas_call(
        functools.partial(kernel, nsteps=BPC),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((B, TC), lambda j, k: (0, k)),
                pl.BlockSpec((TC, 128), lambda j, k: ((j * BPC + k) * TC, 0)),
                pl.BlockSpec((TC,), lambda j, k: ((j * BPC + k) * TC,)),
            ],
            out_specs=pl.BlockSpec((B, TC), lambda j, k: (0, j)),
            scratch_shapes=[pltpu.VMEM((B, TC), jnp.float32)],
            grid=(TOUT, BPC),
        ),
        out_shape=jax.ShapeDtypeStruct((B, OUT), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")),
    )(x, R, S)


print("building...")
t0 = time.perf_counter()
o = fused(X, R, S)
jax.block_until_ready(o)
print(f"build+exec {time.perf_counter()-t0:.1f}s")
ref = np.asarray(np.asarray(X).astype(np.float32) @ np.asarray(Wq))
got = np.asarray(o)
d = float(np.abs(ref - got).max())
print(f"fused vs proven maxdiff: {d:.3e} {'OK' if d < 0.5 else 'MISMATCH'}")
t0 = time.perf_counter()
for _ in range(3):
    jax.block_until_ready(fused(X, R, S))
ms = (time.perf_counter() - t0) / 3 * 1000
print(f"52M-param fused matmul (B={B}): {ms:.2f}ms")
print(f"extrapolated 27B step: {ms * 5400 / 1000:.2f}s "
      f"-> {B / (ms * 5400 / 1000):.1f} tok/s aggregate")

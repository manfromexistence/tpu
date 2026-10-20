"""Fused ternary-matvec, one full column (5120 trits, 40 blocks).

Proves multi-block accumulation. Grid=1. Compares to proven XLA.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print("jax:", jax.__version__, "backend:", jax.default_backend())
src = open("/content/04.py", encoding="utf-8").read()
ns = {"__name__": "x"}
exec(compile(src.split("st = {")[0], "04.py", "exec"), ns)
g = ns
print("staged")

W = g["decode_layer"](0)
Wq = W["attn_qkv.weight"]  # [5120, 10240] Fortran
print("Wq:", Wq.shape)
# column 0 of Fortran [5120,10240] = flat[0:5120] = first 40 blocks
NB = 40
raw = g["LAYER_RAW"][0][:NB * 28]
rb = np.asarray(raw).reshape(NB, 28)
R = jnp.asarray(np.concatenate([rb, np.zeros((NB, 100), np.uint8)],
                               1).reshape(-1))
sc = (rb[:, 26:28].copy().view(np.uint16).reshape(-1)
      .view(np.float16).astype(np.float32))
S = jnp.asarray(np.concatenate([sc, np.ones(128 - NB, np.float32)]))
x = jax.random.normal(jax.random.key(0), (5120,), dtype=jnp.float32)


def col_kernel(raw_ref, sc_ref, x_ref, out_ref):
    acc = jnp.zeros((1,), jnp.float32)
    for b in range(NB):
        blk = raw_ref[b * 128:(b + 1) * 128][:28]
        qs = blk[:24].astype(jnp.int32)
        qh = blk[24:26].astype(jnp.int32)
        scc = sc_ref[b:b + 1]
        q0, q1 = qs[:16], qs[16:24]

        def _p5(q):
            def _t(v):
                return (((((v + (v >> 1)) >> 1) >> 6).astype(jnp.float32))
                        - 1.0)
            return jnp.concatenate([_t((q * 1) & 0xFF),
                                    _t((q * 3) & 0xFF),
                                    _t((q * 9) & 0xFF),
                                    _t((q * 27) & 0xFF),
                                    _t((q * 81) & 0xFF)])

        t = jnp.concatenate([_p5(q0), _p5(q1)])
        h = []
        for p in (1, 3, 9, 27):
            v = (qh * np.int32(p)) & np.int32(0xFF)
            h.append(((((v + (v >> 1)) >> 1) >> 6).astype(jnp.float32)
                      - 1.0))
        w = jnp.concatenate([t, jnp.concatenate(h)]) * scc
        xv = x_ref[b * 128:(b + 1) * 128]
        acc = acc + jnp.sum(w * xv, keepdims=True)
    out_ref[:] = jnp.broadcast_to(acc, (128,))


f = pl.pallas_call(
    col_kernel,
    out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((NB * 128,), lambda i: (0,)),
              pl.BlockSpec((128,), lambda i: (0,)),
              pl.BlockSpec((8192,), lambda i: (0,))],
    out_specs=pl.BlockSpec((128,), lambda i: (0,)))
o = f(R, S, x)
jax.block_until_ready(o)
ref = float(np.asarray(Wq[:, 0]) @ np.asarray(x))
o = o[:1]
print(f"fused column vs proven XLA: got {float(o[0]):.4f} ref {ref:.4f} "
      f"diff {abs(float(o[0]) - ref):.3e}")
t0 = time.perf_counter()
for _ in range(10):
    jax.block_until_ready(f(R, S, x))
print(f"one column (5120 dots): {(time.perf_counter()-t0)/10*1000:.3f}ms")

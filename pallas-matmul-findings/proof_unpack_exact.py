"""Pallas unpack-only (no reductions/scalars): 1 launch vs 14 ops."""
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print("jax:", jax.__version__, "backend:", jax.default_backend())
NR = 32

# proven decoder (pure functions, no weights needed)
import re
src4 = open("/content/06src.py", encoding="utf-8").read() \
    if False else None
# 04src.py has _trit_dev + decode_ptq1_dev; extract them
s4 = open("/content/04src.py", encoding="utf-8").read()
out = []
for name in ["_trit_dev", "decode_ptq1_dev"]:
    m = re.search(rf"^def {name}\(.*?(?=^def |^# %% cell|\Z)",
                  s4, re.M | re.S)
    out.append(m.group(0))
ns4 = {"jnp": jnp, "jax": jax, "np": np, "BLK": 28,
       "POW3": (1, 3, 9, 27, 81)}
exec("\n\n".join(out), ns4)
proven_decode = jax.jit(ns4["decode_ptq1_dev"])


def unpack_kernel(raw_ref, sc_ref, out_ref):
    for r in range(NR):
        blk = raw_ref[r * 128:(r + 1) * 128][:28]
        qs = blk[:24].astype(jnp.int32)
        qh = blk[24:26].astype(jnp.int32)
        sc = sc_ref[r:r + 1]
        q0, q1 = qs[:16], qs[16:24]

        def _p5(q):
            def _t(v):
                return (((((v + (v >> 1)) >> 1) >> 6).astype(jnp.float32))
                        - 1.0)
            return jnp.concatenate([_t((q * 1) & 0xFF), _t((q * 3) & 0xFF), _t((q * 9) & 0xFF),
                                    _t((q * 27) & 0xFF), _t((q * 81) & 0xFF)])

        t = jnp.concatenate([_p5(q0), _p5(q1)])
        h = []
        for p in (1, 3, 9, 27):
            v = (qh * np.int32(p)) & np.int32(0xFF)
            h.append(((((v + (v >> 1)) >> 1) >> 6).astype(jnp.float32)
                      - 1.0))
        w = jnp.concatenate([t, jnp.concatenate(h)]) * sc
        out_ref[r * 128:(r + 1) * 128] = w


raw28 = np.random.randint(0, 256, size=(NR, 28)).astype(np.uint8)
raw = jnp.asarray(
    np.concatenate([raw28, np.zeros((NR, 100), np.uint8)], 1).reshape(-1))
sc_host = (raw28[:, 26:28].copy().view(np.uint16).reshape(-1)
           .view(np.float16).astype(np.float32))
f = pl.pallas_call(
    unpack_kernel,
    out_shape=jax.ShapeDtypeStruct((NR * 128,), jnp.float32),
    grid=(1,),
    in_specs=[pl.BlockSpec((NR * 128,), lambda i: (0,)),
              pl.BlockSpec((128,), lambda i: (0,))],
    out_specs=pl.BlockSpec((NR * 128,), lambda i: (0,)))
o = f(raw, jnp.asarray(sc_host))
jax.block_until_ready(o)
print("pallas unpack exec OK")
# reference
exp = np.zeros((NR, 128), np.float32)
for i in range(NR):
    b = raw28[i]
    qs, qh = b[:24].astype(np.int32), b[24:26].astype(np.int32)
    vals = []
    for j in list(range(16)) + list(range(16, 24)):
        for p in (1, 3, 9, 27, 81):
            v = (int(qs[j]) * p) & 0xFF
            vals.append(((((v + (v >> 1)) >> 1) >> 6) - 1))
    for j in range(2):
        for p in (1, 3, 9, 27):
            v = (int(qh[j]) * p) & 0xFF
            vals.append(((((v + (v >> 1)) >> 1) >> 6) - 1))
    exp[i] = np.asarray(vals, np.float32) * sc_host[i]
got = np.asarray(o).reshape(NR, 128)
ok = bool(np.allclose(exp, got, equal_nan=True))
print(f"allclose(equal_nan=True): {ok}")
finite = np.isfinite(exp) & np.isfinite(got)
print(f"finite-element maxdiff: "
      f"{float(np.abs(exp[finite] - got[finite]).max()):.3e}")
# ground truth: proven decoder on the same raw28
prov = np.asarray(proven_decode(jnp.asarray(raw28)).reshape(NR, 128))
print(f"pallas vs PROVEN maxdiff: "
      f"{float(np.abs(prov - got).max()):.3e}")
print(f"numpy-ref vs PROVEN maxdiff: "
      f"{float(np.abs(prov - exp).max()):.3e}")

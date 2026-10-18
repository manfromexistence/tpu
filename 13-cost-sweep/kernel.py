"""Cost breakdown for the fused PTQ1_0 ternary matmul (notebook 13).

**STEP 2 OF A SESSION, NEVER THE WHOLE SESSION.** The standing rule
(`tpu/README.md` §6, `DO-NOT-RETEST.md` §0) is that every TPU session runs the
real 64-layer Bonsai 2 model end to end. Run `11-bonsai-fused-full/kernel.py`
first, then this file, in the same session, with a `colab restart-kernel`
between the two. This file measures weights. It does not measure the model, and
a session that ran only this file would be a wasted session.

`DO-NOT-RETEST.md` §7 and `07-pallas-fusion/FINDINGS.md` §7 both end at the same
open question. The fused kernel runs ~300x below the HBM roofline and ~50x below
the VPU; every arithmetic explanation has been excluded by measurement; two
survivors remain -- **program count** versus **the unpack/cast/scale chain** --
and they have never been separated.

This script separates them on **real layer-0 weights**, and adds the one
measurement nobody has taken: what this Colab v5e-1 actually delivers as raw
HBM bandwidth. That number decides whether ~2 s per forward pass is a property
of the kernel or a property of the slice we are handed.

Questions, in the order each one can kill the next:

  Q0  Raw HBM bandwidth, and the per-dispatch floor.      (jnp only, no risk)
  Q1  The proven configuration, to cross-check nb 10.     mode="dot"  TO=128
  Q2  Does cost scale with program count?                 mode="dot"  TO x4
  Q3  How much is the weight DMA alone?                   mode="dma"
  Q4  How much does the unpack arithmetic add?            mode="dec"
  Q5  What inside the unpack costs?                       mode="dec_nc"/"dec2"/"dec3"
      `cast`   the int32 cast of the 26 payload bytes, nothing else
      `dec_nc` the decode with the final `concatenate` removed (no (TO,128) tile)
      `dec2`   the same trits in the same order, spelled with `concatenate`
               instead of `stack(axis=1).reshape`
      `dec3`   byte-major order from one broadcast per group, no `stack` at all
      `dec16`  `dec2` at int16 width; every intermediate fits
               (`max q*p = 255*81 = 20655 < 32767`), so identical values

**`TO` is the lever nobody has pulled.** The out-tile height has always been
pinned to `TC=128`, the number of trits in a PTQ1_0 block. That is an accident
of the reference layout: the block is `(TO, 28)` and the array is `(ROWS, 28)`,
which is legal for any `TO` divisible by 8 (`FINDINGS.md` §3, the
last-dim-equals-array rule). `FINDINGS.md` §7 itself names "a 256-row out tile"
as *the* separating experiment, and notebook 12 did not run it -- nb 12 moved
`KB` 1->8 **and** `TO` 128->64 at once, so it changed the program count in the
wrong direction while changing the step count in the right one. Its 15 %
regression is therefore evidence about `TO`, not about the k-loop.

Nothing here is a new algorithm. Every variant uses an unpack that is already
proven bit-exact (`FINDINGS.md` §5, 409600/409600 blocks matched). Ablations
(`dma`, `dec`) are deliberately *not* numerically checked -- they exist to price
a stage, not to produce a result.

Compile failures do not wedge the core (`FINDINGS.md` §1), so every cell is
wrapped in `try/except` and one bad shape cannot cost the session. Runtime
halts do wedge it, so the proven configuration is measured first and every line
is flushed as it is printed.
"""

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print(f"jax {jax.__version__} | backend {jax.default_backend()}", flush=True)

BATCH = int(os.environ.get("BATCH", "32"))
REPS = int(os.environ.get("REPS", "10"))
ROWS = []


def emit(row):
    ROWS.append(row)
    print(f"  {row['label']:34s} {row['ms']:9.3f}ms "
          f"{row.get('gb_s', 0):8.1f} GB/s "
          f"{row.get('gtrit_s', 0):8.2f} Gtrit/s", flush=True)


def bench(f, args, n=REPS):
    """Throughput of n back-to-back calls, one final sync."""
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n


# ===================================================================== Q0
print("\n--- Q0  raw HBM bandwidth and dispatch floor (no Pallas) ---",
      flush=True)

N32 = 64 * 1024 * 1024                                   # 64 Mi f32 = 256 MB
a32 = jnp.ones(N32, jnp.float32)
a8 = jnp.ones(2 * N32, jnp.int8)                         # 128 Mi int8 = 128 MB
b32 = jnp.zeros(N32, jnp.float32)

for label, fn, mbytes, arg in (
        ("read  f32 (sum)", jax.jit(lambda x: jnp.sum(x)), 256, a32),
        ("copy  f32 (x*2)", jax.jit(lambda x: x * 2), 512, a32),
        ("read  int8 (sum)", jax.jit(lambda x: jnp.sum(x.astype(jnp.int32))),
         128, a8),
        ("write f32 (zeros)", jax.jit(lambda x: jnp.zeros_like(x)), 256, a32),
):
    s = bench(fn, (arg,), n=50)
    emit({"label": label, "ms": s * 1e3, "gb_s": mbytes / 1e3 / s})

del a32, a8, b32

# the floor: a 4 KB elementwise op, which measures dispatch, not hardware
tiny = jnp.ones(1024, jnp.float32)
us = bench(jax.jit(lambda x: x * 2), (tiny,), n=200) * 1e6
print(f"  dispatch floor: {us:.0f} us per jitted call", flush=True)
ROWS.append({"label": "dispatch floor (us)", "ms": us / 1e3, "gb_s": 0})
del tiny

# ===================================================================== setup
print("\n--- GGUF reader (proven, notebook 04) ---", flush=True)
src = open("/content/04.py", encoding="utf-8").read()
ns = {"__name__": "nb04"}
exec(compile(src.split("n_layers = ws.meta")[0], "04m.py", "exec"), ns)
jax.config.update("jax_default_matmul_precision", "default")   # FINDINGS 4.2

WANT = "blk.0.attn_qkv.weight"
raw, dims = ns["ws"].read(WANT)[:2]
IN, OUT = int(dims[0]), int(dims[1])
BLK, TRIT = 28, 128
BPC = IN // TRIT
NTOT = OUT * BPC * TRIT
r8 = np.frombuffer(raw, np.uint8).copy()
print(f"{WANT}: dims {dims} IN {IN} OUT {OUT} BPC {BPC} "
      f"trits {NTOT/1e6:.1f} M raw {len(raw)} B", flush=True)


def host_tiles(TO):
    """Raw GGUF bytes -> tile order for out-tile height TO.

    Block (out `o`, in-block `k`) is at flat block index `o*BPC + k`
    (`FINDINGS.md` §4.3), and tile row `(m*BPC + k)*TO + c` must be out column
    `m*TO + c` of in-block `k`. Pure byte reorder -- no trit decode.
    """
    assert OUT % TO == 0, f"TO {TO} does not divide OUT {OUT}"
    cols = r8.reshape(OUT, BPC, BLK)                         # [o, k, byte]
    rv = cols.reshape(OUT // TO, TO, BPC, BLK).transpose(0, 2, 1, 3)
    rep = np.ascontiguousarray(rv.reshape(-1, BLK))
    sc = (rv[..., 26:28].copy().reshape(-1, 2)
          .view(np.uint16).reshape(-1).view(np.float16)
          .astype(np.float32).reshape(-1))
    assert rep.shape == (OUT * BPC, BLK), rep.shape
    assert sc.shape == (OUT * BPC,), sc.shape
    return rep, sc


def perm_ctg():
    """The host permutation that makes the contiguous decode order correct.

    `dec_stack` column `16*p + b` is power `P5[p]` of `qs` byte `b` (and
    `80 + 8*p + b`, and `120 + 2*p + b`).  `dec_contig` puts the same value at
    `5*b + p` (and `80 + 5*b + p`, and `120 + 4*b + p`).  The dot needs
    `x'[k*128 + i] = x[k*128 + sigma(i)]`, so sigma maps a contiguous column to
    the stack column that holds the same trit.
    """
    s = np.empty(TRIT, np.int32)
    for b in range(16):                       # qs bytes 0..15, powers 0..4
        for p in range(5):
            s[5 * b + p] = 16 * p + b
    for b in range(8):                        # qs bytes 16..23, powers 0..4
        for p in range(5):
            s[80 + 5 * b + p] = 80 + 8 * p + b
    for b in range(2):                        # qh bytes 0..1, powers 0..3
        for p in range(4):
            s[120 + 4 * b + p] = 120 + 2 * p + b
    assert sorted(s.tolist()) == list(range(TRIT)), "not a permutation"
    return s


print(f"contiguous-order permutation is valid: "
      f"{sorted(perm_ctg().tolist()) == list(range(TRIT))}", flush=True)

# ===================================================================== Q1-Q5
P5 = (1, 3, 9, 27, 81)
P4 = (1, 3, 9, 27)


def _d(q, p):
    """`(3*((q*p) mod 256)) >> 8` -- bit-identical to the shift chain (nb 11)."""
    if isinstance(p, int) and p == 1:
        return (q * 3) >> 8
    return (((q * p) & 0xFF) * 3) >> 8


def dec_parts(b, TO):
    """The three power groups of a PTQ1_0 block, decoded to trits 0..2."""
    qs = b[:, :24].astype(jnp.int32)
    qh = b[:, 24:26].astype(jnp.int32)
    t80 = jnp.stack([_d(qs[:, :16], p) for p in P5], 1).reshape(TO, 80)
    t40 = jnp.stack([_d(qs[:, 16:], p) for p in P5], 1).reshape(TO, 40)
    h8 = jnp.concatenate([
        jnp.concatenate([_d(qh[:, :1], p), _d(qh[:, 1:], p)], -1)
        for p in P4], -1).reshape(TO, 8)
    return t80, t40, h8


def dec_stack(b, TO):
    """Notebook 11's decode: five power planes stacked into the middle axis."""
    return jnp.concatenate(dec_parts(b, TO), -1)


def dec_cat(b, TO, dt=jnp.int32):
    """The same trits in the same column order, spelled with `concatenate`
    instead of `stack(axis=1).reshape`. `stack`+`reshape` and `concatenate`
    both place power `p` of byte `b` at column `16*p + b`, so this is an A/B on
    whether Mosaic lowers the 3-D reshape as a shuffle.

    `dt` is the integer width the arithmetic runs at. Every intermediate fits in
    int16 (`max q*p = 255*81 = 20655 < 32767`), so `dt=jnp.int16` is exactly
    equivalent and asks whether narrower elements buy wider vregs.
    """
    qs = b[:, :24].astype(dt)
    qh = b[:, 24:26].astype(dt)
    t80 = jnp.concatenate([_d(qs[:, :16], p) for p in P5], -1)
    t40 = jnp.concatenate([_d(qs[:, 16:], p) for p in P5], -1)
    h8 = jnp.concatenate([
        jnp.concatenate([_d(qh[:, :1], p), _d(qh[:, 1:], p)], -1)
        for p in P4], -1)
    return jnp.concatenate([t80, t40, h8], -1)


def dec_contig(b, TO):
    """Same trits, byte-major order: one broadcast per group, no stack.

    Column `5*b + p` is power `p` of `qs` byte `b`, so the trailling-dims
    flatten of the broadcast is a plain reshape. The order differs from
    `dec_stack`, which is free because `x` is permuted on the host to match
    (`perm_ctg`).

    The power vectors are built from `jnp.arange` rather than captured from
    module scope: this stack rejects captured array constants
    ("captures constants [i32[1,1,5], i32[1,1,4]]").
    """
    qs = b[:, :24].astype(jnp.int32)[:, :, None]              # (TO, 24, 1)
    qh = b[:, 24:26].astype(jnp.int32)[:, :, None]
    p5 = (3 ** jnp.arange(5, dtype=jnp.int32))[None, None, :]
    p4 = (3 ** jnp.arange(4, dtype=jnp.int32))[None, None, :]
    t80 = _d(qs[:, :16], p5).reshape(TO, 80)
    t40 = _d(qs[:, 16:], p5).reshape(TO, 40)
    h8 = _d(qh, p4).reshape(TO, 8)
    return jnp.concatenate([t80, t40, h8], -1)


def kernel(xt_ref, rep_ref, sc_ref, z_ref, acc_ref, *, mode, TO, nsteps):
    b = rep_ref[...]

    @pl.when(pl.program_id(2) == 0)
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    if mode == "dma":
        acc_ref[...] += jnp.sum(b.astype(jnp.float32), 1)[:, None]
    elif mode == "cast":
        acc_ref[...] += jnp.sum(b[:, :26].astype(jnp.int32), 1)[:, None]
    elif mode == "dec":
        acc_ref[...] += jnp.sum(dec_stack(b, TO).astype(jnp.float32), 1)[:, None]
    elif mode == "dec_nc":
        # decode, but never build the (TO, 128) tile
        t80, t40, h8 = dec_parts(b, TO)
        acc_ref[...] += (jnp.sum(t80.astype(jnp.float32), 1)
                         + jnp.sum(t40.astype(jnp.float32), 1)
                         + jnp.sum(h8.astype(jnp.float32), 1))[:, None]
    elif mode == "dec2":
        acc_ref[...] += jnp.sum(dec_cat(b, TO).astype(jnp.float32), 1)[:, None]
    elif mode == "dec16":
        # identical values, 16-bit lanes: does the vreg get wider?
        acc_ref[...] += jnp.sum(
            dec_cat(b, TO, jnp.int16).astype(jnp.float32), 1)[:, None]
    elif mode == "dec3":
        acc_ref[...] += jnp.sum(dec_contig(b, TO).astype(jnp.float32), 1)[:, None]
    else:
        w = (dec_stack(b, TO) - 1).astype(jnp.bfloat16) \
            * sc_ref[...].reshape(TO, 1)
        acc_ref[...] += jax.lax.dot_general(
            w, xt_ref[...], (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _store():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def build(TO, mode, B):
    return pl.pallas_call(
        functools.partial(kernel, mode=mode, TO=TO, nsteps=max(BPC, 1)),
        out_shape=jax.ShapeDtypeStruct((OUT, B), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((TRIT, B), lambda m, n, k: (k, 0)),
                pl.BlockSpec((TO, BLK), lambda m, n, k: (m * BPC + k, 0)),
                pl.BlockSpec((TO,), lambda m, n, k: (m * BPC + k,)),
            ],
            out_specs=pl.BlockSpec((TO, B), lambda m, n, k: (m, 0)),
            scratch_shapes=[pltpu.VMEM((TO, B), jnp.float32)],
            grid=(OUT // TO, 1, max(BPC, 1))),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )


def cell(label, TO, mode):
    """One (TO, mode) cell. Compile failures are caught, halts are not."""
    t0 = time.perf_counter()
    rep = sc = R = S = xt = f = None
    try:
        rep, sc = host_tiles(TO)
        R, S = jnp.asarray(rep), jnp.asarray(sc)
        xt = jnp.asarray(np.random.randn(IN, BATCH).astype(np.float32) * 0.5,
                         jnp.bfloat16)
        f = jax.jit(build(TO, mode, BATCH))
        s = bench(f, (xt, R, S))
    except Exception as exc:                          # compile errors only
        print(f"  {label:34s} FAILED {type(exc).__name__}: "
              f"{str(exc)[:160]}", flush=True)
        ROWS.append({"label": label, "ms": float("nan"),
                     "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return
    finally:
        del rep, sc, R, S, xt, f

    progs = (OUT // TO) * BPC
    nbytes = OUT * BPC * BLK
    emit({"label": label, "ms": s * 1e3,
          "gb_s": nbytes / s / 1e9,
          "gtrit_s": NTOT / s / 1e9,
          "programs": progs, "trit_per_prog": TO * TRIT,
          "compile_s": round(time.perf_counter() - t0 - s, 1)})


print(f"\n--- Q1  the proven configuration (expect nb 10's 4.04 ms) | "
      f"B={BATCH} ---", flush=True)
cell("dot   TO=128  (nb 11 config)", 128, "dot")

print("\n--- Q2  program count: vary TO only ---", flush=True)
for TO in (64, 256, 512, 1024):
    cell(f"dot   TO={TO}", TO, "dot")

print("\n--- Q3  weight DMA alone ---", flush=True)
for TO in (128, 512):
    cell(f"dma   TO={TO}", TO, "dma")

print("\n--- Q4  + unpack arithmetic ---", flush=True)
for TO in (128, 512):
    cell(f"dec   TO={TO}", TO, "dec")

print("\n--- Q5  what inside the unpack costs ---", flush=True)
for TO in (128, 512):
    cell(f"cast  TO={TO}", TO, "cast")
for TO in (128, 512):
    cell(f"dec_nc TO={TO}", TO, "dec_nc")
for TO in (128, 512, 1024):
    cell(f"dec2  TO={TO} (concat spelling)", TO, "dec2")
for TO in (512, 1024):
    cell(f"dec3  TO={TO} (broadcast spelling)", TO, "dec3")
for TO in (512, 1024):
    cell(f"dec16 TO={TO} (int16, identical values)", TO, "dec16")

# ===================================================================== report
print("\n=== notebook 13 receipt ===", flush=True)
base = next((r for r in ROWS if r["label"].startswith("dot   TO=128")), None)
for r in ROWS:
    if r["ms"] != r["ms"]:
        print(f"  {r['label']:34s} FAILED")
    else:
        rel = ""
        if base and r is not base and r.get("gtrit_s"):
            rel = f"  {r['gtrit_s']/base['gtrit_s']:.2f}x nb11"
        print(f"  {r['label']:34s} {r['ms']:9.3f}ms "
              f"{r.get('gtrit_s', 0):8.2f} Gtrit/s{rel}")

json.dump({
    "notebook": "13-cost-sweep",
    "backend": str(jax.default_backend()),
    "jax": jax.__version__,
    "batch": BATCH,
    "weight": WANT,
    "trits": NTOT,
    "rows": ROWS,
    "claim": "[exec] measured on Colab TPU v5e-1",
}, open(os.environ.get("RECEIPT", "/content/receipt-colab-tpu-13.json"),
        "w"), indent=2)
print("receipt -> /content/receipt-colab-tpu-13.json", flush=True)

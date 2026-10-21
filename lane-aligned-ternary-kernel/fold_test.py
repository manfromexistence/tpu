"""The decode fold: the same projection with the scale and the `- 1` off the
per-weight path.

What this file is
-----------------
`lane-aligned-ternary-kernel/kernel.py` turns the trit decode into weights
with

    w = (jnp.concatenate([t80, t40, h8], 0) - 1).astype(jnp.bfloat16) * sc

and hands `w` to the MXU. Two things about that line matter:

1. `.astype(bfloat16) * sc` promotes **back to fp32**, so the MXU is fed an
   fp32 weight tile -- 4 bytes per element, not 2. The tile is the widest
   array the projection touches (its own comment calls it "the largest single
   term in the projection's memory traffic"), so it is being carried at twice
   the width it needs.
2. The `- 1` and the `* sc` are two extra full passes over that tile, and
   neither of them belongs there.

The fold
--------
The decode yields a *level* in {0, 1, 2}; the shipped weight is `level - 1`,
in {-1, 0, 1}. So for an out-column o with scale sc[o]:

    sum_in (level-1) * x * sc[o]
        = sc[o] * ( sum_in level * x  -  sum_in x )

`sum_in x` is **one scalar per token**, not one operation per weight, and
`sc[o]` is per out-column, so it multiplies the *accumulator* after the
matmul. What is left on the per-weight path is the level itself, cast to
bf16 -- an exact cast, since 0, 1 and 2 are representable -- and the tile is
bf16, half the width.

The price
---------
The folded accumulator carries `sum_in level * x`, roughly twice the magnitude
of `sum_in (level-1) * x`, before the subtraction cancels it: about one bit
of fp32. The correction is done inside the Pallas kernel, in the fp32
accumulator, so only the final result is rounded to bf16 -- the same rounding
the shipped path ends with. This file checks the *algebra* on CPU; the real
gate is the byte-identical first token on the TPU.

Run: `python fold_test.py`
"""

import numpy as np

# The decode, copied from the kernel (`_d`, `POW3`, `QH3`).
POW3 = (1, 3, 9, 27, 81)
QH3 = (1, 3, 9, 27)


def _d(q, p):
    """The trit decode: `floor(3 * ((byte * p) & 0xFF) / 256)`."""
    q = np.asarray(q, np.int64)
    if p == 1:
        return (q * 3) >> 8
    return (((q * p) & 0xFF) * 3) >> 8


def decode_levels(b):
    """The packed bytes of one 128-in-dim block -> the levels, (128, TC).

    `b` is (26, TC): 24 bytes of `qs` and 2 of `qh`, in the kernel's order.
    The decode's spelling is the kernel's: 5 groups of 16 rows (group A),
    5 of 8 (group B), 4 of 2 (group C), one `concatenate` per group and one
    for the three groups together.
    """
    b = np.asarray(b, np.int64)
    qs, qh = b[:24], b[24:26]
    g80 = [_d(qs[:16], p) for p in POW3]        # (16, TC) x 5
    g40 = [_d(qs[16:], p) for p in POW3]        # (8, TC) x 5
    gh = [_d(qh, p) for p in QH3]               # (2, TC) x 4
    t80 = np.concatenate(g80, 0)               # (80, TC)
    t40 = np.concatenate(g40, 0)               # (40, TC)
    h8 = np.concatenate(gh, 0)                 # (8, TC)
    return np.concatenate([t80, t40, h8], 0)   # (128, TC), values in {0,1,2}


def test_fold(seed=20261021, B=4, TC=16, verbose=True):
    """The folded projection must equal the shipped one to fp32 rounding."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(26, TC), dtype=np.int64)
    levels = decode_levels(b).astype(np.float32)
    assert levels.min() >= 0 and levels.max() <= 2, "the decode is not 3-level"

    sc = rng.standard_normal((1, TC)).astype(np.float32)
    x = rng.standard_normal((B, 128)).astype(np.float32)

    # the shipped path: bf16 levels minus one, scaled, then the matmul
    w_ship = (levels - 1).astype(np.float32) * sc
    out_ship = x @ w_ship

    # the folded path: the levels alone into the matmul, then the correction
    acc = x @ levels
    out_fold = (acc - x.sum(-1, keepdims=True)) * sc

    err = float(np.max(np.abs(out_ship - out_fold)))
    scale = float(np.max(np.abs(out_ship)))
    rel = err / scale if scale else 0.0

    # the same thing in bf16, the way the MXU sees it
    def to_bf16(a):
        # round-to-nearest-even through the float32 bit pattern
        u = a.astype(np.float32).view(np.uint32)
        r = ((u >> 16) & 1) + 0x7FFF
        u = (u + r) & 0xFFFF0000
        return u.view(np.float32)

    w_ship_bf = to_bf16((levels - 1).astype(np.float32) * sc)
    w_fold_bf = to_bf16(levels)
    out_ship_bf = x @ w_ship_bf
    out_fold_bf = ((x @ w_fold_bf) - x.sum(-1, keepdims=True)) * sc
    err_bf = float(np.max(np.abs(out_ship_bf - out_fold_bf)))

    if verbose:
        print(f"levels          : {levels.shape}, values "
              f"{sorted(set(levels.ravel().tolist()))}")
        print(f"shipped  weight tile: fp32 (4 bytes/element)")
        print(f"folded   weight tile: bf16 (2 bytes/element)")
        print(f"max |shipped - folded|        : {err:.3e}  (relative {rel:.3e})")
        print(f"same, through bf16 rounding   : {err_bf:.3e}")
        print("PASS" if rel < 1e-5 else "FAIL")
    return rel < 1e-5


if __name__ == "__main__":
    ok = test_fold()
    raise SystemExit(0 if ok else 1)

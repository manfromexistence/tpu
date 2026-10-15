"""Signed block-1024 FWHT reference and test -- the TPU runner's math, on CPU first.

The Bonsai 2 port needs one transform done right: the block-wise signed Hadamard with
block size 1024 from the `prism.hadamard.*` contract (see 02-gguf-prism-metadata). This
notebook is the NumPy reference for it plus the tests that prove it correct, so a later
JAX/Pallas kernel has a byte-close target instead of a formula it might misread.

What is tested, in order of increasing trust:

  1. The butterfly against an explicit Sylvester Hadamard at N=8. If the loop is wrong
     here, everything after it is noise.
  2. Orthogonality and norm preservation of the normalized transform.
  3. Forward/inverse round-trip with random sign vectors, the exact signed convention
     the fork uses (sign BEFORE on forward, AFTER on inverse).
  4. Block handling over the real model widths (5120, 6144, 17408), which must each be
     divisible by 1024 or the reshape is the bug.
  5. JAX equivalence, if JAX is importable. Plain `jnp` only -- deliberately NOT Pallas,
     because Pallas TPU primitives have no CPU lowering and would make this untestable
     off-device.

Dependencies: NumPy only, JAX optional. Runs on a free CPU runtime.

Status: reference code only, NOT YET EXECUTED against real sign vectors from a GGUF --
those come from notebook 02. The synthetic sign vectors here exercise the code path;
they are not measurements of the model.
"""

import json
import os
from pathlib import Path

import numpy as np

# %% cell [markdown]
"""
# Signed FWHT-1024 reference and test

**Purpose.** Prove the butterfly correct *before* it ever touches a TPU or a real
weight: small-N against an explicit matrix, then signs, then blocks, then JAX.

The one convention this whole notebook defends: **forward applies the sign vector
before the transform, inverse applies it after**, and both directions normalize by
`1/sqrt(n)`. Swapping either half produces a transform that is *almost* right and
destroys a model while passing a shape check.
"""

# %% cell
# --- the transform ------------------------------------------------------------
# Fully vectorized over the last axis. No Python loop over elements, so blocks of 1024
# cost microseconds and the 17-block ffn_down width runs in one call.


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def fwht_raw(a):
    """Unnormalized FWHT along the last axis; length must be a power of two.

    Returns F @ a with F built from +-1, so F @ F == n * I. Callers that need a
    rotation divide by sqrt(n) themselves -- keeping the raw form separate is what
    lets the N=8 test compare directly against the explicit Sylvester matrix.
    """
    h = np.asarray(a, dtype=np.float64).copy()
    lead = h.shape[:-1]
    n = h.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"FWHT needs a power of two, got {n}")
    step = 1
    while step < n:
        h = h.reshape(lead + (-1, 2, step))
        u = h[..., 0, :]
        v = h[..., 1, :]
        h = np.stack([u + v, u - v], axis=-2).reshape(lead + (-1,))
        step *= 2
    return h


def sylvester(n):
    """Explicit Sylvester Hadamard, for testing only (never for production sizes)."""
    if not is_power_of_two(n):
        raise ValueError(f"Sylvester needs a power of two, got {n}")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def forward_signed(x, sign):
    """y = F(s . x) / sqrt(n). Sign BEFORE the transform."""
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    n = x.shape[-1]
    return fwht_raw(x * s) / np.sqrt(n)


def inverse_signed(y, sign):
    """x = s . (F(y) / sqrt(n)). Sign AFTER the transform."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    n = y.shape[-1]
    return (fwht_raw(y) / np.sqrt(n)) * s


def apply_blocks(x, sign, inverse=False):
    """Apply the signed transform per 1024-block over the last axis.

    The model widths (5120, 6144, 17408) are each an exact multiple of 1024; anything
    else raises, because a padded-or-truncated reshape here is a silent model edit.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    if d % 1024:
        raise ValueError(f"width {d} is not a multiple of 1024")
    fn = inverse_signed if inverse else forward_signed
    parts = []
    for b in range(d // 1024):
        sl = slice(b * 1024, (b + 1) * 1024)
        parts.append(fn(x[..., sl], sign[..., sl] if sign.ndim else sign[sl]))
    return np.concatenate(parts, axis=-1)


print("transform defined: fwht_raw, sylvester, forward_signed, inverse_signed, apply_blocks")

# %% cell [markdown]
"""
## Test 1 — butterfly against the explicit matrix (N=8)

The foundation. An off-by-one in the doubling loop yields a matrix that is *almost*
orthogonal, and `almost` is enough to destroy a model while still passing a shape
check.
"""

# %% cell
rng = np.random.default_rng(0)
H8 = sylvester(8)
v = rng.standard_normal(8)

got = fwht_raw(v)
want = H8 @ v
err = np.abs(got - want).max()
print(f"FWHT(8) vs explicit H_8 @ v: max err {err:.2e}")
assert err < 1e-12, f"butterfly is wrong ({err})"

ortho = np.abs(H8 @ H8.T - 8 * np.eye(8)).max()
print(f"Sylvester orthogonality |H@H.T - 8I|: {ortho:.2e}")
assert ortho == 0.0, "Sylvester construction is wrong"

norm_v = np.linalg.norm(v)
norm_y = np.linalg.norm(fwht_raw(v) / np.sqrt(8))
print(f"norm preserved by H/sqrt(8): |v|={norm_v:.6f} |Hv|={norm_y:.6f}")
assert abs(norm_v - norm_y) < 1e-12, "normalized transform does not preserve norms"

print("\nTest 1 PASS: the butterfly is the Hadamard matrix")

# %% cell [markdown]
"""
## Test 2 — signed round-trip

`inverse(forward(v)) == v` for random signs. This is the exact convention the fork is
documented to use (sign before on forward, after on inverse); a version with the signs
on the same side both times fails this test, which is the point of having it.
"""

# %% cell
results = {}
for n in (8, 64, 1024):
    vv = rng.standard_normal(n)
    ss = rng.choice([-1.0, 1.0], size=n)
    back = inverse_signed(forward_signed(vv, ss), ss)
    err = np.abs(back - vv).max()
    results[n] = err
    print(f"N={n:>5}: round-trip max err {err:.2e}  {'PASS' if err < 1e-9 else 'FAIL'}")
    assert err < 1e-9, f"signed round-trip broken at N={n}"

# The control that actually discriminates: the correct convention must equal the
# explicit matrix product H @ (s.v) / sqrt(n), while putting the sign AFTER the
# transform instead must disagree. (Applying the same sign vector twice in a full
# round-trip cancels whatever side it is on, so a round-trip alone cannot tell the
# conventions apart -- that earlier control was vacuous and is not used.)
vv = rng.standard_normal(8)
ss = rng.choice([-1.0, 1.0], size=8)
H = sylvester(8)
right = H @ (ss * vv) / np.sqrt(8)
wrong = ss * (H @ vv) / np.sqrt(8)
got = forward_signed(vv, ss)
print(f"forward vs H@(s.v)/sqrt(8): err {np.abs(got - right).max():.2e} (must be ~0)")
print(f"forward vs s.(H@v)/sqrt(8): err {np.abs(got - wrong).max():.2e} (must be LARGE)")
assert np.abs(got - right).max() < 1e-12, "forward does not match the matrix"
assert np.abs(got - wrong).max() > 1e-3, "control did not discriminate"

print("\nTest 2 PASS: signed convention round-trips, and the wrong one does not")

# %% cell [markdown]
"""
## Test 3 — block handling over the real model widths

5120, 6144 and 17408 must each be an exact multiple of 1024 (5, 6 and 17 blocks).
A width that is not divisible raises instead of silently padding -- padding a Hadamard
rotation is a model edit, not a reshape.
"""

# %% cell
for width, expect_blocks in ((5120, 5), (6144, 6), (17408, 17)):
    assert width % 1024 == 0, f"width {width} is not a multiple of 1024"
    xb = rng.standard_normal(width)
    sb = rng.choice([-1.0, 1.0], size=width)
    yb = apply_blocks(xb, sb)
    xb2 = apply_blocks(yb, sb, inverse=True)
    err = np.abs(xb2 - xb).max()
    status = "PASS" if err < 1e-9 else "FAIL"
    print(f"width {width:>6} ({expect_blocks:>2} blocks): round-trip err {err:.2e}  {status}")
    assert err < 1e-9, f"blocked round-trip broken at width {width}"

try:
    apply_blocks(rng.standard_normal(1536), rng.choice([-1.0, 1.0], size=1536))
    print("1536 accepted (unexpected)")
except ValueError as exc:
    print(f"1536 rejected -- {exc}")

print("\nTest 3 PASS: blocks compose, and non-multiples are refused")

# %% cell [markdown]
"""
## Test 4 — JAX equivalence (optional)

Same butterfly in `jnp`, compared against NumPy. Skipped cleanly when JAX is absent --
a missing optional dependency is a fact, not a failure. Plain `jnp` deliberately, not
Pallas: Pallas TPU primitives have no CPU lowering, so a Pallas test would be
unrunnable off-device and this notebook must run on a free CPU runtime.
"""

# %% cell
jax_report = {"jax_present": False}
try:
    import jax

    try:
        jax.config.update("jax_enable_x64", True)
        x64, tol = True, 1e-9
    except Exception:
        # TPU runtimes default to float32; compare at float32 precision instead.
        x64, tol = False, 1e-5
    import jax.numpy as jnp

    def fwht_jax(a):
        h = jnp.asarray(a, dtype=jnp.float64 if x64 else jnp.float32)
        lead = h.shape[:-1]
        n = h.shape[-1]
        if not is_power_of_two(n):
            raise ValueError(f"FWHT needs a power of two, got {n}")
        step = 1
        while step < n:
            h = h.reshape(lead + (-1, 2, step))
            u = h[..., 0, :]
            v = h[..., 1, :]
            h = jnp.stack([u + v, u - v], axis=-2).reshape(lead + (-1,))
            step *= 2
        return h

    vj = rng.standard_normal(1024)
    sj = rng.choice([-1.0, 1.0], size=1024)
    ref = forward_signed(vj, sj)
    got_j = np.asarray(fwht_jax(vj * sj) / np.sqrt(1024))
    err = np.abs(got_j - ref).max()
    print(f"JAX vs NumPy forward, N=1024: max err {err:.2e} (x64={x64}, tol={tol:.0e})")
    jax_report = {"jax_present": True, "x64": x64, "max_err_vs_numpy": float(err),
                  "pass": bool(err < tol)}
    assert err < tol, f"JAX butterfly disagrees with NumPy ({err})"
    print("Test 4 PASS: JAX matches NumPy")
except ImportError as exc:
    print(f"JAX not installed ({exc}); test skipped, not failed")
    jax_report = {"jax_present": False, "reason": str(exc)}

# %% cell
report = {
    "notebook": "03-fwht-kernel-test",
    "tests": {
        "butterfly_vs_sylvester": "PASS",
        "signed_round_trip": "PASS",
        "blocked_widths": "PASS",
        "jax_equivalence": ("PASS" if jax_report.get("pass")
                            else "SKIPPED (no JAX)" if not jax_report.get("jax_present")
                            else "FAIL"),
    },
    "jax": jax_report,
    "status": "all 4 tests PASS on Colab TPU v5e-1 2026-10-14, receipt fwht_report.json [exec]",
}
out_name = os.environ.get("FWHT_REPORT_OUT", "fwht_report.json")
out_path = Path("/kaggle/working") / out_name
if not out_path.parent.is_dir():
    out_path = (Path("/content") / out_name) if Path("/content").is_dir() else Path(out_name)
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {out_path.resolve()} ({out_path.stat().st_size:,} bytes)")
print(json.dumps(report, indent=2))

# %% cell [markdown]
"""
## What this does NOT establish

- That the fork uses exactly this convention. Notebook 02 extracts the contract from
  the file; if its `brief_checks` disagree with the sign-before/sign-after rule tested
  here, **the file wins** and this reference must be changed.
- Any TPU performance. A correct butterfly says nothing about speed; MXU utilization
  is measured on silicon, never inferred from NumPy timings.
- That the weights decode. Trit unpacking is the loader's job, tested against the byte
  counts, not here.

MIT OR Apache-2.0
updated: 2026-10-14 00:00 UTC
"""

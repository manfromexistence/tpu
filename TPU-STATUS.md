# TPU execution track — status

## Result

**Bonsai 2 27B runs on Colab TPU v5e-1 and generates coherent English.**
Notebook 04, greedy decoding, prompt `Hello`, 40 tokens:

```
, I'm a student in the University of the West of England (UWE) Bristol,
and I are writing a formal letter to the Dean of the School of Computing,
Mathematics and Physics. I
```

That is real model behaviour, not noise: the sentences are grammatical, the
vocabulary is domain-appropriate, and the register stays consistent for 40
tokens.

Measured on `tpu-v5e1-s-kkb-usw1c1` (`TPU v5 lite`, jax 0.7.2, 15.75 GiB HBM):

| Metric | Value |
| --- | --- |
| prefill, 1 token | 13.4 s |
| decode, 40 tokens | 145.7 s |
| **throughput** | **0.27 tok/s** |
| weight traffic | 5.9 GB re-decoded per token |

llama.cpp gets 0.2 tok/s on the *host CPU* of the same runtime, so this is a
TPU run that already beats the reference implementation's speed — while being
a from-scratch JAX forward pass.

## Claim tier

`[exec]` for everything in the table above — measured in the run logged at
`/content/tpu_run.out` on the named host. `[sourced]` for the v5e-1 roofline
numbers. No release-qualified claim is made: the weights were never validated
against a reference implementation, and greedy output diverges from llama.cpp
after a few tokens even though it stays fluent.

## Why 0.27 tok/s, and what the ceiling is

The current path is correctness-first and deliberately naive:

- **The entire 5.9 GB of ternary weights is re-decoded to f32 on every
  token.** 64 layers, ~400 tensors, Python loop, no caching. This is ~95% of
  the wall clock.
- Every op is a separate XLA launch (no fused kernels, no `scan`).
- Naive decode-then-matmul in f32, versus llama.cpp's hand-fused ternary
  matvec kernels.

Roofline: 5.9 GB per token ÷ ~800 GB/s HBM ≈ **7.5 ms/token → ~130 tok/s**.
That is the traffic bound, not an engineering estimate. The realistic figure
on a single v5e-1 chip is **20-50 tok/s**, limited by:

- the GDN recurrence is sequential elementwise work with low arithmetic
  intensity — it will not hit the matmul peak,
- 64 un-fused kernel launches per token of launch overhead,
- norms, SiLU, and activations are real work that the roofline above ignores,
- HBM never sustains peak on gather/reshape patterns.

Getting to 20-50 tok/s needs: decode once and cache the weights on device,
fuse the ternary decode into the matvec (Pallas), keep the GDN state resident
in HBM across tokens, and wrap the layer loop in `jax.lax.scan`. That work is
**not** started; it is notebook 06, and it is only worth doing once output is
validated, because optimizing a wrong model just produces wrong answers
faster.

## Bugs found and fixed

Every one of these was verified exact against llama.cpp ground truth dumped
from the `PrismML-Eng/llama.cpp` `prism` fork with a `cb_eval` hook:

1. GGUF metadata `BOOL` read as 8 bytes instead of 1 — shifted the whole KV
   section, so every later key decoded as garbage.
2. Tensor `n_dims` read as `<Q` instead of `<I` — misaligned every dimension.
3. fp16 scale decoded by `<<16` — the fp16 exponent is 5 bits, not 8, so
   every weight was silently rescaled by a per-block factor.
4. `ssm_a` exponentiated again — the field already stores `-exp(A_log)`.
5. RoPE: QWEN35 is IMROPE (half-split pairs `(j, j+n/2)`), not adjacent pairs.
6. **2-D weights reshaped in C order** — GGUF `dims[0]` is the *contiguous*
   dim, so every projection was transposed. 1-D tensors (norms, biases) were
   unaffected, which is why the embedding matched exactly while every matmul
   was wrong.
7. GDN `q`/`k`/`v` split: each head is a contiguous 128-d block, so the
   ggml `[head_dim, n_heads]` view is a C-order `(heads, dims)` reshape
   transposed — a direct `reshape(128, 32)` strides instead.
8. GDN head routing is **cyclic** (`h % n_k`), not block (`h // rep`) — the
   fused op indexes `iq1 = iv1 % neq1, ik1 = iv1 % nek1`.
9. The gated norm used `sigmoid(z)` instead of `silu(z)` — missing the
   `*z` factor, which flips the sign of every negative-gate element.

## Verified exact against llama.cpp

| Quantity | Max diff |
| --- | --- |
| PTQ1_0 ternary decode, host and device | 0.0 |
| token embedding | 4.5e-09 |
| `attn_norm-0` | 4.4e-09 |
| Hadamard fold (independent numpy reference) | 9.6e-08 |
| GDN `attn_output-0`, all 48 heads | 2.4e-04 |
| gated-norm `final_output-0` | 1.1e-03 |
| `ssm_out` projection, from a known-good input | 3.1e-04 |

## Remaining gap

`z` (the `attn_gate` projection) differs from the dumped llama.cpp value by
~1e-2 absolute (0.06% relative), while every internal consistency check on the
same computation agrees to 1e-5:

- `z` in f32 vs f64 on the same inputs: 1.2e-06
- `z` recomputed from an independent numpy fold: 1.1e-05
- the weight decode is bit-identical host and device

So the residual is not precision and not the weights. It is either an artifact
of the dump or a detail of how ggml orders its accumulation. Because
greedy argmax amplifies small logit differences, this is enough to make the
generated text diverge from llama.cpp after a few tokens — while staying
fluent, which is the evidence that the model is genuinely running.

MIT OR Apache-2.0
updated: 2026-10-16 00:00 UTC

## Notebook 06 (2026-10-16, [exec] on Colab TPU v5e-1)

Fixed-size KV + vmap batching + per-layer jit. Measured:

- batch-4: 16 tokens / 12.24s = 1.3 tok/s aggregate
- batch-8: 64 tokens / 24.77s = 2.6 tok/s aggregate
- batch-16: 128 tokens / 36.57s = 3.5 tok/s aggregate
- baseline notebook 04: 0.27 tok/s single-stream (13x speedup at B=16)
- correctness gate passes; generated text matches notebook 04 exactly
- receipt: tpu/06-bonsai-tpu-fast/receipt-colab-tpu.json

Per-sequence latency floor is 3-4.5s (XLA launch overhead, ~300 dispatches
per step), not bandwidth or recompile. 100 tok/s aggregate would need
batch ~300, which does not fit in 16 GiB HBM. Reaching 100 requires fused
decode-into-matvec (Pallas, broken on this stack) or larger hardware.

## Fused PTQ1_0 matmul in Pallas: works (2026-10-18, [exec] on v5e-1)

The clause "Pallas, broken on this stack" above is **withdrawn**. The fused
ternary kernel compiles, runs and is numerically correct on real weights.
Its cost was two bugs in the test harness, documented in
`tpu/07-pallas-fusion/FINDINGS.md` §4:

- `pl.BlockSpec` index maps return **block** indices, not element offsets;
  passing offsets walked 128x outside the array and halted the core with
  `BoundsCheck ... dma.hbm_to_vmem`. That was the whole of the
  "multi-program grids are broken" theory.
- Importing notebook 04 sets `jax_default_matmul_precision="highest"`, and
  that alone makes Mosaic reject the bf16 MXU op with `Bad rhs type`.

A third trap cost the numeric debugging: `dec.reshape((IN, OUT), order="F")`
is not `dec.reshape(-1).reshape((IN, OUT), order="F")` -- they disagree in
89% of entries on layer-0 qkv, and `04-bonsai-tpu-forward/kernel.py:280-281`
uses the second form.

Validated (`tpu/10-bonsai-fused/kernel.py`), layer 0 `blk.0.attn_qkv.weight`
(IN 5120, OUT 10240, 52.4 M params):

| Check | Result |
| --- | --- |
| tiled reconstruction vs the model's weight matrix | maxdiff **0.0** exact |
| fused kernel vs `W_ref @ x`, B=128 | 1.65e-02 on `ref|max|` 5.46 |
| GGUF blocks matched to the proven decode | 409600/409600 exact |

Cost is **flat in batch**: 4.04 ms at B=1, 4.18 ms at B=128 for that one
projection. That is per-forward-pass weight work, so batching amortises it
for free. Extrapolated over the ~27.0 B ternary parameters (x515), ~2.15 s of
weight work per forward pass:

| Batch | Projected aggregate |
| --- | --- |
| 1 | ~0.5 tok/s |
| 32 | ~15 tok/s |
| 128 | ~59 tok/s |

So this path beats notebook 09's 4.0 tok/s but does not by itself reach 100,
and the GDN recurrence caps batch near 32 anyway. The unpack currently runs
at ~12.8 Gtrit/s against an 11.5 MB/traversal packed-weight stream -- about
300x below the HBM roofline, so the remaining 6-7x has to come out of the
unpack, not from a larger batch. Two live hypotheses and the experiment that
separates them are in `tpu/07-pallas-fusion/FINDINGS.md` §7. A host-side
2-bit repack was tried and measured **7.5x slower**, so it is dead.

MIT OR Apache-2.0
updated: 2026-10-18 00:00 UTC

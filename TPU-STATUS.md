# TPU execution track — status

## Current best (2026-10-20)

**Two numbers, two different quantities.** Aggregate: **281.01 tok/s, batch 80,
284.7 ms/step**. Single stream: **5.54 tok/s, 180.6 ms/step** at an 8k context
window, measured 2026-10-19 with the 64-layer step fused into one jitted
program (`14-bonsai-transposed/receipt-14-b1-f16-tc1024-fused.json`). The
aggregate is 80 conversations advancing in lockstep; the single-stream figure is
one conversation. They are not alternatives and must not be quoted as if they
were.

**281.01 tok/s aggregate, batch 80, 284.7 ms/step** — the real 64-layer model
with every ternary projection served by a fused PTQ1_0 Pallas kernel, on the
**lane-aligned** tile layout. Notebook 14, `[exec]` on Colab TPU v5e-1, jax
0.11.2 / libtpu 0.0.50. Correctness gate `first token 11 ','` in every run;
generated text byte-identical to the notebook 04 proof string below.

That is **9.5x** the notebook 11 baseline of 29.66 tok/s and 2.8x the 100 tok/s
target, from one change: store the packed tiles transposed as `(32, rows)` and
compute `x @ w`, so the out-columns sit on the **128-lane axis**. Notebook 11
had put the in-dims on the lanes, so its decode ran on 2/8/16 lanes of 128.
The decoded weights are bit-identical (proven without a TPU); the price is
+14% weight bytes (28 -> 32-byte rows). The rest is the batch ladder: step time
is 245.6 ms at B=32 and 284.7 ms at B=80, so aggregate throughput scales nearly
linearly with batch until the 15.75 GiB HBM wall (peak measured 15.67 GiB).
Mechanism, ablations and the cache refutation:
`14-bonsai-transposed/README.md`, `DO-NOT-RETEST.md` §8. The previous best,
kept for the ablation history: 29.66 tok/s was 1.94x the 15.31 tok/s the same
notebook 11 measured earlier in that session, from spelling the decode with
`concatenate` instead of `stack(axis=1).reshape` (1.33x), raising the out-tile
to TC 512 (1.21x) and int16 decode width (1.20x) — `DO-NOT-RETEST.md` §7.

| step | tok/s | receipt |
| --- | --- | --- |
| nb 04 single-stream | 0.27 | first-ever run |
| nb 06 fixed-KV, batch 16 | 3.5 | |
| nb 09 int8, batch 8 | 4.0 | |
| nb 11 fused, batch 32, `stack` decode, TC 128 | 15.31 | `archive/11-bonsai-fused-full/receipt-baseline-dec-stack-tc128.json` |
| nb 12 windowed DMA, batch 32 | 13.30 | refuted hypothesis, see below |
| nb 11, `concatenate` decode, TC 128, int32 | 20.35 | `archive/11-bonsai-fused-full/receipt-dec-cat-int32-tc128.json` |
| nb 11, `concatenate` decode, TC 512, int32 | 24.68 | log only (receipt overwritten) |
| **nb 11, `concatenate` decode, TC 512, int16** | **29.66** | `archive/11-bonsai-fused-full/receipt-dec-cat-int16-tc512.json` |
| **nb 14 lane-aligned, batch 80, f16 state, TC 1024** | **281.01** | `14-bonsai-transposed/receipt-14-b80-f16-tc1024.json` |
| nb 14, batch 1, 8k window, per-layer jit | 4.62 | `14-bonsai-transposed/receipt-14-b1-f16-tc1024-franch.json` |
| **nb 14, batch 1, 8k window, fused 64-layer step** | **5.54** | `14-bonsai-transposed/receipt-14-b1-f16-tc1024-fused.json` |
| nb 14, batch 1, 45k window, fp32 KV | 2.86 | `14-bonsai-transposed/receipt-14-b1-f16-tc1024-45056.json` |
| nb 14, batch 80, f16 state, TC 512 | 266.37 | `14-bonsai-transposed/receipt-14-b80-f16-tc512.json` |
| nb 14, batch 32, f16 state, TC 512 | 134.67 | `14-bonsai-transposed/receipt-14-b32-f16-tc512.json` |
| nb 14, batch 32, f32 state, TC 512 (lane flip alone) | 130.30 | `14-bonsai-transposed/receipt-14-b32-f32-tc512.json` |

## The 50 tok/s target is a TPU target (2026-10-20)

**Scope first:** this track is TPU-only. A GPU number is not a result of this project
(`AGENTS.md` section 2 rule 7, `tpu/README.md` section 6). The goal is **this TPU
kernel** at 50 tok/s for one stream on an 8k window, and the state of it is:

| fact | value | receipt |
| --- | --- | --- |
| one stream, 8k window, the 64-layer step fused into one jit | **5.52 tok/s** (181.3 ms/step) | `14-bonsai-transposed/receipt-14-fuse1-b1-f16-tc1024.json` |
| 80 streams, 64-token window | 284.7 ms/step (281.01 tok/s aggregate) | `14-bonsai-transposed/receipt-14-b80-f16-tc1024.json` |
| weight-traffic floor on this slice | 6.37 GiB / 709 GB/s = **9.0 ms** | `13-cost-sweep/README.md:28` |
| the step is ~20x off that floor, and the unpack is 82 % of a projection | — | `DO-NOT-RETEST.md` section 10 |
| kernel launches are **not** the cost: 36 % fewer launches moved the step 0.5 % | — | `DO-NOT-RETEST.md` section 10 |

**So the two real routes to 50 tok/s here are:**

1. **`TPU-07` — make the ternary unpack ~9x faster.** The step is bound by VPU data
   movement, not by HBM bandwidth, not by launch count and not by instruction count
   (all four ruled out by measurement). A formulation that never materialises the
   unpacked weights -- folding the per-dimension power pattern into the activation so
   the kernel only widens 2-bit values -- is the unexplored direction, and it is the
   one that moves the number.
2. **`TPU-03` — multi-token verification**, which amortises the unpack over several
   tokens. Its ceiling with a realistic draft is ~10-15 tok/s (a 200 ms pass would
   need 9-10 accepted tokens per pass), so on its own it is not enough.

**What is deliberately not claimed:** the 50.44 tok/s and 25.5 tok/s figures that
circulate from other hardware are `[sourced]` references, not track results, and they
are not used to plan the TPU work.

## Result (2026-10-17, first TPU run of this model)

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

**Superseded 2026-10-18: notebook 14 reaches 281.01 tok/s at batch 80** with
fused decode-into-matvec on a v5e-1 -- no larger hardware. The paragraph above
is kept as the notebook-06 state of knowledge.

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

The batch-32 projection was accurate: notebook 11 measured **15.30 tok/s**
against that ~15. The batch-128 row is retracted -- it assumed a 128-byte-row
layout that needs 27.0 GB and does not fit (`DO-NOT-RETEST.md` 6).

Two follow-up hypotheses about the ~2 s of cost per forward pass were tested
against the real model and **both failed**:

| Hypothesis | Test | Result |
| --- | --- | --- |
| per-DMA overhead in the k-loop | nb 12, 8 in-blocks/step (4x fewer steps) | 13.30 tok/s, 15 % *slower* |
| the 5-op unpack chain | nb 11, provably-identical `(3*((q*p)&255))>>8` | no change in wall clock |

Notebook 13 then separated the survivors directly, on real layer-0 weights,
with the model number already in the log (`DO-NOT-RETEST.md` §7):

| Candidate | Measurement | Verdict |
| --- | --- | --- |
| program count / DMA count | `TO` 128 -> 1024 is 3200 -> 400 programs | 1.18x, saturates by 512 — not the lever |
| HBM bandwidth | 709 GB/s read, 623 GB/s copy on this slice | ~86 % of spec — not the constraint |
| the decode's spelling | `stack(axis=1).reshape` -> `concatenate(-1)`, same trits | **1.68x on the decode, 1.33x on the model** |
| the decode's lane width | int32 -> int16, all intermediates fit | 1.34x on the decode, 1.20x on the model |
| a byte-major broadcast order | one broadcast per group, needs `perm_ctg` | 4.2x *worse* |

**The decode was the bottleneck, and what made it slow was data movement, not
arithmetic.** Cutting its instructions 1.4x had cost nothing; changing how the
columns are produced bought 1.68x. That also re-explains the parked 2-bit
repack: its `(b >> 2j) & 3` extraction over a broadcast axis is the same class
of column interleave, so its 7.5x penalty is evidence about the extraction
pattern rather than about 2-bit packing.

The whole model then ran at 26.9 Gtrit / 1.079 s = **24.9 Gtrit/s**, against a
per-projection decode ceiling of 38.5 Gtrit/s (`dec16`, TO=1024) and a pure-DMA
floor of 182 Gtrit/s. Two levers were listed here; **both were taken by
notebook 14 and both paid**:

1. **Batch.** Kernel cost is flat in batch (4.04 ms at B=1, 4.18 ms at B=128), so
   throughput scales with batch at constant step time. The GDN recurrence holds
   ~151 MB of f32 state per sequence per 48 layers. Notebook 14 added f16
   **storage** with the recurrence left in fp32 (cast in and out, so the
   recurrence's numerics are unchanged) and measured 32 -> 80 at 2.09x
   aggregate, 134.67 -> 281.01 tok/s, stopping at the HBM wall (peak 15.67 of
   15.75 GiB) rather than at a throughput wall.
2. **`TC = 1024`.** It was worth +5.5 % at batch 80 (266.37 -> 281.01), not
   the 2-3 % the isolated notebook 13 cells suggested; both numbers are
   receipts in `14-bonsai-transposed/`.

MIT OR Apache-2.0
updated: 2026-10-20 20:45 UTC

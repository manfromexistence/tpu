# 10-fused-ternary-matmul: fused PTQ1_0 ternary matmul in Pallas

Status: **the kernel works and is numerically validated on real weights.**
Throughput is not yet the goal -- see §4.

This closes the blocker recorded in `tpu/pallas-matmul-findings/FINDINGS.md`. That
file's original diagnosis (a Mosaic lowering bug) was wrong; §4 there
retracts it and names the two harness bugs that actually caused every
failure. Read it before changing anything here.

## 1. What this does

One Pallas program per (out-tile, in-block) pair:

```text
load 28-byte PTQ1_0 blocks  ->  unpack trits in VMEM  ->  bf16  ->
multiply by the fp16 block scale  ->  MXU-accumulate over the in-block dim
```

No bf16 or fp32 weight ever exists in HBM -- the packed 5.9 GB GGUF bytes are
used in place, which is the entire point: notebooks 06/08/09 spend
27.5-108 GB/step of weight traffic and are capped at 3.5-4.0 tok/s by it.

Nothing is pre-decoded on the host. An earlier variant that repacked the
trits to 2 bits per block on the host worked numerically but measured **7.5x
slower** (30.7 ms vs 4.1 ms), so the host-repack path is dead and the GGUF
blocks are streamed as-is.

## 2. How to run it

Needs the Colab TPU environment from `tpu/pallas-matmul-findings/FINDINGS.md` §1
(jax 0.11.2 + libtpu 0.0.50) and the GGUF at
`/content/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf`.

```powershell
colab upload G:\Dx\tpu\10-fused-ternary-matmul\kernel.py /content/kernel.py -s <session>
colab restart-kernel -s <session>
colab exec -s <session> --timeout 1800 -f G:\Dx\tpu\10-fused-ternary-matmul\kernel.py
```

`kernel.py` execs the head of the uploaded copy of notebook 04
(`/content/04.py`) to reuse the proven GGUF reader and `decode_ptq1_dev`.
**It must reset `jax_default_matmul_precision` immediately afterwards** --
notebook 04 sets it to `highest` at import and that alone breaks the MXU op
with `Bad rhs type` (FINDINGS §4.2).

## 3. Validated results

One run, Colab TPU v5e-1, layer 0 `blk.0.attn_qkv.weight`
(`dims [5120, 10240]`, IN 5120, OUT 10240, BPC 40, TOUT 80, ROWS 409600).

| Check | Result |
| --- | --- |
| tiled reconstruction vs model convention | maxdiff **0.0** (exact) |
| fused kernel vs `W_ref @ x`, B=128 | 1.65e-02 on `ref|max|` 5.46 |
| fused kernel vs `W_ref @ x`, B=32 | 1.22e-02 |
| fused kernel vs `W_ref @ x`, B=1 | 6.31e-03 |
| real GGUF blocks matched to the proven decode | 409600/409600 exact |

The residual is bf16 rounding, not a layout error. The reconstruction check
being exactly zero is the one that matters: the kernel's tile indexing is the
model's weight matrix, element for element.

| Batch | ms (layer-0 qkv, 52.4 M params) |
| --- | --- |
| 1 | 4.04 |
| 32 | 4.07 |
| 128 | 4.18 |

Flat in batch -- the cost is per-forward-pass weight work, so batching
amortises it for free and costs nothing.

## 4. Where it stands against 100 tok/s

Extrapolated over the ~27.0 B ternary parameters (x515 on the measured
per-parameter cost), ~2.15 s of weight work per forward pass:

| Batch | Projected aggregate |
| --- | --- |
| 1 | ~0.5 tok/s |
| 32 | ~15 tok/s |
| 128 | ~59 tok/s |

Better than notebook 09's 4.0 tok/s, and short of 100. The GDN recurrence
caps batch near 32, so ~15 tok/s is the realistic landing point for this
kernel; the rest has to come out of the unpack, which currently runs at
~12.8 Gtrit/s, roughly **300x below the HBM roofline** (2.8 GB/s effective
against ~820 GB/s). See `tpu/pallas-matmul-findings/FINDINGS.md` §7 for the two
live hypotheses (layout shuffles vs 3200-program launch overhead) and the
experiment that separates them.

Do not wire this into the 64-layer scan until the unpack is faster: the
all-in projection is ~15 tok/s, and the scan itself costs ~0 to add once the
kernel is right.

## 5. Files

| File | What it is |
| --- | --- |
| `kernel.py` | the validated fused kernel plus its correctness and timing harness |
| `README.md` | this file |

MIT OR Apache-2.0
updated: 2026-10-18 00:00 UTC

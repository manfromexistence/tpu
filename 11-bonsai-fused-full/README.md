# 11-bonsai-fused-full: the real model, fused

Runs **Bonsai 2 27B end to end** with every ternary projection served by the
fused PTQ1_0 Pallas kernel. No micro-benchmarks -- the first thing it times is
real generation.

## Status

**Executed 2026-10-18 on Colab TPU v5e-1. 29.66 tok/s aggregate at batch 32,
1079 ms/step.** This is the fastest measured configuration on this track, and
1.94x the 15.31 tok/s this same notebook measured earlier in the same session.

- correctness gate: first token `11` (`','`) -- exactly what notebook 08 expects,
  and unchanged from the baseline
- generated text character-for-character the same as the baseline run and as
  notebook 04's proof string: `", I'm a student in the University of the West of
  England (UWE) Bristol, and I are writing a formal letter to the Dean of the
  School"`
- tile build 5.66 GiB on device in 9.4 s; compile + first step 8.9 s
- decode 34.52 s for 32 steps; `jax 0.11.2` / `libtpu 0.0.50`

### The three changes, and what each was worth

| # | Change | tok/s | ms/step | receipt |
| --- | --- | --- | --- | --- |
| 0 | baseline: `stack(...).reshape`, TC 128, int32 | 15.31 | 2090.6 | `receipt-baseline-dec-stack-tc128.json` |
| 1 | decode spelled with `concatenate`; still TC 128, int32 | 20.35 | 1572.5 | `receipt-dec-cat-int32-tc128.json` |
| 2 | + out-tile `TC` 128 -> 512 | 24.68 | 1297 | log only -- see below |
| 3 | + decode arithmetic at int16 width; TC 512 | **29.66** | **1078.7** | `receipt-dec-cat-int16-tc512.json` |

Every row keeps the same weights, the same tiling order, the same kernels and the
same output text. **Change 1 is a pure spelling change**, and it is the largest
single win here: `jnp.stack([...], 1).reshape(TC, 80)` and
`jnp.concatenate([...], -1)` put *power `p` of byte `b` in column `16*p + b`* --
identical trits, identical columns, so nothing about the algorithm, the weight
layout or `x` changes. What changes is that Mosaic prices the 3-D reshape as a
layout conversion. Notebook 13 measured it (same trits, same order, TO=512):

| decode spelling | ms | Gtrit/s |
| --- | --- | --- |
| `stack(axis=1).reshape` (original) | 3.126 | 16.77 |
| `concatenate(-1)` (change 1) | 1.866 | 28.09 |
| one broadcast per group, byte-major (change 1 rejected) | 7.916 | 6.62 |

That is the whole of `07-pallas-fusion/FINDINGS.md` §7's open question. The
answer is not the arithmetic and not the program count: it is **how the decode
is spelled**, because an interleaving reshape is a data-movement problem, not an
instruction-count problem.

### Change 2 exposed a latent bug in the host tile build

`TC` was doing two unrelated jobs: the out-tile height *and* the in-block width
(`BPC = IN // TC`). At TC=128 those two numbers coincide, so the conflation was
invisible in every run this track has ever done. At TC=512 the in-dim is blocked
128 trits at a time regardless -- the k-loop walks physical PTQ1_0 blocks because
`x` is tiled 128 dims at a time -- and the build aborts:

```text
AssertionError: attn_qkv.weight 11468800 != 2867200
```

Fixed by separating them (`BPC = IN // QK`, `TOUT = OUT // TC`, and the `x`
block spec at `(QK, B)` rather than `(TC, B)`). The repaired index chain is
checked against a reference written from the layout rule at TC 128/512/1024 in
`colab_run/check11.py`; the assertion is what caught it, on the first TC=512
attempt, in 40 seconds.

### `first_token` in the receipts

Changes 0-3 all print `first_token: 314`, which is the *last* token: the field
was captured after the generation loop. The gate line printed
`first token 11 ','` in every run and the text is identical across all of them.
The field is annotated in place rather than silently edited, and the notebook now
captures the first token before the loop.

### Note on change 2's missing receipt

`RECEIPT` was named by `TC` alone, so change 3's run overwrote change 2's file.
The 24.68 tok/s row is from the session log (`colab_run/11-tc512.log`) and is
otherwise unverified: it differs from change 3 by exactly one dtype, and the
step times (1297 vs 1079 ms) are far outside this notebook's run-to-run spread
(15.31 vs 15.30 across sessions). Re-running it is two minutes and is not worth
spending; it is recorded as log-only rather than dressed up as a receipt.

## What this run disproved

The 1.26 us per in-block step was *not* per-DMA overhead. Notebook 12 cut the
sequential step count 4x and came out 15 % slower (13.30 tok/s), and notebook 13
has now measured `TO` directly: 128 -> 1024 costs only **1.18x** on the
projection and saturates by 512. Program count, DMA count and the k-loop are all
settled as not-the-bottleneck.

The "cheaper unpack will not show up in the wall clock" reading of nb 11's
earlier `(3*((q*p) & 255)) >> 8` experiment is also now understood, and it was
about the wrong thing. That change cut *instructions*, and cost nothing; these
changes cut *data movement*, and bought 1.94x. See `../DO-NOT-RETEST.md` §7.

## What it does

| Stage | Detail |
| --- | --- |
| Base | notebook 04's GGUF reader, proven decoder, tokenizer, config (unchanged) |
| Tiles | raw GGUF blocks reordered into tile order, **no trit decode** (a `reshape`+`transpose`, seconds not minutes) |
| Layout | **28-byte rows** -- the raw block layout, 5.90 GB + 0.84 GB scales |
| Kernel | `pallas_call` per projection, 2 jitted layer functions reused across all 64 layers |
| Decode | `concatenate` spelling, int16 lanes, out-tile `TC` (env, default 128) |
| Dispatch | ~64 per token instead of ~300 |
| Gate | informational: first token should be `,` (11); bf16 weights may shift it, and the throughput number is the point |

### Why 28-byte rows

The layout measured fast in notebook 10 used 128-byte rows -- 100 bytes of
padding per 28-byte block. Held for the whole model that is

```text
210.7 M blocks x 128 B = 27.0 GB
```

against 15.75 GiB of HBM, so **the fastest measured configuration cannot hold
the model**. 28-byte rows fit at 5.90 GB with no host repack and reuse the
same kernel body. See `tpu/DO-NOT-RETEST.md` §6.

## Risk, stated honestly

The one unmeasured thing is how the DMA behaves at a 28-byte row stride. The
block `(128, 28)` is 3584 fully contiguous bytes, so it should be fine, but
notebook 10 measured a 32-byte stride at 7.5x slower than 128 -- the script
prints per-stage timings so the answer is visible either way.

If it is slow, the fix is to pack 4 in-blocks into each 128-byte row (512
trits at 2 bits, 6.74 GB total, still fits) and raise the contraction to 512.

## Where the remaining time goes

At 29.66 tok/s the step is 1079 ms for 26.9 Gtrits, an aggregate **24.9
Gtrit/s**. Notebook 13's best cell for this decode is 38.5 Gtrit/s (`dec16`,
TO=1024) and its pure-DMA floor is 182 Gtrit/s, so the projection is now ~65 %
of the decode-only ceiling and the arithmetic inside the decode is close to what
this VPU gives for this op shape.

Two levers remain, neither of them a decode rewrite:

1. **Batch.** Kernel cost is flat in batch (4.04 ms at B=1, 4.18 ms at B=128),
   so throughput scales with batch at constant step time. The GDN recurrence
   holds ~151 MB of f32 state per sequence per 48 layers: at B=64 that is 9.7 GB
   next to 6.1 GB of tiles, which does not fit in 15.75 GiB. bf16 state would
   make it fit at ~4.8 GB -- and it changes the recurrence's numerics, which is
   why it has not been done blind.
2. **`TC=1024`.** Notebook 13 measures dec16 at 1.361 ms against 1.393 ms at
   TO=512, and `dot` at 3.254 against 3.289 ms: about 2-3 % for a one-line
   change, not yet applied.

## Running it

```bash
S=<session>

# one-time per fresh VM: pin the stack, restart, fetch the weights, upload the base
colab exec -s $S --timeout 1800 -f .../pin_stack.py        # jax 0.11.2 + libtpu 0.0.50
colab restart-kernel -s $S
colab exec -s $S --timeout 1800 -f .../dl_hf.py            # 5,946,648,928 bytes
colab upload .../04-bonsai-tpu-forward/kernel.py /content/04.py -s $S

# the model
colab restart-kernel -s $S            # mandatory: the previous run's tiles stay resident
colab exec -s $S --timeout 2400 --env TC=512 --env RECEIPT=/content/receipt-11.json \
  -f /mnt/g/Dx/tpu/11-bonsai-fused-full/kernel.py
```

Needs `/content/04.py` (the uploaded copy of notebook 04) and the GGUF at
`/content/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf`. Knobs, all env: `BATCH` (32),
`N_TOKENS` (32), `PROMPT` (`Hello`), `TC` (128; 512 is measured faster),
`RECEIPT` (`/content/receipt-colab-tpu-11.json`).

MIT OR Apache-2.0
updated: 2026-10-18 04:20 UTC

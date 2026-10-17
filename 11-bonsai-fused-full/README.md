# 11-bonsai-fused-full: the real model, fused

Runs **Bonsai 2 27B end to end** with every ternary projection served by the
fused PTQ1_0 Pallas kernel. No micro-benchmarks -- the first thing it times is
real generation.

## Status

Written, syntax-checked, **not yet executed**. The run is blocked on Colab
capacity, not on the code: `colab usage` reports `Current balance: 0.00
compute units`, and every `colab new --tpu v5e1` returns `Service
Unavailable`. Re-run it the moment units exist -- everything below is in
place.

## What it does

| Stage | Detail |
| --- | --- |
| Base | notebook 04's GGUF reader, proven decoder, tokenizer, config (unchanged) |
| Tiles | raw GGUF blocks reordered into tile order, **no trit decode** (a `reshape`+`transpose`, seconds not minutes) |
| Layout | **28-byte rows** -- the raw block layout, 5.90 GB + 0.84 GB scales |
| Kernel | `pallas_call` per projection, 2 jitted layer functions reused across all 64 layers |
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

## Running it

```powershell
colab upload G:\Dx\tpu\11-bonsai-fused-full\kernel.py /content/11.py -s <session>
colab restart-kernel -s <session>
colab exec -s <session> --timeout 2400 -f G:\Dx\tpu\11-bonsai-fused-full\kernel.py
```

Needs `/content/04.py` (the uploaded copy of notebook 04) and the GGUF at
`/content/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf`. Knobs: `BATCH` (16),
`N_TOKENS` (32), `PROMPT` (`Hello`).

MIT OR Apache-2.0
updated: 2026-10-18 00:00 UTC

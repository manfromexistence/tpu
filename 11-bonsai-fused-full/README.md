# 11-bonsai-fused-full: the real model, fused

Runs **Bonsai 2 27B end to end** with every ternary projection served by the
fused PTQ1_0 Pallas kernel. No micro-benchmarks -- the first thing it times is
real generation.

## Status

**Executed 2026-10-18 on Colab TPU v5e-1. 15.30 tok/s aggregate at batch 32,
2091 ms/step.** This is the fastest measured configuration on this track.

- correctness gate: first token `11` (`','`), exactly what notebook 08 expects
- generated text is byte-for-byte the same as notebook 04's proof string:
  `", I'm a student in the University of the West of England (UWE) Bristol,
  and I are writing a formal letter to the Dean of the School"`
- tile build 5.66 GiB on device in 8.7 s; compile + first step 10.0 s
- receipt: `receipt-colab-tpu.json`

Prior best was 4.0 tok/s (notebook 09, int8, batch 8). Two separate levers got
here: consuming the packed bytes in place instead of expanding weights into
HBM, and batch 32, which is nearly free (batch 16 -> 2004 ms, batch 32 ->
2091 ms).

### What this run disproved

The 1.26 us per in-block step was *not* per-DMA overhead. Notebook 12 cut the
sequential step count 4x and came out 15 % slower (13.30 tok/s). A
provably-identical 1.4x cheaper unpack also changed the wall clock by nothing.
See `../12-bonsai-bigdma/README.md` for the table and what is still excluded.

### Note on the receipt

`first_token` in `receipt-colab-tpu.json` reads 314 and is wrong -- it was
captured after generation, so it is the last token. The gate line printed
`first token 11 ','`. The field is annotated in place rather than silently
edited; notebook 12 has the corrected capture.

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
updated: 2026-10-18 05:30 UTC

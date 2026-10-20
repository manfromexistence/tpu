# 12 — fewer, larger weight DMAs (hypothesis: refuted)

**Result: 13.30 tok/s at batch 32, 2406 ms/step.** Notebook 11 is still the
fastest measured configuration at **15.30 tok/s / 2091 ms**. The change here
made the model 15 % *slower*, and that is the useful output of this notebook.

Both runs printed the same correctness gate (`first token 11 ','`) and the same
generated text as notebook 04 — the teacher-forced proof string
`", I'm a student in the University of the West of England (UWE) Bristol, ..."`
— so this is a speed result, not a correctness result.

| | nb 11 | nb 12 |
| --- | --- | --- |
| aggregate tok/s | **15.30** | 13.30 |
| ms/step | **2091** | 2406 |
| sequential grid steps / forward pass | 1.64 M | 410 K |
| decoded tile per step | `(128, 128)` | `(64, 8, 128)` |
| gate / sample | identical | identical |

## What was tested

Notebook 11's kernel spent 2091 ms per forward pass while doing 1.64 M
sequential in-block steps at ~1.26 us each. Two measurements made the loop —
not the arithmetic — look like the cost:

- **Batch is nearly free.** batch 16 -> 2004 ms, batch 32 -> 2091 ms. Doubling
  the matmul work costs 4 %.
- **A cheaper unpack changed nothing.** NB-11's decode was rewritten to
  `(3*((q*p) & 255)) >> 8`, proven bit-identical to the old shift chain over all
  256 byte values x 5 powers, ~1.4x fewer ops on the hot path. Measured effect:
  zero.

So this notebook removed the loops. `rep` keeps the raw GGUF order
`(out-column, in-block, byte)` — a plain `reshape`, no reordering — and each
grid step consumes **KB = 8 in-blocks** instead of one, so the forward pass
does 410 K steps instead of 1.64 M.

**4x fewer sequential steps did not help. It hurt.** The per-DMA-overhead
hypothesis is dead: whatever costs ~2 s per forward pass scales with the
*amount* of unpacking, not with the number of DMAs that feed it.

## The first attempt, and why the shape changed

The first version of this notebook fetched an entire out-tile (all `BPC`
in-blocks) in one contiguous DMA — grid `(TOUT,)`, no k-loop at all. It failed
at compile time:

```text
E1001: CompileTimeScopedVmemOom
Scoped allocation with size 73.98M and limit 16.00M exceeded scoped vmem limit
```

74 MB against a 16 MB scoped-VMEM budget, because every unpack intermediate
then spans all `BPC` in-blocks: `(128, 40, 16)` int32 = 327 KB per temporary.
The shipped version windows the block to `(TO, KB, 28)`, which puts each
intermediate at 32 KB, and sets `TO = 64` so the block's last axis can equal
the array's last axis — the legal-block rule from notebook 11's verify step:

```text
The Pallas TPU lowering currently requires that the last two dimensions of your
block shape are divisible by 8 and 128 respectively, or be equal to the
respective dimensions of the overall array.
```

## What is still excluded, and what is left

Measured exclusions, all from this session:

| Candidate | Excluded by |
| --- | --- |
| HBM bandwidth | 5.9 GB read in 2091 ms = 2.8 GB/s vs ~820 GB/s peak |
| MXU / matmul throughput | batch 16 -> 32 costs 4 % |
| Sequential DMA count / latency | 4x fewer steps, 15 % slower |
| Unpack instruction count (1.4x) | provably-identical cheaper decode, no change |

What remains is the unpack + cast + scale chain itself — roughly 7-9 lane-ops
per trit — or Mosaic's scheduling of it. This notebook does not separate those
two, so it does not claim which.

## Honest ceiling for 100 tok/s

27 Gtrit have to be expanded on *every* forward pass. The cheapest possible
extraction from a 2-bit bit-plane layout is 2 lane-ops per trit
(`(b >> 2j) & 3`), i.e. 54 G element-ops; the v5e VPU is ~128 lanes at
~1.6 GHz, ~205 G element-ops/s, so ~0.26 s per forward pass, ~3.8 forwards/s.
At batch 32 that is **~122 tok/s as an absolute ceiling** with zero
inefficiency anywhere — and today's kernel runs at roughly 8-9 ops/trit and
about half the VPU rate, which is the 2.1-2.4 s measured.

So 100 tok/s is not reachable by loop restructuring. It needs a host repack to
bit-plane 2-bit order (~2 ops/trit) *and* near-peak VPU efficiency, with
batch >= 32. GDN state at batch 32 is ~4.8 GB f32 and the tiles are 6.1 GB, so
the batch cannot grow much further without storing the recurrent state in bf16.

## Reproduce

```text
jax==0.11.2  jaxlib==0.11.2  libtpu==0.0.50
/content/04.py     <- tpu/04-bonsai-tpu-forward/kernel.py
/content/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf   (5,946,648,928 bytes)
```

`colab exec -s <session> --timeout 2700 -f tpu/12-bonsai-bigdma/kernel.py`

Restart the kernel between runs: a `colab exec` reuses the process, so the
previous run's 6 GB of tiles stay alive and fake a `RuntimeBufferAllocationFailure`
about 80 MB short of the real limit.

MIT OR Apache-2.0
updated: 2026-10-18 05:25 UTC

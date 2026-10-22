# dflash2-draft — the drafter for the 50 tok/s path

**Status: ported and running on CPU (2026-10-22). Not yet run on the TPU, and
not yet run with the real weights.** What is here is the drafter's forward
pass, its candidate selector, and its loader, with the published shapes; the
contract test builds the whole thing from random parameters and checks the
shapes, the selector's path, and the rope convention.

## What it is

The drafter for multi-token verification. It is a *block-diffusion* drafter: it
proposes a whole **block of 8 tokens** in a single pass, keeps the top **16**
candidates at every position, and a lightweight selector traces one coherent
path through them. The target then verifies the block in **one** pass — which
is our chunked path, already proven exact on the real model — and the standard
speculative rule accepts the longest matching prefix plus one.

It is small: **five layers**, hidden 5120 (our target's width), 32 query heads
over 8 kv heads, head dim 128, sliding window 2048, and the target's own
vocabulary of 248,320. It reuses the **target's** embedding and the **target's**
LM head, so it carries no embedding of its own.

## Why this drafter

Its acceptance length for *this exact base architecture* is published — the
number that decides the whole project (sourced:
`huggingface.co/z-lab/Qwen3.8-27B-DFlash2`, measured by them against the bf16
Qwen3.8-27B on an H200; all three drafters propose seven draft tokens per
step):

| drafter | GSM8K | MATH-500 | MBPP | HumanEval | MT-Bench |
| --- | --- | --- | --- | --- | --- |
| **DFlash 2** | **5.46** | **5.28** | **4.79** | **4.39** | **4.10** |
| MTP | 5.02 | 4.72 | 3.99 | 3.91 | 3.74 |
| DSpark | 4.36 | 3.92 | 3.51 | 3.30 | 3.01 |

Acceptance length is completed tokens per verification step. A chunked pass
costs about what one sequential step costs — the weights are unpacked once for
the whole chunk — so 8 verified tokens per ~135 ms is 59 tok/s if every token
were accepted, and the published acceptance turns that into **30.6-40.7 tok/s**
on the shipped path and **34.3-45.7 tok/s** with `FOLD=3`'s 119.5 ms pass.

So **50 tok/s needs the acceptance at the top of that range *and* a pass at
109 ms or less**: `FOLD=3` reaches 45.7, and the last ~10 % is the unpack
itself — which is what `TODO.md` `TPU-08`'s 2-bit `PQ2_0` port is for. The
honest range for the parts that exist is **30-46 tok/s**.

## What the mechanism is, precisely

1. The drafter reads the **target's hidden states at layers 5, 19, 33, 47 and
   61** (`dflash_config.target_layer_ids`), concatenated into a 5H feature row
   per committed token, plus the anchor token and seven mask tokens.
2. The five draft layers run with **bidirectional** attention over the context
   and the block (the block is a diffusion draft, not a causal continuation),
   and each layer has a **two-tap dynamic convolution**: a learned base kernel
   plus a kernel predicted from the hidden state itself, which is what stops the
   draft decaying toward the end of the block.
3. The target's **LM head** is applied to the draft's hidden states — the drafter
   has no head of its own — giving per-position logits.
4. The selector scores each candidate by a unary score plus a pairwise score
   from two codebooks (a predecessor codebook indexed by the previous token and
   a successor codebook indexed by the candidate), and walks the block greedily.
5. The target verifies the block in one chunked pass; `acc` is the length of the
   matching prefix and the pass emits `acc + 1` tokens (the prefix plus the
   target's own bonus token).

## What it still needs

- **The kernel must expose the taps.** Our fused 64-layer step does not return
  the hidden states at layers 5/19/33/47/61. It needs a rolling feature buffer
  (the drafter's context is 1536 tokens, so 78 MB at bf16) and a write of the
  five taps per token.
- **The real weights.** 3.85 GB of safetensors from
  `huggingface.co/z-lab/Qwen3.8-27B-DFlash2`, Apache-2.0.
- **A TPU run** to measure the acceptance *on our ternary target*, which may
  differ from the published numbers: the drafter was trained against the bf16
  Qwen3.8-27B, and our target's hidden states are not bit-identical to that
  model's.

## Provenance and licence

Ported from `Inferact/tpu-megakernels`, `qwen/dflash.py` (Apache-2.0,
github.com/Inferact/tpu-megakernels) — the reference implementation behind
"700 TPS on Kimi K3: A Case for TPU Megakernels". The structure (the two-tap
dynamic convolutions, the bidirectional block attention, the codebook-scored
candidate selector) is that work; the rewiring onto this track's ternary
target, its embedding and its LM head is ours. The Apache-2.0 notice travels
with the source, in `kernel.py`'s docstring.

The draft's own weights are Apache-2.0.

MIT OR Apache-2.0
updated: 2026-10-22 15:55 UTC

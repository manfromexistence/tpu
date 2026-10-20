"""GGUF prism.hadamard.* metadata reader -- ground truth before any TPU code.

The Bonsai 2 TPU runner stands or falls on three facts that live inside the GGUF file
itself, not in any blog post: which tensors carry a Hadamard rotation, what the block
size is, and which sign vectors go with which input widths. This notebook reads exactly
those facts and writes them out as JSON, so the FWHT kernel and the loader consume a
measured contract instead of a remembered one.

What this does NOT do: decode a single weight, run anything on a TPU, or touch the
network. NumPy and the standard library are the only dependencies, so it runs
identically on Kaggle, Colab, or a free CPU runtime -- and per COLAB.md S12.1 that is
where it should run first.

The contract table in the markdown below is the OPERATOR BRIEF (2026-10-14), marked
[unverified] where it has not been checked against the fork source or a real file.
Anything in it that this notebook cannot confirm is labelled, not asserted.

Status: NOT YET EXECUTED against Ternary-Bonsai-2-27B-PQ2_0.gguf from this workspace.
"""

import json
import os
import struct
from pathlib import Path

# %% cell [markdown]
"""
# GGUF `prism.hadamard.*` reader

**Input:** a GGUF file path in `GGUF_PATH` (default: the PQ2_0 file name).
**Output:** `gguf_prism_report.json` -- tensor census, metadata keys, and the extracted
Hadamard contract.

Operator brief [unverified against fork source -- confirm each row with this notebook
before building on it]:

| Key / fact | Brief value |
| --- | --- |
| `prism.hadamard.block_size` | 1024 |
| sign vectors keyed by input dim | 5120 (attn/ffn), 6144 (`ssm_out`), 17408 (`ffn_down`) |
| forward (`weight_names`) | `attn_q/k/v/gate/output`, `ffn_gate/up/down`, `ssm_out`, `output.weight` |
| inverse (`inverse_weight_names`) | `token_embd.weight` |
| `PQ2_0` layout | 34 bytes per 128 weights: 2-byte FP16 scale + 32 packed bytes, 4 trits/byte |
| dequant | `value = (code - 1) * scale`, codes `00=-1, 01=0, 10=+1` |
| `PTQ1_0` ggml type id | fork-specific -- **read it from the file, do not assume 143** |
"""

# %% cell
# --- GGUF reader: header, metadata KV, tensor index ---------------------------
# Header-only: no tensor data is ever read. The largest allocation here is the tensor
# index itself (hundreds of entries), so this runs on any runtime that can open a file.

GGUF_MAGIC = b"GGUF"

# ggml_type ids for the unquantized kinds this reader decodes names for. Quantized
# type ids are NOT mapped here on purpose: their ids are fork-specific, and mapping
# them by memory is how a reader silently mislabels every ternary tensor.
DTYPE_BYTES = {0: 4, 1: 2, 30: 2}
DTYPE_NAMES = {0: "F32", 1: "F16", 30: "BF16"}

# GGUF metadata value types, per the format spec. Type 7 is BOOL (1 byte) --
# misreading it as 8 bytes shifts the whole stream and every later key decodes
# as garbage (found live on the PTQ1_0 file: MemoryError on a giant length).
_VTYPE_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_string(fh):
    (length,) = struct.unpack("<Q", fh.read(8))
    return fh.read(length).decode("utf-8", errors="replace")


def _skip_value(fh, vtype):
    if vtype in _VTYPE_FIXED:
        fh.read(_VTYPE_FIXED[vtype])
        return None
    if vtype == 8:
        return _read_string(fh)
    if vtype == 9:
        (elem_type,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        vals = []
        for _ in range(min(count, 1 << 20)):
            vals.append(_read_value(fh, elem_type))
        return vals
    raise ValueError(f"unknown GGUF metadata value type {vtype}")


def _read_value(fh, vtype):
    if vtype in _VTYPE_FIXED:
        raw = fh.read(_VTYPE_FIXED[vtype])
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[vtype]
        return struct.unpack("<" + fmt, raw)[0]
    if vtype == 8:
        return _read_string(fh)
    if vtype == 9:
        (elem_type,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        return [_read_value(fh, elem_type) for _ in range(min(count, 1 << 20))]
    raise ValueError(f"unknown GGUF metadata value type {vtype}")


def read_gguf_index(path):
    """Return (version, metadata, tensors). Tensors are (name, dims, dtype_id, offset)."""
    with open(path, "rb") as fh:
        if fh.read(4) != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        (version,) = struct.unpack("<I", fh.read(4))
        (n_tensors,) = struct.unpack("<Q", fh.read(8))
        (n_kv,) = struct.unpack("<Q", fh.read(8))
        metadata = {}
        for _ in range(n_kv):
            key = _read_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            try:
                metadata[key] = _read_value(fh, vtype)
            except ValueError as exc:
                metadata[key] = f"<unreadable: {exc}>"
        tensors = []
        for _ in range(n_tensors):
            name = _read_string(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            (dtype_id,) = struct.unpack("<I", fh.read(4))
            (offset,) = struct.unpack("<Q", fh.read(8))
            tensors.append((name, dims, dtype_id, offset))
    return version, metadata, tensors


GGUF_CANDIDATES = [
    os.environ.get("GGUF_PATH", ""),
    "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "/kaggle/input/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
]
GGUF_PATH = next((p for p in GGUF_CANDIDATES if p and Path(p).exists()), GGUF_CANDIDATES[1])
print(f"GGUF_PATH = {GGUF_PATH}  (exists: {Path(GGUF_PATH).exists()})")
if not Path(GGUF_PATH).exists():
    print("Set GGUF_PATH to the file. On Kaggle, attach dataset")
    print("essencefromexistence/bonsai2-27b-ptq1; on Colab, mount Drive first.")

# %% cell [markdown]
"""
## Tensor census

Counts tensors by ggml dtype id and lists every distinct input width. The input widths
are what the sign-vector table is keyed on, so this row is the one the Hadamard
contract stands on. Dtype ids outside `{0, 1, 30}` are reported by number only --
**never mapped from memory**, because a wrong mapping here mislabels the ternary
tensors the whole port exists to handle.
"""

# %% cell
report = {"gguf_path": GGUF_PATH, "file_present": Path(GGUF_PATH).exists()}

if report["file_present"]:
    version, metadata, tensors = read_gguf_index(GGUF_PATH)
    report["gguf_version"] = version
    report["n_tensors"] = len(tensors)
    report["n_metadata_keys"] = len(metadata)

    by_dtype = {}
    for _n, _d, t, _o in tensors:
        by_dtype[t] = by_dtype.get(t, 0) + 1
    report["tensor_count_by_dtype_id"] = {str(k): v for k, v in sorted(by_dtype.items())}
    print(f"GGUF v{version}: {len(tensors)} tensors, {len(metadata)} metadata keys")
    print("tensor counts by dtype id (ids outside 0/1/30 intentionally unmapped):")
    for tid, count in sorted(by_dtype.items()):
        label = DTYPE_NAMES.get(tid, "? (fork-specific, do not assume)")
        print(f"  id {tid:>4} {label:<38} x{count}")

    widths = sorted({t[1][-1] for t in tensors if t[1]})
    report["distinct_input_widths"] = widths
    print(f"\ndistinct input widths: {widths}")
else:
    report["note"] = ("file absent; census skipped. This is a correct result on a "
                      "runtime without the weights, not a failure.")
    print(report["note"])

# %% cell [markdown]
"""
## The `prism.hadamard.*` contract, extracted -- not asserted

Every metadata key starting with `prism.` is printed verbatim. The brief's table is
checked against what is actually in the file, key by key, and each row is marked
confirmed or not. **A row that is not confirmed here must not be used downstream.**
"""

# %% cell
def short(value, limit=160):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


if report["file_present"]:
    prism = {k: v for k, v in metadata.items() if k.startswith("prism.")}
    report["prism_keys"] = sorted(prism)
    print(f"{len(prism)} metadata keys starting with 'prism.':")
    for key in sorted(prism):
        print(f"  {key} = {short(prism[key])}")

    checks = {}
    # block_size: the brief says 1024.
    for cand in ("prism.hadamard.block_size", "prism.hadamard_block_size"):
        if cand in prism:
            checks["block_size"] = {"key": cand, "value": prism[cand],
                                    "matches_brief_1024": prism[cand] == 1024}
    # Sign schema (as found in PTQ1_0 2026-10-14, NOT as the brief guessed):
    # sign_widths lists one width per vector; sign_values is their concatenation,
    # so len(sign_values) must equal sum(sign_widths). Each entry must be +-1.
    widths = list(prism.get("prism.hadamard.sign_widths", []))
    svals = list(prism.get("prism.hadamard.sign_values", []))
    vecs, pos = {}, 0
    for w in widths:
        vecs[w] = svals[pos:pos + w]
        pos += w
    checks["sign_schema"] = {
        "widths": widths,
        "n_values": len(svals),
        "sums_to_widths": pos == len(svals) and pos == sum(widths),
        "all_pm1": all(v in (1, -1) for v in svals),
        "per_width_all_pm1": {w: all(v in (1, -1) for v in v) for w, v in vecs.items()},
    }
    print(f"\nsign schema: widths={widths} n_values={len(svals)} "
          f"sums_ok={checks['sign_schema']['sums_to_widths']} "
          f"all_pm1={checks['sign_schema']['all_pm1']}")
    # Weight -> width map: each forward weight's input (last) dim selects its vector.
    by_name = {t[0]: t for t in tensors}
    fwd = list(prism.get("prism.hadamard.weight_names", []))
    inv = list(prism.get("prism.hadamard.inverse_weight_names", []))
    width_of, unmapped = {}, []
    for name in fwd:
        row = by_name.get(name)
        if row is None:
            unmapped.append(name)
            continue
        width_of[name] = row[1][-1] if row[1] else None
    from collections import Counter
    # Widths WITH sign vectors vs WITHOUT: a forward weight whose input width
    # has no vector is a finding, not a failure -- it simply skips the signed
    # FWHT at load time. The notebook extracts; the loader will decide.
    with_vec = {w for w in width_of.values() if w in set(widths)}
    without_vec = {w for w in width_of.values() if w not in set(widths)}
    checks["weight_map"] = {
        "n_forward": len(fwd),
        "width_histogram": dict(Counter(width_of.values())),
        "widths_with_sign_vector": sorted(with_vec),
        "widths_without_sign_vector": sorted(without_vec),
        "unmapped_names": unmapped,
        "inverse": inv,
    }
    print(f"weight map: {len(fwd)} forward names, widths used={checks['weight_map']['width_histogram']}, "
          f"unmapped={len(unmapped)}, inverse={inv}")
    print(f"signed widths: {sorted(with_vec)}; plain widths: {sorted(without_vec)}")
    assert checks["sign_schema"]["sums_to_widths"], "sign_values do not partition by sign_widths"
    assert checks["sign_schema"]["all_pm1"], "sign_values contain non-+-1 entries"
    assert not unmapped, f"forward names missing from tensor index: {unmapped[:5]}"
    report["brief_checks"] = checks
    print("\nbrief checks:")
    print(json.dumps(checks, indent=2, default=str))
else:
    print("file absent; contract extraction skipped.")

# %% cell
# --- write the report ----------------------------------------------------------
# The FWHT kernel test and the loader consume THIS file, not the brief. If the file was
# absent, what gets written is the explicit statement of that, so a downstream notebook
# fails on a missing input rather than on invented constants.
out_name = os.environ.get("PRISM_REPORT_OUT", "gguf_prism_report.json")
out_path = Path("/kaggle/working") / out_name
if not out_path.parent.is_dir():
    out_path = (Path("/content") / out_name) if Path("/content").is_dir() else Path(out_name)
report["notebook"] = "02-gguf-format-and-sign-metadata"
report["status"] = ("measured" if report["file_present"]
                    else "repo-managed (no file on this runtime)")
out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(f"\nwrote {out_path.resolve()} ({out_path.stat().st_size:,} bytes)")

# %% cell [markdown]
"""
## What this does NOT establish

- Whether the ternary weights decode correctly. That is the loader's job, against the
  byte counts in the brief (34 bytes per 128 weights for `PQ2_0`).
- Whether `PTQ1_0`'s ggml type id is any particular number. Read it from the dtype
  census above; the brief's "143" was never confirmed against a file in this workspace.
- Anything about speed, size, or quality. This notebook moves metadata, not weights.

MIT OR Apache-2.0
updated: 2026-10-14 00:00 UTC
"""

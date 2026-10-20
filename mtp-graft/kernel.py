"""Build a kernel that grafts the base model's MTP head into Bonsai 2 and runs it.

Two routes are tried, because the fork may support either:

  A. **graft** -- copy the head's 15 `blk.64.*` tensors into the PTQ1_0 file
     and add `qwen35.nextn_predict_layers`, so the target file itself carries
     the nextn block. The head is published as its own GGUF
     (`unsloth/Qwen3.8-27B-GGUF`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, 18 tensors).
  B. **separate file** -- hand the fork the MTP file directly as the draft
     model, if it accepts that.

Every stage runs at -c 8192 on one T4, with the no-speculation baseline
measured in the same session, so the speedup is same-file, same-card.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN = WORK / "run"
RUN.mkdir(parents=True, exist_ok=True)
LOG = RUN / "graft.log"
log = open(LOG, "a", buffering=1)

MTP_URL = ("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/"
           "MTP/mtp-Qwen3.8-27B-Q4_0.gguf")
MTP_SIZE = 1369590656
CTX = 8192
NTOK = 128
PROMPT = "Hello, Who are you?"


def say(m):
    print(m)
    log.write(str(m).rstrip() + "\n")


def sh(cmd, timeout=None, cwd=None, check=True):
    say(f"\n$ {cmd}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, cwd=cwd, check=False)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    log.write(out[-20000:] + "\n")
    say(out[-1200:])
    say(f"[exit {p.returncode}]")
    if check and p.returncode:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p


# ------------------------------------------------------------------ the graft tool
"""Graft a multi-token-prediction head into a GGUF that has none.

Bonsai 2 27B's PTQ1_0 file ships 851 tensors and no MTP head (measured from
the header). The base model's MTP head is published as its own GGUF
(`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, 18 tensors, `qwen35.nextn_predict_layers: 1`).
This copies the head's block tensors into the target file, so llama.cpp's
`--spec-type draft-mtp` finds the nextn block inside the model.

GGUF v3 layout: magic, version, tensor_count, kv_count, then the KV pairs,
then the tensor infos (name, n_dims, dims, type, offset), then the tensor
data. `offset` is relative to the start of the data section and every tensor
starts on a `general.alignment` boundary.

Growing the header moves the data section, but the offsets are RELATIVE to
that section, so the existing tensors keep their offsets and the new tensors
are appended after the old data. The header's *size* does not depend on the
offset values (each is a fixed u64), so the size can be computed first and the
offsets filled in afterwards.
"""

import struct
import sys

SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
           4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
ELEM = {0: 4, 1: 2, 28: 8, 30: 2}
# GGML quantised types: type id -> (elements per block, bytes per block)
QUANT = {
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
    21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (32, 34),
    25: (32, 64), 26: (32, 128), 27: (32, 256), 29: (256, 56),
}
TYPES = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
         "f32": 6, "bool": 7, "string": 8, "array": 9, "u64": 10,
         "i64": 11, "f64": 12}


def enc_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_value(v):
    if isinstance(v, str):
        return struct.pack("<I", TYPES["string"]) + enc_string(v)
    if isinstance(v, bool):
        return struct.pack("<I", TYPES["bool"]) + struct.pack("<B", 1 if v else 0)
    if isinstance(v, int):
        for name, (lo, hi) in {"u8": (0, 255), "i8": (-128, 127),
                                "u16": (0, 65535), "i16": (-32768, 32767),
                                "u32": (0, 2**32 - 1), "i32": (-2**31, 2**31 - 1),
                                "u64": (0, 2**64 - 1), "i64": (-2**63, 2**63 - 1)}.items():
            if lo <= v <= hi:
                return struct.pack("<I", TYPES[name]) + struct.pack(SCALARS[name][0], v)
        raise ValueError(f"int out of range: {v}")
    if isinstance(v, float):
        return struct.pack("<I", TYPES["f32"]) + struct.pack("<f", v)
    raise TypeError(f"unsupported KV type: {type(v)}")


def align_up(n, a):
    return (n + a - 1) // a * a


def tensor_info_bytes(name, dims, typ, offset):
    return (enc_string(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", typ) + struct.pack("<Q", offset))


class GGUF:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(96 * 1024 * 1024)
        if head[:4] != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        self.version = struct.unpack_from("<I", head, 4)[0]
        self.n_tensors = struct.unpack_from("<Q", head, 8)[0]
        self.n_kv = struct.unpack_from("<Q", head, 16)[0]
        off = 24
        self.kv = {}
        for _ in range(self.n_kv):
            key, off = self._read_string(head, off)
            (t,) = struct.unpack_from("<I", head, off)
            off += 4
            val, off = self._read_value(head, off, t)
            self.kv[key] = val
        self.kv_start = 24
        self.kv_raw = head[24:off]
        self.tensors = []
        for _ in range(self.n_tensors):
            name, off = self._read_string(head, off)
            (ndims,) = struct.unpack_from("<I", head, off)
            off += 4
            dims = struct.unpack_from("<%dQ" % ndims, head, off)
            off += 8 * ndims
            (typ,) = struct.unpack_from("<I", head, off)
            off += 4
            (toff,) = struct.unpack_from("<Q", head, off)
            off += 8
            self.tensors.append({"name": name, "dims": dims, "type": typ, "offset": toff})
        self.alignment = int(self.kv.get("general.alignment", 32))
        # the tensor data section starts on an alignment boundary
        self.data_start = align_up(off, self.alignment)
        with open(path, "rb") as f:
            f.seek(0, 2)
            self.file_size = f.tell()

    @staticmethod
    def _read_string(buf, off):
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        return buf[off:off + n].decode("utf-8", "replace"), off + n

    @staticmethod
    def _read_value(buf, off, t):
        if t == 8:
            return GGUF._read_string(buf, off)
        if t == 9:
            (et,) = struct.unpack_from("<I", buf, off)
            off += 4
            (n,) = struct.unpack_from("<Q", buf, off)
            off += 8
            vals = []
            for _ in range(n):
                v, off = GGUF._read_value(buf, off, et)
                vals.append(v)
            return vals, off
        fmt, size = SCALARS[t]
        (v,) = struct.unpack_from(fmt, buf, off)
        return v, off + size

    def nbytes(self, t):
        n = 1
        for d in t["dims"]:
            n *= d
        if t["type"] in ELEM:
            return n * ELEM[t["type"]]
        if t["type"] in QUANT:
            elems, nbytes = QUANT[t["type"]]
            if n % elems:
                raise ValueError(f"{t['name']}: {n} elements is not a multiple "
                                 f"of the {elems}-element block")
            return n // elems * nbytes
        raise ValueError(f"unknown GGML type {t['type']} for tensor {t['name']}")

    def copy_bytes(self, t, out):
        start = self.data_start + t["offset"]
        nbytes = self.nbytes(t)
        with open(self.path, "rb") as f:
            f.seek(start)
            left = nbytes
            while left > 0:
                chunk = f.read(min(1 << 22, left))
                if not chunk:
                    break
                out.write(chunk)
                left -= len(chunk)


def graft(target_path, head_path, out_path, copy_names, extra_kv,
          head_kv_names=None, dry_run=False):
    tgt = GGUF(target_path)
    head = GGUF(head_path)
    print(f"target: {tgt.n_tensors} tensors, data at {tgt.data_start}, "
          f"file {tgt.file_size}, align {tgt.alignment}")
    print(f"head  : {head.n_tensors} tensors, data at {head.data_start}, "
          f"file {head.file_size}")

    by_name = {t["name"]: t for t in head.tensors}
    chosen = []
    for n in copy_names:
        if n not in by_name:
            raise KeyError(f"{n} not in the head file")
        chosen.append(by_name[n])
    print(f"copying {len(chosen)} tensors: {[t['name'] for t in chosen]}")

    new_kv_bytes = b""
    added_kv = 0
    for k, v in extra_kv.items():
        new_kv_bytes += enc_string(k) + enc_value(v)
        added_kv += 1
    for k in (head_kv_names or []):
        if k in head.kv and k not in tgt.kv:
            new_kv_bytes += enc_string(k) + enc_value(head.kv[k])
            added_kv += 1
            print(f"  + KV {k} = {head.kv[k]}")

    n_new = len(chosen)
    # header size first (offsets are fixed-width, so the size is known now)
    size = 24 + len(tgt.kv_raw) + len(new_kv_bytes)
    for t in tgt.tensors:
        size += len(tensor_info_bytes(t["name"], t["dims"], t["type"], 0))
    for t in chosen:
        size += len(tensor_info_bytes(t["name"], t["dims"], t["type"], 0))
    new_data_start = align_up(size, tgt.alignment)
    shift = new_data_start - tgt.data_start
    old_data_len = tgt.file_size - tgt.data_start
    print(f"header {size} -> data start {new_data_start} (was {tgt.data_start}, "
          f"shift {shift}, old data {old_data_len})")

    # the new tensors go after the old data, each aligned
    cursor = align_up(old_data_len, tgt.alignment)
    new_offsets = []
    for t in chosen:
        new_offsets.append(cursor)
        cursor = align_up(cursor + head.nbytes(t), tgt.alignment)
    new_data_len = cursor

    infos = b"".join(tensor_info_bytes(t["name"], t["dims"], t["type"],
                                       t["offset"]) for t in tgt.tensors)
    new_infos = b"".join(tensor_info_bytes(t["name"], t["dims"], t["type"], o)
                         for t, o in zip(chosen, new_offsets))
    pre = (b"GGUF" + struct.pack("<I", tgt.version)
           + struct.pack("<Q", tgt.n_tensors + n_new)
           + struct.pack("<Q", tgt.n_kv + added_kv)
           + tgt.kv_raw + new_kv_bytes + infos + new_infos)
    pad = (-len(pre)) % tgt.alignment
    assert len(pre) + pad == new_data_start, (len(pre) + pad, new_data_start)

    if dry_run:
        print("dry run: nothing written")
        return {"new_tensors": [t["name"] for t in chosen], "shift": shift,
                "new_size": new_data_start + new_data_len}

    with open(out_path, "wb") as out:
        out.write(pre)
        out.write(b"\x00" * pad)
        with open(target_path, "rb") as f:
            f.seek(tgt.data_start)
            left = old_data_len
            while left > 0:
                chunk = f.read(min(1 << 22, left))
                if not chunk:
                    break
                out.write(chunk)
                left -= len(chunk)
        for t, o in zip(chosen, new_offsets):
            want = o + new_data_start
            have = out.tell()
            if want > have:
                out.write(b"\x00" * (want - have))
            head.copy_bytes(t, out)
    print(f"wrote {out_path}")
    return {"new_tensors": [t["name"] for t in chosen], "shift": shift,
            "new_size": new_data_start + new_data_len}


# ---------------------------------------------------------------- build cache
def find_cache():
    for depth in (1, 2, 3, 4):
        for suffix in ("bin/llama-cli", "llama-cli"):
            hits = list(Path("/kaggle/input").glob("*/" * depth + suffix))
            hits = [h for h in hits if "bonsai2-llamacpp-build" in str(h)]
            if hits:
                return sorted(hits, key=lambda h: len(str(h)))[0].parent
    return None


cached = find_cache()
CLI = Path("llama.cpp/build/bin/llama-cli").resolve()
if cached and not (CLI.exists() and os.access(CLI, os.X_OK)):
    say("restoring cached build")
    target = Path("llama.cpp/build/bin")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(cached.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
    for versioned in sorted(target.glob("*.so.*.*.*")):
        base, _, rest = str(versioned.name).partition(".so.")
        for alias in (f"{base}.so.{rest.split('.')[0]}", f"{base}.so"):
            link = target / alias
            if not link.exists():
                link.symlink_to(versioned.name)
    for exe in ("llama-cli", "llama-server", "llama-bench"):
        p = target / exe
        if p.exists():
            os.chmod(p, 0o755)
    say(f"cache restored; llama-cli usable: {CLI.exists() and os.access(CLI, os.X_OK)}")
if not (CLI.exists() and os.access(CLI, os.X_OK)):
    raise RuntimeError("no usable llama-cli")

# ------------------------------------------------------------------- the model
MODEL = None
for base in sorted(Path("/kaggle/input").glob("*")):
    for p in sorted(base.rglob("*PTQ1_0*.gguf")):
        MODEL = p
        break
    if MODEL:
        break
if MODEL is None:
    raise RuntimeError("Bonsai PTQ1_0 not found")
LOCAL_MODEL = RUN / MODEL.name
if not (LOCAL_MODEL.exists() and LOCAL_MODEL.stat().st_size == MODEL.stat().st_size):
    say("copying the target model into the working dir (it will be rewritten)")
    shutil.copyfile(MODEL, LOCAL_MODEL)
say(f"target: {LOCAL_MODEL} ({LOCAL_MODEL.stat().st_size / 2**30:.2f} GiB)")

# ------------------------------------------------------------- the MTP head
MTP = RUN / "mtp-Qwen3.8-27B-Q4_0.gguf"
if not (MTP.exists() and MTP.stat().st_size == MTP_SIZE):
    say("=== stage: download the MTP head ===")
    for attempt in range(1, 6):
        if MTP.exists():
            MTP.unlink()
        sh(f"curl -L --retry 3 -C - -o {MTP} '{MTP_URL}'", timeout=2400, check=False)
        if MTP.exists() and MTP.stat().st_size == MTP_SIZE:
            say(f"mtp verified on attempt {attempt}")
            break
        say(f"attempt {attempt}: got {MTP.stat().st_size if MTP.exists() else 0}")
        time.sleep(10)
    if not (MTP.exists() and MTP.stat().st_size == MTP_SIZE):
        raise RuntimeError("MTP download failed after 5 attempts")
say(f"mtp head: {MTP} ({MTP.stat().st_size / 2**20:.0f} MB)")

# ------------------------------------------------------------------ the graft
GRAFTED = RUN / "bonsai2-ptq1_0-mtp.gguf"
if not GRAFTED.exists():
    say("=== stage: graft the MTP head into the target ===")
    g = graft(
        str(LOCAL_MODEL), str(MTP), str(GRAFTED),
        copy_names=[
            "blk.64.attn_k.weight", "blk.64.attn_k_norm.weight",
            "blk.64.attn_norm.weight", "blk.64.attn_output.weight",
            "blk.64.attn_q.weight", "blk.64.attn_q_norm.weight",
            "blk.64.attn_v.weight", "blk.64.ffn_down.weight",
            "blk.64.ffn_gate.weight", "blk.64.ffn_up.weight",
            "blk.64.nextn.eh_proj.weight", "blk.64.nextn.enorm.weight",
            "blk.64.nextn.hnorm.weight", "blk.64.nextn.shared_head_norm.weight",
            "blk.64.post_attention_norm.weight",
        ],
        extra_kv={"qwen35.nextn_predict_layers": 1},
        head_kv_names=["qwen35.nextn_predict_layers"])
    say(f"grafted: {g}")
say(f"grafted file: {GRAFTED} ({GRAFTED.stat().st_size / 2**30:.2f} GiB)")

# ------------------------------------------------------------------- the runs
BASE = [str(CLI), "-ngl", "99", "-c", str(CTX), "-n", str(NTOK), "--temp", "0",
        "-p", PROMPT, "--single-turn", "--reasoning-budget", "0",
        "--reasoning-format", "none"]


def parse(out):
    accept = re.search(r"draft acceptance rate = ([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated", out)
    tok = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    pp = re.search(r"Prompt:\s*([\d.]+)\s*t/s", out)
    refuse = re.search(r"does not have|vocab.*must match|no mtp|requires", out, re.I)
    return {
        "acceptance": float(accept.group(1)) if accept else None,
        "accepted": int(accept.group(2)) if accept else None,
        "generated": int(accept.group(3)) if accept else None,
        "gen_tok_s": float(tok.group(1)) if tok else None,
        "prompt_tok_s": float(pp.group(1)) if pp else None,
        "refused": bool(refuse),
        "refuse_message": refuse.group(0) if refuse else None,
    }


def run_stage(label, model, extra):
    say(f"=== stage: {label} ===")
    cmd = [str(CLI), "-m", str(model)] + BASE + extra
    say(" ".join(cmd))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400,
                       check=False, env=env)
    elapsed = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    log.write(out[-20000:] + "\n")
    say(out[-2500:])
    r = parse(out)
    r["wall_seconds"] = round(elapsed, 1)
    r["exit_code"] = p.returncode
    r["cmd"] = " ".join(cmd)
    return r


results = {}
results["baseline_original"] = run_stage("baseline (no speculation)", LOCAL_MODEL, [])
results["mtp_grafted"] = run_stage("MTP head grafted into the target", GRAFTED,
                                   ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8"])
results["mtp_separate_file"] = run_stage(
    "MTP head as a separate draft file", LOCAL_MODEL,
    ["--spec-type", "draft-mtp", "--spec-draft-model", str(MTP),
     "--spec-draft-n-max", "8"])

base_tps = results["baseline_original"].get("gen_tok_s")
for k, v in results.items():
    if v.get("gen_tok_s") and base_tps:
        v["speedup_vs_measured_baseline"] = round(v["gen_tok_s"] / base_tps, 2)

receipt = {
    "platform": "kaggle",
    "accelerator": "T4",
    "target": LOCAL_MODEL.name,
    "target_bytes": LOCAL_MODEL.stat().st_size,
    "grafted_file": GRAFTED.name,
    "grafted_bytes": GRAFTED.stat().st_size,
    "mtp_head_source": MTP_URL,
    "mtp_head_bytes": MTP_SIZE,
    "context_tokens": CTX,
    "generated_requested": NTOK,
    "prompt": PROMPT,
    "stages": results,
    "best_tok_s": max([v["gen_tok_s"] for v in results.values()
                       if v.get("gen_tok_s")] or [None]),
    "target_has_mtp_head": False,
    "target_mtp_evidence": "GGUF header: 851 tensors, 0 named nextn/mtp, no qwen35.nextn_block_count (measured 2026-10-20)",
    "reference_not_measured_here": "pich, RTX PRO 4000 24GB, Qwen3.8-27B GGUF: 50.44 tok/s @ 48.3% acceptance",
    "tier": "exec",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(WORK / "graft-receipt.json").write_text(json.dumps(receipt, indent=2))
say("=== verdict ===")
say(json.dumps(receipt, indent=2))

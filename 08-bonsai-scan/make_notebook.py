"""Generate notebook.ipynb from kernel.py (small notebook generator, hub convention)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "# %% cell"
MD_MARKER = "# %% cell [markdown]"
Q = chr(34) * 3


def split_cells(text):
    lines = text.splitlines()
    cells, i = [], 0
    lead = []
    while i < len(lines) and not lines[i].startswith(MARKER):
        lead.append(lines[i])
        i += 1

    body = list(lead)
    if body and body[0].lstrip().startswith(Q):
        close = next((j for j in range(1, len(body))
                      if body[j].rstrip().endswith(Q)), None)
        if close is not None:
            tail = body[close + 1:]
            doc = [ln for ln in body[1:close] if ln.strip()]
            body = ["# " + ln for ln in doc] + tail
    if any(ln.strip() for ln in body):
        cells.append(("code", "\n".join(body).strip("\n")))

    while i < len(lines):
        if not lines[i].startswith(MARKER):
            i += 1
            continue
        kind = "markdown" if lines[i].strip() == MD_MARKER else "code"
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith(MARKER):
            buf.append(lines[i])
            i += 1
        src = "\n".join(buf).strip("\n")
        if src:
            cells.append((kind, src))
    return cells


def main():
    cells = split_cells((HERE / "kernel.py").read_text(encoding="utf-8"))
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)

    nb = {"cells": out,
          "metadata": {
              "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
              "language_info": {"name": "python", "version": "3.12.0"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(out):
        cell["id"] = f"fwht-{i:03d}"

    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    target = HERE / "notebook.ipynb"
    if "--check" in sys.argv and target.exists():
        if target.read_text(encoding="utf-8") == text:
            print(f"up to date: {len(out)} cells")
            return 0
        print(f"OUT OF DATE: {len(out)} cells")
        return 1
    target.write_text(text, encoding="utf-8")
    md = sum(1 for c in out if c["cell_type"] == "markdown")
    print(f"wrote notebook.ipynb: {len(out)} cells ({len(out) - md} code, {md} markdown), "
          f"{len(text):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

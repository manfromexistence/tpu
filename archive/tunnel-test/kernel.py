"""Tunnel test only: does a public URL reach back into this Kaggle session?

Purpose is narrow on purpose. This is NOT the Bonsai kernel and has nothing to do with
it: no model, no GGUF, no CUDA build, no GPU session. The whole question it answers is
whether cloudflared can hand this notebook a public hostname that your own machine can
reach, because that is the one assumption every "serve from Kaggle" plan rests on.

Costs: CPU-only, so Kaggle's unlimited CPU quota. Wall clock about 60 seconds.

If this notebook prints a hostname and the verification cell gets a 200 back through
it, the tunnel works and the rest of the idea is viable. If it fails here, no amount of
model work will help and the plan is dead on arrival.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# %% cell [markdown]
"""
# Kaggle tunnel test

Tests one thing: **can this session be reached from the public internet?**

| Step | What it proves |
| --- | --- |
| local HTTP server | something is listening to tunnel |
| cloudflared quick tunnel | Kaggle allows the outbound connection |
| hostname appears | the edge allocated a route |
| **public 200 response** | the route actually forwards traffic back in |

The last row is the only one that matters. A hostname printing without a working
response means the tunnel half-connected, which is the common failure.
"""

# %% cell
# --- 1. a trivial HTTP server -------------------------------------------------
# Deliberately not a model server. The point is the network path, and a 20-line
# http.server removes every variable that a real inference server would introduce.
STATE = {"requests": 0, "started": time.time()}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        STATE["requests"] += 1
        if self.path in ("/", "/health"):
            self._reply(200, {"status": "ok",
                              "uptime_s": round(time.time() - STATE["started"], 1),
                              "requests": STATE["requests"],
                              "note": "hello from a Kaggle notebook"})
        else:
            self._reply(404, {"error": "not found"})

    def log_message(self, *args):  # keep the notebook log clean
        pass


httpd = HTTPServer(("127.0.0.1", 8080), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()

probe = subprocess.run("curl -sf http://127.0.0.1:8080/health", shell=True,
                       capture_output=True, text=True)
print(f"loopback health: {probe.stdout.strip()}")
assert probe.returncode == 0, "local server did not come up"
print("local server is listening on 127.0.0.1:8080")

# %% cell [markdown]
"""
## Download cloudflared

A *quick* tunnel (`--url`) needs no Cloudflare account, no token, and no DNS setup.
That is what makes it the right tool for a test: nothing to configure, and nothing in
this notebook that could be misconfigured. The trade-off is a random hostname that
changes on every run, which is fine here and useless for a real deployment.
"""

# %% cell
BIN = Path("/kaggle/working/bin")
BIN.mkdir(parents=True, exist_ok=True)
CF = BIN / "cloudflared"

if not CF.exists():
    print("downloading cloudflared...")
    subprocess.run(
        f"curl -L --fail -o {CF} https://github.com/cloudflare/cloudflared/"
        f"releases/latest/download/cloudflared-linux-amd64",
        shell=True, check=True, timeout=600)
    os.chmod(CF, 0o755)
print(f"cloudflared: {CF} ({CF.stat().st_size:,} bytes)")

# %% cell
# --- 2. start the tunnel ------------------------------------------------------
# --no-autoupdate matters: an auto-update replacing the binary mid-session would drop
# the tunnel with no warning. --metrics is only so the port is explicit and reserved.
CF_LOG = Path("/kaggle/working/cloudflared.log")
cf = subprocess.Popen(
    [str(CF), "tunnel", "--url", "http://127.0.0.1:8080",
     "--no-autoupdate", "--metrics", "127.0.0.1:45678"],
    stdout=open(CF_LOG, "w"), stderr=subprocess.STDOUT)

print(f"cloudflared pid {cf.pid}; waiting for a hostname...")

# Poll for the hostname instead of sleeping a fixed time. cloudflared prints it to
# stderr the moment the edge connection settles, typically 3-8 seconds.
TUNNEL_URL = None
for _ in range(45):
    try:
        text = CF_LOG.read_text()
    except OSError:
        text = ""
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
    if m:
        TUNNEL_URL = m.group(0)
        break
    if cf.poll() is not None:
        print("cloudflared exited early. Log tail:")
        print(text[-1200:])
        break
    time.sleep(2)

if not TUNNEL_URL:
    print("\n!!! NO HOSTNAME. Tunnel failed. See cloudflared.log above.")
    print("This is the answer: the approach does not work on this account.")
else:
    print(f"\n{'=' * 68}\n  PUBLIC URL:  {TUNNEL_URL}\n{'=' * 68}\n")

# %% cell [markdown]
"""
## Verify through the **public** URL

Hitting loopback proves nothing about the tunnel — it never leaves the machine. So the
check below curls the public hostname from inside the same session. That exercises the
full path: edge → tunnel → local server.

If this passes, the tunnel genuinely works and you can reach it from your own PC.
"""

# %% cell
RESULT = {"url": TUNNEL_URL, "verified": False}

if TUNNEL_URL:
    # Retry a few times: the first request through a just-created quick tunnel is
    # sometimes refused while the route propagates, and that is not a failure.
    for attempt in range(1, 8):
        proc = subprocess.run(
            f"curl -s -o /tmp/body.txt -w '%{{http_code}}' --max-time 30 "
            f"{TUNNEL_URL}/health", shell=True, capture_output=True, text=True)
        code = (proc.stdout or "").strip()
        body = ""
        try:
            body = Path("/tmp/body.txt").read_text()
        except OSError:
            pass
        print(f"attempt {attempt}: HTTP {code}  {body}")
        if code == "200" and "hello from a Kaggle notebook" in body:
            RESULT["verified"] = True
            RESULT["attempt"] = attempt
            RESULT["body"] = json.loads(body)
            break
        time.sleep(4)

print()
if RESULT["verified"]:
    print("=" * 68)
    print("  TUNNEL WORKS.")
    print(f"  {TUNNEL_URL}")
    print("  Reachable from your PC with:")
    print(f"    curl {TUNNEL_URL}/health")
    print("=" * 68)
else:
    print("=" * 68)
    print("  TUNNEL DID NOT VERIFY.")
    print("  A hostname may have printed, but no response came back through it.")
    print("  So the approach is not viable on this account/plan.")
    print("=" * 68)

Path("/kaggle/working/tunnel_result.json").write_text(
    json.dumps(RESULT, indent=2), encoding="utf-8")
print(f"\nresult written to /kaggle/working/tunnel_result.json")
print(json.dumps(RESULT, indent=2))

# %% cell [markdown]
"""
## Hold the session open

Kaggle ends the session when the notebook finishes, which would tear down both the
server and the tunnel. This loop keeps them alive so you have time to open a terminal
on your own machine and actually curl the URL.

Capped at **55 minutes**, not 11 hours: this is a test, and holding a session for half
a day to answer a yes/no question would waste Kaggle resources for no benefit. If the
tunnel works you can re-run with a longer cap when you actually want to serve something.
"""

# %% cell
print(f"\nHolding the tunnel open for 55 minutes so you can test from your PC.")
print(f"URL: {TUNNEL_URL}")
print(f"Press stop in the Kaggle UI to end early.\n")

deadline = time.time() + 55 * 60
beat = 0
while time.time() < deadline and cf.poll() is None:
    beat += 1
    if beat % 30 == 0:
        print(f"[t+{int((time.time() - STATE['started']) / 60)}m] still up, "
              f"requests served: {STATE['requests']}  url={TUNNEL_URL}")
    time.sleep(10)

print(f"\nSession hold finished. cloudflared alive: {cf.poll() is None}")
print(f"Total requests through the tunnel: {STATE['requests']}")
print(f"URL was: {TUNNEL_URL}")

# %% cell [markdown]
"""
## What this notebook establishes

**It establishes:** whether a Kaggle CPU session can be reached from the public internet
via a cloudflared quick tunnel, tested by an actual HTTP 200 fetched through the public
hostname.

**It does not establish:**
- Anything about GPU sessions, which have a separate 12 h cap and their own queue.
- That the practice is sanctioned. It is not — Kaggle's product-feedback thread names
  tunnelling GPU notebooks out to the internet as the pattern they are working to block.
  This test answers "does it work", not "is it allowed".
- That the hostname is stable. Quick tunnels get a new random name every run.

MIT OR Apache-2.0
updated: 2026-10-12 00:00 UTC
"""

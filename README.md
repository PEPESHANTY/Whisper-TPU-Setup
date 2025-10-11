


# Whisper-TPU-Setup

# Whisper-JAX (TPU) — Reproducible Setup Guide

This is a clean, copy‑pasteable guide for running **whisper-jax** on a TPU VM.  
Everything is stored under:

```
/mnt/node5_tpu_data_code_1/whisper
```

It includes the exact versions, commands, and a tiny compatibility shim we used.

---

## 0) Prereqs

- Python **3.10**
- `ffmpeg` available on the machine (for the quick test)
- You’re on a **TPU VM** (v4-* etc.)

Optional (avoids accidental cross‑env imports):

```bash
export PYTHONNOUSERSITE=1
```

---

## 1) Create an isolated venv

```bash
export WHISPER_HOME="/mnt/node5_tpu_data_code_1/whisper"
mkdir -p "$WHISPER_HOME"/{hf,xdg_cache,jax_cache,tmp,models}

python3 -m venv "$WHISPER_HOME/whisper-venv"
source "$WHISPER_HOME/whisper-venv/bin/activate"

python -m pip install --upgrade pip wheel setuptools
```

---

## 2) Install **JAX for TPU** (pinned)

```bash
pip install --no-cache-dir --force-reinstall \
  "jax[tpu]==0.4.31" "jaxlib==0.4.31" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Verify:

```bash
python - <<'PY'
import jax, jaxlib, numpy as np
print("JAX   :", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("NumPy :", np.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY
```
Expect `Backend: tpu` with 4/8 devices listed.

---

## 3) Install a compatible **Flax** stack (no deps → JAX stays pinned)

```bash
pip install --no-cache-dir --no-deps \
  "flax==0.10.7" "optax==0.2.6" "chex==0.1.90"
```

---

## 4) Transformers + Audio stack

```bash
pip install "transformers==4.41.1" "tokenizers==0.19.1" \
            "huggingface-hub==0.23.0" "safetensors==0.4.2" \
            "filelock>=3.12" "fsspec>=2024.3.1" "packaging>=23.2" "tqdm>=4.66" "regex>=2024.5.15"

pip install "soundfile>=0.12.1" "librosa==0.10.2.post1" "soxr>=0.3.2"
```

---

## 5) Install **whisper-jax** (known‑good commit) **without deps**

```bash
pip install --no-cache-dir --no-deps \
  "git+https://github.com/sanchit-gandhi/whisper-jax.git@f983178"
pip install "cached-property==1.5.2"
```

> Explanation: that commit still references `jax.core.NamedShape`; we’ll add a tiny shim next.

---

## 6) Compatibility shim for `jax.core.NamedShape`

Create a `.pth` startup hook so older `whisper-jax` code keeps working:

```bash
SITE="$WHISPER_HOME/whisper-venv/lib/python3.10/site-packages"
cat > "$SITE/00_whisper_shim.pth" <<'PY'
import jax; setattr(jax.core, "NamedShape", getattr(jax.core, "NamedShape", type("NamedShape", (), {})))
PY
```

---

## 7) Runtime environment (all caches under `/mnt/node5…`)

```bash
export HF_HOME="$WHISPER_HOME/hf"; unset TRANSFORMERS_CACHE
export XDG_CACHE_HOME="$WHISPER_HOME/xdg_cache"
export TMPDIR="$WHISPER_HOME/tmp"
export JAX_CACHE_DIR="$WHISPER_HOME/jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export BATCH_SIZE=8
# Optional override:
# export WHISPER_MODEL_ID="openai/whisper-large-v2"
```

---

## 8) Quick smoke test (first call compiles on TPU)

Create a 1s test tone:

```bash
ffmpeg -f lavfi -i sine=frequency=1000:duration=1 -c:a pcm_s16le -ar 16000 -ac 1 "$WHISPER_HOME/test.wav" -y
```

Transcribe:

```bash
python - <<'PY'
import os, jax.numpy as jnp
try:
    from whisper_jax import FlaxWhisperPipline as Pipeline
except ImportError:
    from whisper_jax import FlaxWhisperPipeline as Pipeline

wav = os.path.join(os.environ["WHISPER_HOME"], "test.wav")
pipe = Pipeline(os.getenv("WHISPER_MODEL_ID","openai/whisper-large-v2"),
                dtype=jnp.bfloat16, batch_size=int(os.getenv("BATCH_SIZE","8")))
out  = pipe(wav, task="transcribe")
print("Transcript:", out if isinstance(out, str) else out.get("text"))
PY
```

First run may take ~minutes (XLA JIT). Subsequent runs are fast. The test tone usually gives something like `Transcript: BEEP ...`

---

## 9) Optional: start a simple HTTP service

```bash
pip install fastapi "uvicorn[standard]" python-multipart
```

Create server:

```bash
cat > "$WHISPER_HOME/serve_whisper.py" <<'PY'
import os, time, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import jax, jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline as Pipeline

MODEL_ID   = os.getenv("WHISPER_MODEL_ID", "openai/whisper-large-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
TMP_DIR    = os.getenv("TMPDIR", "/tmp")
pipe = Pipeline(MODEL_ID, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)

app = FastAPI(title="Whisper-JAX TPU Service")

@app.get("/health")
def health():
    return {"status":"ok","jax":jax.__version__,"backend":jax.default_backend(),
            "devices":len(jax.devices()),"model_id":MODEL_ID,"batch_size":BATCH_SIZE}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...),
                     task: str = Form("transcribe"),
                     return_timestamps: bool = Form(False)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_DIR) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    t0 = time.time()
    try:
        out = pipe(tmp_path, task=task, return_timestamps=return_timestamps)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
    elapsed = time.time() - t0
    text = out if isinstance(out, str) else out.get("text", "")
    chunks = None if isinstance(out, str) else out.get("chunks")
    return JSONResponse({"text": text, "chunks": chunks, "seconds": round(elapsed, 3)})
PY
```

Run it:

```bash
cd "$WHISPER_HOME"
uvicorn serve_whisper:app --host 0.0.0.0 --port 7860 --workers 1
```

Then tunnel port **7860** to your laptop and open `http://localhost:7860/docs`.

---

## Troubleshooting (what we hit & how we fixed)

| Error | Cause | Fix |
|---|---|---|
| `Config has no attribute 'define_bool_state'` | Old Flax expected a JAX flag API removed in 0.4.x | Use **flax==0.10.7** |
| `cannot import name 'maps' / 'linear_util'` | Old Flax importing modules removed in newer JAX | Same: upgrade Flax |
| `jax.core has no attribute 'NamedShape'` | Old whisper‑jax references removed JAX type | Add the `.pth` **NamedShape** shim |
| `ModuleNotFoundError: cached_property` | Expected by that commit | `pip install cached-property==1.5.2` |
| Pip warnings about `vllm-tpu` / `setuptools` | Another env on disk | Ignore; this venv is isolated; optionally `export PYTHONNOUSERSITE=1` |

---

## Verified versions

- `jax==0.4.31`, `jaxlib==0.4.31` (TPU backend)
- `flax==0.10.7`, `optax==0.2.6`, `chex==0.1.90`
- `transformers==4.41.1`, `tokenizers==0.19.1`, `huggingface-hub==0.23.0`, `safetensors==0.4.2`
- `librosa==0.10.2.post1`, `soundfile==0.13.1`, `soxr>=0.3.2`
- `cached-property==1.5.2`



## Usage & Operations Cheatsheet

### Log in & jump to the project
```bash
cd ../..
cd /mnt/node5_tpu_data_code_1/whisper
source whisper-venv/bin/activate
```

### Start Whisper (single TPU, localhost only)
```bash
# choose ONE chip (0..3). Do this in the SAME shell before starting uvicorn.
export PJRT_DEVICE=TPU
export TPU_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS   # important: prevents HF/Flax CPU-backend error

# run the API (don’t use --limit-concurrency here)
uvicorn serve_whisper:app --host 127.0.0.1 --port 7860 --workers 1
```

Health check (on the TPU):
```bash
curl -s http://127.0.0.1:7860/health || curl -s http://127.0.0.1:7860
```

### SSH tunneling from your laptop
> Example only — **replace** user/IP with yours.

```bash
ssh -NT -4 -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
  -i "C:\Users\you\.ssh\my-tpu-key" \
  -L 7860:127.0.0.1:7860 you@203.0.113.10
# then open http://127.0.0.1:7860 (or /docs)
```
If port 7860 is busy locally, pick another: `-L 8786:127.0.0.1:7860` → visit `http://127.0.0.1:8786`.

### See which TPU chips are in use
```bash
sudo fuser -v /dev/accel*
```
You’ll see holders of `/dev/accel0..3` with PIDs and command names.

### Free a stuck TPU process (example PID: 2928424)
```bash
ps -fp 2928424
sudo kill -15 2928424; sleep 5 || true
sudo kill -9 2928424   # if it didn’t exit

# verify TPU backend comes up
python - <<'PY'
import jax
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY
```

### Verify your “single-TPU” env actually applied
```bash
python - <<'PY'
import os, jax
print("TPU_VISIBLE_DEVICES =", os.environ.get("TPU_VISIBLE_DEVICES"))
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("local_device_count:", jax.local_device_count())
PY
# Expect: TPU_VISIBLE_DEVICES=0, backend=tpu, local_device_count=1
```

### Quick client test (from your laptop, via the tunnel)
```bash
curl -s http://127.0.0.1:7860/transcribe \
  -F 'file=@english_sample.wav;type=audio/wav' \
  -F 'task=transcribe' -F 'return_timestamps=false'
```

### Optional: run in tmux (keeps it alive after you disconnect)
```bash
tmux new -s whisper -d 'bash -lc "
cd /mnt/node5_tpu_data_code_1/whisper &&
source whisper-venv/bin/activate &&
export PJRT_DEVICE=TPU &&
export TPU_VISIBLE_DEVICES=0 &&
export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
unset JAX_PLATFORMS &&
uvicorn serve_whisper:app --host 127.0.0.1 --port 7860 --workers 1
"'
tmux attach -t whisper   # view logs
# tmux detach:  Ctrl+b then d
# stop: inside tmux press Ctrl+C, or: tmux kill-session -t whisper
```

# Whisper‑JAX on TPU — Pin to **One** TPU Chip

This mini‑guide shows exactly how to make a Whisper‑JAX service use **one** TPU chip (out of 4) instead of all chips on a TPU VM.

> Works for Uvicorn/FastAPI servers like `serve_whisper.py` and other JAX programs.

---

## 1) Pick a single chip with env vars

Run these **in the same shell** you will start Uvicorn from (before any JAX import):

```bash
export PJRT_DEVICE=TPU
export TPU_VISIBLE_DEVICES=0        # choose 0..3 for accel0..accel3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS                  # important: don't force 'tpu' only
```

Start your server (bound to loopback for safety):
```bash
uvicorn serve_whisper:app --host 127.0.0.1 --port 7860 --workers 1
```

> `--workers` and `--limit-concurrency` control HTTP requests, **not** TPU selection. The device selection happens only via `TPU_VISIBLE_DEVICES`.

---

## 2) Verify it really uses just one chip

```bash
python - <<'PY'
import os, jax
print("TPU_VISIBLE_DEVICES =", os.environ.get("TPU_VISIBLE_DEVICES"))
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("local_device_count:", jax.local_device_count())
PY
```
Expected:
```
TPU_VISIBLE_DEVICES = 0
backend: tpu
local_device_count: 1
```

---

## 3) Make it foolproof (set in code too)

Put this at the **top** of `serve_whisper.py` (before importing JAX/whisper/transformers):

```python
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")       # choose 0..3
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Do NOT set JAX_PLATFORMS here (HF/Flax may briefly query CPU backend)
```

---

## 4) Run multiple isolated servers (optional)

Pin each process to a different chip:

```bash
# chip 0 on port 7860
TPU_VISIBLE_DEVICES=0 uvicorn serve_whisper:app --host 127.0.0.1 --port 7860 --workers 1

# chip 1 on port 7861 (another shell/tmux)
TPU_VISIBLE_DEVICES=1 uvicorn serve_whisper:app --host 127.0.0.1 --port 7861 --workers 1
```

---

## 5) Ops helpers

**See which processes are using the chips:**
```bash
sudo fuser -v /dev/accel*
```

**Free a stuck process (example PID: 2928424):**
```bash
ps -fp 2928424
sudo kill -15 2928424; sleep 5 || true
sudo kill -9 2928424   # if needed
```

**Run under tmux so it keeps running after logout:**
```bash
tmux new -s whisper -d 'bash -lc "
export PJRT_DEVICE=TPU &&
export TPU_VISIBLE_DEVICES=0 &&
export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
unset JAX_PLATFORMS &&
cd /mnt/node5_tpu_data_code_1/whisper &&
source whisper-venv/bin/activate &&
uvicorn serve_whisper:app --host 127.0.0.1 --port 7860 --workers 1
"'
tmux attach -t whisper   # view logs; detach: Ctrl+b then d
```

---

## 6) Common pitfalls

- **Setting `JAX_PLATFORMS=tpu`** → breaks HF/Flax init (they briefly query CPU backend). Leave it **unset**.
- **Setting env vars in a different shell** → JAX enumerates all devices. Export them **in the same shell/process** that starts Uvicorn.
- **Using `--limit-concurrency 1`** → can reject `/health` & parallel calls with 503. Prefer an app‑level mutex if you truly need single‑flight on `/transcribe`:

```python
import asyncio
transcribe_lock = asyncio.Lock()

@app.post("/transcribe")
async def transcribe(...):
    async with transcribe_lock:
        # do the work
        ...
```

That’s all—you’re now running Whisper‑JAX on **one** TPU chip, reliably.

# Process of Whisper-JAX (Whisper-large-v3) — Persistent TPU/CPU Service


# Whisper-JAX Persistent TPU/CPU Service Setup

## Overview

This guide explains how we successfully deployed **Whisper-JAX (openai/whisper-large-v3)** as a **persistent FastAPI service** running on **TPU or CPU**, using `systemd` for reliability.  
It documents all dependency conflicts, fixes, and configuration steps to make the setup run indefinitely and restart automatically.

---

## 0. Environment Setup

**Location:** `/mnt/node5_tpu_data_code_1/whisper`  
**Virtualenv:** `/mnt/node5_tpu_data_code_1/whisper/whisper-venv`  
**Python:** 3.10

```bash
mkdir -p /mnt/node5_tpu_data_code_1/whisper
python3 -m venv /mnt/node5_tpu_data_code_1/whisper/whisper-venv
source /mnt/node5_tpu_data_code_1/whisper/whisper-venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

> (Optional) Install TPU-compatible JAX:
```bash
pip install --no-cache-dir --force-reinstall "jax[tpu]==0.4.31" "jaxlib==0.4.31"     -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## 1. Stable Working Versions

We used a **manually pinned version set** to avoid HuggingFace dependency conflicts:

```bash
export PYTHONNOUSERSITE=1

pip uninstall -y transformers tokenizers huggingface-hub safetensors whisper-jax || true

pip install --no-cache-dir --no-deps   "transformers==4.41.1"   "tokenizers==0.19.1"   "huggingface-hub==0.23.0"   "safetensors==0.4.2"

pip install --no-cache-dir --no-deps   "git+https://github.com/sanchit-gandhi/whisper-jax.git@f983178"   "cached-property==1.5.2"
```

### Why these versions?
- Compatible with `whisper-jax` commit `f983178`
- Avoids the tokenizers `>=0.22` enforcement from newer Transformers
- Stable on both TPU and CPU

---

## 2. Environment Isolation

We removed `vllm_installation` from `sys.path` to prevent importing wrong Transformers versions.

```bash
SITE="/mnt/node5_tpu_data_code_1/whisper/whisper-venv/lib/python3.10/site-packages"
cat > "$SITE/sitecustomize.py" <<'PY'
import sys
sys.path = [p for p in sys.path if "vllm_installation" not in p]
PY
```

Sanity check:
```bash
python - <<'PY'
import transformers, tokenizers, sys
print(transformers.__version__, tokenizers.__version__)
print("vllm path present?", any("vllm_installation" in p for p in sys.path))
PY
```

---

## 3. FastAPI Server

**File:** `/mnt/node5_tpu_data_code_1/whisper/serve_whisper.py`

```python
import os, time, tempfile, sys
sys.path = [p for p in sys.path if "vllm_installation" not in p]

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import jax, jax.numpy as jnp

try:
    from whisper_jax import FlaxWhisperPipline as Pipeline
except ImportError:
    from whisper_jax import FlaxWhisperPipeline as Pipeline

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

app = FastAPI(title="Whisper-JAX TPU Service")

_pipe = None
def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = Pipeline(MODEL_ID, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
    return _pipe

@app.get("/health")
def health():
    return {
        "status": "ok",
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "model_id": MODEL_ID,
        "batch_size": BATCH_SIZE
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), task: str = Form("transcribe")):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    t0 = time.time()
    try:
        out = get_pipe()(tmp_path, task=task)
    finally:
        os.remove(tmp_path)
    return JSONResponse({"text": out["text"], "seconds": round(time.time() - t0, 3)})
```

Install server deps:
```bash
pip install fastapi "uvicorn[standard]" python-multipart
```

---

## 4. Systemd Configuration

**File:** `/etc/default/whisper`
```bash
WHISPER_HOME=/mnt/node5_tpu_data_code_1/whisper
VENV=/mnt/node5_tpu_data_code_1/whisper/whisper-venv
HOST=0.0.0.0
PORT=9000
MODEL_ID=openai/whisper-large-v3
BATCH_SIZE=8
PYTHONNOUSERSITE=1
```

**File:** `/etc/systemd/system/whisper.service`
```bash
[Unit]
Description=Whisper-JAX TPU API (uvicorn)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=shantanu_tpu
WorkingDirectory=/mnt/node5_tpu_data_code_1/whisper
EnvironmentFile=/etc/default/whisper
ExecStart=/bin/bash -lc 'exec /mnt/node5_tpu_data_code_1/whisper/whisper-venv/bin/python -m uvicorn serve_whisper:app --host "$HOST" --port "$PORT" --workers 1'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable + start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable whisper
sudo systemctl restart whisper
curl -s http://127.0.0.1:9000/health
```

---

## 5. Troubleshooting Summary

| Issue | Cause | Fix |
|-------|--------|-----|
| Port already in use | Nginx running on 8000 | Stop nginx or switch to port 9000 |
| systemd “bad unit file” | Broken ExecStart line | Use explicit full Python path |
| tokenizers version error | Imported from wrong site-packages | Added `sitecustomize.py` cleanup |
| Transformers conflict | Huggingface-hub pin mismatch | Manually pinned compatible versions |
| Service dies on logout | Running uvicorn manually | Moved to `systemd` daemon |
| Backend shows CPU | TPU env not exported | Set `PJRT_DEVICE=TPU` and install `jax[tpu]` |

## 5. B Troubleshooting (Detailed)

### A) “Bad unit file” or `status=203/EXEC`
Fix malformed ExecStart:
```bash
ExecStart=/bin/bash -lc 'exec /path/to/venv/bin/python -m uvicorn serve_whisper:app --host "$HOST" --port "$PORT" --workers 1'
sudo sed -i 's/\r$//' /etc/systemd/system/whisper.service /etc/default/whisper
sudo systemctl daemon-reload && sudo systemctl restart whisper
```

### B) `tokenizers>=0.22.0,<=0.23.0 required but found 0.19.1`
You’re mixing incompatible versions.  
Reinstall with compatible set:
```bash
pip install --no-cache-dir --no-deps   "transformers==4.41.1" "tokenizers==0.19.1" "huggingface-hub==0.23.0" "safetensors==0.4.2"
```

### C) “backend”: “cpu” instead of “tpu”
Add to `/etc/default/whisper`:
```
PJRT_DEVICE=TPU
TPU_VISIBLE_DEVICES=0
XLA_PYTHON_CLIENT_PREALLOCATE=false
```
Then reinstall TPU JAX and restart:
```bash
pip install --no-cache-dir --force-reinstall "jax[tpu]==0.4.31" "jaxlib==0.4.31" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo systemctl restart whisper
```

### D) Port already in use
```bash
sudo fuser -k 9000/tcp
sudo systemctl restart whisper
```

### E) Nothing responds on port 9000
```bash
journalctl -u whisper -n 200 --no-pager
# Look for Python tracebacks
```

### F) Check environment inside running service
```bash
PID=$(systemctl show -p MainPID --value whisper)
tr '\0' '\n' < /proc/$PID/environ | egrep 'HOST|PORT|MODEL|TPU'
```

---

## Verification Commands

```bash
systemctl status whisper --no-pager
journalctl -u whisper -n 100 --no-pager
sudo ss -lntp | grep :9000
curl -s http://127.0.0.1:9000/health
```


---

## 6. Validation

```bash
curl -s http://127.0.0.1:9000/health
# {"status":"ok","jax":"0.4.31","backend":"cpu","devices":1,"model_id":"openai/whisper-large-v3","batch_size":8}
```

Persistent service ✅  
Correct dependency stack ✅  
Survives reboots ✅  
No tokenizers conflict ✅  

---

## 7. Quick Recreate Script

```bash
python3 -m venv /mnt/node5_tpu_data_code_1/whisper/whisper-venv
source /mnt/node5_tpu_data_code_1/whisper/whisper-venv/bin/activate
pip install --upgrade pip setuptools wheel

pip install --no-cache-dir --no-deps   "transformers==4.41.1" "tokenizers==0.19.1" "huggingface-hub==0.23.0" "safetensors==0.4.2"
pip install --no-cache-dir --no-deps   "git+https://github.com/sanchit-gandhi/whisper-jax.git@f983178" "cached-property==1.5.2"
pip install fastapi "uvicorn[standard]" python-multipart
```

---

✅ **Final Result:**  
The service is now **production-stable**, automatically restarts, isolates dependencies, and exposes `/health` and `/transcribe` endpoints indefinitely.



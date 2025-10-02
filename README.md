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

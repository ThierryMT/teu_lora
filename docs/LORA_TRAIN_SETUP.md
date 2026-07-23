# Quasar LoRA Training — Setup & Run Guide

End-to-end guidance to stand up the LoRA train job (`train_lora.py`) on a
fresh Vast/H100 box **without re-hitting the bugs we already fixed**.

Validated stack (this instance):

| Piece | Version / note |
|-------|----------------|
| GPU | NVIDIA H100 80GB |
| Python | 3.11 (project `.venv`) |
| torch | `2.5.1+cu121` |
| transformers | `5.14.1` |
| peft | `0.19.1` |
| flash-linear-attention (`fla`) | `0.5.1` |
| **triton** | **`3.3.0` only** (do not use ≥3.4) |
| causal-conv1d | `1.6.2.post1` (build with `--no-build-isolation`) |
| datasets | `5.x` (no dataset scripts) |

---

## 0. Paths & prerequisites

```bash
# Repo lives here on this box (symlink used by modeling code):
#   /workspace/teutonic  ==  /root/teutonic
ln -sfn /workspace/teutonic /root/teutonic

cd /workspace/teutonic
source .venv/bin/activate   # or: uv venv .venv && source .venv/bin/activate
```

You need:

1. **Base weights** at `/root/teutonic/newking` (or set `TEUTONIC_MODEL_DIR`).
   - If shards are age-encrypted (`teutonic_encryption.json` present), decrypt
     first with `age` + `keys/validator_model_decryption.key` before training.
2. **Disk** for HF cache + synth shard (~1.2 GB npy).
3. **Node/pm2** for the managed process (optional but recommended).

```bash
. /opt/nvm/nvm.sh
pm2 --version
```

---

## 1. Install Python deps (do this first, in order)

### 1.1 PyTorch (CUDA wheel index)

```bash
source /workspace/teutonic/.venv/bin/activate
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 HuggingFace / training stack

```bash
uv pip install \
  "transformers>=4.47.0" \
  "datasets>=2.18.0" \
  "accelerate>=0.27.0" \
  "peft>=0.10.0" \
  safetensors tokenizers huggingface_hub \
  bitsandbytes tensorboard \
  numpy pandas tqdm sentencepiece protobuf einops
```

### 1.3 FLA + Triton (pin Triton)

```bash
# Package name on PyPI is flash-linear-attention (imports as `fla`)
uv pip install flash-linear-attention

# CRITICAL: fla 0.5.x GLA backward dies on Triton ≥ 3.4
# (CUDA illegal memory access during autotune). Pin 3.3.0.
uv pip install "triton==3.3.0"
```

### 1.4 causal-conv1d (Quasar modeling hard-requires it)

Build against the **already-installed** torch (isolated builds pull a mismatched CUDA torch):

```bash
uv pip install causal-conv1d --no-build-isolation
# Re-pin triton afterward if the build pulled an older/newer wheel:
uv pip install "triton==3.3.0"
```

### 1.5 Sanity imports

```bash
export PYTHONPATH=/workspace/teutonic/newking
python - <<'PY'
import torch, transformers, peft, fla, triton, causal_conv1d, configuration_qwen3_5
assert tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 4)
print("OK", torch.__version__, triton.__version__, torch.cuda.is_available())
PY
```

---

## 2. Known bugs we hit (and the fix)

Keep these in mind if you rewrite `train_lora.py` or upgrade packages.

| # | Failure | Root cause | Fix (already in tree) |
|---|---------|------------|------------------------|
| 1 | `No module named causal_conv1d` / `configuration_qwen3_5` | Quasar remote code imports | Install `causal-conv1d`; set `PYTHONPATH=…/newking` |
| 2 | `Dataset scripts are no longer supported, but found peS2o.py` | `datasets` 5.x dropped scripts | Load peS2o via raw JSON: `hf://…/data/v2/train-*.json.gz` |
| 3 | `TrainingArguments … unexpected keyword include_tokens_per_second` | Removed in this transformers | Do not pass that kwarg; use `warmup_steps` not `warmup_ratio` |
| 4 | `create_causal_mask() got unexpected keyword cache_position` | HF masking_utils older than Quasar | `patch_transformers_masking_compat()` in `train_lora.py` |
| 5 | GLA `Expected attention_mask … [B,S]` got `[B,S,S]` | GLA sits in `full_attention` slots but FLA needs 2D pad mask | `uses_linear_mask` routing in `newking/modeling_qwen3_5.py` + runtime GLA mask coerce |
| 6 | CUDA OOM at step 0 (batch 8 × 2048) | Activations + LoRA on all MLP/attn | `batch=1`, `grad_accum=32`, GC on, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 7 | `CUDA error: illegal memory access` in `fla` GLA backward | Triton 3.7 autotune incompatible | **Pin `triton==3.3.0`**; script aborts if ≥3.4 |
| 8 | Safetensors `header too large` | Encrypted king shards | Decrypt with `age` before `from_pretrained` |
| 9 | Synth manifest `shard_prefix` points at `quasar-synth-run` (404) | Stale prefix in manifest | Resolve shard next to manifest: `…/quasar-synth-v1/shards/<file>` |

Modeling patch location (commit/keep in sync with HF dynamic-module cache):

- Source of truth: `newking/modeling_qwen3_5.py` (GLA → `linear_attn_mask`)
- Cache copies under `HF_HOME/modules/transformers_modules/newking/*/modeling_qwen3_5.py` are regenerated from the local folder on load — **edit the source under `newking/`**.

---

## 3. Training mix (current defaults)

| Source | Count | How loaded |
|--------|------:|------------|
| `nvidia/OpenMathReasoning` (`cot`) | 5,000 | HF streaming → tokenize |
| `allenai/peS2o` (v2 json.gz) | 5,000 | HF `json` streaming → tokenize |
| `quasar-synth-v1` | 120,000 | Pre-tokenized packed `uint32` `.npy` @ 2048 |

- Manifest: `https://us-east-1.hippius.com/tokens-here/dataset/quasar-synth-v1/manifest.json`
- Cached shard: `/workspace/teutonic/data/quasar-synth-v1/shard_000000.npy` (~1.19 GB)
- Total: **130,000** sequences, 1 epoch, seq_len **2048**
- Effective batch: `1 × 32 accum = 32` → ~**4,062** optimizer steps

Override via env (also set in `ecosystem.train.config.js`):

```bash
TEUTONIC_LORA_MATH_N=5000
TEUTONIC_LORA_SCIENCE_N=5000
TEUTONIC_LORA_SYNTH_N=20000
TEUTONIC_LORA_BATCH=1
TEUTONIC_LORA_GRAD_ACCUM=16
TEUTONIC_LORA_SEQ_LEN=2048
TEUTONIC_SYNTH_MANIFEST_URL=https://us-east-1.hippius.com/tokens-here/dataset/quasar-synth-v1/manifest.json
TEUTONIC_MODEL_DIR=/root/teutonic/newking
```

---

## 4. Run with PM2 (recommended)

Config: `ecosystem.train.config.js`  
Script: `train_lora.py`  
Process name: `teutonic-train-lora` (one-shot, `autorestart: false`)

```bash
. /opt/nvm/nvm.sh
cd /workspace/teutonic

# Fresh start
pm2 delete teutonic-train-lora 2>/dev/null || true
pm2 start ecosystem.train.config.js
pm2 save

# Monitor
pm2 list
pm2 logs teutonic-train-lora
pm2 logs teutonic-train-lora --err --lines 50 --nostream
nvidia-smi
```

Healthy early logs should show:

1. Env banner with mix counts + `Batch/Accum : 1 / 32`
2. Model + LoRA load (~17 GB VRAM idle)
3. `Micro check OK` (fwd/bwd smoke)
4. Math / Science / Synth counts
5. `Starting mixed-corpus training…` then `{'loss': …}` every 10 steps

Stop / restart:

```bash
pm2 stop teutonic-train-lora
pm2 delete teutonic-train-lora
pm2 start ecosystem.train.config.js
```

Foreground (debug) alternative:

```bash
cd /workspace/teutonic
source .venv/bin/activate
export PYTHONPATH=/workspace/teutonic/newking
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_lora.py
```

---

## 5. Outputs

| Path | Contents |
|------|----------|
| `./lora-smoke-test-output/` | checkpoints + TensorBoard |
| `./lora-smoke-test-adapter/` | final LoRA adapter + tokenizer |
| PM2 logs | `/root/.pm2/logs/teutonic-train-lora-*.log` |

Base weights under `newking/` are snapshotted before/after; script prints
`PASSED — 0 base weights changed` if LoRA-only training held.

---

## 6. Do / Don’t checklist

**Do**

- Install torch from the CUDA index **before** other CUDA extensions.
- Build `causal-conv1d` with `--no-build-isolation`.
- Keep **triton==3.3.0**.
- Keep `PYTHONPATH` pointing at `newking` for local Quasar modules.
- Use batch 1 + grad accum on 80 GB for this LoRA target set.

**Don’t**

- `uv pip install fla` (wrong name → use `flash-linear-attention`).
- `load_dataset("allenai/peS2o")` on datasets≥4.
- Let `causal-conv1d` / random upgrades bump Triton past 3.3.
- Train on encrypted safetensors without decrypting.
- Trust the synth manifest `shard_prefix` blindly (use colocated `…/v1/shards/`).

---

## 7. Quick reinstall one-liner (after venv exists)

```bash
source /workspace/teutonic/.venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install "transformers>=4.47.0" "datasets>=2.18.0" "accelerate>=0.27.0" "peft>=0.10.0" \
  safetensors tokenizers huggingface_hub bitsandbytes tensorboard \
  numpy pandas tqdm sentencepiece protobuf einops flash-linear-attention
uv pip install causal-conv1d --no-build-isolation
uv pip install "triton==3.3.0"
ln -sfn /workspace/teutonic /root/teutonic
```

Then: `pm2 start /workspace/teutonic/ecosystem.train.config.js`

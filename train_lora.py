"""
LoRA train for QuasarForCausalLM
────────────────────────────────
Mix:
  - 5,000  nvidia/OpenMathReasoning (cot, streamed + tokenized)
  - 5,000  allenai/peS2o (json shards, streamed + tokenized)
  - 120,000 quasar-synth-v1 (pre-tokenized seq_packed uint32 .npy @ 2048)
Total: 130,000 sequences / 1 epoch
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import types
import urllib.error
import urllib.request
import yaml
from pathlib import Path

# Reduce CUDA allocator fragmentation before torch initializes CUDA context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import transformers
import peft as peft_lib
import triton

# fla 0.5.x GLA backward autotune is unstable on Triton>=3.4 (illegal memory access).
_triton_ver = tuple(int(x) for x in triton.__version__.split(".")[:2])
if _triton_ver >= (3, 4):
    raise RuntimeError(
        f"triton {_triton_ver} is too new for fla GLA training; "
        "install triton==3.3.0 (uv pip install 'triton==3.3.0')"
    )

from torch.utils.data import Dataset as TorchDataset, ConcatDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset

MODEL_DIR = os.environ.get("TEUTONIC_MODEL_DIR", "/root/teutonic/newking")
TOKENIZER_ID = os.environ.get("TEUTONIC_TOKENIZER_ID", "silx-ai/Quasar-10B")

# ── YAML config (optional) ────────────────────────────────────────────────────
# Set TEUTONIC_TRAIN_CONFIG=/path/to/configs/v6_baseline.yaml to load a run
# config.  Values in the YAML file take precedence over env vars.  If no config
# is provided the legacy 3-dataset env-var path is used unchanged.
_cfg: dict = {}
_cfg_path = os.environ.get("TEUTONIC_TRAIN_CONFIG", "")
if _cfg_path:
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f) or {}
    print(f"[config] loaded: {_cfg_path}  (experiment: {_cfg.get('experiment_name', '?')})")

def _cv(yaml_key: str, env_var: str, default):
    """Config value: YAML key → env var → default (type-cast to default's type)."""
    if yaml_key in _cfg:
        return type(default)(_cfg[yaml_key])
    val = os.environ.get(env_var)
    return type(default)(val) if val is not None else default

EXPERIMENT_NAME: str = _cfg.get("experiment_name", "lora-run")

# LoRA hyperparameters
LORA_R         = _cv("lora_r",        "TEUTONIC_LORA_R",            16)
LEARNING_RATE  = _cv("learning_rate", "TEUTONIC_LORA_LR",           2e-4)
MAX_GRAD_NORM  = _cv("max_grad_norm", "TEUTONIC_LORA_MAX_GRAD_NORM", 1.0)
MAX_STEPS      = _cv("max_steps",     "TEUTONIC_LORA_MAX_STEPS",    -1)   # -1 = use num_train_epochs

# Sequence / batch parameters
SEQ_LEN    = _cv("seq_len",    "TEUTONIC_LORA_SEQ_LEN",    2048)
BATCH_SIZE = _cv("batch_size", "TEUTONIC_LORA_BATCH",       1)
GRAD_ACCUM = _cv("grad_accum", "TEUTONIC_LORA_GRAD_ACCUM", 16)

# ── Dataset config ────────────────────────────────────────────────────────────
# Config-driven path: dataset_counts maps dataset name → number of sequences.
# Legacy path (no YAML): three env vars as before.
DATASET_COUNTS: dict[str, int]
DATASET_GROUPS: dict[str, list[str]]
HIPPIUS_BASE_URL: str
DATASET_CACHE_DIR: Path

if _cfg:
    DATASET_COUNTS = {k: int(v) for k, v in _cfg.get("dataset_counts", {}).items()}
    DATASET_GROUPS = {
        grp: list(names)
        for grp, names in _cfg.get("dataset_groups", {}).items()
    }
    HIPPIUS_BASE_URL = str(
        _cfg.get("hippius_base_url",
                 "https://us-east-1.hippius.com/tokens-here/dataset")
    )
    DATASET_CACHE_DIR = Path(
        _cfg.get("dataset_cache_dir",
                 os.environ.get("TEUTONIC_SYNTH_CACHE_DIR", "/workspace/teutonic/data"))
    )
else:
    # Legacy 3-dataset mode — kept for backward compatibility.
    _MATH_N    = int(os.environ.get("TEUTONIC_LORA_MATH_N",    "5000"))
    _SCIENCE_N = int(os.environ.get("TEUTONIC_LORA_SCIENCE_N", "5000"))
    _SYNTH_N   = int(os.environ.get("TEUTONIC_LORA_SYNTH_N",   "20000"))
    DATASET_COUNTS = {
        "openmathreasoning": _MATH_N,
        "pes2o":             _SCIENCE_N,
        "quasar-synth-v1":  _SYNTH_N,
    }
    DATASET_GROUPS = {}
    HIPPIUS_BASE_URL = "https://us-east-1.hippius.com/tokens-here/dataset"
    DATASET_CACHE_DIR = Path(
        os.environ.get("TEUTONIC_SYNTH_CACHE_DIR",
                       "/workspace/teutonic/data/quasar-synth-v1")
    )
    # Legacy manifest URL (used only in the legacy dataset-loading branch below).
    _LEGACY_SYNTH_MANIFEST_URL = os.environ.get(
        "TEUTONIC_SYNTH_MANIFEST_URL",
        "https://us-east-1.hippius.com/tokens-here/dataset/quasar-synth-v1/manifest.json",
    )

# Per-dataset sampling multipliers (name → float, default 1.0).
# Applied as:  effective_count = round(base_count * multiplier)
# If effective_count > available sequences, sampling is done with replacement.
# Empty dict → no-op (identical to all multipliers = 1.0).
DATASET_SAMPLING_MULTIPLIERS: dict[str, float] = {
    k: float(v)
    for k, v in _cfg.get("dataset_sampling_multipliers", {}).items()
}

# Per-dataset loss multipliers (name → float, default 1.0).
# Applied as:  loss = (per_example_loss * weight).mean()
# Empty dict → no-op; all samples weighted equally.
DATASET_LOSS_MULTIPLIERS: dict[str, float] = {
    k: float(v)
    for k, v in _cfg.get("dataset_loss_multipliers", {}).items()
}

# Per-dataset manifest URL overrides (name → full manifest URL).
# When set, overrides the default {hippius_base_url}/{name}/manifest.json.
# Use this when a dataset lives under a non-standard Hippius path.
DATASET_MANIFEST_URLS: dict[str, str] = {
    k: str(v)
    for k, v in _cfg.get("dataset_manifest_urls", {}).items()
}

# Per-dataset shard index to download (name → int, default -1 = last shard).
# -1  : always use the newest/last shard in the manifest  ← default
#  0  : first shard (oldest)
#  N  : Nth shard (0-based, negative indices wrap from the end)
DATASET_SHARD_INDICES: dict[str, int] = {
    k: int(v)
    for k, v in _cfg.get("dataset_shard_indices", {}).items()
}
_DEFAULT_SHARD_INDEX: int = int(_cfg.get("default_shard_index", -1))

def patch_transformers_masking_compat() -> None:
    """Quasar modeling passes cache_position; some masking_utils builds reject it."""
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return
    fn = getattr(masking_utils, "create_causal_mask", None)
    if fn is None or getattr(fn, "_quasar_compat", False):
        return
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return
    if "cache_position" in params:
        return

    def create_causal_mask_compat(*args, **kwargs):
        cache_position = kwargs.pop("cache_position", None)
        past_key_values = kwargs.get("past_key_values")
        original_get_mask_sizes = None
        if cache_position is not None and hasattr(past_key_values, "get_mask_sizes"):
            original_get_mask_sizes = past_key_values.get_mask_sizes

            def get_mask_sizes_compat(self, query_length, layer_idx):
                try:
                    return original_get_mask_sizes(cache_position, layer_idx)
                except Exception:
                    return int(query_length), 0

            past_key_values.get_mask_sizes = types.MethodType(
                get_mask_sizes_compat, past_key_values
            )
        try:
            return fn(*args, **kwargs)
        finally:
            if original_get_mask_sizes is not None:
                past_key_values.get_mask_sizes = original_get_mask_sizes

    create_causal_mask_compat._quasar_compat = True
    masking_utils.create_causal_mask = create_causal_mask_compat


def patch_quasar_gla_mask_routing(model) -> None:
    """
    GLA occupies some 'full_attention' slots but FLA requires a 2D padding mask.
    Ensure every loaded QuasarTextModel routes GLA layers to linear_attn_mask.
    """
    seen = set()

    def _patch_module(mod) -> None:
        if mod is None or id(mod) in seen:
            return
        seen.add(id(mod))
        forward = getattr(mod, "forward", None)
        if forward is None or getattr(forward, "_quasar_gla_mask_patch", False):
            return
        # Only patch Quasar text towers that expose the hybrid layer loop.
        if not hasattr(mod, "_update_linear_attn_mask") or not hasattr(mod, "layers"):
            return

        original_forward = forward

        def forward_compat(self, *args, **kwargs):
            # Prefer calling original but with a monkeypatched layer loop if needed.
            # Safer: wrap create path by temporarily patching attribute used in source.
            return original_forward(*args, **kwargs)

        # Direct source-level fix may already be present; verify runtime behavior
        # by wrapping decoder layer mask selection via a pre-hook on layers.
        # If modeling already contains uses_linear_mask, this is a no-op safety net
        # that coerces any 3D/4D mask entering a GLA module down to 2D/None.
        for layer in getattr(mod, "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            gla = getattr(self_attn, "gla", None) if self_attn is not None else None
            if gla is None:
                continue
            gla_forward = gla.forward
            if getattr(gla_forward, "_quasar_gla_mask_patch", False):
                continue

            def make_wrapped(orig):
                def wrapped(*args, attention_mask=None, **kwargs):
                    if attention_mask is not None and getattr(attention_mask, "ndim", 0) != 2:
                        attention_mask = None
                    return orig(*args, attention_mask=attention_mask, **kwargs)

                wrapped._quasar_gla_mask_patch = True
                return wrapped

            gla.forward = make_wrapped(gla_forward)

        forward_compat._quasar_gla_mask_patch = True

    # Walk common wrappers: PeftModel -> base_model -> model / model.language_model / model.model
    roots = [model]
    for attr in ("base_model", "model", "language_model"):
        nxt = []
        for root in roots:
            child = getattr(root, attr, None)
            if child is not None:
                nxt.append(child)
        roots.extend(nxt)

    for root in roots:
        _patch_module(root)
        child = getattr(root, "model", None)
        _patch_module(child)


patch_transformers_masking_compat()

# ─────────────────────────────────────────────
# 0. ENVIRONMENT CHECK
# ─────────────────────────────────────────────
print("=" * 55)
print(f"  Experiment   : {EXPERIMENT_NAME}")
print(f"  PyTorch      : {torch.__version__}")
print(f"  Transformers : {transformers.__version__}")
print(f"  PEFT         : {peft_lib.__version__}")
print(f"  CUDA         : {torch.version.cuda}")
print(f"  GPU          : {torch.cuda.get_device_name(0)}")
print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  BF16         : {torch.cuda.is_bf16_supported()}")
print(f"  LoRA R       : {LORA_R}")
print(f"  LR           : {LEARNING_RATE}")
print(f"  Max grad norm: {MAX_GRAD_NORM}")
print(f"  Max steps    : {MAX_STEPS if MAX_STEPS > 0 else '(full epoch)'}")
print(f"  Batch/Accum  : {BATCH_SIZE} / {GRAD_ACCUM}  (effective {BATCH_SIZE * GRAD_ACCUM})")
print(f"  Seq length   : {SEQ_LEN}")
print(f"  Datasets     : {list(DATASET_COUNTS.keys())}")
total_seqs = sum(DATASET_COUNTS.values())
print(f"  Total seqs   : {total_seqs:,}")
if DATASET_GROUPS:
    for grp, members in DATASET_GROUPS.items():
        print(f"  Group [{grp}]: {members}")
if DATASET_SAMPLING_MULTIPLIERS:
    print(f"  Sampling ×    : {DATASET_SAMPLING_MULTIPLIERS}")
print("=" * 55)

assert torch.cuda.is_available(), "❌ CUDA not available!"
assert torch.cuda.is_bf16_supported(), "❌ BF16 not supported!"


def get_weight_hash(model):
    """Hash base (non-LoRA) weights without fp32 upcast (avoids huge RAM/VRAM spikes)."""
    hashes = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        tensor = param.detach().contiguous()
        if tensor.device.type == "cuda":
            tensor = tensor.cpu()
        hashes[name] = hashlib.md5(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    return hashes


def format_math(example):
    text = f"Problem: {example['problem']}\n\nSolution: {example['generated_solution']}"
    return {"text": text}


def format_science(example):
    return {"text": example["text"]}


def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding=False,
    )


class CausalLmTorchDataset(TorchDataset):
    """Wrap token-id rows for Trainer (provides input_ids / attention_mask / labels)."""

    def __init__(self, rows: list[list[int]] | np.ndarray):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        ids = torch.as_tensor(self.rows[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


class MmapPackedSynthDataset(TorchDataset):
    """Random-access view over a seq-packed uint32 .npy shard (shape [n_tokens])."""

    def __init__(self, path: Path, seq_indices: np.ndarray, seq_len: int, dataset_id: int = 0):
        self.mmap = np.load(path, mmap_mode="r")
        self.seq_indices = np.asarray(seq_indices, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.dataset_id = int(dataset_id)

    def __len__(self) -> int:
        return int(self.seq_indices.shape[0])

    def __getitem__(self, idx: int) -> dict:
        start = int(self.seq_indices[idx]) * self.seq_len
        ids = torch.from_numpy(
            np.asarray(self.mmap[start : start + self.seq_len], dtype=np.int64)
        )
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }


def synth_shard_url(manifest_url: str, shard_key: str) -> str:
    """
    Manifest may list an outdated shard_prefix (quasar-synth-run). Prefer shards
    colocated next to the manifest (…/quasar-synth-v1/shards/<file>).
    """
    fname = shard_key.rstrip("/").rsplit("/", 1)[-1]
    base = manifest_url.rsplit("/", 1)[0]
    return f"{base}/shards/{fname}"


def ensure_synth_shard(
    manifest_url: str,
    cache_dir: Path,
    shard_index: int = -1,
) -> tuple[Path, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(manifest_url, timeout=60) as resp:
        manifest = json.loads(resp.read().decode("utf-8"))
    shards = manifest.get("shards") or []
    if not shards:
        raise RuntimeError(f"no shards in manifest: {manifest_url}")
    shard = shards[shard_index]
    key = shard["key"]
    url = synth_shard_url(manifest_url, key)
    expected = int(shard.get("size_bytes") or 0)
    dest = cache_dir / Path(key).name
    if dest.exists() and (expected <= 0 or dest.stat().st_size == expected):
        print(f"      Synth shard cached: {dest} ({dest.stat().st_size:,} bytes)")
        return dest, manifest
    idx_label = f"[{shard_index}]" if shard_index != -1 else "[last]"
    print(f"      Downloading shard {idx_label} of {len(shards)}:\n        {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    if expected > 0 and tmp.stat().st_size != expected:
        raise RuntimeError(
            f"synth shard size mismatch: got {tmp.stat().st_size}, expected {expected}"
        )
    tmp.replace(dest)
    print(f"      Synth shard ready: {dest} ({dest.stat().st_size:,} bytes)")
    return dest, manifest


def _load_or_download_shard(
    ds_name: str,
    manifest_url: str,
    cache_dir: Path,
    seq_len: int,
    shard_index: int = -1,
) -> tuple[Path, dict]:
    """Return (shard_path, manifest).

    Resolution order:
      1. Any *.npy already in cache_dir  — use it; read manifest.json if
         present beside it, else synthesise a minimal one from seq_len.
      2. Network download via manifest_url / ensure_synth_shard.
         shard_index selects which shard to download (-1 = last/newest).
         On HTTP 404, raises a RuntimeError with actionable instructions.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Local-first: skip network if a shard is already on disk ──────────────
    npy_files = sorted(cache_dir.glob("*.npy"))
    if npy_files:
        npy = npy_files[0]
        local_mf = cache_dir / "manifest.json"
        if local_mf.exists():
            manifest: dict = json.loads(local_mf.read_text())
        else:
            manifest = {
                "seq_len": seq_len,
                "shards": [{"key": npy.name, "size_bytes": npy.stat().st_size}],
            }
        print(f"      [{ds_name}] local shard: {npy.name}  ({npy.stat().st_size:,} bytes)")
        return npy, manifest

    # ── Network download ──────────────────────────────────────────────────────
    try:
        return ensure_synth_shard(manifest_url, cache_dir, shard_index=shard_index)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"\n\n  ✗  [{ds_name}] Manifest not found — HTTP 404\n"
                f"     URL tried : {manifest_url}\n\n"
                f"  Fix options (choose one):\n\n"
                f"  A) Override the manifest URL in your config:\n"
                f"       dataset_manifest_urls:\n"
                f"         {ds_name}: <correct_hippius_manifest_url>\n\n"
                f"  B) Pre-download the .npy shard and place it here:\n"
                f"       {cache_dir}/\n"
                f"     The trainer will detect it automatically on next start.\n"
                f"     (Use hippius-hub or direct S3 download to obtain the shard.)\n"
            ) from exc
        raise# ─────────────────────────────────────────────
# 1. TOKENIZER
# ─────────────────────────────────────────────
print(f"\n[1/8] Loading tokenizer from {TOKENIZER_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    trust_remote_code=True,
    use_fast=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"      Vocab size : {tokenizer.vocab_size:,}")
print(f"      Pad token  : {tokenizer.pad_token!r}  id={tokenizer.pad_token_id}")

# ─────────────────────────────────────────────
# 2. BASE MODEL
# ─────────────────────────────────────────────
print(f"\n[2/8] Loading base model from {MODEL_DIR} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)
model.config.use_cache = False  # required for gradient checkpointing

vram_used = torch.cuda.memory_allocated(0) / 1e9
print(f"      VRAM used  : {vram_used:.2f} GB / 80.0 GB")
print(f"      VRAM free  : {80.0 - vram_used:.2f} GB")

# ─────────────────────────────────────────────
# 3. LORA ADAPTERS
# ─────────────────────────────────────────────
print("\n[3/8] Injecting LoRA adapters...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_R * 2,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        # ── GatedDeltaNet layers ──
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
        # ── GatedLinearAttention layers ──
        "self_attn.gla.q_proj",
        "self_attn.gla.k_proj",
        "self_attn.gla.v_proj",
        "self_attn.gla.g_proj",
        "self_attn.gla.gk_proj.0",
        "self_attn.gla.gk_proj.1",
        "self_attn.gla.o_proj",
        # ── MLP ──
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# PEFT + gradient checkpointing requires input grads.
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
patch_quasar_gla_mask_routing(model)

vram_after_lora = torch.cuda.memory_allocated(0) / 1e9
print(f"      VRAM after LoRA : {vram_after_lora:.2f} GB / 80.0 GB")

# ─────────────────────────────────────────────
# 4. SAFETY SNAPSHOT + MICRO FORWARD CHECK
# ─────────────────────────────────────────────
print("\n[4/8] 🛡️  Snapshotting base weights BEFORE training...")
snapshot_before = get_weight_hash(model)
torch.cuda.empty_cache()
print(f"      Captured {len(snapshot_before):,} base weight tensors.")

print("      Micro forward/backward check (batch=1, seq=128)...")
model.train()
micro_ids = torch.randint(
    low=0,
    high=min(tokenizer.vocab_size, 1000),
    size=(1, 128),
    device="cuda",
)
micro_mask = torch.ones_like(micro_ids)
out = model(input_ids=micro_ids, attention_mask=micro_mask, labels=micro_ids)
loss = out.loss
assert torch.isfinite(loss), f"non-finite micro loss: {loss}"
loss.backward()
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
print(f"      Micro check OK  loss={float(loss):.4f}  "
      f"VRAM={torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# ─────────────────────────────────────────────
# 5. DATASETS — mixed corpus
# ─────────────────────────────────────────────
print("\n[5/8] Building mixed training set...")
print(f"      target mix: {DATASET_COUNTS}")

per_ds_datasets: list[TorchDataset] = []

if _cfg:
    # ── Config-driven path ────────────────────────────────────────────────────
    # All datasets are Hippius-hosted seq-packed uint32 .npy shards.
    # Manifest URL: {HIPPIUS_BASE_URL}/{dataset_name}/manifest.json
    _sampling_plan: list[tuple[str, int, float, int]] = []  # (name, base, mult, effective)

    for ds_idx, (ds_name, ds_count) in enumerate(DATASET_COUNTS.items()):
        manifest_url = DATASET_MANIFEST_URLS.get(
            ds_name, f"{HIPPIUS_BASE_URL}/{ds_name}/manifest.json"
        )
        cache_dir = DATASET_CACHE_DIR / ds_name
        print(f"      [{ds_name}] base={ds_count:,}  manifest: {manifest_url}")
        shard_path, shard_manifest = _load_or_download_shard(
            ds_name, manifest_url, cache_dir, SEQ_LEN,
            shard_index=DATASET_SHARD_INDICES.get(ds_name, _DEFAULT_SHARD_INDEX),
        )
        manifest_seq_len = int(shard_manifest.get("seq_len") or SEQ_LEN)
        if manifest_seq_len != SEQ_LEN:
            raise RuntimeError(
                f"{ds_name}: manifest seq_len={manifest_seq_len} != SEQ_LEN={SEQ_LEN}"
            )
        ds_mmap = np.load(shard_path, mmap_mode="r")
        n_seqs_avail = int(ds_mmap.shape[0]) // SEQ_LEN
        if n_seqs_avail == 0:
            raise RuntimeError(
                f"{ds_name}: shard has no complete sequences at seq_len={SEQ_LEN}"
            )

        # Apply sampling multiplier.
        multiplier = DATASET_SAMPLING_MULTIPLIERS.get(ds_name, 1.0)
        effective_count = max(1, round(ds_count * multiplier))
        # Oversample with replacement when effective_count exceeds available sequences.
        replace = effective_count > n_seqs_avail
        if not replace and n_seqs_avail < ds_count:
            raise RuntimeError(
                f"{ds_name}: only {n_seqs_avail:,} seqs available, need {ds_count:,}"
            )

        indices = np.random.default_rng(42).choice(n_seqs_avail, size=effective_count, replace=replace)
        per_ds_datasets.append(MmapPackedSynthDataset(shard_path, indices, SEQ_LEN, dataset_id=ds_idx))
        _sampling_plan.append((ds_name, ds_count, multiplier, effective_count))

        tag = f" (×{multiplier} → {effective_count:,})" if multiplier != 1.0 else ""
        resample_tag = " [oversample]" if replace else ""
        si = DATASET_SHARD_INDICES.get(ds_name, _DEFAULT_SHARD_INDEX)
        si_tag = f"  shard={'last' if si == -1 else si}"
        print(f"      {ds_name}: {ds_count:,}{tag}  avail={n_seqs_avail:,}{resample_tag}{si_tag} ✅")

    # ── Effective sampling plan summary ──────────────────────────────────────
    _eff_total = sum(e for _, _, _, e in _sampling_plan)
    col = max(len(n) for n, *_ in _sampling_plan)
    print(f"\n      {'─' * (col + 42)}")
    print(f"      {'Dataset':<{col}}  {'base':>8}  {'×mult':>6}  {'effective':>10}  {'%':>6}")
    print(f"      {'─' * (col + 42)}")
    for ds_name, base, mult, eff in _sampling_plan:
        pct = 100.0 * eff / _eff_total
        mult_str = f"×{mult:.2f}"
        print(f"      {ds_name:<{col}}  {base:>8,}  {mult_str:>6}  {eff:>10,}  {pct:>5.1f}%")
    print(f"      {'─' * (col + 42)}")
    print(f"      {'TOTAL':<{col}}  {'':>8}  {'':>6}  {_eff_total:>10,}  {'100.0%':>6}")
    print(f"      {'─' * (col + 42)}\n")

else:
    # ── Legacy 3-dataset path (no YAML config) ────────────────────────────────
    print(f"      [legacy] math={_MATH_N:,}  science={_SCIENCE_N:,}  synth={_SYNTH_N:,}")

    print(f"      Streaming nvidia/OpenMathReasoning cot ({_MATH_N:,})...")
    ds_math_stream = load_dataset(
        "nvidia/OpenMathReasoning",
        split="cot",
        streaming=True,
    )
    ds_math_stream = ds_math_stream.shuffle(seed=42, buffer_size=10_000).take(_MATH_N)
    ds_math = Dataset.from_list([format_math(x) for x in ds_math_stream])
    ds_math = ds_math.map(tokenize_fn, batched=True, remove_columns=["text"])
    math_rows = [list(x) for x in ds_math["input_ids"]]
    print(f"      Math done    : {len(math_rows):,} samples ✅")

    print(f"      Streaming allenai/peS2o json shards ({_SCIENCE_N:,})...")
    ds_sci_stream = load_dataset(
        "json",
        data_files="hf://datasets/allenai/peS2o/data/v2/train-*.json.gz",
        split="train",
        streaming=True,
    )
    ds_sci_stream = ds_sci_stream.shuffle(seed=42, buffer_size=10_000).take(_SCIENCE_N)
    ds_science = Dataset.from_list([format_science(x) for x in ds_sci_stream])
    ds_science = ds_science.map(tokenize_fn, batched=True, remove_columns=["text"])
    science_rows = [list(x) for x in ds_science["input_ids"]]
    print(f"      Science done : {len(science_rows):,} samples ✅")

    print(f"      Loading quasar-synth-v1 packed npy ({_SYNTH_N:,})...")
    print(f"      manifest: {_LEGACY_SYNTH_MANIFEST_URL}")
    synth_path, synth_manifest = ensure_synth_shard(_LEGACY_SYNTH_MANIFEST_URL, DATASET_CACHE_DIR)
    manifest_seq_len = int(synth_manifest.get("seq_len") or SEQ_LEN)
    if manifest_seq_len != SEQ_LEN:
        raise RuntimeError(
            f"synth seq_len={manifest_seq_len} != training SEQ_LEN={SEQ_LEN}"
        )
    synth_mmap = np.load(synth_path, mmap_mode="r")
    n_tokens = int(synth_mmap.shape[0])
    n_seqs_avail = n_tokens // SEQ_LEN
    if n_seqs_avail < _SYNTH_N:
        raise RuntimeError(
            f"synth only has {n_seqs_avail:,} sequences, need {_SYNTH_N:,}"
        )
    rng = np.random.default_rng(42)
    synth_indices = rng.choice(n_seqs_avail, size=_SYNTH_N, replace=False)
    synth_ds = MmapPackedSynthDataset(synth_path, synth_indices, SEQ_LEN)
    print(
        f"      Synth done   : {len(synth_ds):,} / {n_seqs_avail:,} available "
        f"({n_tokens:,} tokens) ✅"
    )

    per_ds_datasets = [
        CausalLmTorchDataset(math_rows),
        CausalLmTorchDataset(science_rows),
        synth_ds,
    ]


combined = ConcatDataset(per_ds_datasets)


class ShuffledDataset(TorchDataset):
    def __init__(self, base: TorchDataset, seed: int = 42):
        self.base = base
        order = np.arange(len(base))
        np.random.default_rng(seed).shuffle(order)
        self.order = order

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[int(self.order[idx])]


train_dataset = ShuffledDataset(combined, seed=42)
print(f"\n      Total : {len(train_dataset):,} samples  ✅")

# ─────────────────────────────────────────────
# 6. TRAINING ARGS
# ─────────────────────────────────────────────
print("\n[6/8] Configuring training args...")
steps_per_epoch = max(1, len(train_dataset) // max(1, BATCH_SIZE * GRAD_ACCUM))
warmup_steps = max(1, int(0.03 * steps_per_epoch))

training_args = TrainingArguments(
    output_dir=f"./{EXPERIMENT_NAME}-output",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1,
    max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="tensorboard",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    max_grad_norm=MAX_GRAD_NORM,
)

# ─────────────────────────────────────────────
# 7. TRAINER
# ─────────────────────────────────────────────

# ── Loss weight tensor (one float per dataset_id index) ──────────────────────
# Index i = position of dataset in DATASET_COUNTS; value = loss multiplier.
# Shape [num_datasets], float32.  Moved to GPU lazily inside compute_loss.
_ds_names_ordered = list(DATASET_COUNTS.keys())
_id_to_loss_weight = torch.ones(len(_ds_names_ordered))
for _i, _name in enumerate(_ds_names_ordered):
    _id_to_loss_weight[_i] = DATASET_LOSS_MULTIPLIERS.get(_name, 1.0)

if DATASET_LOSS_MULTIPLIERS:
    print("\n  Active dataset loss multipliers:")
    col = max(len(n) for n in _ds_names_ordered)
    for _name, _w in DATASET_LOSS_MULTIPLIERS.items():
        print(f"    {_name:<{col}} : ×{_w}")
    # Warn on any configured name not present in the active dataset mix.
    for _name in DATASET_LOSS_MULTIPLIERS:
        if _name not in DATASET_COUNTS:
            print(
                f"  ⚠️  Warning: loss multiplier configured for '{_name}' "
                "but it is not in dataset_counts — multiplier has no effect."
            )

_use_weighted_loss = bool(DATASET_LOSS_MULTIPLIERS) and _cfg


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-dataset loss multipliers.

    Set ``loss_weights`` to a 1-D float tensor of shape [num_datasets] before
    training.  Index i must correspond to the integer ``dataset_id`` stored in
    each batch.  When ``loss_weights`` is None every sample is weighted equally
    and the behaviour is identical to the base Trainer.
    """

    loss_weights: torch.Tensor | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        dataset_ids = inputs.pop("dataset_id", None)
        outputs = model(**inputs)

        # Fast path: no per-dataset weights configured, or metadata not present.
        if self.loss_weights is None or dataset_ids is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        weights = self.loss_weights.to(dataset_ids.device)[dataset_ids]  # [B]

        # Skip the expensive per-example path when all weights in this batch are 1.
        if weights.eq(1.0).all():
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # Per-example cross-entropy (sequences are fully packed; no -100 labels).
        logits = outputs.logits                 # [B, T, V]
        labels = inputs["labels"]               # [B, T]
        B, T = labels.shape
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token = loss_fct(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )                                        # [B * (T-1)]
        per_sample = per_token.view(B, T - 1).mean(dim=-1)  # [B]
        loss = (per_sample * weights).mean()
        return (loss, outputs) if return_outputs else loss


print("\n[7/8] Building Trainer...")
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)
trainer.loss_weights = _id_to_loss_weight if _use_weighted_loss else None

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
print("\n[8/8] 🚀 Starting mixed-corpus training...")
print(f"      Planned steps ≈ {steps_per_epoch}  warmup={warmup_steps}")
trainer.train()

# ─────────────────────────────────────────────
# VERIFY BASE WEIGHTS UNCHANGED
# ─────────────────────────────────────────────
print("\n🛡️  Verifying base weights AFTER training...")
snapshot_after = get_weight_hash(model)
changed = [n for n in snapshot_before if snapshot_before[n] != snapshot_after[n]]

print("\n" + "=" * 55)
if len(changed) == 0:
    print("  ✅ PASSED — 0 base weights changed!")
    print(f"  ❄️  {MODEL_DIR} is 100% intact.")
else:
    print(f"  ❌ FAILED — {len(changed)} base weights changed!")
    for name in changed[:10]:
        print(f"     CHANGED: {name}")
print("=" * 55)

# ─────────────────────────────────────────────
# SAVE ADAPTER ONLY
# ─────────────────────────────────────────────
model.save_pretrained(f"./{EXPERIMENT_NAME}-adapter")
tokenizer.save_pretrained(f"./{EXPERIMENT_NAME}-adapter")

peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
print(f"\n  Peak VRAM usage : {peak_vram:.2f} GB / 80.0 GB")
print(f"  Adapter saved   : ./{EXPERIMENT_NAME}-adapter/")
print(f"  Original model  : {MODEL_DIR}  ← untouched ✅")

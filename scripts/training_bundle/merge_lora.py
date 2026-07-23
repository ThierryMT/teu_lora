#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a standalone Teutonic challenger snapshot.

Produces the layout validators accept before eval:
  - config.json (king-identical for Quasar lock keys / auto_map)
  - configuration_qwen3_5.py + modeling_qwen3_5.py (byte-identical to king)
  - model.safetensors OR model-NNNNN-of-NNNNN.safetensors + index
  - tokenizer files
  - no adapter_*, optimizer.pt, trainer_state, or other training debris
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QUASAR_CODE_FILES = (
    "configuration_qwen3_5.py",
    "modeling_qwen3_5.py",
)
KING_COPY_FILES = (
    "config.json",
    "generation_config.json",
    *QUASAR_CODE_FILES,
)


def _copy_king_files(base_model: Path, output_dir: Path) -> None:
    """Overwrite save_pretrained artifacts with king-locked files."""
    for name in KING_COPY_FILES:
        src = base_model / name
        if not src.exists():
            if name in QUASAR_CODE_FILES or name == "config.json":
                raise FileNotFoundError(
                    f"base model missing required king file: {src}"
                )
            continue
        shutil.copy2(src, output_dir / name)

    cfg_path = output_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    # Eval loads bf16; keep dtype consistent with king snapshots.
    cfg["dtype"] = "bfloat16"
    cfg["use_cache"] = True
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def _strip_training_debris(output_dir: Path) -> None:
    junk_exact = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
        "README.md",
    }
    for path in output_dir.iterdir():
        if path.name in junk_exact or path.name.startswith("checkpoint-"):
            path.unlink() if path.is_file() else shutil.rmtree(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True, help="King / base model directory")
    ap.add_argument("--adapter-dir", required=True, help="PEFT adapter / Trainer checkpoint")
    ap.add_argument("--output-dir", required=True, help="Submission-ready merged snapshot")
    ap.add_argument(
        "--tokenizer",
        default="",
        help="Tokenizer source (default: adapter-dir, else base-model)",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    base = Path(args.base_model).resolve()
    adapter = Path(args.adapter_dir).resolve()
    out = Path(args.output_dir).resolve()
    tok_src = Path(args.tokenizer).resolve() if args.tokenizer else (
        adapter if (adapter / "tokenizer.json").exists() or (adapter / "tokenizer_config.json").exists()
        else base
    )

    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"output dir is not empty: {out}\n"
            "Pick a fresh path (e.g. challenger-600) so the adapter is not overwritten."
        )
    out.mkdir(parents=True, exist_ok=True)

    # Quasar snapshots import sibling modules (`from configuration_qwen3_5 import …`).
    # Local dirs are not always put on sys.path by transformers dynamic loading.
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    print(f"[1/4] Loading base model from {base} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        use_safetensors=True,
        device_map={"": args.device} if args.device != "cpu" else None,
    )

    print(f"[2/4] Loading adapter from {adapter} ...")
    model = PeftModel.from_pretrained(model, str(adapter))
    print("[3/4] Merging LoRA into base weights ...")
    merged = model.merge_and_unload()
    # Keep weights on device briefly; save_pretrained streams from state_dict.
    merged.config.use_cache = True

    print(f"[4/4] Saving standalone snapshot to {out} ...")
    merged.save_pretrained(str(out), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(
        str(tok_src),
        trust_remote_code=True,
        use_fast=True,
    )
    tok.save_pretrained(str(out))

    # Critical for Quasar: auto_map + local code must match the king byte-for-byte.
    _copy_king_files(base, out)
    _strip_training_debris(out)

    st = sorted(p.name for p in out.glob("*.safetensors"))
    has_index = (out / "model.safetensors.index.json").exists()
    has_single = (out / "model.safetensors").exists()
    print("Done.")
    print(f"  safetensors : {st[:6]}{'...' if len(st) > 6 else ''}")
    print(f"  layout      : {'single' if has_single else 'sharded' if has_index else 'INVALID'}")
    print(f"  code files  : {[n for n in QUASAR_CODE_FILES if (out / n).exists()]}")
    print(f"  upload this directory to Hippius Hub, then submit the v4 reveal.")


if __name__ == "__main__":
    main()

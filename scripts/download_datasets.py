#!/usr/bin/env python3
"""
Pre-download all dataset shards required by a training config.
Run this BEFORE pm2 start to ensure all shards are on disk.

Usage:
    # Download all datasets in a config:
    python scripts/download_datasets.py configs/v7_hard_preserve.yaml

    # Download one specific dataset only:
    python scripts/download_datasets.py configs/v7_hard_preserve.yaml --dataset dendrite-synth-run

    # Dry-run (show what would be downloaded, no network calls):
    python scripts/download_datasets.py configs/v7_hard_preserve.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def _shard_url(manifest_url: str, shard_key: str) -> str:
    """Resolve shard download URL relative to the manifest URL."""
    fname = shard_key.rstrip("/").rsplit("/", 1)[-1]
    base = manifest_url.rsplit("/", 1)[0]
    return f"{base}/shards/{fname}"


def download_dataset(
    ds_name: str,
    manifest_url: str,
    cache_dir: Path,
    shard_index: int = -1,
    dry_run: bool = False,
) -> bool:
    """
    Download one shard for a dataset.
    shard_index selects which shard: -1 = last/newest (default), 0 = first.
    Saves manifest.json beside the shard so the trainer can skip future
    network calls (local-first resolution in _load_or_download_shard).

    Returns True on success, False on any error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch manifest ────────────────────────────────────────────────────────
    idx_label = "last" if shard_index == -1 else str(shard_index)
    print(f"  [{ds_name}]  shard={idx_label}")
    print(f"    manifest : {manifest_url}")
    if dry_run:
        print(f"    (dry-run — skipping download)")
        return True

    try:
        with urllib.request.urlopen(manifest_url, timeout=60) as resp:
            manifest: dict = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"    ✗ manifest fetch failed: HTTP {exc.code}")
        return False
    except Exception as exc:
        print(f"    ✗ manifest fetch failed: {exc}")
        return False

    # Save manifest locally so trainer can skip this URL on next run.
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    shards = manifest.get("shards") or []
    if not shards:
        print(f"    ✗ manifest has no shards")
        return False

    seq_len = int(manifest.get("seq_len") or 2048)
    resolved_idx = shard_index % len(shards)  # handle negative indices
    print(f"    seq_len  : {seq_len}")
    print(f"    shards   : {len(shards)} total — downloading shard[{shard_index}] (index {resolved_idx})")

    # ── Download selected shard ───────────────────────────────────────────────
    shard = shards[shard_index]
    key = shard["key"]
    url = _shard_url(manifest_url, key)
    expected = int(shard.get("size_bytes") or 0)
    dest = cache_dir / Path(key).name

    if dest.exists() and (expected <= 0 or dest.stat().st_size == expected):
        print(f"    ✓ already cached: {dest.name}  ({dest.stat().st_size:,} bytes)")
        return True

    size_str = f"{expected / 1e9:.2f} GB" if expected > 0 else "unknown size"
    print(f"    downloading: {dest.name}  ({size_str})")
    print(f"    url: {url}")

    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                print(f"\r    progress: {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
    except Exception as exc:
        print(f"\n    ✗ download failed: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False

    actual = tmp.stat().st_size
    if expected > 0 and actual != expected:
        print(f"    ✗ size mismatch: got {actual:,} bytes, expected {expected:,}")
        tmp.unlink()
        return False

    tmp.replace(dest)
    print(f"    ✓ saved: {dest.name}  ({actual:,} bytes)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download dataset shards required by a training config."
    )
    parser.add_argument(
        "config",
        help="Path to YAML training config (e.g. configs/v7_hard_preserve.yaml)",
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        action="append",
        dest="datasets",
        default=None,
        help="Download only this dataset (by name). Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without making any network requests.",
    )
    parser.add_argument(
        "--shard",
        metavar="INDEX",
        type=int,
        default=None,
        help="Shard index to download: -1=last/newest (default), 0=first. "
             "Overrides default_shard_index from config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    experiment   = cfg.get("experiment_name", "?")
    ds_counts    = {k: int(v) for k, v in cfg.get("dataset_counts", {}).items()}
    mf_overrides = {k: str(v) for k, v in cfg.get("dataset_manifest_urls", {}).items()}
    shard_indices = {k: int(v) for k, v in cfg.get("dataset_shard_indices", {}).items()}
    default_shard = int(cfg.get("default_shard_index", -1))
    base_url     = cfg.get("hippius_base_url", "https://us-east-1.hippius.com/tokens-here/dataset")
    cache_root   = Path(cfg.get("dataset_cache_dir", "/workspace/teutonic/data"))

    # CLI --shard flag overrides config default for all datasets
    if args.shard is not None:
        default_shard = args.shard

    print("=" * 60)
    print(f"  Config     : {args.config}")
    print(f"  Experiment : {experiment}")
    print(f"  Cache root : {cache_root}")
    if args.dry_run:
        print("  Mode       : DRY RUN")
    print("=" * 60)
    print()

    # Validate --dataset filter
    if args.datasets:
        for name in args.datasets:
            if name not in ds_counts:
                print(f"✗ Dataset '{name}' not found in config.")
                print(f"  Available : {list(ds_counts.keys())}")
                sys.exit(1)

    targets = args.datasets if args.datasets else list(ds_counts.keys())

    # Warn about missing manifest URL overrides (will use default pattern)
    missing_overrides = [n for n in targets if n not in mf_overrides]
    if missing_overrides:
        print(f"  ⚠  No manifest URL override for: {missing_overrides}")
        print(f"     Will use default: {base_url}/{{name}}/manifest.json")
        print()

    ok: list[str] = []
    failed: list[str] = []

    for ds_name in targets:
        manifest_url = mf_overrides.get(ds_name, f"{base_url}/{ds_name}/manifest.json")
        cache_dir = cache_root / ds_name
        success = download_dataset(
            ds_name, manifest_url, cache_dir,
            shard_index=shard_indices.get(ds_name, default_shard),
            dry_run=args.dry_run,
        )
        (ok if success else failed).append(ds_name)
        print()

    print("=" * 60)
    print(f"  Done   : {len(ok)} / {len(targets)}")
    if failed:
        print(f"  Failed : {failed}")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

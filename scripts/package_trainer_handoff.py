"""Build a self-contained handoff archive for an external SL-0 trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]

FILES = (
    "requirements-train.txt",
    "src/train/__init__.py",
    "src/train/shared_data.py",
    "src/train/shared_model.py",
    "src/train/train_shared.py",
    "tests/test_shared_training.py",
    "data/training/training_manifest_v1.json",
    "data/training/README.md",
    "data/cards.json",
    "data/card_tags.json",
    "data/card_tag_full_audit.json",
    "data/manual_overrides.json",
)

ALIASES = {
    "docs/operations/服务器共享模型训练指南.md": "docs/operations/server_training_guide_zh.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release_assets/trainer_handoff_v2")
    parser.add_argument("--archive", type=Path, default=ROOT / "release_assets/pokemon-tcg-sl0-trainer-handoff-v2.tar.gz")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_asset = ROOT / "release_assets/training_data_v2/data/training/training_decisions_v1.jsonl.gz"
    if not data_asset.is_file():
        raise FileNotFoundError(f"run scripts/package_training_release.py first: {data_asset}")
    staged_data = output / "data/training/training_decisions_v1.jsonl.gz"
    staged_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_asset, staged_data)

    items = []
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        items.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256(source)})
    for source_relative, destination_relative in ALIASES.items():
        source = ROOT / source_relative
        destination = output / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        items.append({
            "path": destination_relative,
            "source_path": source_relative,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })
    items.append({
        "path": "data/training/training_decisions_v1.jsonl.gz",
        "bytes": staged_data.stat().st_size,
        "sha256": sha256(staged_data),
        "restored_path": "data/training/training_decisions_v1.jsonl",
        "restored_sha256": "E8DC4DC2784A3505EAA159255A735A2C50B907DB66A5F9AB7759BEC326062370",
    })

    start_here = """# SL-0-shared 外部训练交接包

这个目录可以独立交给训练人员，不需要发送原始 Kaggle replay 或中间转换数据。

## 必做步骤

1. 使用 Python 3.11，并安装与服务器 CUDA 匹配的 PyTorch。
2. 在本目录运行 `gzip -dk data/training/training_decisions_v1.jsonl.gz`。
3. 校验解压文件 SHA-256：`E8DC4DC2784A3505EAA159255A735A2C50B907DB66A5F9AB7759BEC326062370`。
4. 运行 `python tests/test_shared_training.py`，必须得到 `OK`。
5. 先按 `docs/operations/server_training_guide_zh.md` 运行 10,000/2,000 样本冒烟。
6. 冒烟报告发回确认后，才启动完整训练。

## 必须交回的训练产物

- `artifacts/sl0_shared_*/run_config.json`
- `artifacts/sl0_shared_*/metrics.jsonl`
- `artifacts/sl0_shared_*/best.pt`
- `artifacts/sl0_shared_*/last.pt`
- GPU/CPU/内存/磁盘配置
- PyTorch、CUDA、驱动版本
- 完整训练命令、开始/结束时间、samples/s、峰值显存

## 不要做

- 不要修改训练集、manifest、特征顺序或 split。
- 不要跳过冒烟直接跑全量。
- 不要用测试集选超参数。
- 不要把 checkpoint 静默用于不同 SHA-256 的数据集。
"""
    (output / "START_HERE.md").write_text(start_here, encoding="utf-8")
    items.append({"path": "START_HERE.md", "bytes": (output / "START_HERE.md").stat().st_size, "sha256": sha256(output / "START_HERE.md")})

    manifest = {
        "schema_version": "sl0_trainer_handoff_v2",
        "purpose": "external training of SL-0-shared",
        "ruleset": "ptcg_abc_2026_simulation_designated_pool_v1",
        "dataset_samples": 861939,
        "dataset_games": 6552,
        "dataset_sha256_restored": "E8DC4DC2784A3505EAA159255A735A2C50B907DB66A5F9AB7759BEC326062370",
        "intermediate_datasets_required": False,
        "items": items,
    }
    (output / "HANDOFF_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    print(json.dumps({
        "directory": str(output),
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256(args.archive),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

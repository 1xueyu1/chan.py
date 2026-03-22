from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive one framework version into "
            "backup/<version_name>/"
        ),
    )
    parser.add_argument("--version-name", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--predict-dir", default="")
    parser.add_argument("--backtest-dir", default="")
    parser.add_argument("--notes", default="")
    return parser


def _copy_dir(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _default_code_files(project_root: Path) -> List[Path]:
    rels = [
        "Debug/xgboost_shap_train.py",
        "Debug/xgboost_shap_predict.py",
        "Debug/run_full_pipeline.py",
        "Debug/run_predict_backtest.py",
        "Debug/PIPELINE_USAGE.md",
        "Backtest/config.py",
        "Backtest/engine.py",
        "Backtest/model_gate.py",
        "Backtest/reporter.py",
        "Backtest/examples/run_vectorbt_backtest.py",
    ]
    return [project_root / rel for rel in rels]


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    backup_root = project_root / "backup"
    version_dir = backup_root / args.version_name

    code_dir = version_dir / "code"
    artifacts_dir = version_dir / "artifacts"
    docs_dir = version_dir / "docs"

    for d in [code_dir, artifacts_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_src = (project_root / args.train_dir).resolve()
    predict_src = (
        (project_root / args.predict_dir).resolve()
        if args.predict_dir
        else None
    )
    backtest_src = (
        (project_root / args.backtest_dir).resolve()
        if args.backtest_dir
        else None
    )

    copied_artifacts: Dict[str, str] = {}
    if _copy_dir(train_src, artifacts_dir / "train"):
        copied_artifacts["train"] = str(artifacts_dir / "train")
    if predict_src and _copy_dir(predict_src, artifacts_dir / "predict"):
        copied_artifacts["predict"] = str(artifacts_dir / "predict")
    if backtest_src and _copy_dir(backtest_src, artifacts_dir / "backtest"):
        copied_artifacts["backtest"] = str(artifacts_dir / "backtest")

    copied_code: List[str] = []
    for src in _default_code_files(project_root):
        rel = src.relative_to(project_root)
        dst = code_dir / rel
        if _copy_file(src, dst):
            copied_code.append(str(rel))

    notes_path = docs_dir / "version_notes.md"
    notes_text = (
        f"# {args.version_name}\n\n"
        f"- archived_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- train_source: {args.train_dir}\n"
        f"- predict_source: {args.predict_dir or '--'}\n"
        f"- backtest_source: {args.backtest_dir or '--'}\n\n"
        f"## Notes\n\n"
        f"{args.notes or 'N/A'}\n"
    )
    notes_path.write_text(notes_text, encoding="utf-8")

    manifest = {
        "version_name": args.version_name,
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "train_dir": args.train_dir,
            "predict_dir": args.predict_dir,
            "backtest_dir": args.backtest_dir,
        },
        "copied_artifacts": copied_artifacts,
        "copied_code_files": copied_code,
        "notes_file": str(notes_path),
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[Backup] done")
    print(f"[Backup] version_dir={version_dir}")


if __name__ == "__main__":
    main()

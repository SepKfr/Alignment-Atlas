from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from src.app.services import AtlasService
from src.ingest.pipeline import IngestPipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
RAW_HTML_DIR = DATA_DIR / "raw_html"


def _print_result(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _cmd_reingest(args: argparse.Namespace) -> int:
    if args.clean:
        print("[clean] removing processed + index artifacts", flush=True)
        _rm(PROCESSED_DIR)
        _rm(INDEX_DIR)
        if args.wipe_raw:
            print("[clean] removing raw download cache", flush=True)
            _rm(RAW_PDFS_DIR)
            _rm(RAW_HTML_DIR)

    pipeline = IngestPipeline()
    run = pipeline.run_full(
        run_relations=not args.skip_relations,
        progress_callback=None,
        claims_env=None,
    )
    _print_result(
        {
            "ok": run.ok,
            "failed_module": run.failed_module,
            "stage_results": run.stage_results,
        }
    )
    return 0 if run.ok else 1


def _cmd_ingest_url(args: argparse.Namespace) -> int:
    service = AtlasService()
    out = service.ingest_source(
        title=args.title or "",
        source_url=args.url,
        source_type=args.source_type,
        year=args.year,
        run_relations=not args.skip_relations,
        incremental=not args.full_rebuild,
    )
    _print_result(out)
    return 0 if bool(out.get("ok")) else 1


def _cmd_run_stage(args: argparse.Namespace) -> int:
    pipeline = IngestPipeline()
    ok, rec = pipeline.run_single_stage(args.stage)
    _print_result(rec)
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reingest = sub.add_parser("reingest", help="Run full ingest pipeline.")
    p_reingest.add_argument("--clean", action="store_true", help="Delete processed/index artifacts before reingest.")
    p_reingest.add_argument("--wipe-raw", action="store_true", help="Also delete raw downloaded PDFs/HTML before run.")
    p_reingest.add_argument("--skip-relations", action="store_true", help="Skip relation stages 06/07.")
    p_reingest.set_defaults(func=_cmd_reingest)

    p_ingest = sub.add_parser("ingest-url", help="Ingest a single source URL via AtlasService.")
    p_ingest.add_argument("--url", required=True, help="Source URL to ingest.")
    p_ingest.add_argument("--title", default="", help="Optional title.")
    p_ingest.add_argument("--source-type", default="auto", choices=["auto", "pdf", "html"])
    p_ingest.add_argument("--year", type=int, default=None)
    p_ingest.add_argument("--skip-relations", action="store_true")
    p_ingest.add_argument("--full-rebuild", action="store_true", help="Use full rebuild mode instead of incremental.")
    p_ingest.set_defaults(func=_cmd_ingest_url)

    p_stage = sub.add_parser("run-stage", help="Run one ingest stage module.")
    p_stage.add_argument("--stage", required=True, help='Stage id or module path, e.g. "build_kg"')
    p_stage.set_defaults(func=_cmd_run_stage)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


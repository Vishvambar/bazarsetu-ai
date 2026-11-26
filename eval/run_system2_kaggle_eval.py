"""
BazarSetu V3.0 - Step 4 Live Kaggle Benchmark Runner for System 2.

Runs Gemma 4 12B QAT (GPU 1) over all 43 products in eval/golden_dataset.jsonl,
evaluates all 12 release gates via eval/system2_eval.py, and saves a detailed
audit report to storage/eval_reports/step4_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.prepare_system2_golden import find_restore_archive, restore_golden_zip
from eval.system2_eval import RELEASE_GATES, evaluate_pipeline
from system2.engine import System2Config, System2Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BazarSetu.System2KaggleEval")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run System 2 Gemma 4 12B live benchmark on Kaggle.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "eval" / "golden_dataset.jsonl"),
        help="Path to golden_dataset.jsonl",
    )
    parser.add_argument(
        "--auto-restore",
        action="store_true",
        help="Restore storage/golden/ from zip archive if present.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional HuggingFace repo override (default: unsloth/gemma-4-12B-it-qat-GGUF).",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="Optional GGUF model filename override.",
    )
    parser.add_argument(
        "--mmproj-file",
        default=None,
        help="Optional mmproj filename override (default: mmproj-F16.gguf).",
    )
    parser.add_argument(
        "--report-out",
        default=str(PROJECT_ROOT / "storage" / "eval_reports" / "step4_report.json"),
        help="Path to write full evaluation JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N records (e.g., --limit 3 for quick smoke check).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama.cpp timing & debug output.",
    )
    args = parser.parse_args()

    golden_dir = PROJECT_ROOT / "storage" / "golden"
    if args.auto_restore:
        archive = find_restore_archive(None)
        if archive:
            restore_golden_zip(archive, golden_dir)

    cfg = System2Config(verbose=args.verbose)
    if args.repo_id:
        cfg.repo_id = args.repo_id
    if args.model_file:
        cfg.model_filename = args.model_file
    if args.mmproj_file:
        cfg.mmproj_filename = args.mmproj_file

    logger.info("Loading System2Engine (%s / %s) on cuda:%d...", cfg.repo_id, cfg.model_filename, cfg.main_gpu)
    engine = System2Engine(config=cfg)

    logger.info("Running evaluation across %s...", args.dataset)
    report = evaluate_pipeline(args.dataset, engine.predict, limit=args.limit, progress=True)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print(" BAZARSETU V3.0 - SYSTEM 2 STEP 4 RELEASE GATE SCORECARD")
    print("=" * 68)
    for key, val in report.items():
        if key in ("failures", "attribute_mismatches"):
            continue
        print(f"  {key:30s}: {val}")
    print("-" * 68)
    print(f"  Release Gates Target          : p95 <= {RELEASE_GATES['p95_latency_ms_max']} ms, schema == 100%, seals_halluc == 0")
    print(f"  Failures ({len(report['failures'])})               :")
    for f in report["failures"][:15]:
        print(f"    - {f}")
    mismatches = report.get("attribute_mismatches") or {}
    for m_key in ("color", "scale", "composition", "craft", "material"):
        items = mismatches.get(m_key) or []
        if items:
            print(f"  {m_key.title()} mismatches ({len(items)})     : {', '.join(items[:8])}")
    print("=" * 68 + "\n")
    logger.info("Saved full report -> %s", report_path)
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

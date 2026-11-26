"""
BazarSetu V3.0 - System 2 Automated Evaluation Harness.

Evaluates structured VLM / System 2 outputs against V3.0/eval/golden_dataset.jsonl
across all accuracy, schema, seal-hallucination, seller-review, and latency gates.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system2.schemas import ProductIntelligenceOutput


RELEASE_GATES = {
    "schema_validity_pct": 100.0,
    "domain_accuracy_pct": 95.0,
    "color_accuracy_pct": 90.0,
    "scale_accuracy_pct": 90.0,
    "composition_accuracy_pct": 90.0,
    "material_accuracy_pct": 85.0,
    "craft_recall_pct": 85.0,
    "review_trigger_recall_pct": 75.0,
    "review_false_positive_pct_max": 20.0,
    "seal_recall_pct": 85.0,
    "seal_hallucination_count_max": 0,
    "p95_latency_ms_max": 8500.0,
}


def load_golden_records(dataset_path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        records.append(json.loads(line))
    return records


def evaluate_pipeline(
    dataset_path: str | Path,
    predict_fn: Callable[[str, Dict[str, str]], Dict[str, Any]],
    limit: int | None = None,
    progress: bool = False,
) -> Dict[str, Any]:
    records = load_golden_records(dataset_path)
    if limit is not None and limit > 0:
        records = records[:limit]
    if not records:
        raise ValueError(f"No valid records found in {dataset_path}")

    counts = {
        "schema_valid": 0,
        "domain_match": 0,
        "color_match": 0,
        "scale_match": 0,
        "composition_match": 0,
        "material_match": 0,
        "craft_hits": 0,
        "craft_total": 0,
        "review_expected": 0,
        "review_triggered_hits": 0,
        "clear_expected": 0,
        "review_false_positives": 0,
        "seal_expected_total": 0,
        "seal_hits": 0,
        "seal_hallucinations": 0,
    }
    latencies_ms: List[float] = []
    failures: List[str] = []

    total_records = len(records)
    for idx, row in enumerate(records, start=1):
        pid = row["product_id"]
        gt = row["ground_truth"]

        t0 = time.perf_counter()
        try:
            raw_out = predict_fn(pid, row["views"])
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            failures.append(f"{pid}: predictor raised exception -> {exc}")
            if progress:
                print(f"[{idx:2d}/{total_records}] {pid:15s} | {elapsed_ms/1000:.2f}s | ERROR: {exc}", flush=True)
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        try:
            pred = ProductIntelligenceOutput.model_validate(raw_out)
            counts["schema_valid"] += 1
        except Exception as exc:
            failures.append(f"{pid}: schema validation failed -> {exc}")
            if progress:
                print(f"[{idx:2d}/{total_records}] {pid:15s} | {elapsed_ms/1000:.2f}s | SCHEMA FAIL: {exc}", flush=True)
            continue

        if pred.ondc_taxonomy.domain_code.value == gt["domain_code"]:
            counts["domain_match"] += 1
        else:
            failures.append(
                f"{pid}: domain {pred.ondc_taxonomy.domain_code.value} != {gt['domain_code']}"
            )

        if (
            pred.color_profile.primary_color.standard_family.strip().lower()
            == gt["primary_color_family"].strip().lower()
        ):
            counts["color_match"] += 1

        if pred.product_identity.physical_scale.value == gt["physical_scale"]:
            counts["scale_match"] += 1

        if pred.product_identity.composition_type.value == gt["composition_type"]:
            counts["composition_match"] += 1

        if (
            pred.materials[0].material_family.strip().lower()
            == gt["primary_material_family"].strip().lower()
        ):
            counts["material_match"] += 1

        pred_crafts = {c.strip().lower() for c in pred.pattern_and_craft.craft_techniques}
        for exp_craft in gt.get("expected_craft_techniques", []):
            counts["craft_total"] += 1
            if exp_craft.strip().lower() in pred_crafts:
                counts["craft_hits"] += 1

        # Seller review trigger check (Recall on ambiguous + False-Positive rate on clear)
        flagged_review = len(pred.audit_telemetry.needs_seller_review) > 0
        if gt.get("expect_seller_review", False):
            counts["review_expected"] += 1
            if flagged_review:
                counts["review_triggered_hits"] += 1
            else:
                failures.append(f"{pid}: missed needs_seller_review on ambiguous item")
        else:
            counts["clear_expected"] += 1
            if flagged_review:
                counts["review_false_positives"] += 1

        # Certification seals: both positive recall and zero-hallucination precision
        pred_seals = {
            s.strip().lower()
            for s in pred.text_and_seals.certification_seals
            if s and s.strip().lower() != "none"
        }
        gt_seals = {s.strip().lower() for s in gt.get("expected_seals", []) if s}
        for exp_seal in gt_seals:
            counts["seal_expected_total"] += 1
            if exp_seal in pred_seals:
                counts["seal_hits"] += 1
            else:
                failures.append(f"{pid}: missed expected seal '{exp_seal}'")

        hallucinated = pred_seals - gt_seals
        if hallucinated:
            counts["seal_hallucinations"] += 1
            failures.append(f"{pid}: hallucinated seals {sorted(hallucinated)}")

        if progress:
            print(
                f"[{idx:2d}/{total_records}] {pid:15s} | {elapsed_ms/1000:5.2f}s | "
                f"domain={pred.ondc_taxonomy.domain_code.value} | "
                f"color={pred.color_profile.primary_color.standard_family} | "
                f"seals={sorted(pred_seals)} | review={flagged_review}",
                flush=True,
            )

    n = len(records)
    report: Dict[str, Any] = {
        "total_samples": n,
        "schema_validity_pct": round(100.0 * counts["schema_valid"] / n, 2),
        "domain_accuracy_pct": round(100.0 * counts["domain_match"] / n, 2),
        "color_accuracy_pct": round(100.0 * counts["color_match"] / n, 2),
        "scale_accuracy_pct": round(100.0 * counts["scale_match"] / n, 2),
        "composition_accuracy_pct": round(100.0 * counts["composition_match"] / n, 2),
        "material_accuracy_pct": round(100.0 * counts["material_match"] / n, 2),
        "craft_recall_pct": round(
            100.0 * counts["craft_hits"] / max(1, counts["craft_total"]), 2
        ),
        "review_trigger_recall_pct": round(
            100.0 * counts["review_triggered_hits"] / max(1, counts["review_expected"]),
            2,
        ),
        "review_false_positive_pct": round(
            100.0 * counts["review_false_positives"] / max(1, counts["clear_expected"]),
            2,
        ),
        "seal_recall_pct": (
            round(100.0 * counts["seal_hits"] / counts["seal_expected_total"], 2)
            if counts["seal_expected_total"] > 0
            else 100.0
        ),
        "seal_hallucination_count": counts["seal_hallucinations"],
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "failures": failures,
    }
    report["all_gates_passed"] = check_release_gates(report)
    return report


def check_release_gates(report: Dict[str, Any]) -> bool:
    return (
        report["schema_validity_pct"] >= RELEASE_GATES["schema_validity_pct"]
        and report["domain_accuracy_pct"] >= RELEASE_GATES["domain_accuracy_pct"]
        and report["color_accuracy_pct"] >= RELEASE_GATES["color_accuracy_pct"]
        and report["scale_accuracy_pct"] >= RELEASE_GATES["scale_accuracy_pct"]
        and report["composition_accuracy_pct"] >= RELEASE_GATES["composition_accuracy_pct"]
        and report["material_accuracy_pct"] >= RELEASE_GATES["material_accuracy_pct"]
        and report["craft_recall_pct"] >= RELEASE_GATES["craft_recall_pct"]
        and report["review_trigger_recall_pct"] >= RELEASE_GATES["review_trigger_recall_pct"]
        and report["review_false_positive_pct"] <= RELEASE_GATES["review_false_positive_pct_max"]
        and report["seal_recall_pct"] >= RELEASE_GATES["seal_recall_pct"]
        and report["seal_hallucination_count"] <= RELEASE_GATES["seal_hallucination_count_max"]
        and report["p95_latency_ms"] <= RELEASE_GATES["p95_latency_ms_max"]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate golden_dataset.jsonl schema & mock smoke test")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "eval" / "golden_dataset.jsonl"),
        help="Path to golden_dataset.jsonl",
    )
    args = parser.parse_args()
    loaded = load_golden_records(args.dataset)
    print(f"Loaded {len(loaded)} golden evaluation records from {args.dataset}.")


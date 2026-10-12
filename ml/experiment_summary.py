"""
Experiment Summary and Audit CLI for Multimodal Health Screening.

Summarizes all recorded training runs, checkpoints, and held-out test evaluations
directly from machine-readable metadata. Missing metrics are labeled unavailable.

Usage:
    python ml/experiment_summary.py               # Human-readable tabular summary
    python ml/experiment_summary.py --json        # Machine-readable JSON summary
    python ml/experiment_summary.py --run <id>    # Deep-dive into specific run
    python ml/experiment_summary.py --legacy     # Audit baseline checkpoints in ml/weights
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.config import get_default_experiments_dir
from ml.experiment_tracker import (
    ExperimentTracker,
    audit_legacy_checkpoints,
    verify_checkpoint_integrity,
)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats list of rows into an aligned ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    body_lines = []
    for row in rows:
        body_lines.append(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))

    return f"{header_line}\n{separator}\n" + "\n".join(body_lines)


def summarize_experiments(
    experiments_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    output_json: bool = False,
    include_legacy: bool = True,
) -> int:
    """
    Executes experiment discovery, metadata parsing, and report generation.
    """
    tracker = ExperimentTracker(experiments_dir=experiments_dir)
    runs = tracker.list_runs()
    legacy_ckpts = audit_legacy_checkpoints() if include_legacy else []

    if run_id:
        # Deep inspection of a single run
        run = tracker.get_run(run_id)
        if run is None:
            print(f"Error: Run '{run_id}' not found in {tracker.runs_root}", file=sys.stderr)
            return 1

        if output_json:
            print(json.dumps(run.metadata, indent=2))
        else:
            meta = run.metadata
            print("=" * 80)
            print(f"EXPERIMENT RUN DETAILS: {run_id}")
            print("=" * 80)
            print(f"Experiment Name:   {meta.get('experiment_name')}")
            print(f"Status:            {meta.get('status')}")
            print(f"Start (UTC):       {meta.get('timestamps', {}).get('start_utc')}")
            print(f"Completed (UTC):   {meta.get('timestamps', {}).get('completion_utc') or 'In progress / Interrupted'}")
            print(f"Duration:          {meta.get('timestamps', {}).get('total_duration_seconds') or 'N/A'} seconds")
            print(f"Seed:              {meta.get('reproducibility', {}).get('seed')}")
            print(f"Configured Device: {meta.get('device', {}).get('configured')}")
            print(f"Dataset Manifest:  {meta.get('dataset', {}).get('manifest_path')} (SHA: {str(meta.get('dataset', {}).get('manifest_sha256'))[:10]}...)")
            print(f"Total Samples:     {meta.get('dataset', {}).get('total_images')}")
            
            # Epochs
            history = meta.get("epoch_history", [])
            print(f"\nTraining History ({len(history)} epochs recorded):")
            if history:
                ep_headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR", "Time (s)"]
                ep_rows = []
                for ep in history:
                    ep_rows.append([
                        str(ep.get("epoch")),
                        f"{ep.get('train_loss', 0):.4f}",
                        f"{ep.get('train_acc', 0)*100:.2f}%",
                        f"{ep.get('val_loss', 0):.4f}",
                        f"{ep.get('val_acc', 0)*100:.2f}%",
                        f"{ep.get('learning_rate', 0):.6f}",
                        f"{ep.get('epoch_duration_seconds', 0):.1f}",
                    ])
                print(format_table(ep_headers, ep_rows))

            # Checkpoints
            print("\nCheckpoints:")
            ckpts = meta.get("checkpoints", {})
            for ckpt_type, cdata in ckpts.items():
                if cdata:
                    print(f"  [{ckpt_type}] File: {cdata.get('filename')} | SHA-256: {cdata.get('sha256')} | Val Acc: {cdata.get('best_val_accuracy')}")
                else:
                    print(f"  [{ckpt_type}] None recorded")

            # Evaluations
            evals = meta.get("evaluations", [])
            print(f"\nHeld-Out Evaluations ({len(evals)} linked):")
            for ev in evals:
                acc_val = ev.get("overall_accuracy")
                acc_str = f"{acc_val * 100:.2f}%" if acc_val is not None else "N/A"
                f1_val = ev.get("macro_f1")
                f1_str = f"{f1_val * 100:.2f}%" if f1_val is not None else "N/A"
                print(f"  - Split: {ev.get('dataset_split')} | Samples: {ev.get('evaluated_images')} | Accuracy: {acc_str} | Macro F1: {f1_str}")
        return 0

    if output_json:
        payload = {
            "experiments_root": tracker.experiments_root,
            "tracked_runs_count": len(runs),
            "total_runs": len(runs),
            "runs": runs,
            "legacy_checkpoints": legacy_ckpts,
            "legacy_checkpoints_audit": legacy_ckpts,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("=" * 80)
    print("MULTIMODAL HEALTH SCREENING - EXPERIMENT TRACKING REGISTRY")
    print("=" * 80)
    print(f"Experiments Root: {tracker.experiments_root}")
    print(f"Total Runs Discovered: {len(runs)}")
    print()

    if runs:
        headers = ["Run ID", "Status", "Started (UTC)", "Epochs", "Best Val Acc", "Test Acc", "Test F1", "Best SHA256", "Resumable"]
        rows = []
        for r in runs:
            val_acc_str = f"{r['best_val_accuracy']*100:.2f}%" if r.get("best_val_accuracy") is not None else "N/A"
            test_acc_str = f"{r['test_accuracy']*100:.2f}%" if r.get("test_accuracy") is not None else "N/A"
            test_f1_str = f"{r['test_macro_f1']*100:.2f}%" if r.get("test_macro_f1") is not None else "N/A"
            rows.append([
                r["run_id"],
                r["status"],
                r.get("start_utc", "N/A")[:19],
                str(r.get("epochs_completed", 0)),
                val_acc_str,
                test_acc_str,
                test_f1_str,
                r.get("best_checkpoint_sha256") or "None",
                "Yes" if r.get("has_recovery_checkpoint") else "No",
            ])
        print(format_table(headers, rows))
    else:
        print("No new experiment runs recorded yet in experiments/runs/.")
        print("To record a run, execute: python ml/train_eye.py")

    if legacy_ckpts:
        print("\n" + "=" * 80)
        print("LEGACY PRODUCTION CHECKPOINTS AUDIT (ml/weights)")
        print("=" * 80)
        leg_headers = ["Checkpoint File", "Size (MB)", "Integrity", "SHA-256 (Prefix)", "Best Val Acc", "Resumable"]
        leg_rows = []
        for c in legacy_ckpts:
            sha_pref = (c.get("sha256") or "")[:12] + "..." if c.get("sha256") else "N/A"
            acc_str = f"{c['best_val_accuracy']*100:.2f}%" if c.get("best_val_accuracy") is not None else "N/A"
            leg_rows.append([
                c["filename"],
                str(c["size_mb"]),
                "VALID" if c["valid"] else f"FAIL ({c.get('error', '')[:15]})",
                sha_pref,
                acc_str,
                "Yes" if c.get("is_resumable") else "No",
            ])
        print(format_table(leg_headers, leg_rows))

    return 0


def main():
    parser = argparse.ArgumentParser(description="Multimodal Health Screening Experiment Tracker Summary")
    parser.add_argument("--experiments-dir", type=str, default=None, help="Root experiments directory")
    parser.add_argument("--run", type=str, default=None, help="Specific run ID to inspect in detail")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--legacy", action="store_true", help="Audit baseline legacy checkpoints")
    parser.add_argument("--no-legacy", action="store_true", help="Skip legacy checkpoint audit")

    args = parser.parse_args()
    sys.exit(
        summarize_experiments(
            experiments_dir=args.experiments_dir,
            run_id=args.run,
            output_json=args.json,
            include_legacy=not args.no_legacy,
        )
    )


if __name__ == "__main__":
    main()

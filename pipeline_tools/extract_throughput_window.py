#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOPK_RE = re.compile(r"(?:^|_)k(\d+)$")

_KIND_ORDER = [
    "lora",
    "cola",
    "hydralora",
    "adamole",
    "moelora",
    "mola",
    "movlora",
    "hmora",
    "mixlora",
    "moelpr",
]


@dataclass(frozen=True)
class StepStats:
    step: int
    pct: float | None
    total_steps: int | None
    samples: list[float]

    @property
    def mean(self) -> float | None:
        if not self.samples:
            return None
        return float(statistics.mean(self.samples))

    @property
    def median(self) -> float | None:
        if not self.samples:
            return None
        return float(statistics.median(self.samples))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _load_summary_map(summary_csv: Path) -> dict[str, dict[str, str]]:
    if not summary_csv.exists():
        return {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            out[run_id] = {str(k): str(v) for k, v in row.items() if k is not None and v is not None}
        return out


def _parse_topk(run_id: str) -> int | None:
    match = _TOPK_RE.search(run_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _kind_sort_key(kind: str) -> tuple[int, str]:
    kind_norm = str(kind or "").strip().lower()
    try:
        idx = _KIND_ORDER.index(kind_norm)
    except ValueError:
        idx = len(_KIND_ORDER)
    return idx, kind_norm


def _topk_sort_value(value: object) -> int:
    if value is None:
        return -1
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return -1
    try:
        return int(text)
    except ValueError:
        return -1


def _collect_step_stats(trainer_log_jsonl: Path) -> dict[int, StepStats]:
    by_step: dict[int, list[float]] = {}
    pct_by_step: dict[int, float] = {}
    total_steps_by_step: dict[int, int] = {}
    for payload in _read_jsonl(trainer_log_jsonl):
        step = _safe_int(payload.get("current_steps"))
        if step is None:
            continue
        throughput = _safe_float(payload.get("throughput"))
        if throughput is None:
            continue
        by_step.setdefault(step, []).append(throughput)
        pct = _safe_float(payload.get("percentage"))
        if pct is not None:
            pct_by_step[step] = pct
        total_steps = _safe_int(payload.get("total_steps"))
        if total_steps is not None:
            total_steps_by_step[step] = total_steps
    out: dict[int, StepStats] = {}
    for step, samples in by_step.items():
        out[step] = StepStats(
            step=step,
            pct=pct_by_step.get(step),
            total_steps=total_steps_by_step.get(step),
            samples=samples,
        )
    return dict(sorted(out.items(), key=lambda item: item[0]))


def _select_stable_steps(
    steps: dict[int, StepStats],
    *,
    stable_min_pct: float,
    stable_max_pct: float,
    fallback_last_n: int,
    stable_last_n_steps: int | None = None,
) -> list[int]:
    if stable_last_n_steps is not None and stable_last_n_steps > 0:
        available = [step for step, stats in steps.items() if step > 0 and stats.median is not None]
        if not available:
            return []
        return available[-stable_last_n_steps:]
    chosen = [
        step
        for step, stats in steps.items()
        if step > 0
        and stats.pct is not None
        and stable_min_pct <= float(stats.pct) <= stable_max_pct
        and stats.median is not None
    ]
    if chosen:
        return chosen
    available = [step for step, stats in steps.items() if step > 0 and stats.median is not None]
    if not available:
        return []
    return available[-fallback_last_n:]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract per-step and stable-window throughput metrics from pipeline outputs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/pipeline_runs/moe_study_profile_topk_sweep"),
        help="Pipeline output root (contains runs/, logs/, summary.csv).",
    )
    parser.add_argument("--stable-min-pct", type=float, default=60.0)
    parser.add_argument("--stable-max-pct", type=float, default=100.0)
    parser.add_argument(
        "--stable-last-n-steps",
        type=int,
        default=None,
        help="If set, ignore pct window and use last N steps (excluding step 0).",
    )
    parser.add_argument("--fallback-last-n", type=int, default=5)
    parser.add_argument("--out-steps-csv", type=Path, default=None)
    parser.add_argument("--out-summary-csv", type=Path, default=None)
    args = parser.parse_args()

    output_root: Path = args.output_root
    runs_root = output_root / "runs"
    summary_map = _load_summary_map(output_root / "summary.csv")
    allowed_run_ids = set(summary_map) if summary_map else None

    out_steps_csv: Path = args.out_steps_csv or (output_root / "throughput_steps.csv")
    out_summary_csv: Path = args.out_summary_csv or (output_root / "throughput_stable_avg.csv")
    out_steps_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for run_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if allowed_run_ids is not None and run_id not in allowed_run_ids:
            continue
        trainer_log = run_dir / "trainer_log.jsonl"
        if not trainer_log.exists():
            continue
        steps = _collect_step_stats(trainer_log)
        stable_steps = _select_stable_steps(
            steps,
            stable_min_pct=float(args.stable_min_pct),
            stable_max_pct=float(args.stable_max_pct),
            fallback_last_n=int(args.fallback_last_n),
            stable_last_n_steps=(int(args.stable_last_n_steps) if args.stable_last_n_steps is not None else None),
        )
        stable_step_set = set(stable_steps)

        kind = summary_map.get(run_id, {}).get("kind", "")
        top_k = _parse_topk(run_id)

        per_step_medians: list[float] = []
        total_steps_value: int | None = None
        for step, stats in steps.items():
            median = stats.median
            mean = stats.mean
            if total_steps_value is None and stats.total_steps is not None:
                total_steps_value = stats.total_steps
            if median is not None and step in stable_step_set:
                per_step_medians.append(median)
            step_rows.append(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "top_k": top_k if top_k is not None else "",
                    "step": step,
                    "pct": stats.pct if stats.pct is not None else "",
                    "total_steps": stats.total_steps if stats.total_steps is not None else "",
                    "throughput_tok_s_median": f"{median:.6f}" if median is not None else "",
                    "throughput_tok_s_mean": f"{mean:.6f}" if mean is not None else "",
                    "samples": len(stats.samples),
                    "in_stable_window": "1" if step in stable_step_set else "0",
                }
            )

        stable_mean = _mean(per_step_medians)
        stable_std = _stdev(per_step_medians)

        effective_train_results = None
        train_results_path = run_dir / "train_results.json"
        if train_results_path.exists():
            try:
                train_results = json.loads(train_results_path.read_text(encoding="utf-8"))
                effective_train_results = _safe_float(train_results.get("effective_tokens_per_sec"))
            except Exception:
                effective_train_results = None

        effective_summary = None
        try:
            effective_summary = float(summary_map.get(run_id, {}).get("effective_tokens_per_sec", ""))  # type: ignore[arg-type]
        except Exception:
            effective_summary = None

        summary_rows.append(
            {
                "run_id": run_id,
                "kind": kind,
                "top_k": top_k if top_k is not None else "",
                "total_steps": total_steps_value if total_steps_value is not None else "",
                "stable_min_pct": float(args.stable_min_pct),
                "stable_max_pct": float(args.stable_max_pct),
                "stable_last_n_steps": int(args.stable_last_n_steps) if args.stable_last_n_steps is not None else "",
                "stable_steps": ",".join(str(s) for s in stable_steps),
                "stable_step_count": len(stable_steps),
                "stable_tok_s_mean": f"{stable_mean:.6f}" if stable_mean is not None else "",
                "stable_tok_s_std": f"{stable_std:.6f}" if stable_std is not None else "",
                "effective_tokens_per_sec_train_results": f"{effective_train_results:.6f}" if effective_train_results is not None else "",
                "effective_tokens_per_sec_summary_csv": f"{effective_summary:.6f}" if effective_summary is not None else "",
            }
        )

    step_fieldnames = [
        "run_id",
        "kind",
        "top_k",
        "step",
        "pct",
        "total_steps",
        "throughput_tok_s_median",
        "throughput_tok_s_mean",
        "samples",
        "in_stable_window",
    ]
    summary_rows.sort(
        key=lambda row: (
            _kind_sort_key(str(row.get("kind", ""))),
            _topk_sort_value(row.get("top_k")),
            str(row.get("run_id", "")),
        )
    )
    step_rows.sort(
        key=lambda row: (
            _kind_sort_key(str(row.get("kind", ""))),
            _topk_sort_value(row.get("top_k")),
            str(row.get("run_id", "")),
            int(row.get("step", 0)),
        )
    )
    with out_steps_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=step_fieldnames)
        writer.writeheader()
        for row in step_rows:
            writer.writerow(row)

    summary_fieldnames = [
        "run_id",
        "kind",
        "top_k",
        "total_steps",
        "stable_min_pct",
        "stable_max_pct",
        "stable_last_n_steps",
        "stable_steps",
        "stable_step_count",
        "stable_tok_s_mean",
        "stable_tok_s_std",
        "effective_tokens_per_sec_train_results",
        "effective_tokens_per_sec_summary_csv",
    ]
    with out_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[ok] wrote {out_steps_csv}")
    print(f"[ok] wrote {out_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

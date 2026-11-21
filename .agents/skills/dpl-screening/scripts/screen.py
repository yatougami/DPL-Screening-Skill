#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen ONE small-molecule target with the paper's fixed Strategy I.

Strategy I: independently select the Top 30% (including every boundary tie) by
AF3 ranking_score, Boltz-2 pocket pLDDT (5 Å), and Chai-1 ligand–protein
chain-pair ipTM (B). The final set is the union of the three selections.

No training labels, binding affinities or reference dataset are needed.
Python standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TOP_PERCENT = 30
# (score column from the user's original extractor, QC status, output column)
SPECS = {
    "af3": ("ranking_score", "af3_status", "af3_ranking_score"),
    "boltz2": ("boltz2_pocket_mean_plddt_5A", "boltz2_status", "boltz2_pocket_plddt_5A"),
    "chai1": ("chai_ligand_protein_chain_pair_iptm_B", "result_status", "chai_ligand_protein_chain_pair_iptm_B"),
}
GOOD_STATUSES = {"ok"}
AF3_AUXILIARY_WARNINGS = {
    'protein_plddt_fallback_atom_to_residue', 'protein_residue_plddt_unavailable',
    'pocket_residue_plddt_unavailable', 'pocket_plddt_fallback_atom_to_residue',
    'pocket_plddt_mixed_native_and_fallback', 'ligand_atom_plddt_unavailable',
}
SAMPLE_COLUMNS = {
    'af3': {'mode': 'score_mode', 'seed': 'ranking_score_seed', 'sample': 'ranking_score_sample'},
    'boltz2': {'mode': 'boltz2_score_mode', 'sample': 'boltz2_samples_used'},
    'chai1': {'mode': 'chai_score_mode', 'seed': 'seed', 'sample': 'model_idx'},
}


def validate_model_score(model: str, row: dict, *, metadata: bool = True) -> tuple[float, list[str]]:
    """Shared gates for raw output readiness, collection and selection. Never rewrite QC."""
    column, status_column, _ = SPECS[model]
    try:
        score = float(row.get(column, ''))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{model}: missing/invalid mandatory {column}') from exc
    if not math.isfinite(score):
        raise ValueError(f'{model}: nonfinite mandatory {column}')
    status = str(row.get(status_column) or '').strip()
    warnings = []
    if status.lower() not in GOOD_STATUSES:
        flags = status.split(';')
        if model != 'af3' or not flags or any(flag not in AF3_AUXILIARY_WARNINGS for flag in flags):
            raise ValueError(f'{model}: invalid/incomplete status {status!r}')
        warnings = flags
    if metadata:
        rule = SAMPLE_COLUMNS[model]
        if str(row.get(rule['mode']) or '').strip().lower() != 'single':
            raise ValueError(f'{model}: only single-mode scores are accepted')
        for attr, expected in (('seed', '1'), ('sample', '0')):
            key = rule.get(attr)
            if key and str(row.get(key) or '').strip() not in {expected, expected + '.0'}:
                raise ValueError(f'{model}: invalid {key}; expected {expected}')
        if model == 'boltz2' and str(row.get('boltz2_sample_status') or '').strip().lower() != 'model_0:ok':
            raise ValueError('boltz2: incomplete model_0 sample status')
    return score, warnings


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Empty/duplicate CSV headers: {path}")
        rows = list(reader)
        if any(None in r for r in rows):
            raise ValueError(f"Too many fields in at least one row of {path}")
        return rows, columns


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def index_model(path: Path, key_col: str, model: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows, columns = read_csv(path)
    score_col, status_col, _ = SPECS[model]
    rule = SAMPLE_COLUMNS[model]
    required = {key_col, score_col, status_col, *rule.values()}
    if model == 'boltz2':
        required.add('boltz2_sample_status')
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path}: missing {model} columns {sorted(missing)}")
    indexed = {}
    for row in rows:
        key = (row[key_col] or "").strip()
        if not key or key in indexed:
            raise ValueError(f"{path}: empty or duplicate {key_col}: {key!r}")
        validate_model_score(model, row)
        if (row.get(rule['mode']) or '').strip().lower() != 'single':
            raise ValueError(f'{model}/{key}: only single-mode scores are accepted')
        for attr, expected in (('seed', '1'), ('sample', '0')):
            column = rule.get(attr)
            if column and str(row.get(column) or '').strip() not in {expected, expected + '.0'}:
                raise ValueError(f'{model}/{key}: invalid {column}; expected {expected}')
        if model == 'boltz2' and (row.get('boltz2_sample_status') or '').strip().lower() != 'model_0:ok':
            raise ValueError(f'{model}/{key}: incomplete Boltz sample status')
        indexed[key] = row
    return indexed, columns


def validate_single_target(original: list[dict], original_cols: list[str], target_col: str) -> str:
    """Never silently mix multiple target groups in a single screening run."""
    labels = []
    for field in (target_col, "ligand_name", "ligand_smiles"):
        if field not in original_cols or field in labels:
            continue
        values = [(row.get(field) or "").strip() for row in original]
        if not all(values):
            raise ValueError(f"Single-target input contains blank {field} values")
        if len(set(values)) != 1:
            raise ValueError(f"Only ONE target is supported, but {field} contains {len(set(values))} values")
        labels.append(field)
    if not labels:
        raise ValueError("Need at least one target identifier: Target, ligand_name, or ligand_smiles")
    chosen = target_col if target_col in labels else ("ligand_name" if "ligand_name" in labels else "ligand_smiles")
    return (original[0][chosen] or "").strip()


def _screen(input_csv: Path, score_files: dict[str, Path], output_dir: Path,
           key_col: str = "task_id", target_col: str = "Target") -> dict:
    """Write selection CSVs/JSON; raise without selecting if any score is incomplete."""
    if set(score_files) != set(SPECS):
        raise ValueError("Exactly three score files required: af3, boltz2 and chai1")
    original, original_cols = read_csv(input_csv)
    if key_col not in original_cols:
        raise ValueError(f"Input CSV missing unique ID column: {key_col}")
    ids = [(r.get(key_col) or "").strip() for r in original]
    input_by_id = dict(zip(ids, original))
    if not ids or any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Input CSV must have at least one row and nonempty, unique task IDs")
    target_name = validate_single_target(original, original_cols, target_col)

    # Clean outputs before attempting a new run, so stale selections cannot look current.
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("selected_candidates.csv", "excluded_candidates.csv", "all_scores.csv",
                     "screening_summary.json", "qc_issues.csv"):
        (output_dir / filename).unlink(missing_ok=True)

    lookup = {}
    unused = {}
    for model, path in score_files.items():
        by_id, columns = index_model(path, key_col, model)
        if set(by_id) != set(ids):
            raise ValueError(f'{model}: score IDs must exactly match the input candidates')
        lookup[model] = by_id
        unused[model] = len(set(by_id) - set(ids))
        # Score extractors merge the original task metadata; verify it when available.
        for key in ids:
            result = by_id.get(key)
            if result is not None:
                source = input_by_id[key]
                for field in (target_col, "ligand_name", "ligand_smiles", "Protein_seq"):
                    if field in original_cols and field in columns:
                        if (result.get(field) or "").strip() != (source.get(field) or "").strip():
                            raise ValueError(f"{model} score task {key!r}: {field} disagrees with input")

    combined = []
    issues = []
    score_warnings = []
    for source in original:
        key = (source[key_col] or "").strip()
        row = dict(source)
        for model, (score_col, status_col, output_col) in SPECS.items():
            result = lookup[model].get(key)
            status = "MISSING_ROW" if result is None else (result.get(status_col) or "").strip()
            row[f"{model}_result_status"] = status
            score = None
            if result is not None:
                try:
                    score, warnings = validate_model_score(model, result)
                    score_warnings.extend({'task_id': key, 'model': model, 'source_status': status,
                                           'warning': warning} for warning in warnings)
                except (ValueError, TypeError):
                    pass
            if score is None:
                issues.append({"task_id": key, "model": model, "status": status,
                               "details": "Missing score, invalid status, incomplete mean5 or nonfinite value"})
                row[output_col] = ""
            else:
                row[output_col] = score
        combined.append(row)

    write_csv(output_dir / "qc_issues.csv", issues, ["task_id", "model", "status", "details"])
    write_csv(output_dir / 'score_warnings.csv', score_warnings,
              ['task_id', 'model', 'source_status', 'warning'])
    if issues:
        raise ValueError(f"Screening ABORTED: {len(issues)} score/status issues. "
                         f"See {output_dir / 'qc_issues.csv'}; no candidate selection produced.")

    n = len(combined)
    k = (TOP_PERCENT * n + 99) // 100  # ceil(0.30*n), exactly as original script
    selected_union: set[str] = set()
    thresholds = {}
    for model, (_, _, output_col) in SPECS.items():
        ordered = sorted(combined, key=lambda r: (-r[output_col], str(r[key_col])))
        cutoff = ordered[k - 1][output_col]
        selected = {str(row[key_col]).strip() for row in combined if row[output_col] >= cutoff}
        selected_union |= selected
        thresholds[model] = {
            "score_column": SPECS[model][0],
            "cutoff_score": cutoff,
            "nominal_top_k": k,
            "actual_retained": len(selected),
            "boundary_tie_inflation": len(selected) - k,
        }
        last_score = None
        min_rank = 0
        for position, row in enumerate(ordered, 1):
            if last_score != row[output_col]:
                min_rank = position
                last_score = row[output_col]
            row[f"{model}_rank"] = min_rank  # shared rank for tied scores
            row[f"{model}_top_selected"] = int(row[output_col] >= cutoff)

    for row in combined:
        row["selected_union"] = int(str(row[key_col]).strip() in selected_union)
        row["selected_by_models"] = ";".join(m for m in SPECS if row[f"{m}_top_selected"])
    appended = [v[2] for v in SPECS.values()]
    appended += [f"{m}_result_status" for m in SPECS]
    appended += [c for m in SPECS for c in (f"{m}_rank", f"{m}_top_selected")]
    appended += ["selected_by_models", "selected_union"]
    result_cols = list(dict.fromkeys(original_cols + appended))

    write_csv(output_dir / "all_scores.csv", combined, result_cols)
    selected = [r for r in combined if r["selected_union"]]
    excluded = [r for r in combined if not r["selected_union"]]
    write_csv(output_dir / "selected_candidates.csv", selected, result_cols)
    write_csv(output_dir / "excluded_candidates.csv", excluded, result_cols)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "Strategy I (A4 + B2 + C5)",
        "method": "Each model Top 30%, include all ties at cutoff, then union",
        "target": target_name,
        "target_count": 1,
        "top_percent": TOP_PERCENT,
        "auxiliary_warning_count": len(score_warnings),
        "input_count": n,
        "nominal_top_k_each_model": k,
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "reduction_fraction": len(excluded) / n,
        "selection": thresholds,
        "score_rows_outside_input_ignored": unused,
        "input_sha256": sha256(input_csv),
        "score_sha256": {model: sha256(path) for model, path in score_files.items()},
        "notes": ["This is a predictive candidate ranking procedure; no binding labels or Kd are used.",
                  "Candidate reduction describes this run only and does not guarantee experimental binding."],
    }
    (output_dir / "screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def screen(input_csv: Path, score_files: dict[str, Path], output_dir: Path,
           key_col: str = 'task_id', target_col: str = 'Target') -> dict:
    """Publish atomically; archive old selections before any attempted re-screen."""
    output_dir = Path(output_dir).expanduser().resolve()
    inputs = [Path(input_csv).resolve(), *[Path(p).resolve() for p in score_files.values()]]
    if any(p.is_relative_to(output_dir) for p in inputs):
        raise ValueError('Output directory must not contain the input or score CSVs')
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    def archive(path: Path, kind: str):
        base = output_dir.parent / 'backups' / kind
        base.mkdir(parents=True, exist_ok=True)
        dest = base / (output_dir.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        shutil.move(str(path), str(dest))
        return dest

    if output_dir.exists():
        if any(output_dir.iterdir()):
            archive(output_dir, 'screen_results')
        else:
            output_dir.rmdir()
    stage = Path(tempfile.mkdtemp(prefix='.selection-', dir=output_dir.parent))
    try:
        result = _screen(input_csv, score_files, stage, key_col, target_col)
        os.replace(stage, output_dir)
        return result
    except BaseException:
        dest = archive(stage, 'failed_screen')
        print(f'[FAILED SCREEN] Diagnostics preserved at {dest}', file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed Strategy I for ONE target, Top 30% union")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--af3-scores", type=Path, required=True)
    parser.add_argument("--boltz2-scores", type=Path, required=True)
    parser.add_argument("--chai1-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-col", default="task_id")
    parser.add_argument("--target-col", default="Target")
    args = parser.parse_args()
    result = screen(args.input_csv, {
        "af3": args.af3_scores,
        "boltz2": args.boltz2_scores,
        "chai1": args.chai1_scores,
    }, args.output_dir, args.id_col, args.target_col)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

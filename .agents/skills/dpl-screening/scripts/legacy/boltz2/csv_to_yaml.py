#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a CSV file into standard Boltz-2 YAML inputs.

Example:
    python csv_to_yaml.py \
      -i /path/to/project/candidates.csv \
      -o /path/to/project/boltz2/input \
      --name-col task_id \
      --seq-col Protein_seq \
      --smiles-col ligand_smiles \
      --no-msa
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with python -m pip install pyyaml."
    ) from exc


INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
VALID_SEQUENCE = re.compile(r"^[A-Z]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV rows into Boltz-2 YAML inputs for protein–ligand complexes."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for YAML files",
    )
    parser.add_argument(
        "--name-col",
        default="task_id",
        help="CSV column used as the YAML filename stem (default: task_id)",
    )
    parser.add_argument(
        "--seq-col",
        default="Protein_seq",
        help="CSV column containing the protein sequence (default: Protein_seq)",
    )
    parser.add_argument(
        "--smiles-col",
        default="ligand_smiles",
        help="CSV column containing the ligand SMILES (default: ligand_smiles)",
    )
    parser.add_argument(
        "--protein-id",
        default="A",
        help="Protein chain ID (default: A)",
    )
    parser.add_argument(
        "--ligand-id",
        default="B",
        help="Ligand chain ID, also used as the affinity binder (default: B)",
    )
    parser.add_argument(
        "--no-msa",
        action="store_true",
        help="Set msa: empty in the protein entry to force single-sequence mode",
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Omit the affinity property. Boltz-2 affinity prediction is enabled by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML files with the same name",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid rows and continue. By default, conversion stops at the first invalid row.",
    )
    parser.add_argument(
        "--no-rdkit-check",
        action="store_true",
        help="Skip RDKit SMILES validation and atom counting",
    )

    return parser.parse_args()


def normalize_task_name(value: Any, row_number: int) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Row {row_number}: the name column is empty")

    task_name = INVALID_FILENAME_CHARS.sub("_", raw).strip("._")
    if not task_name:
        raise ValueError(f"Row {row_number}: cannot generate a valid filename from {raw!r}")

    return task_name


def normalize_sequence(value: Any, row_number: int) -> str:
    sequence = re.sub(r"\s+", "", str(value or "")).upper()

    if not sequence:
        raise ValueError(f"Row {row_number}: the protein sequence is empty")

    if not VALID_SEQUENCE.fullmatch(sequence):
        raise ValueError(
            f"Row {row_number}: the protein sequence contains invalid characters: {sequence[:100]}"
        )

    return sequence


def normalize_smiles(value: Any, row_number: int) -> str:
    smiles = str(value or "").strip()

    if not smiles:
        raise ValueError(f"Row {row_number}: SMILES is empty")

    return smiles


def get_rdkit(disabled: bool):
    if disabled:
        return None

    try:
        from rdkit import Chem  # type: ignore
        return Chem
    except ImportError:
        print(
            "[WARNING] 当前环境没有 RDKit，自动跳过 SMILES 检查。",
            file=sys.stderr,
        )
        return None


def validate_smiles(smiles: str, row_number: int, chem_module) -> tuple[int | None, str]:
    if chem_module is None:
        return None, ""

    mol = chem_module.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Row {row_number}: RDKit could not parse the SMILES: {smiles}")

    atom_count = int(chem_module.RemoveHs(mol).GetNumAtoms())

    if atom_count > 128:
        raise ValueError(
            f"Row {row_number}: the ligand has {atom_count} atoms, "
            "exceeding the Boltz-2 affinity limit of 128"
        )

    warning = ""
    if atom_count > 56:
        warning = (
            f"配体原子数为 {atom_count}，大于 56；"
            "结构预测仍可运行，但 affinity 结果需谨慎解释。"
        )

    return atom_count, warning


def build_yaml(
    sequence: str,
    smiles: str,
    protein_id: str,
    ligand_id: str,
    no_msa: bool,
    enable_affinity: bool,
) -> dict[str, Any]:
    protein_data: dict[str, Any] = {
        "id": protein_id,
        "sequence": sequence,
    }

    if no_msa:
        protein_data["msa"] = "empty"

    document: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {
                "protein": protein_data,
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": smiles,
                }
            },
        ],
    }

    if enable_affinity:
        document["properties"] = [
            {
                "affinity": {
                    "binder": ligand_id,
                }
            }
        ]

    return document


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if args.protein_id == args.ligand_id:
        raise ValueError("protein-id and ligand-id must be different")

    output_dir.mkdir(parents=True, exist_ok=True)

    chem_module = get_rdkit(args.no_rdkit_check)

    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[tuple[int, str]] = []
    seen_names: set[str] = set()

    created = 0
    failed = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("The CSV has no header")

        required_columns = [
            args.name_col,
            args.seq_col,
            args.smiles_col,
        ]
        missing = [col for col in required_columns if col not in reader.fieldnames]

        if missing:
            raise ValueError(
                "Required CSV columns are missing: "
                + ", ".join(missing)
                + f"\nAvailable CSV columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                task_name = normalize_task_name(
                    row.get(args.name_col),
                    row_number,
                )

                if task_name in seen_names:
                    raise ValueError(f"Row {row_number}: duplicate task name: {task_name}")
                seen_names.add(task_name)

                sequence = normalize_sequence(
                    row.get(args.seq_col),
                    row_number,
                )
                smiles = normalize_smiles(
                    row.get(args.smiles_col),
                    row_number,
                )

                atom_count, warning = validate_smiles(
                    smiles,
                    row_number,
                    chem_module,
                )

                yaml_path = output_dir / f"{task_name}.yaml"

                if yaml_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"File already exists: {yaml_path}. Use --overwrite to replace it."
                    )

                yaml_data = build_yaml(
                    sequence=sequence,
                    smiles=smiles,
                    protein_id=args.protein_id,
                    ligand_id=args.ligand_id,
                    no_msa=args.no_msa,
                    enable_affinity=not args.no_affinity,
                )

                with yaml_path.open("w", encoding="utf-8", newline="\n") as out:
                    yaml.safe_dump(
                        yaml_data,
                        out,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False,
                        width=10000,
                    )

                manifest_row = dict(row)
                manifest_row.update(
                    {
                        "boltz2_task_name": task_name,
                        "boltz2_yaml_path": str(yaml_path),
                        "protein_length": len(sequence),
                        "ligand_atom_count": (
                            "" if atom_count is None else atom_count
                        ),
                        "msa_mode": "empty" if args.no_msa else "not_specified",
                        "affinity_enabled": not args.no_affinity,
                        "affinity_binder": (
                            args.ligand_id if not args.no_affinity else ""
                        ),
                        "conversion_warning": warning,
                    }
                )
                manifest_rows.append(manifest_row)

                created += 1

                message = (
                    f"[CREATED] {yaml_path.name} | "
                    f"protein={len(sequence)} aa"
                )
                if atom_count is not None:
                    message += f" | ligand_atoms={atom_count}"
                if warning:
                    message += f" | WARNING: {warning}"

                print(message, flush=True)

            except Exception as exc:
                failed += 1
                failed_rows.append((row_number, str(exc)))
                print(f"[FAILED] CSV 第 {row_number} 行：{exc}", file=sys.stderr)

                if not args.skip_invalid:
                    raise

    manifest_path = output_dir.parent / "boltz2_input_manifest.csv"
    failed_path = output_dir.parent / "boltz2_conversion_failed.tsv"

    if manifest_rows:
        fieldnames: list[str] = []
        for row in manifest_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    if failed_rows:
        with failed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["csv_row", "error"])
            writer.writerows(failed_rows)
    elif failed_path.exists():
        failed_path.unlink()

    print("=" * 80)
    print("Boltz-2 YAML conversion finished")
    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Name column: {args.name_col}")
    print(f"Sequence column: {args.seq_col}")
    print(f"SMILES column: {args.smiles_col}")
    print(f"MSA mode: {'empty' if args.no_msa else 'not specified'}")
    print(f"Affinity enabled: {not args.no_affinity}")
    print(f"Created: {created}")
    print(f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")
    if failed:
        print(f"Failed rows: {failed_path}")
    print("=" * 80)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

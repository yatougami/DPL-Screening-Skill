#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a CSV file into JSON inputs for AlphaFold 3.

Input and output conventions
1. The input CSV can have any filename.
2. Columns such as label, KD_nM and relation are optional.
3. The command-line options specify the following:
   - task ID column
   - protein name column
   - protein sequence column
   - ligand SMILES column
   - ligand name column (optional)
   - modelSeeds
4. Each row produces one task_id.json file.
5. The JSON name is set to task_id. AF3 uses the task name for its output directory.

Example
python csv_to_json.py \
    -i input.csv \
    -o af3_inputs \
    --task-id-col task_id \
    --protein-name-col Protein_ID \
    --protein-seq-col Protein_seq \
    --ligand-smiles-col ligand_smiles \
    --seeds 1

Multiple seeds
python csv_to_json.py -i input.csv --seeds 1,2,3,4,5
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VALID_AA_PATTERN = re.compile(r"^[A-Z]+$")


def safe_filename(name: str) -> str:
    """Convert a task_id into a filename-safe string."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed_task"


def clean_sequence_for_af3(sequence: str) -> str:
    """Remove whitespace and uppercase the sequence while preserving the amino acids."""
    return "".join(str(sequence).split()).upper()


def parse_seeds(seed_text: str) -> List[int]:
    """Parse a seed or comma-separated list of seeds, such as 1 or 1,2,3."""
    values: List[int] = []

    for item in str(seed_text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seed = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid seed {item!r}. Seeds must be integers.") from exc

        if seed < 0:
            raise ValueError(f"Invalid seed {seed}. Seeds must be nonnegative.")
        values.append(seed)

    if not values:
        raise ValueError("Specify at least one seed, for example --seeds 1.")

    # Remove duplicates while preserving input order.
    return list(dict.fromkeys(values))


def get_required_value(row: Dict[str, str], column: str) -> str:
    """Read a required column and strip leading and trailing whitespace."""
    return str(row.get(column, "") or "").strip()


def build_af3_json(
    task_id: str,
    protein_name: str,
    protein_sequence: str,
    ligand_smiles: str,
    ligand_name: Optional[str],
    seeds: List[int],
) -> Dict:
    """Build the JSON input for one AlphaFold 3 task."""
    protein_block = {
        "id": "A",
        "sequence": protein_sequence,
        # Disable MSA for single-sequence prediction of designed proteins.
        "unpairedMsa": "",
        "pairedMsa": "",
        # Use no templates.
        "templates": [],
    }

    if protein_name:
        protein_block["description"] = protein_name

    ligand_block = {
        "id": "B",
        "smiles": ligand_smiles,
    }

    if ligand_name:
        ligand_block["description"] = ligand_name

    return {
        "name": task_id,
        "modelSeeds": seeds,
        "sequences": [
            {"protein": protein_block},
            {"ligand": ligand_block},
        ],
        "dialect": "alphafold3",
        "version": 4,
    }


def check_columns(fieldnames: Optional[Iterable[str]], required_columns: List[str]) -> None:
    """Check only the CSV columns used for this conversion."""
    if not fieldnames:
        raise ValueError("The CSV has no header. Columns cannot be read by name.")

    available = list(fieldnames)
    missing = [column for column in required_columns if column not in available]

    if missing:
        raise ValueError(
            "The CSV is missing the following columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
            + "\n\nAvailable CSV columns:\n  "
            + ", ".join(available)
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to AlphaFold 3 JSON input files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="af3_inputs",
        help="Output directory for JSON files",
    )

    parser.add_argument(
        "--task-id-col",
        default="task_id",
        help="Column used for the JSON name and filename stem",
    )
    parser.add_argument(
        "--protein-name-col",
        default="Protein_ID",
        help="Protein or sequence name column. Pass an empty string to omit the description.",
    )
    parser.add_argument(
        "--protein-seq-col",
        default="Protein_seq",
        help="Column containing the protein amino acid sequence",
    )
    parser.add_argument(
        "--ligand-smiles-col",
        default="ligand_smiles",
        help="Column containing the ligand SMILES",
    )
    parser.add_argument(
        "--ligand-name-col",
        default="",
        help="Optional ligand name column. If omitted, the CSV does not need a ligand_name column.",
    )
    parser.add_argument(
        "--seeds",
        default="1",
        help="Model seeds separated by commas, for example 1,2,3,4,5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files with the same name. The default is to skip them.",
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    seeds = parse_seeds(args.seeds)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [
        args.task_id_col,
        args.protein_seq_col,
        args.ligand_smiles_col,
    ]

    if args.protein_name_col:
        required_columns.append(args.protein_name_col)
    if args.ligand_name_col:
        required_columns.append(args.ligand_name_col)

    # Check each column only once.
    required_columns = list(dict.fromkeys(required_columns))

    generated = 0
    skipped_invalid = 0
    skipped_existing = 0
    seen_filenames = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        check_columns(reader.fieldnames, required_columns)

        for line_number, row in enumerate(reader, start=2):
            task_id = get_required_value(row, args.task_id_col)
            protein_sequence = clean_sequence_for_af3(
                get_required_value(row, args.protein_seq_col)
            )
            ligand_smiles = get_required_value(row, args.ligand_smiles_col)

            protein_name = (
                get_required_value(row, args.protein_name_col)
                if args.protein_name_col
                else ""
            )
            ligand_name = (
                get_required_value(row, args.ligand_name_col)
                if args.ligand_name_col
                else ""
            )

            missing_values = []
            if not task_id:
                missing_values.append(args.task_id_col)
            if not protein_sequence:
                missing_values.append(args.protein_seq_col)
            if not ligand_smiles:
                missing_values.append(args.ligand_smiles_col)

            if missing_values:
                print(
                    f"[跳过] CSV 第 {line_number} 行缺少必需内容："
                    + ", ".join(missing_values)
                )
                skipped_invalid += 1
                continue

            if not VALID_AA_PATTERN.fullmatch(protein_sequence):
                print(
                    f"[警告] {task_id}: 蛋白序列含有非英文字母字符，"
                    "仍会按原内容写入，请检查该序列。"
                )

            filename = safe_filename(task_id) + ".json"

            if filename in seen_filenames:
                print(
                    f"[跳过] CSV 第 {line_number} 行：task_id {task_id!r} "
                    f"与前面的任务生成了重复文件名 {filename!r}"
                )
                skipped_invalid += 1
                continue

            seen_filenames.add(filename)
            output_path = output_dir / filename

            if output_path.exists() and not args.overwrite:
                print(f"[跳过已有] {output_path.name}")
                skipped_existing += 1
                continue

            af3_json = build_af3_json(
                task_id=task_id,
                protein_name=protein_name,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles,
                ligand_name=ligand_name or None,
                seeds=seeds,
            )

            with output_path.open("w", encoding="utf-8") as handle:
                # json.dump escapes backslashes in SMILES strings.
                json.dump(af3_json, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            generated += 1

    print("\n转换完成")
    print(f"输入 CSV：{input_csv}")
    print(f"输出目录：{output_dir}")
    print(f"成功生成：{generated}")
    print(f"无效行跳过：{skipped_invalid}")
    print(f"已有文件跳过：{skipped_existing}")
    print(f"modelSeeds：{seeds}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
from pathlib import Path


def safe_filename(name):
    """
    Convert the CSV filename column to a filename-safe string.
    A name such as bind_0001 already meets the naming rules.
    Replace whitespace, slashes and other special characters with underscores.
    """
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._-")
    return name


def clean_protein_sequence(seq):
    """
    Clean the protein sequence.
    Remove spaces, line breaks and tabs, and convert letters to uppercase.
    Preserve the amino acid sequence.
    """
    seq = str(seq).strip()
    seq = re.sub(r"\s+", "", seq)
    return seq.upper()


def clean_smiles(smiles):
    """
    Clean the SMILES string.
    Remove leading, trailing and internal whitespace.
    Preserve the molecular structure encoded by the SMILES.
    """
    smiles = str(smiles).strip()
    smiles = re.sub(r"\s+", "", smiles)
    return smiles


def write_chai_input(out_file, task_name, protein_seq, ligand_smiles):
    """
    Write one FASTA-like input file for Chai-1.
    """
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f">protein|name={task_name}_protein\n")
        f.write(f"{protein_seq}\n\n")
        f.write(f">ligand|name={task_name}_ligand\n")
        f.write(f"{ligand_smiles}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV rows to Chai-1 input fasta files."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for Chai-1 input files"
    )

    parser.add_argument(
        "--name-col",
        required=True,
        help="CSV column used as the output filename stem, for example task_id"
    )

    parser.add_argument(
        "--protein-col",
        required=True,
        help="CSV column containing the protein sequence, for example Protein_seq"
    )

    parser.add_argument(
        "--smiles-col",
        required=True,
        help="CSV column containing the ligand SMILES, for example ligand_smiles"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] 找不到 CSV 文件：{csv_path}", file=sys.stderr)
        sys.exit(1)

    used_names = set()
    n_total = 0
    n_written = 0
    n_skipped = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            print("[ERROR] CSV 没有表头。", file=sys.stderr)
            sys.exit(1)

        headers = reader.fieldnames

        required_cols = [
            args.name_col,
            args.protein_col,
            args.smiles_col,
        ]

        missing = [c for c in required_cols if c not in headers]

        if missing:
            print("[ERROR] CSV 里找不到这些列名：", file=sys.stderr)
            for c in missing:
                print(f"  - {c}", file=sys.stderr)

            print("\n[INFO] 当前 CSV 实际表头是：", file=sys.stderr)
            for h in headers:
                print(f"  - {h}", file=sys.stderr)

            sys.exit(1)

        for row_idx, row in enumerate(reader, start=2):
            n_total += 1

            raw_name = row.get(args.name_col, "")
            raw_protein = row.get(args.protein_col, "")
            raw_smiles = row.get(args.smiles_col, "")

            task_name = safe_filename(raw_name)
            protein_seq = clean_protein_sequence(raw_protein)
            ligand_smiles = clean_smiles(raw_smiles)

            if not task_name:
                print(f"[WARN] 第 {row_idx} 行文件名为空，跳过。", file=sys.stderr)
                n_skipped += 1
                continue

            if not protein_seq:
                print(f"[WARN] 第 {row_idx} 行蛋白序列为空，跳过：{raw_name}", file=sys.stderr)
                n_skipped += 1
                continue

            if not ligand_smiles:
                print(f"[WARN] 第 {row_idx} 行 SMILES 为空，跳过：{raw_name}", file=sys.stderr)
                n_skipped += 1
                continue

            final_name = task_name

            if final_name in used_names:
                final_name = f"{task_name}_row{row_idx}"
                print(
                    f"[WARN] 文件名重复：{task_name}，已改为：{final_name}",
                    file=sys.stderr
                )

            used_names.add(final_name)

            out_file = out_dir / f"{final_name}.fasta"

            write_chai_input(
                out_file=out_file,
                task_name=final_name,
                protein_seq=protein_seq,
                ligand_smiles=ligand_smiles,
            )

            n_written += 1

    print("[DONE] CSV 已转换为 Chai-1 读入文件")
    print(f"[INFO] CSV 文件：{csv_path}")
    print(f"[INFO] 输出目录：{out_dir}")
    print(f"[INFO] 总行数：{n_total}")
    print(f"[INFO] 成功生成：{n_written}")
    print(f"[INFO] 跳过：{n_skipped}")


if __name__ == "__main__":
    main()
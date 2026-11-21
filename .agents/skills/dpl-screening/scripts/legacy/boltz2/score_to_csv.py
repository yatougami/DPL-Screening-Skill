#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_to_csv.py

Purpose
----
1. Read the Boltz-2 output directory. Each task uses the following path:

       <input_dir>/<task>/predictions/<task>/

   Example:

       ALL_cleaned/bind_0001/predictions/bind_0001/

2. Two scoring modes are available:
   - --mode single reads one model_N.
   - --mode mean5 reads model_0 through model_4 and averages each numerical score.
     Each structure is scored using its own coordinates.

3. Extract confidence, pLDDT, PAE, PDE, affinity and geometry scores.
   Match task directory names to the specified input CSV column when merging scores.

4. Calculate mean residue pLDDT for the 5 Å pocket:
   - Use the affinity ligand in the predicted structure as the reference.
   - Select protein residues with a heavy atom within --pocket-cutoff of any ligand heavy atom.
   - Use one native per-residue pLDDT token per residue.
   - Give each pocket residue equal weight, regardless of its atom count.

5. Calculate overall protein confidence as a mean over residues:
   - Each Boltz-2 protein pLDDT token corresponds to one residue.
   - Each protein residue contributes one value with equal weight.
   - Store the result in boltz2_protein_mean_plddt.
   Average ligand pLDDT over atom-level tokens and store it in boltz2_ligand_mean_plddt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


PROTEIN_MOL_TYPE = 0
LIGAND_MOL_TYPE = 3
CONTACT_CUTOFF_ANGSTROM = 4.0


@dataclass
class PDBAtom:
    record: str
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: str
    insertion_code: str
    element: str
    coord: np.ndarray
    bfactor: float

    @property
    def residue_key(self) -> tuple[str, str, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in {"H", "D"}


@dataclass
class ChainInfo:
    index: int
    name: str
    mol_type: int
    role: str
    token_indices: np.ndarray
    atom_weights: np.ndarray

    @property
    def atom_count(self) -> int:
        return int(np.sum(self.atom_weights))


def natural_key(text: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Boltz-2 scores from one model or the mean of five models, "
            "calculate residue-weighted protein and pocket pLDDT, and append scores to an original CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help=(
            "Boltz-2 output directory containing task directories such as bind_0001 and nobind_0001."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        help="Output CSV containing the merged scores",
    )
    parser.add_argument(
        "--merge-csv",
        required=True,
        help="Input CSV to which Boltz-2 scores will be appended",
    )
    parser.add_argument(
        "--merge-key",
        required=True,
        help="Column in the input CSV that matches task directory names, for example task_id",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "mean5"],
        default="single",
        help="single reads one model. mean5 averages numerical scores across five individually scored models.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Model index N to read as model_N in single mode (default: 0). Ignored in mean5 mode.",
    )
    parser.add_argument(
        "--mean5-samples",
        type=int,
        nargs=5,
        metavar=("S0", "S1", "S2", "S3", "S4"),
        default=[0, 1, 2, 3, 4],
        help="Five model indices to use in mean5 mode (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        default=5.0,
        help="Pocket distance cutoff between predicted ligand and protein heavy atoms, in Å (default: 5.0).",
    )
    parser.add_argument(
        "--float-digits",
        type=int,
        default=6,
        help="Decimal places for newly extracted floating-point scores (default: 6)",
    )
    parser.add_argument(
        "--ligand-chain-id",
        default=None,
        help=(
            "Optional affinity ligand chain ID. Usually inferred from input_used.yaml "
            "or from the unique ligand chain."
        ),
    )
    parser.add_argument(
        "--include-unmatched-results",
        action="store_true",
        help="Append tasks found in the output directory that have no matching input CSV row",
    )
    return parser.parse_args()


def read_csv_dicts(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader), list(reader.fieldnames)


def find_file(preferred_dir: Path, exact_names: Iterable[str], patterns: Iterable[str]) -> Path | None:
    for name in exact_names:
        path = preferred_dir / name
        if path.is_file():
            return path

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(preferred_dir.glob(pattern))

    candidates = sorted(set(candidates), key=lambda path: natural_key(path.name))
    return candidates[0] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at the top level of {path}")
    return data


def load_npz_array(path: Path, preferred_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if preferred_key in archive.files:
            return np.asarray(archive[preferred_key])
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])
        raise KeyError(
            f"{path} has no key {preferred_key!r}. Available keys: {archive.files}"
        )


def parse_pdb_atoms(path: Path) -> list[PDBAtom]:
    atoms: list[PDBAtom] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue

            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue

            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26:27].strip()

            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValueError(f"Cannot parse PDB coordinates in {path}:\n{line.rstrip()}") from exc

            try:
                bfactor = float(line[60:66])
            except ValueError:
                bfactor = float("nan")

            element = line[76:78].strip()
            if not element:
                letters = re.sub(r"[^A-Za-z]", "", atom_name)
                element = letters[:1] if letters else ""

            atoms.append(
                PDBAtom(
                    record=record,
                    atom_name=atom_name,
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=residue_number,
                    insertion_code=insertion_code,
                    element=element,
                    coord=coord,
                    bfactor=bfactor,
                )
            )

    if not atoms:
        raise ValueError(f"No ATOM or HETATM records found in {path}")

    return atoms


def get_affinity_binder_from_yaml(task_dir: Path) -> str | None:
    if yaml is None:
        return None

    candidates = [
        task_dir / "input_used.yaml",
        task_dir / "input_used.yml",
    ]

    for path in candidates:
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)

        if not isinstance(document, dict):
            continue

        properties = document.get("properties")
        if not isinstance(properties, list):
            continue

        binders: list[str] = []
        for item in properties:
            if not isinstance(item, dict):
                continue
            affinity = item.get("affinity")
            if isinstance(affinity, dict) and affinity.get("binder") is not None:
                binders.append(str(affinity["binder"]))

        if len(binders) == 1:
            return binders[0]

    return None


def infer_binder_chain(
    task_dir: Path,
    explicit_chain: str | None,
    pre_chains: np.ndarray | None,
    pdb_atoms: list[PDBAtom],
) -> str:
    if explicit_chain:
        return explicit_chain

    yaml_binder = get_affinity_binder_from_yaml(task_dir)
    if yaml_binder:
        return yaml_binder

    if pre_chains is not None:
        ligand_names = [
            str(chain["name"])
            for chain in pre_chains
            if int(chain["mol_type"]) == LIGAND_MOL_TYPE
        ]
        if len(ligand_names) == 1:
            return ligand_names[0]

    hetatm_chains = sorted(
        {atom.chain_id for atom in pdb_atoms if atom.record == "HETATM"}
    )
    if len(hetatm_chains) == 1:
        return hetatm_chains[0]

    raise ValueError(
        "Cannot identify a unique affinity ligand chain. Specify --ligand-chain-id."
    )


def chain_infos_from_pre_affinity(
    pre_affinity_path: Path,
    binder_chain: str,
    expected_token_count: int,
) -> list[ChainInfo]:
    with np.load(pre_affinity_path, allow_pickle=True) as archive:
        if "chains" not in archive.files or "residues" not in archive.files:
            raise KeyError(
                f"{pre_affinity_path} is missing the chains or residues array"
            )
        chains = np.asarray(archive["chains"])
        residues = np.asarray(archive["residues"])

    infos: list[ChainInfo] = []
    token_offset = 0

    for chain_index, chain in enumerate(chains):
        name = str(chain["name"])
        mol_type = int(chain["mol_type"])
        atom_num = int(chain["atom_num"])
        res_idx = int(chain["res_idx"])
        res_num = int(chain["res_num"])

        if mol_type == PROTEIN_MOL_TYPE:
            token_count = res_num
            residue_slice = residues[res_idx : res_idx + res_num]
            atom_weights = np.asarray(
                residue_slice["atom_num"],
                dtype=np.int64,
            )
            role = "protein"
        elif name == binder_chain or mol_type == LIGAND_MOL_TYPE:
            # Boltz-2 represents a non-polymer ligand with one token per atom.
            token_count = atom_num
            atom_weights = np.ones(token_count, dtype=np.int64)
            role = "ligand" if name == binder_chain else "other_ligand"
        else:
            token_count = res_num
            atom_weights = np.asarray(
                residues[res_idx : res_idx + res_num]["atom_num"],
                dtype=np.int64,
            )
            role = "other"

        token_indices = np.arange(
            token_offset,
            token_offset + token_count,
            dtype=np.int64,
        )
        token_offset += token_count

        infos.append(
            ChainInfo(
                index=chain_index,
                name=name,
                mol_type=mol_type,
                role=role,
                token_indices=token_indices,
                atom_weights=atom_weights,
            )
        )

    if token_offset != expected_token_count:
        raise ValueError(
            "The pre_affinity chain token count does not match the pLDDT/PAE dimensions: "
            f"{token_offset} != {expected_token_count}"
        )

    return infos


def chain_infos_from_pdb(
    pdb_atoms: list[PDBAtom],
    binder_chain: str,
    expected_token_count: int,
) -> list[ChainInfo]:
    chain_order: list[str] = []
    for atom in pdb_atoms:
        if atom.chain_id not in chain_order:
            chain_order.append(atom.chain_id)

    infos: list[ChainInfo] = []
    token_offset = 0

    for chain_index, chain_name in enumerate(chain_order):
        chain_atoms = [atom for atom in pdb_atoms if atom.chain_id == chain_name]
        atom_records = [atom for atom in chain_atoms if atom.record == "ATOM"]
        het_records = [atom for atom in chain_atoms if atom.record == "HETATM"]

        if chain_name == binder_chain:
            token_count = len(het_records) if het_records else len(chain_atoms)
            atom_weights = np.ones(token_count, dtype=np.int64)
            role = "ligand"
            mol_type = LIGAND_MOL_TYPE
        elif atom_records:
            residue_order: list[tuple[str, str, str, str]] = []
            residue_counts: dict[tuple[str, str, str, str], int] = {}
            for atom in atom_records:
                key = atom.residue_key
                if key not in residue_counts:
                    residue_order.append(key)
                    residue_counts[key] = 0
                residue_counts[key] += 1

            atom_weights = np.array(
                [residue_counts[key] for key in residue_order],
                dtype=np.int64,
            )
            token_count = len(residue_order)
            role = "protein"
            mol_type = PROTEIN_MOL_TYPE
        else:
            token_count = len(chain_atoms)
            atom_weights = np.ones(token_count, dtype=np.int64)
            role = "other_ligand"
            mol_type = LIGAND_MOL_TYPE

        token_indices = np.arange(
            token_offset,
            token_offset + token_count,
            dtype=np.int64,
        )
        token_offset += token_count

        infos.append(
            ChainInfo(
                index=chain_index,
                name=chain_name,
                mol_type=mol_type,
                role=role,
                token_indices=token_indices,
                atom_weights=atom_weights,
            )
        )

    if token_offset != expected_token_count:
        raise ValueError(
            "The token count inferred from the PDB does not match the pLDDT/PAE dimensions: "
            f"{token_offset} != {expected_token_count}"
        )

    return infos


def weighted_values(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.int64)

    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be one-dimensional and have the same length")
    if np.any(weights < 0):
        raise ValueError("atom_weights must be nonnegative")

    return np.repeat(values, weights)


def summary_statistics(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {}

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_max": float(np.max(values)),
    }


def combine_indices(infos: list[ChainInfo], role: str) -> np.ndarray:
    arrays = [info.token_indices for info in infos if info.role == role]
    if not arrays:
        return np.array([], dtype=np.int64)
    return np.concatenate(arrays)


def get_matrix_value(
    matrix_dict: dict[str, Any],
    row_index: int,
    column_index: int,
) -> float:
    row = matrix_dict.get(str(row_index), matrix_dict.get(row_index))
    if not isinstance(row, dict):
        return float("nan")
    value = row.get(str(column_index), row.get(column_index))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def extract_confidence_metrics(
    confidence: dict[str, Any],
    chain_infos: list[ChainInfo],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in confidence.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"boltz2_{key}"] = float(value)

    chains_ptm = confidence.get("chains_ptm")
    if isinstance(chains_ptm, dict):
        for info in chain_infos:
            value = chains_ptm.get(str(info.index), chains_ptm.get(info.index))
            if value is not None:
                out[f"boltz2_chain_ptm_{info.name}"] = float(value)

        protein_values = [
            get_matrix_value(
                {str(k): {"0": v} for k, v in chains_ptm.items()},
                info.index,
                0,
            )
            for info in chain_infos
            if info.role == "protein"
        ]
        protein_values = [value for value in protein_values if math.isfinite(value)]
        if protein_values:
            out["boltz2_protein_chain_ptm_mean"] = float(np.mean(protein_values))

        ligand_infos = [info for info in chain_infos if info.role == "ligand"]
        if len(ligand_infos) == 1:
            ligand_info = ligand_infos[0]
            value = chains_ptm.get(
                str(ligand_info.index),
                chains_ptm.get(ligand_info.index),
            )
            if value is not None:
                out["boltz2_ligand_chain_ptm"] = float(value)

    pair_matrix = confidence.get("pair_chains_iptm")
    if isinstance(pair_matrix, dict):
        for row_info in chain_infos:
            for column_info in chain_infos:
                value = get_matrix_value(
                    pair_matrix,
                    row_info.index,
                    column_info.index,
                )
                if math.isfinite(value):
                    out[
                        "boltz2_pair_chain_iptm_"
                        f"{row_info.name}_to_{column_info.name}"
                    ] = value

        protein_infos = [info for info in chain_infos if info.role == "protein"]
        ligand_infos = [info for info in chain_infos if info.role == "ligand"]

        if protein_infos and len(ligand_infos) == 1:
            ligand = ligand_infos[0]

            # A: ligand row -> protein column
            direction_a = [
                get_matrix_value(pair_matrix, ligand.index, protein.index)
                for protein in protein_infos
            ]
            # B: protein row -> ligand column
            direction_b = [
                get_matrix_value(pair_matrix, protein.index, ligand.index)
                for protein in protein_infos
            ]

            direction_a = [value for value in direction_a if math.isfinite(value)]
            direction_b = [value for value in direction_b if math.isfinite(value)]

            if direction_a:
                out[
                    "boltz2_ligand_protein_chain_pair_iptm_A"
                ] = float(np.mean(direction_a))
            if direction_b:
                out[
                    "boltz2_ligand_protein_chain_pair_iptm_B"
                ] = float(np.mean(direction_b))
            if direction_a and direction_b:
                out[
                    "boltz2_ligand_protein_chain_pair_iptm_bidirectional_mean"
                ] = float(np.mean(direction_a + direction_b))

    if {
        "confidence_score",
        "complex_plddt",
        "iptm",
    }.issubset(confidence):
        recomputed = (
            0.8 * float(confidence["complex_plddt"])
            + 0.2 * float(confidence["iptm"])
        )
        out["boltz2_confidence_score_recomputed"] = recomputed
        out["boltz2_confidence_score_recomputed_abs_diff"] = abs(
            recomputed - float(confidence["confidence_score"])
        )

    return out


def extract_plddt_metrics(
    plddt: np.ndarray,
    chain_infos: list[ChainInfo],
) -> dict[str, Any]:
    """Extract pLDDT statistics with equal residue weights for protein.

    Boltz-2 uses one pLDDT token per protein residue and one token per ligand
    atom. Therefore protein confidence is calculated directly from protein
    token values without repeating them by residue atom count. Ligand
    confidence remains an atom-level mean.
    """
    plddt = np.asarray(plddt, dtype=np.float64).reshape(-1)

    out: dict[str, Any] = {
        "boltz2_token_count": int(plddt.size),
    }

    # Retained only for the original complex-wide atom-weighted diagnostic.
    all_atom_values: list[np.ndarray] = []

    # Protein: one token = one residue, so these are residue-equally weighted.
    protein_residue_values: list[np.ndarray] = []

    # Ligand: one token = one atom, so these remain atom-level values.
    ligand_atom_values: list[np.ndarray] = []

    for info in chain_infos:
        chain_token_values = np.asarray(
            plddt[info.token_indices],
            dtype=np.float64,
        )
        chain_token_values = chain_token_values[np.isfinite(chain_token_values)]

        # Complex-wide atom-weighted diagnostic is retained for compatibility.
        chain_atom_values = weighted_values(
            plddt[info.token_indices],
            info.atom_weights,
        )
        all_atom_values.append(chain_atom_values)

        out[f"boltz2_chain_{info.name}_token_count"] = int(
            len(info.token_indices)
        )
        out[f"boltz2_chain_{info.name}_atom_count"] = int(info.atom_count)

        if info.role == "protein":
            if chain_token_values.size:
                out[f"boltz2_chain_{info.name}_residue_mean_plddt"] = float(
                    np.mean(chain_token_values)
                )
                protein_residue_values.append(chain_token_values)
        elif info.role == "ligand":
            if chain_token_values.size:
                out[f"boltz2_chain_{info.name}_atom_mean_plddt"] = float(
                    np.mean(chain_token_values)
                )
                ligand_atom_values.append(chain_token_values)
        else:
            if chain_token_values.size:
                out[f"boltz2_chain_{info.name}_token_mean_plddt"] = float(
                    np.mean(chain_token_values)
                )

    if all_atom_values:
        all_atoms = np.concatenate(all_atom_values)
        out.update(
            summary_statistics(
                "boltz2_complex_atom_plddt",
                all_atoms,
            )
        )

    if protein_residue_values:
        protein_residues = np.concatenate(protein_residue_values)
        protein_stats = summary_statistics(
            "boltz2_protein_residue_plddt",
            protein_residues,
        )
        out.update(protein_stats)
        out["boltz2_protein_mean_plddt"] = float(np.mean(protein_residues))
        out["boltz2_protein_residue_count"] = int(protein_residues.size)
        out["boltz2_protein_plddt_source"] = "NPZ_PER_RESIDUE_TOKEN_EQUAL_WEIGHT"

    if ligand_atom_values:
        ligand_atoms = np.concatenate(ligand_atom_values)
        ligand_stats = summary_statistics(
            "boltz2_ligand_atom_plddt",
            ligand_atoms,
        )
        out.update(ligand_stats)
        out["boltz2_ligand_mean_plddt"] = float(np.mean(ligand_atoms))
        out["boltz2_ligand_atom_count"] = int(ligand_atoms.size)
        out["boltz2_ligand_plddt_source"] = "NPZ_PER_LIGAND_ATOM_TOKEN"

    return out


def matrix_off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        return matrix.reshape(-1)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def extract_error_matrix_metrics(
    matrix: np.ndarray,
    matrix_name: str,
    chain_infos: list[ChainInfo],
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{matrix_name.upper()} must be a square matrix. Actual shape: {matrix.shape}"
        )

    protein_indices = combine_indices(chain_infos, "protein")
    ligand_indices = combine_indices(chain_infos, "ligand")

    if protein_indices.size == 0 or ligand_indices.size == 0:
        raise ValueError(
            f"Cannot summarize {matrix_name.upper()}: missing protein or affinity ligand"
        )

    out: dict[str, Any] = {}
    prefix = f"boltz2_{matrix_name}"

    out.update(
        summary_statistics(
            f"{prefix}_complex_offdiagonal",
            matrix_off_diagonal_values(matrix),
        )
    )

    protein_block = matrix[np.ix_(protein_indices, protein_indices)]
    out.update(
        summary_statistics(
            f"{prefix}_protein_self_offdiagonal",
            matrix_off_diagonal_values(protein_block),
        )
    )

    ligand_block = matrix[np.ix_(ligand_indices, ligand_indices)]
    out.update(
        summary_statistics(
            f"{prefix}_ligand_self_offdiagonal",
            matrix_off_diagonal_values(ligand_block),
        )
    )

    # A: ligand row -> protein column
    ligand_to_protein = matrix[np.ix_(ligand_indices, protein_indices)].reshape(-1)
    # B: protein row -> ligand column
    protein_to_ligand = matrix[np.ix_(protein_indices, ligand_indices)].reshape(-1)
    bidirectional = np.concatenate([ligand_to_protein, protein_to_ligand])

    out.update(
        summary_statistics(
            f"{prefix}_ligand_to_protein_A",
            ligand_to_protein,
        )
    )
    out.update(
        summary_statistics(
            f"{prefix}_protein_to_ligand_B",
            protein_to_ligand,
        )
    )
    out.update(
        summary_statistics(
            f"{prefix}_ligand_protein_bidirectional",
            bidirectional,
        )
    )

    # “Ligand-specific” = matrix elements where at least one endpoint is ligand.
    ligand_mask = np.zeros(matrix.shape[0], dtype=bool)
    ligand_mask[ligand_indices] = True
    ligand_specific_mask = ligand_mask[:, None] | ligand_mask[None, :]
    ligand_specific_values = matrix[ligand_specific_mask]

    out.update(
        summary_statistics(
            f"{prefix}_ligand_specific",
            ligand_specific_values,
        )
    )

    return out


def extract_affinity_metrics(affinity: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in affinity.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"boltz2_{key}"] = float(value)

    for suffix in ("", "1", "2"):
        key = f"affinity_pred_value{suffix}"
        if key not in affinity:
            continue

        value = float(affinity[key])
        out[f"boltz2_affinity_pred_ic50_uM{suffix}"] = float(10.0 ** value)
        out[f"boltz2_affinity_pred_pIC50{suffix}"] = float(6.0 - value)

    if {
        "affinity_pred_value1",
        "affinity_pred_value2",
    }.issubset(affinity):
        out["boltz2_affinity_pred_value_model_disagreement_abs"] = abs(
            float(affinity["affinity_pred_value1"])
            - float(affinity["affinity_pred_value2"])
        )

    if {
        "affinity_probability_binary1",
        "affinity_probability_binary2",
    }.issubset(affinity):
        out["boltz2_affinity_probability_model_disagreement_abs"] = abs(
            float(affinity["affinity_probability_binary1"])
            - float(affinity["affinity_probability_binary2"])
        )

    return out


def extract_pdb_geometry_metrics(
    pdb_atoms: list[PDBAtom],
    binder_chain: str,
    protein_chain_names: set[str],
) -> dict[str, Any]:
    protein_atoms = [
        atom
        for atom in pdb_atoms
        if atom.record == "ATOM"
        and atom.chain_id in protein_chain_names
        and not atom.is_hydrogen
    ]
    ligand_atoms = [
        atom
        for atom in pdb_atoms
        if atom.record == "HETATM"
        and atom.chain_id == binder_chain
        and not atom.is_hydrogen
    ]

    out: dict[str, Any] = {
        "boltz2_pdb_protein_heavy_atom_count": len(protein_atoms),
        "boltz2_pdb_ligand_heavy_atom_count": len(ligand_atoms),
    }

    if not protein_atoms or not ligand_atoms:
        return out

    protein_coords = np.vstack([atom.coord for atom in protein_atoms])
    ligand_coords = np.vstack([atom.coord for atom in ligand_atoms])

    # Shape: ligand heavy atoms x protein heavy atoms.
    displacement = (
        ligand_coords[:, None, :]
        - protein_coords[None, :, :]
    )
    distances = np.linalg.norm(displacement, axis=2)

    ligand_nearest = np.min(distances, axis=1)
    protein_nearest = np.min(distances, axis=0)
    contact_mask = distances <= CONTACT_CUTOFF_ANGSTROM

    contacting_ligand = np.any(contact_mask, axis=1)
    contacting_protein = np.any(contact_mask, axis=0)

    contacting_residues = {
        protein_atoms[index].residue_key
        for index in np.flatnonzero(contacting_protein)
    }

    out.update(
        {
            "boltz2_ligand_protein_min_heavy_atom_distance_A": float(
                np.min(distances)
            ),
            "boltz2_ligand_mean_nearest_protein_heavy_atom_distance_A": float(
                np.mean(ligand_nearest)
            ),
            "boltz2_ligand_median_nearest_protein_heavy_atom_distance_A": float(
                np.median(ligand_nearest)
            ),
            "boltz2_protein_mean_nearest_ligand_heavy_atom_distance_A": float(
                np.mean(protein_nearest)
            ),
            "boltz2_ligand_protein_heavy_atom_pair_contacts_4A": int(
                np.count_nonzero(contact_mask)
            ),
            "boltz2_ligand_contact_heavy_atom_count_4A": int(
                np.count_nonzero(contacting_ligand)
            ),
            "boltz2_ligand_contact_heavy_atom_fraction_4A": float(
                np.mean(contacting_ligand)
            ),
            "boltz2_protein_contact_heavy_atom_count_4A": int(
                np.count_nonzero(contacting_protein)
            ),
            "boltz2_protein_contact_residue_count_4A": int(
                len(contacting_residues)
            ),
            "boltz2_contacts_per_ligand_heavy_atom_4A": float(
                np.count_nonzero(contact_mask) / len(ligand_atoms)
            ),
        }
    )

    return out


def _pdb_protein_residues_by_chain(
    pdb_atoms: list[PDBAtom],
    protein_chain_names: set[str],
) -> dict[str, dict[tuple[str, str, str, str], list[PDBAtom]]]:
    """Collect protein residues in PDB encounter order, retaining heavy atoms only."""
    residues_by_chain: dict[
        str,
        dict[tuple[str, str, str, str], list[PDBAtom]],
    ] = {}

    for atom in pdb_atoms:
        if (
            atom.record != "ATOM"
            or atom.chain_id not in protein_chain_names
            or atom.is_hydrogen
        ):
            continue
        chain_residues = residues_by_chain.setdefault(atom.chain_id, {})
        chain_residues.setdefault(atom.residue_key, []).append(atom)

    return residues_by_chain


def _residue_plddt_from_pdb_bfactor(
    residues_by_chain: dict[str, dict[tuple[str, str, str, str], list[PDBAtom]]],
) -> dict[tuple[str, str, str, str], float]:
    """Fallback: one value per residue, preferring CA and normalizing to 0-1."""
    raw_values: dict[tuple[str, str, str, str], float] = {}

    for chain_residues in residues_by_chain.values():
        for residue_key, atoms in chain_residues.items():
            ca_values = [
                atom.bfactor
                for atom in atoms
                if atom.atom_name.upper() == "CA" and math.isfinite(atom.bfactor)
            ]
            heavy_values = [
                atom.bfactor for atom in atoms if math.isfinite(atom.bfactor)
            ]
            if ca_values:
                raw_values[residue_key] = float(ca_values[0])
            elif heavy_values:
                raw_values[residue_key] = float(np.mean(heavy_values))

    finite = [value for value in raw_values.values() if math.isfinite(value)]
    scale = 100.0 if finite and max(finite) > 1.5 else 1.0
    return {key: value / scale for key, value in raw_values.items()}


def extract_pocket_residue_plddt_metrics(
    pdb_atoms: list[PDBAtom],
    binder_chain: str,
    chain_infos: list[ChainInfo],
    plddt: np.ndarray,
    pocket_cutoff: float,
) -> dict[str, Any]:
    """
    Calculate ligand-neighborhood protein confidence with equal residue weights.

    The preferred confidence source is Boltz-2's native per-token pLDDT NPZ. For
    protein chains, one token corresponds to one residue. If PDB residue order and
    token count cannot be reconciled, one CA/B-factor value per residue is used as
    a clearly labelled fallback.
    """
    protein_infos = [info for info in chain_infos if info.role == "protein"]
    protein_chain_names = {info.name for info in protein_infos}
    residues_by_chain = _pdb_protein_residues_by_chain(
        pdb_atoms,
        protein_chain_names,
    )
    ligand_atoms = [
        atom
        for atom in pdb_atoms
        if atom.record == "HETATM"
        and atom.chain_id == binder_chain
        and not atom.is_hydrogen
    ]

    out: dict[str, Any] = {
        "boltz2_pocket_cutoff_A": float(pocket_cutoff),
        "boltz2_pocket_mean_plddt_5A": float("nan"),
        "boltz2_pocket_residue_count_5A": 0,
        "boltz2_pocket_plddt_source": "",
        "boltz2_pocket_residue_ids_json": "[]",
    }

    if not ligand_atoms:
        raise ValueError("Cannot calculate pocket pLDDT: no affinity ligand heavy atoms found in the predicted structure")
    if not protein_infos:
        raise ValueError("Cannot calculate pocket pLDDT: no protein chain found")

    plddt = np.asarray(plddt, dtype=np.float64).reshape(-1)
    native_values: dict[tuple[str, str, str, str], float] = {}
    native_mapping_ok = True

    for info in protein_infos:
        residue_items = list(residues_by_chain.get(info.name, {}).items())
        token_values = plddt[info.token_indices]
        if len(residue_items) != len(token_values):
            native_mapping_ok = False
            break
        for (residue_key, _atoms), value in zip(residue_items, token_values):
            if math.isfinite(float(value)):
                native_values[residue_key] = float(value)

    if native_mapping_ok and native_values:
        residue_plddt = native_values
        source = "NPZ_PER_RESIDUE_TOKEN"
    else:
        residue_plddt = _residue_plddt_from_pdb_bfactor(residues_by_chain)
        source = "PDB_CA_BFACTOR_RESIDUE_FALLBACK"

    ligand_coords = np.vstack([atom.coord for atom in ligand_atoms])
    cutoff2 = float(pocket_cutoff) ** 2
    selected_values: list[float] = []
    selected_ids: list[str] = []

    for info in protein_infos:
        for residue_key, residue_atoms in residues_by_chain.get(info.name, {}).items():
            value = residue_plddt.get(residue_key)
            if value is None or not math.isfinite(float(value)):
                continue
            residue_coords = np.vstack([atom.coord for atom in residue_atoms])
            displacement = residue_coords[:, None, :] - ligand_coords[None, :, :]
            min_distance2 = float(np.min(np.sum(displacement * displacement, axis=2)))
            if min_distance2 <= cutoff2:
                selected_values.append(float(value))
                chain_id, residue_number, insertion_code, residue_name = residue_key
                selected_ids.append(
                    f"{chain_id}:{residue_number}{insertion_code}:{residue_name}"
                )

    out["boltz2_pocket_residue_count_5A"] = len(selected_values)
    out["boltz2_pocket_plddt_source"] = source
    out["boltz2_pocket_residue_ids_json"] = json.dumps(
        selected_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if selected_values:
        out["boltz2_pocket_mean_plddt_5A"] = float(np.mean(selected_values))

    return out


def safe_string(value: Any, digits: int) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def collect_one_task(
    task_dir: Path,
    sample_index: int,
    explicit_ligand_chain: str | None,
    pocket_cutoff: float,
) -> dict[str, Any]:
    task_name = task_dir.name
    prediction_dir = task_dir / "predictions" / task_name

    if not prediction_dir.is_dir():
        prediction_dir = task_dir

    confidence_path = find_file(
        prediction_dir,
        [f"confidence_{task_name}_model_{sample_index}.json"],
        [f"confidence*model_{sample_index}.json"],
    )
    structure_path = find_file(
        prediction_dir,
        [
            f"{task_name}_model_{sample_index}.pdb",
            f"{task_name}_model_{sample_index}.cif",
            f"{task_name}_model_{sample_index}.mmcif",
        ],
        [
            f"*model_{sample_index}.pdb",
            f"*model_{sample_index}.cif",
            f"*model_{sample_index}.mmcif",
        ],
    )
    plddt_path = find_file(
        prediction_dir,
        [f"plddt_{task_name}_model_{sample_index}.npz"],
        [f"plddt*model_{sample_index}.npz"],
    )
    pae_path = find_file(
        prediction_dir,
        [f"pae_{task_name}_model_{sample_index}.npz"],
        [f"pae*model_{sample_index}.npz"],
    )
    pde_path = find_file(
        prediction_dir,
        [f"pde_{task_name}_model_{sample_index}.npz"],
        [f"pde*model_{sample_index}.npz"],
    )
    affinity_path = find_file(
        prediction_dir,
        [f"affinity_{task_name}.json"],
        ["affinity*.json"],
    )
    pre_affinity_path = find_file(
        prediction_dir,
        [f"pre_affinity_{task_name}.npz"],
        ["pre_affinity*.npz"],
    )

    row: dict[str, Any] = {
        "boltz2_task_id": task_name,
        "boltz2_sample_index": sample_index,
        "boltz2_prediction_dir": str(prediction_dir),
        "boltz2_confidence_json": str(confidence_path or ""),
        "boltz2_structure_file": str(structure_path or ""),
        "boltz2_plddt_npz": str(plddt_path or ""),
        "boltz2_pae_npz": str(pae_path or ""),
        "boltz2_pde_npz": str(pde_path or ""),
        "boltz2_affinity_json": str(affinity_path or ""),
        "boltz2_pre_affinity_npz": str(pre_affinity_path or ""),
        "boltz2_pocket_cutoff_A": float(pocket_cutoff),
    }

    missing = [
        name
        for name, path in [
            ("confidence", confidence_path),
            ("structure", structure_path),
            ("plddt", plddt_path),
            ("pae", pae_path),
            ("pde", pde_path),
            ("affinity", affinity_path),
            ("pre_affinity", pre_affinity_path),
        ]
        if path is None
    ]

    try:
        if confidence_path is None:
            raise FileNotFoundError("Missing confidence JSON for this sample")
        if structure_path is None or structure_path.suffix.lower() != ".pdb":
            raise FileNotFoundError(
                "Missing PDB for this sample. Atom, geometry and pocket statistics require PDB output."
            )
        if plddt_path is None:
            raise FileNotFoundError("Missing pLDDT NPZ for this sample")
        if pae_path is None:
            raise FileNotFoundError("Missing PAE NPZ for this sample")
        if pde_path is None:
            raise FileNotFoundError("Missing PDE NPZ for this sample")
        if affinity_path is None:
            raise FileNotFoundError("Missing task-level affinity JSON")

        confidence = load_json(confidence_path)
        affinity = load_json(affinity_path)
        plddt = load_npz_array(plddt_path, "plddt").reshape(-1)
        pae = load_npz_array(pae_path, "pae")
        pde = load_npz_array(pde_path, "pde")
        pdb_atoms = parse_pdb_atoms(structure_path)

        pre_chains: np.ndarray | None = None
        if pre_affinity_path is not None:
            with np.load(pre_affinity_path, allow_pickle=True) as archive:
                if "chains" in archive.files:
                    pre_chains = np.asarray(archive["chains"])

        binder_chain = infer_binder_chain(
            task_dir=task_dir,
            explicit_chain=explicit_ligand_chain,
            pre_chains=pre_chains,
            pdb_atoms=pdb_atoms,
        )
        row["boltz2_affinity_binder_chain"] = binder_chain

        if pre_affinity_path is not None:
            chain_infos = chain_infos_from_pre_affinity(
                pre_affinity_path=pre_affinity_path,
                binder_chain=binder_chain,
                expected_token_count=len(plddt),
            )
            row["boltz2_chain_mapping_source"] = "pre_affinity"
        else:
            chain_infos = chain_infos_from_pdb(
                pdb_atoms=pdb_atoms,
                binder_chain=binder_chain,
                expected_token_count=len(plddt),
            )
            row["boltz2_chain_mapping_source"] = "pdb_fallback"

        if pae.shape != (len(plddt), len(plddt)):
            raise ValueError(
                f"PAE shape {pae.shape} does not match the token count {len(plddt)}"
            )
        if pde.shape != (len(plddt), len(plddt)):
            raise ValueError(
                f"PDE shape {pde.shape} does not match the token count {len(plddt)}"
            )

        row["boltz2_chain_count"] = len(chain_infos)
        row["boltz2_protein_chain_count"] = sum(
            info.role == "protein" for info in chain_infos
        )
        row["boltz2_ligand_chain_count"] = sum(
            info.role == "ligand" for info in chain_infos
        )

        row.update(extract_confidence_metrics(confidence, chain_infos))
        row.update(extract_plddt_metrics(plddt, chain_infos))
        row.update(
            extract_error_matrix_metrics(
                pae,
                "pae",
                chain_infos,
            )
        )
        row.update(
            extract_error_matrix_metrics(
                pde,
                "pde",
                chain_infos,
            )
        )
        row.update(extract_affinity_metrics(affinity))

        protein_chain_names = {
            info.name for info in chain_infos if info.role == "protein"
        }
        row.update(
            extract_pdb_geometry_metrics(
                pdb_atoms=pdb_atoms,
                binder_chain=binder_chain,
                protein_chain_names=protein_chain_names,
            )
        )
        row.update(
            extract_pocket_residue_plddt_metrics(
                pdb_atoms=pdb_atoms,
                binder_chain=binder_chain,
                chain_infos=chain_infos,
                plddt=plddt,
                pocket_cutoff=pocket_cutoff,
            )
        )

        row["boltz2_status"] = "OK" if not missing else (
            "PARTIAL_MISSING_" + ",".join(missing)
        )

    except Exception as exc:
        row["boltz2_status"] = f"FAILED: {exc}"

    return row


def _finite_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value in {None, ""}:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def aggregate_one_task(
    task_dir: Path,
    mode: str,
    sample_indices: list[int],
    digits: int,
    explicit_ligand_chain: str | None,
    pocket_cutoff: float,
) -> dict[str, Any]:
    """Calculate each requested model independently, then average numeric fields."""
    all_results: list[tuple[int, dict[str, Any]]] = []
    usable_results: list[tuple[int, dict[str, Any]]] = []

    for sample_index in sample_indices:
        result = collect_one_task(
            task_dir=task_dir,
            sample_index=sample_index,
            explicit_ligand_chain=explicit_ligand_chain,
            pocket_cutoff=pocket_cutoff,
        )
        all_results.append((sample_index, result))
        if not str(result.get("boltz2_status", "")).startswith("FAILED"):
            usable_results.append((sample_index, result))

    row: dict[str, Any] = {
        "boltz2_task_id": task_dir.name,
        "boltz2_score_mode": mode,
        "boltz2_samples_requested": ",".join(str(value) for value in sample_indices),
        "boltz2_sample_count_requested": len(sample_indices),
        "boltz2_sample_count_used": len(usable_results),
        "boltz2_samples_used": ",".join(str(value) for value, _ in usable_results),
        "boltz2_pocket_cutoff_A": float(pocket_cutoff),
        "boltz2_sample_status": ";".join(
            f"model_{sample}:{result.get('boltz2_status', '')}"
            for sample, result in all_results
        ),
    }

    if mode == "single":
        row["boltz2_sample_index"] = sample_indices[0]

    if not usable_results:
        row["boltz2_status"] = "FAILED_NO_SUCCESSFUL_MODEL"
        return {key: safe_string(value, digits) for key, value in row.items()}

    metadata_fields = {
        "boltz2_task_id",
        "boltz2_status",
        "boltz2_score_mode",
        "boltz2_sample_index",
        "boltz2_samples_requested",
        "boltz2_sample_count_requested",
        "boltz2_sample_count_used",
        "boltz2_samples_used",
        "boltz2_sample_status",
        "boltz2_pocket_cutoff_A",
    }
    path_fields = {
        "boltz2_confidence_json",
        "boltz2_structure_file",
        "boltz2_plddt_npz",
        "boltz2_pae_npz",
        "boltz2_pde_npz",
    }
    first_result = usable_results[0][1]
    all_fields = []
    for _sample, result in usable_results:
        for field in result:
            if field not in all_fields:
                all_fields.append(field)

    for field in all_fields:
        if field in metadata_fields:
            continue

        numeric_values = [
            number
            for _sample, result in usable_results
            if (number := _finite_numeric(result.get(field))) is not None
        ]
        present_values = [
            result.get(field)
            for _sample, result in usable_results
            if result.get(field) not in {None, ""}
        ]

        if numeric_values and len(numeric_values) == len(present_values):
            row[field] = float(np.mean(numeric_values))
            continue

        if field in path_fields or field == "boltz2_pocket_residue_ids_json":
            row[field] = " | ".join(
                f"model_{sample}:{result.get(field, '')}"
                for sample, result in usable_results
                if result.get(field) not in {None, ""}
            )
            continue

        unique_text = []
        for value in present_values:
            text_value = str(value)
            if text_value not in unique_text:
                unique_text.append(text_value)
        if len(unique_text) == 1:
            row[field] = present_values[0]
        elif unique_text:
            row[field] = " | ".join(unique_text)

    # The affinity/pre-affinity files are task-level, not diffusion-model-specific.
    for field in ("boltz2_affinity_json", "boltz2_pre_affinity_npz"):
        if first_result.get(field) not in {None, ""}:
            row[field] = first_result[field]

    if len(usable_results) == len(sample_indices):
        row["boltz2_status"] = "OK" if mode == "single" else "OK_MEAN5"
    else:
        row["boltz2_status"] = (
            f"PARTIAL_{len(usable_results)}_OF_{len(sample_indices)}"
        )

    return {key: safe_string(value, digits) for key, value in row.items()}


def collect_scores(
    input_dir: Path,
    mode: str,
    sample_indices: list[int],
    digits: int,
    explicit_ligand_chain: str | None,
    pocket_cutoff: float,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}

    task_dirs = [
        path
        for path in input_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_tmp")
    ]
    task_dirs = sorted(task_dirs, key=lambda path: natural_key(path.name))

    for index, task_dir in enumerate(task_dirs, start=1):
        print(
            f"[{index}/{len(task_dirs)}] Extracting {task_dir.name}; "
            f"mode={mode}; models={sample_indices}",
            flush=True,
        )
        scores[task_dir.name] = aggregate_one_task(
            task_dir=task_dir,
            mode=mode,
            sample_indices=sample_indices,
            digits=digits,
            explicit_ligand_chain=explicit_ligand_chain,
            pocket_cutoff=pocket_cutoff,
        )

    return scores


def preferred_score_fields() -> list[str]:
    return [
        "boltz2_task_id",
        "boltz2_status",
        "boltz2_sample_index",
        "boltz2_affinity_binder_chain",
        "boltz2_chain_mapping_source",

        "boltz2_confidence_score",
        "boltz2_confidence_score_recomputed",
        "boltz2_confidence_score_recomputed_abs_diff",
        "boltz2_ptm",
        "boltz2_iptm",
        "boltz2_ligand_iptm",
        "boltz2_protein_iptm",
        "boltz2_ligand_protein_chain_pair_iptm_A",
        "boltz2_ligand_protein_chain_pair_iptm_B",
        "boltz2_ligand_protein_chain_pair_iptm_bidirectional_mean",
        "boltz2_complex_plddt",
        "boltz2_complex_iplddt",
        "boltz2_complex_pde",
        "boltz2_complex_ipde",

        "boltz2_protein_mean_plddt",
        "boltz2_ligand_mean_plddt",
        "boltz2_protein_residue_plddt_mean",
        "boltz2_protein_residue_plddt_median",
        "boltz2_protein_residue_plddt_std",
        "boltz2_protein_residue_plddt_min",
        "boltz2_protein_residue_plddt_q10",
        "boltz2_protein_residue_plddt_max",
        "boltz2_protein_residue_count",
        "boltz2_protein_plddt_source",
        "boltz2_ligand_atom_plddt_mean",
        "boltz2_ligand_atom_plddt_median",
        "boltz2_ligand_atom_plddt_std",
        "boltz2_ligand_atom_plddt_min",
        "boltz2_ligand_atom_plddt_q10",
        "boltz2_ligand_atom_plddt_max",
        "boltz2_ligand_atom_count",
        "boltz2_ligand_plddt_source",
        "boltz2_complex_atom_plddt_mean",

        "boltz2_pocket_mean_plddt_5A",
        "boltz2_pocket_residue_count_5A",
        "boltz2_pocket_cutoff_A",
        "boltz2_pocket_plddt_source",
        "boltz2_pocket_residue_ids_json",

        "boltz2_pae_ligand_to_protein_A_mean",
        "boltz2_pae_protein_to_ligand_B_mean",
        "boltz2_pae_ligand_protein_bidirectional_mean",
        "boltz2_pae_ligand_protein_bidirectional_median",
        "boltz2_pae_ligand_protein_bidirectional_min",
        "boltz2_pae_ligand_specific_mean",
        "boltz2_pae_ligand_specific_min",

        "boltz2_pde_ligand_to_protein_A_mean",
        "boltz2_pde_protein_to_ligand_B_mean",
        "boltz2_pde_ligand_protein_bidirectional_mean",
        "boltz2_pde_ligand_protein_bidirectional_median",
        "boltz2_pde_ligand_protein_bidirectional_min",
        "boltz2_pde_ligand_specific_mean",
        "boltz2_pde_ligand_specific_min",

        "boltz2_affinity_pred_value",
        "boltz2_affinity_pred_pIC50",
        "boltz2_affinity_pred_ic50_uM",
        "boltz2_affinity_probability_binary",
        "boltz2_affinity_pred_value1",
        "boltz2_affinity_pred_pIC501",
        "boltz2_affinity_pred_ic50_uM1",
        "boltz2_affinity_probability_binary1",
        "boltz2_affinity_pred_value2",
        "boltz2_affinity_pred_pIC502",
        "boltz2_affinity_pred_ic50_uM2",
        "boltz2_affinity_probability_binary2",
        "boltz2_affinity_pred_value_model_disagreement_abs",
        "boltz2_affinity_probability_model_disagreement_abs",

        "boltz2_ligand_protein_min_heavy_atom_distance_A",
        "boltz2_ligand_mean_nearest_protein_heavy_atom_distance_A",
        "boltz2_ligand_protein_heavy_atom_pair_contacts_4A",
        "boltz2_ligand_contact_heavy_atom_count_4A",
        "boltz2_ligand_contact_heavy_atom_fraction_4A",
        "boltz2_protein_contact_residue_count_4A",
        "boltz2_contacts_per_ligand_heavy_atom_4A",

        "boltz2_protein_chain_count",
        "boltz2_ligand_chain_count",
        "boltz2_token_count",
        "boltz2_protein_atom_count",
        "boltz2_ligand_atom_count",

        "boltz2_prediction_dir",
        "boltz2_confidence_json",
        "boltz2_structure_file",
        "boltz2_plddt_npz",
        "boltz2_pae_npz",
        "boltz2_pde_npz",
        "boltz2_affinity_json",
        "boltz2_pre_affinity_npz",
    ]


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    merge_csv = Path(args.merge_csv).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Boltz-2 output directory not found: {input_dir}")
    if not merge_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {merge_csv}")
    if args.sample_index < 0:
        raise ValueError("--sample-index must be nonnegative")
    if any(sample < 0 for sample in args.mean5_samples):
        raise ValueError("--mean5-samples must contain only nonnegative indices")
    if args.pocket_cutoff <= 0:
        raise ValueError("--pocket-cutoff must be greater than 0")
    if args.float_digits < 0:
        raise ValueError("--float-digits must be nonnegative")

    original_rows, original_fields = read_csv_dicts(merge_csv)

    if args.merge_key not in original_fields:
        raise ValueError(
            f"Input CSV has no column matching --merge-key={args.merge_key!r}\n"
            f"Available columns: {original_fields}"
        )

    sample_indices = (
        [args.sample_index]
        if args.mode == "single"
        else list(args.mean5_samples)
    )

    score_by_task = collect_scores(
        input_dir=input_dir,
        mode=args.mode,
        sample_indices=sample_indices,
        digits=args.float_digits,
        explicit_ligand_chain=args.ligand_chain_id,
        pocket_cutoff=args.pocket_cutoff,
    )

    merged_rows: list[dict[str, Any]] = []
    matched_tasks: set[str] = set()

    # Preserve the input CSV row order and all original columns.
    for original_row in original_rows:
        key = str(original_row.get(args.merge_key, "")).strip()
        score = score_by_task.get(key)

        merged = dict(original_row)
        if score is None:
            merged.update(
                {
                    "boltz2_task_id": key,
                    "boltz2_status": "MISSING_RESULT",
                    "boltz2_score_mode": args.mode,
                    "boltz2_sample_index": (
                        args.sample_index if args.mode == "single" else ""
                    ),
                    "boltz2_samples_requested": ",".join(
                        str(value) for value in sample_indices
                    ),
                    "boltz2_sample_count_requested": len(sample_indices),
                    "boltz2_sample_count_used": 0,
                    "boltz2_samples_used": "",
                    "boltz2_pocket_cutoff_A": safe_string(
                        args.pocket_cutoff,
                        args.float_digits,
                    ),
                }
            )
        else:
            matched_tasks.add(key)
            merged.update(score)

        merged_rows.append(merged)

    if args.include_unmatched_results:
        for task_name in sorted(score_by_task, key=natural_key):
            if task_name in matched_tasks:
                continue

            merged = {field: "" for field in original_fields}
            merged[args.merge_key] = task_name
            merged.update(score_by_task[task_name])
            merged_rows.append(merged)

    extra_fields: list[str] = []
    for row in merged_rows:
        for field in row:
            if field not in original_fields and field not in extra_fields:
                extra_fields.append(field)

    ordered_extra: list[str] = []
    for field in preferred_score_fields():
        if field in extra_fields and field not in ordered_extra:
            ordered_extra.append(field)
    for field in extra_fields:
        if field not in ordered_extra:
            ordered_extra.append(field)

    final_fields = original_fields + ordered_extra

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=final_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    ok_count = sum(
        str(row.get("boltz2_status", "")).startswith("OK")
        for row in merged_rows
    )
    failed_count = sum(
        str(row.get("boltz2_status", "")).startswith("FAILED")
        for row in merged_rows
    )
    missing_count = sum(
        row.get("boltz2_status") == "MISSING_RESULT"
        for row in merged_rows
    )

    print("=" * 100)
    print("Boltz-2 score collection finished")
    print(f"Boltz-2 output dir: {input_dir}")
    print(f"Mode: {args.mode}")
    print(f"Models requested: {sample_indices}")
    print(f"Pocket cutoff: {args.pocket_cutoff:.2f} A")
    print(f"Original CSV: {merge_csv}")
    print(f"Merge key: {args.merge_key}")
    print(f"Output CSV: {output_csv}")
    print(f"Original rows: {len(original_rows)}")
    print(f"Matched tasks: {len(matched_tasks)}")
    print(f"OK rows: {ok_count}")
    print(f"FAILED rows: {failed_count}")
    print(f"Missing-result rows: {missing_count}")
    print(f"Float digits: {args.float_digits}")
    print("=" * 100)


if __name__ == "__main__":
    main()

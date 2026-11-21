#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect Chai-1 confidence scores and optionally merge them into the input CSV by task directory name.

Scoring modes and output conventions
1. --mode single reads the selected seed and model_idx (defaults: seed 1, model_idx 0).
2. --mode mean5 reads model_idx 0 through 4 for the same seed, scores each predicted structure
   using its own coordinates, then takes the arithmetic mean of each numerical field.
3. For chai_pocket_mean_plddt_5A, select protein residues in each predicted CIF with a heavy atom
   within 5.0 Å of any ligand heavy atom. Each pocket residue contributes one pLDDT value
   and receives equal weight in the mean.
4. The script also extracts NPZ scores, directional chain-pair iPTM, protein residue pLDDT
   and ligand heavy-atom pLDDT, and supports merging the scores into a CSV.
5. Chai-1 results use the following directory layout:

   <result_dir>/bind_0001/seed_1/
       pred.model_idx_0.cif
       pred.model_idx_1.cif
       ...
       scores.model_idx_0.npz
       scores.model_idx_1.npz
       ...

Task directory names such as bind_0001 match values in the input CSV column specified by --merge-key.
"""

import argparse
import json
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_CIF_VALUES = {"", ".", "?"}
DEFAULT_DECIMALS = 6
DEFAULT_POCKET_CUTOFF_A = 5.0
MEAN5_MODEL_INDICES = (0, 1, 2, 3, 4)


def parse_model_idx(path: Path):
    """Extract model_idx from scores.model_idx_0.npz or pred.model_idx_0.cif."""
    match = re.search(r"model_idx_(\d+)", path.name)
    return int(match.group(1)) if match else None


def parse_seed(seed_dir: Path):
    """Extract seed number from a directory such as seed_1."""
    match = re.search(r"seed[_-]?(\d+)", seed_dir.name)
    return int(match.group(1)) if match else seed_dir.name


def is_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def array_to_python_scalar(value):
    """Convert a NumPy scalar to a normal Python scalar for CSV output."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def rounded_float(value, decimals: int = DEFAULT_DECIMALS):
    """Return a finite float rounded to the requested number of decimals."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return round(number, decimals)


def safe_float(value):
    if value in MISSING_CIF_VALUES or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _tokenize_cif(cif_path: Path) -> Iterable[str]:
    """
    Tokenize a Chai-1 ModelCIF file using only the Python standard library.

    Quoted strings and CIF semicolon-delimited multiline values are handled.
    This avoids requiring Gemmi/Biopython on the analysis server.
    """
    with cif_path.open("r", encoding="utf-8", errors="replace") as handle:
        line_iter = iter(handle)
        for raw_line in line_iter:
            # A semicolon text field begins only when ';' is in column 1.
            if raw_line.startswith(";"):
                block = [raw_line[1:].rstrip("\r\n")]
                for block_line in line_iter:
                    if block_line.startswith(";"):
                        break
                    block.append(block_line.rstrip("\r\n"))
                yield "\n".join(block)
                continue

            lexer = shlex.shlex(raw_line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            for token in lexer:
                yield token


def read_cif_loops(cif_path: Path, wanted_categories: set[str]):
    """Read selected mmCIF loop categories into lists of dictionaries."""
    output = {category: [] for category in wanted_categories}
    tokens = list(_tokenize_cif(cif_path))
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if token.lower() != "loop_":
            i += 1
            continue

        i += 1
        headers = []
        while i < len(tokens) and tokens[i].startswith("_"):
            headers.append(tokens[i])
            i += 1

        if not headers:
            continue

        n_columns = len(headers)
        values = []
        while i < len(tokens):
            current = tokens[i]
            lower = current.lower()
            at_row_boundary = len(values) % n_columns == 0

            if lower in {"loop_", "stop_"} or lower.startswith("data_") or lower.startswith("save_"):
                break
            if at_row_boundary and current.startswith("_"):
                break

            values.append(current)
            i += 1

        category = headers[0].split(".", 1)[0]
        if category not in wanted_categories:
            continue

        if len(values) % n_columns != 0:
            raise ValueError(
                f"Malformed CIF loop {category}: {len(values)} values for "
                f"{n_columns} columns in {cif_path}"
            )

        for start in range(0, len(values), n_columns):
            row_values = values[start : start + n_columns]
            output[category].append(dict(zip(headers, row_values)))

    return output


def _natural_chain_key(chain_id: str):
    """Provide a stable fallback ordering for chain identifiers."""
    parts = re.split(r"(\d+)", str(chain_id))
    return [int(part) if part.isdigit() else part for part in parts]


def _classify_chains(cif_loops):
    """Return matrix chain order, protein chains, and ligand chains."""
    struct_asym_rows = cif_loops.get("_struct_asym", [])
    entity_rows = cif_loops.get("_entity", [])
    entity_poly_rows = cif_loops.get("_entity_poly", [])
    atom_rows = cif_loops.get("_atom_site", [])

    entity_type = {
        row.get("_entity.id"): str(row.get("_entity.type", "")).lower()
        for row in entity_rows
    }
    entity_poly_type = {
        row.get("_entity_poly.entity_id"): str(row.get("_entity_poly.type", "")).lower()
        for row in entity_poly_rows
    }

    chain_order = []
    chain_entity = {}
    for row in struct_asym_rows:
        chain_id = row.get("_struct_asym.id")
        if not chain_id or chain_id in MISSING_CIF_VALUES:
            continue
        chain_order.append(chain_id)
        chain_entity[chain_id] = row.get("_struct_asym.entity_id")

    # Fallback to atom-site first appearance if _struct_asym is unavailable.
    if not chain_order:
        for row in atom_rows:
            chain_id = row.get("_atom_site.label_asym_id")
            if chain_id and chain_id not in MISSING_CIF_VALUES and chain_id not in chain_order:
                chain_order.append(chain_id)

    protein_chains = []
    ligand_chains = []

    for chain_id in chain_order:
        entity_id = chain_entity.get(chain_id)
        e_type = entity_type.get(entity_id, "")
        p_type = entity_poly_type.get(entity_id, "")

        if e_type == "polymer" and "polypeptide" in p_type:
            protein_chains.append(chain_id)
        elif e_type in {"non-polymer", "branched"}:
            ligand_chains.append(chain_id)

    # Conservative fallback based on ATOM/HETATM records.
    if not protein_chains or not ligand_chains:
        groups_by_chain = defaultdict(set)
        for row in atom_rows:
            chain_id = row.get("_atom_site.label_asym_id")
            group = str(row.get("_atom_site.group_PDB", "")).upper()
            if chain_id and chain_id not in MISSING_CIF_VALUES:
                groups_by_chain[chain_id].add(group)

        if not protein_chains:
            protein_chains = [
                chain for chain in chain_order if "ATOM" in groups_by_chain.get(chain, set())
            ]
        if not ligand_chains:
            ligand_chains = [
                chain
                for chain in chain_order
                if chain not in protein_chains
                and "HETATM" in groups_by_chain.get(chain, set())
            ]

    chain_order = sorted(chain_order, key=_natural_chain_key) if not struct_asym_rows else chain_order
    return chain_order, protein_chains, ligand_chains


def _extract_protein_residue_plddt(cif_loops, protein_chains: list[str]):
    """
    Extract one pLDDT value per protein residue.

    Primary source: ModelCIF _ma_qa_metric_local.
    Fallback: CA B_iso_or_equiv per residue; if CA is missing, the within-residue
    heavy-atom mean is used before residues are averaged equally.
    """
    metric_rows = cif_loops.get("_ma_qa_metric", [])
    local_rows = cif_loops.get("_ma_qa_metric_local", [])
    atom_rows = cif_loops.get("_atom_site", [])

    plddt_metric_ids = set()
    for row in metric_rows:
        text = " ".join(
            str(row.get(key, ""))
            for key in (
                "_ma_qa_metric.name",
                "_ma_qa_metric.type",
                "_ma_qa_metric.description",
            )
        ).lower()
        if "plddt" in text or "predicted lddt" in text:
            metric_id = row.get("_ma_qa_metric.id")
            if metric_id not in MISSING_CIF_VALUES:
                plddt_metric_ids.add(metric_id)

    residue_values = defaultdict(list)
    for row in local_rows:
        chain_id = row.get("_ma_qa_metric_local.label_asym_id")
        if chain_id not in protein_chains:
            continue
        metric_id = row.get("_ma_qa_metric_local.metric_id")
        if plddt_metric_ids and metric_id not in plddt_metric_ids:
            continue
        value = safe_float(row.get("_ma_qa_metric_local.metric_value"))
        if value is None:
            continue
        residue_key = (
            chain_id,
            row.get("_ma_qa_metric_local.label_seq_id"),
            row.get("_ma_qa_metric_local.label_comp_id"),
        )
        residue_values[residue_key].append(value)

    if residue_values:
        collapsed = {key: float(np.mean(values)) for key, values in residue_values.items()}
        return collapsed, "_ma_qa_metric_local"

    # Fallback for CIFs without the ModelCIF local-quality loop.
    ca_values = {}
    heavy_values = defaultdict(list)
    for row in atom_rows:
        chain_id = row.get("_atom_site.label_asym_id")
        if chain_id not in protein_chains:
            continue
        element = str(row.get("_atom_site.type_symbol", "")).upper()
        if element in {"H", "D", "T"}:
            continue
        value = safe_float(row.get("_atom_site.B_iso_or_equiv"))
        if value is None:
            continue
        residue_key = (
            chain_id,
            row.get("_atom_site.label_seq_id"),
            row.get("_atom_site.label_comp_id"),
        )
        heavy_values[residue_key].append(value)
        if str(row.get("_atom_site.label_atom_id", "")).upper() == "CA":
            ca_values[residue_key] = value

    collapsed = {}
    for key, values in heavy_values.items():
        collapsed[key] = ca_values.get(key, float(np.mean(values)))
    return collapsed, "_atom_site.CA_B_iso_or_equiv_fallback"


def _extract_ligand_atom_plddt(cif_loops, ligand_chains: list[str]):
    """Extract ligand heavy-atom pLDDT values from atom-site B_iso_or_equiv."""
    values_by_chain = defaultdict(list)
    for row in cif_loops.get("_atom_site", []):
        chain_id = row.get("_atom_site.label_asym_id")
        if chain_id not in ligand_chains:
            continue
        element = str(row.get("_atom_site.type_symbol", "")).upper()
        if element in {"H", "D", "T"}:
            continue
        value = safe_float(row.get("_atom_site.B_iso_or_equiv"))
        if value is not None:
            values_by_chain[chain_id].append(value)
    return values_by_chain


def _atom_is_hydrogen(row: dict) -> bool:
    element = str(row.get("_atom_site.type_symbol", "")).strip().upper()
    if element:
        return element in {"H", "D", "T"}
    atom_name = str(row.get("_atom_site.label_atom_id", "")).strip().upper()
    return atom_name.startswith(("H", "D", "T"))


def _atom_coord(row: dict):
    try:
        return np.array(
            [
                float(row.get("_atom_site.Cartn_x")),
                float(row.get("_atom_site.Cartn_y")),
                float(row.get("_atom_site.Cartn_z")),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None


def _extract_pocket_residue_plddt(
    cif_loops,
    protein_chains: list[str],
    ligand_chains: list[str],
    protein_residue_values: dict,
    cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
):
    """
    Return per-residue pLDDT for protein residues within cutoff_a Å of the predicted ligand.

    Define the pocket by the minimum distance between protein and ligand heavy atoms.
    Use one pLDDT value per residue so that every residue has equal weight in the mean.
    """
    atom_rows = cif_loops.get("_atom_site", [])
    protein_set = set(protein_chains)
    ligand_set = set(ligand_chains)

    ligand_coords = []
    for row in atom_rows:
        if row.get("_atom_site.label_asym_id") not in ligand_set:
            continue
        if _atom_is_hydrogen(row):
            continue
        coord = _atom_coord(row)
        if coord is not None:
            ligand_coords.append(coord)

    if not ligand_coords:
        return [], []

    ligand_coords = np.asarray(ligand_coords, dtype=float)
    residue_coords = defaultdict(list)
    for row in atom_rows:
        chain_id = row.get("_atom_site.label_asym_id")
        if chain_id not in protein_set or _atom_is_hydrogen(row):
            continue
        seq_id = row.get("_atom_site.label_seq_id")
        comp_id = row.get("_atom_site.label_comp_id")
        if seq_id in MISSING_CIF_VALUES or seq_id is None:
            continue
        coord = _atom_coord(row)
        if coord is None:
            continue
        residue_coords[(chain_id, seq_id, comp_id)].append(coord)

    cutoff2 = float(cutoff_a) ** 2
    pocket_keys = []
    for residue_key, coords in residue_coords.items():
        protein_coords = np.asarray(coords, dtype=float)
        delta = protein_coords[:, None, :] - ligand_coords[None, :, :]
        min_distance2 = float(np.min(np.sum(delta * delta, axis=2)))
        if min_distance2 <= cutoff2:
            pocket_keys.append(residue_key)

    # Some ModelCIF local quality tables use comp_id values that differ from those in atom_site.
    # Keep a fallback mapping keyed by chain and sequence position.
    by_chain_seq = {}
    for (chain_id, seq_id, _comp_id), value in protein_residue_values.items():
        by_chain_seq[(chain_id, seq_id)] = value

    pocket_values = []
    matched_keys = []
    for key in pocket_keys:
        value = protein_residue_values.get(key)
        if value is None:
            value = by_chain_seq.get((key[0], key[1]))
        if value is None:
            continue
        pocket_values.append(float(value))
        matched_keys.append(key)

    return pocket_values, matched_keys


def add_cif_values_to_row(
    row: dict,
    cif_path: Path,
    decimals: int = DEFAULT_DECIMALS,
    pocket_cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
):
    """Add residue-level protein and heavy-atom ligand confidence from CIF."""
    wanted = {
        "_struct_asym",
        "_entity",
        "_entity_poly",
        "_atom_site",
        "_ma_qa_metric",
        "_ma_qa_metric_local",
    }
    try:
        cif_loops = read_cif_loops(cif_path, wanted)
        chain_order, protein_chains, ligand_chains = _classify_chains(cif_loops)

        row["chai_chain_order"] = ";".join(chain_order)
        row["chai_protein_chain_ids"] = ";".join(protein_chains)
        row["chai_ligand_chain_ids"] = ";".join(ligand_chains)
        row["chai_num_protein_chains"] = len(protein_chains)
        row["chai_num_ligand_chains"] = len(ligand_chains)

        protein_residue_values, protein_source = _extract_protein_residue_plddt(
            cif_loops, protein_chains
        )
        row["chai_protein_mean_plddt"] = rounded_float(
            np.mean(list(protein_residue_values.values()))
            if protein_residue_values
            else None,
            decimals,
        )
        row["chai_protein_plddt_residue_count"] = len(protein_residue_values)
        row["chai_protein_plddt_source"] = protein_source

        protein_chain_means = {}
        for chain_id in protein_chains:
            values = [
                value
                for (chain, _seq, _comp), value in protein_residue_values.items()
                if chain == chain_id
            ]
            if values:
                protein_chain_means[chain_id] = rounded_float(np.mean(values), decimals)
        row["chai_protein_chain_mean_plddt_json"] = json.dumps(
            protein_chain_means, ensure_ascii=False, sort_keys=True
        )

        # Use one pLDDT value per pocket residue and give all residues equal weight.
        pocket_values, pocket_keys = _extract_pocket_residue_plddt(
            cif_loops=cif_loops,
            protein_chains=protein_chains,
            ligand_chains=ligand_chains,
            protein_residue_values=protein_residue_values,
            cutoff_a=pocket_cutoff_a,
        )
        row["chai_pocket_mean_plddt_5A"] = rounded_float(
            np.mean(pocket_values) if pocket_values else None,
            decimals,
        )
        row["chai_pocket_plddt_residue_count_5A"] = len(pocket_values)
        row["chai_pocket_cutoff_A"] = rounded_float(pocket_cutoff_a, decimals)
        row["chai_pocket_plddt_source"] = (
            f"residue_equal_weighted;residue_source={protein_source}"
        )
        row["chai_pocket_residue_ids_json"] = json.dumps(
            [f"{chain}:{seq}:{comp}" for chain, seq, comp in pocket_keys],
            ensure_ascii=False,
        )

        ligand_values_by_chain = _extract_ligand_atom_plddt(cif_loops, ligand_chains)
        all_ligand_values = [
            value for values in ligand_values_by_chain.values() for value in values
        ]
        row["chai_ligand_mean_plddt"] = rounded_float(
            np.mean(all_ligand_values) if all_ligand_values else None,
            decimals,
        )
        row["chai_ligand_plddt_heavy_atom_count"] = len(all_ligand_values)
        row["chai_ligand_plddt_source"] = "_atom_site.B_iso_or_equiv_heavy_atoms"
        row["chai_ligand_chain_mean_plddt_json"] = json.dumps(
            {
                chain_id: rounded_float(np.mean(values), decimals)
                for chain_id, values in ligand_values_by_chain.items()
                if values
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        # Store internally for chain-pair iPTM mapping; removed before CSV output.
        row["__chai_chain_order"] = chain_order
        row["__chai_protein_chains"] = protein_chains
        row["__chai_ligand_chains"] = ligand_chains
    except Exception as exc:
        row["cif_parse_error"] = str(exc)

    return row


def _squeeze_chain_pair_matrix(value):
    matrix = np.asarray(value, dtype=float)
    while matrix.ndim > 2 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"per_chain_pair_iptm is not a square matrix: shape={matrix.shape}")
    return matrix


def add_directional_chain_pair_iptm(
    row: dict,
    npz_data,
    decimals: int = DEFAULT_DECIMALS,
):
    """
    Add the two directional ligand-protein chain-pair iPTM scores.

    Chai-1 matrix orientation is [query_chain, key_chain].
      A: ligand -> protein
      B: protein -> ligand
    """
    if "per_chain_pair_iptm" not in npz_data:
        row["chai_chain_pair_iptm_error"] = "per_chain_pair_iptm key is missing"
        return row

    chain_order = row.get("__chai_chain_order", [])
    protein_chains = row.get("__chai_protein_chains", [])
    ligand_chains = row.get("__chai_ligand_chains", [])

    try:
        matrix = _squeeze_chain_pair_matrix(npz_data["per_chain_pair_iptm"])
        if len(chain_order) != matrix.shape[0]:
            raise ValueError(
                f"CIF chain count ({len(chain_order)}) does not match iPTM matrix "
                f"dimension ({matrix.shape[0]})"
            )
        if not protein_chains:
            raise ValueError("No protein chain was identified in CIF")
        if not ligand_chains:
            raise ValueError("No ligand chain was identified in CIF")

        chain_to_index = {chain_id: index for index, chain_id in enumerate(chain_order)}
        values_a = {}
        values_b = {}

        for ligand_chain in ligand_chains:
            for protein_chain in protein_chains:
                ligand_index = chain_to_index[ligand_chain]
                protein_index = chain_to_index[protein_chain]

                # Matrix axes: [query_chain, key_chain]
                a_value = matrix[ligand_index, protein_index]
                b_value = matrix[protein_index, ligand_index]

                values_a[f"{ligand_chain}->{protein_chain}"] = rounded_float(
                    a_value, decimals
                )
                values_b[f"{protein_chain}->{ligand_chain}"] = rounded_float(
                    b_value, decimals
                )

        finite_a = [float(value) for value in values_a.values() if value != ""]
        finite_b = [float(value) for value in values_b.values() if value != ""]

        row["chai_ligand_protein_chain_pair_iptm_A"] = rounded_float(
            np.mean(finite_a) if finite_a else None,
            decimals,
        )
        row["chai_ligand_protein_chain_pair_iptm_B"] = rounded_float(
            np.mean(finite_b) if finite_b else None,
            decimals,
        )
        row["chai_ligand_protein_chain_pair_iptm_A_pairs_json"] = json.dumps(
            values_a, ensure_ascii=False, sort_keys=True
        )
        row["chai_ligand_protein_chain_pair_iptm_B_pairs_json"] = json.dumps(
            values_b, ensure_ascii=False, sort_keys=True
        )
    except Exception as exc:
        row["chai_chain_pair_iptm_error"] = str(exc)

    return row


def add_npz_values_to_row(
    row: dict,
    npz_path: Path,
    cif_path: Path | None = None,
    max_array_values: int = 20,
    decimals: int = DEFAULT_DECIMALS,
    pocket_cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
):
    """
    Read scores.model_idx_*.npz and preserve every original NPZ field.

    Scalars are saved directly. Arrays retain shape/size/mean/min/max/std and,
    when small enough, their complete values as JSON.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        row["npz_load_error"] = str(exc)
        return row

    row["npz_keys"] = ";".join(list(data.keys()))

    for key in data.keys():
        value = data[key]
        arr = np.asarray(value)
        col_prefix = f"chai_{key}"

        if arr.shape == ():
            try:
                scalar = array_to_python_scalar(arr.item())
                row[col_prefix] = rounded_float(scalar, decimals) if isinstance(
                    scalar, (float, np.floating)
                ) else scalar
            except Exception:
                row[col_prefix] = str(arr)
            continue

        row[f"{col_prefix}_shape"] = "x".join(map(str, arr.shape))
        row[f"{col_prefix}_size"] = int(arr.size)

        if arr.size == 1:
            try:
                scalar = array_to_python_scalar(arr.reshape(-1)[0])
                row[col_prefix] = rounded_float(scalar, decimals) if isinstance(
                    scalar, (float, np.floating)
                ) else scalar
            except Exception:
                row[col_prefix] = str(arr.reshape(-1)[0])
            continue

        if is_numeric_array(arr):
            flat = arr.astype(float).reshape(-1)
            valid = flat[np.isfinite(flat)]
            if valid.size > 0:
                row[f"{col_prefix}_mean"] = rounded_float(np.mean(valid), decimals)
                row[f"{col_prefix}_min"] = rounded_float(np.min(valid), decimals)
                row[f"{col_prefix}_max"] = rounded_float(np.max(valid), decimals)
                row[f"{col_prefix}_std"] = rounded_float(np.std(valid), decimals)
            else:
                row[f"{col_prefix}_mean"] = ""
                row[f"{col_prefix}_min"] = ""
                row[f"{col_prefix}_max"] = ""
                row[f"{col_prefix}_std"] = ""

            if arr.size <= max_array_values:
                row[f"{col_prefix}_values"] = json.dumps(
                    np.round(arr.astype(float), decimals).tolist(),
                    ensure_ascii=False,
                )
        else:
            if arr.size <= max_array_values:
                row[f"{col_prefix}_values"] = json.dumps(
                    arr.tolist(), ensure_ascii=False
                )
            else:
                row[f"{col_prefix}_values"] = str(
                    arr.reshape(-1)[:max_array_values].tolist()
                )

    # Useful semantic alias; original chai_aggregate_score is still preserved.
    if "chai_aggregate_score" in row:
        row["chai_ranking_score"] = row["chai_aggregate_score"]

    if cif_path is not None and cif_path.exists():
        row = add_cif_values_to_row(
            row,
            cif_path,
            decimals=decimals,
            pocket_cutoff_a=pocket_cutoff_a,
        )
        row = add_directional_chain_pair_iptm(row, data, decimals=decimals)

    # Internal mapping fields must not appear in the final CSV.
    row.pop("__chai_chain_order", None)
    row.pop("__chai_protein_chains", None)
    row.pop("__chai_ligand_chains", None)
    return row


def _find_seed_dir(task_dir: Path, seed: int) -> Path | None:
    """Locate the requested seed directory using the Chai-1 seed_1 or seed-1 naming convention."""
    for name in (f"seed_{seed}", f"seed-{seed}", f"seed{seed}"):
        candidate = task_dir / name
        if candidate.is_dir():
            return candidate

    # Support older layouts with results stored directly in the task directory.
    direct_npz = task_dir / "scores.model_idx_0.npz"
    direct_cif = task_dir / "pred.model_idx_0.cif"
    if direct_npz.exists() or direct_cif.exists():
        return task_dir
    return None


def _available_model_indices(seed_dir: Path) -> list[int]:
    return sorted(
        index
        for index in (
            parse_model_idx(path) for path in seed_dir.glob("scores.model_idx_*.npz")
        )
        if index is not None
    )


def collect_one_sample(
    task_dir: Path,
    seed: int,
    sample_idx: int,
    max_array_values: int,
    decimals: int,
    pocket_cutoff_a: float,
):
    """Extract scores for one task at the specified model_idx."""
    task_id = task_dir.name
    row = {
        "task_id": task_id,
        "seed": seed,
        "model_idx": sample_idx,
        "chai_score_mode": "single",
        "chai_sample_count_used": 0,
        "chai_samples_used": "",
        "chai_pocket_cutoff_A": rounded_float(pocket_cutoff_a, decimals),
    }

    seed_dir = _find_seed_dir(task_dir, seed)
    if seed_dir is None:
        row.update({
            "result_status": f"MISSING_SEED_{seed}",
            "available_model_indices": "",
            "result_dir": "",
            "pred_cif": "",
            "score_npz": "",
        })
        return row

    score_npz = seed_dir / f"scores.model_idx_{sample_idx}.npz"
    pred_cif = seed_dir / f"pred.model_idx_{sample_idx}.cif"
    available_indices = _available_model_indices(seed_dir)

    row.update({
        "available_model_indices": ";".join(map(str, available_indices)),
        "result_dir": str(seed_dir),
        "pred_cif": str(pred_cif) if pred_cif.exists() else "",
        "score_npz": str(score_npz) if score_npz.exists() else "",
    })

    if not score_npz.exists():
        if pred_cif.exists():
            row["result_status"] = "CIF_ONLY"
        elif available_indices:
            row["result_status"] = "SELECTED_SAMPLE_MISSING"
        else:
            row["result_status"] = "NO_SCORE_NO_CIF"
        return row

    # As in the AF3 reader, a sample can contribute scores when its core score file is present.
    # If the CIF is missing, keep the NPZ scores and flag the missing pocket, protein and ligand pLDDT values.
    row["chai_sample_count_used"] = 1
    row["chai_samples_used"] = str(sample_idx)
    row["result_status"] = "OK" if pred_cif.exists() else "NPZ_ONLY"

    row = add_npz_values_to_row(
        row=row,
        npz_path=score_npz,
        cif_path=pred_cif if pred_cif.exists() else None,
        max_array_values=max_array_values,
        decimals=decimals,
        pocket_cutoff_a=pocket_cutoff_a,
    )
    return row


def _as_finite_float(value):
    if value is None or value == "" or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _unique_nonempty(values):
    unique = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in unique:
            continue
        unique.append(text)
    return unique


def aggregate_mean5_rows(
    task_dir: Path,
    seed: int,
    sample_rows: list[tuple[int, dict]],
    decimals: int,
    pocket_cutoff_a: float,
):
    """
    Average each numerical field across model_idx 0 through 4.

    Extract all NPZ and CIF scores from each model independently using its own coordinates.
    Keep matching nonnumeric values and join differing values with " | " for traceability.
    """
    valid = [
        (sample_idx, row)
        for sample_idx, row in sample_rows
        if int(row.get("chai_sample_count_used", 0) or 0) == 1
    ]

    out = {
        "task_id": task_dir.name,
        "seed": seed,
        "model_idx": "",
        "chai_score_mode": "mean5",
        "chai_sample_count_used": len(valid),
        "chai_samples_used": ",".join(str(idx) for idx, _ in valid),
        "chai_pocket_cutoff_A": rounded_float(pocket_cutoff_a, decimals),
        "available_model_indices": ";".join(map(str, _available_model_indices(_find_seed_dir(task_dir, seed))))
        if _find_seed_dir(task_dir, seed) is not None
        else "",
    }

    statuses = [f"model_idx_{idx}:{row.get('result_status', '')}" for idx, row in sample_rows]
    out["chai_sample_status"] = ";".join(statuses)

    if not valid:
        out["result_status"] = "ERROR_NO_MEAN5_SAMPLE"
        errors = []
        for idx, row in sample_rows:
            errors.append(f"model_idx_{idx}:{row.get('result_status', 'UNKNOWN')}")
        out["chai_mean5_error"] = " | ".join(errors)
        return out

    excluded = {
        "task_id",
        "seed",
        "model_idx",
        "chai_score_mode",
        "chai_sample_count_used",
        "chai_samples_used",
        "chai_sample_status",
        "result_status",
        "available_model_indices",
        "result_dir",
        "pred_cif",
        "score_npz",
    }

    all_keys = set()
    for _, row in valid:
        all_keys.update(row.keys())

    for key in sorted(all_keys):
        if key in excluded or key.startswith("__"):
            continue
        raw_values = [row.get(key, "") for _, row in valid]
        numeric_values = [
            number
            for number in (_as_finite_float(value) for value in raw_values)
            if number is not None
        ]
        nonempty_count = sum(value not in {None, ""} for value in raw_values)

        # Average a field only if every nonempty value is numeric. Preserve text such as chain names.
        if numeric_values and len(numeric_values) == nonempty_count:
            out[key] = rounded_float(np.mean(numeric_values), decimals)
        else:
            unique = _unique_nonempty(raw_values)
            if len(unique) == 1:
                out[key] = unique[0]
            elif unique:
                out[key] = " | ".join(
                    f"model_idx_{idx}:{row.get(key, '')}"
                    for idx, row in valid
                    if str(row.get(key, "")).strip()
                )

    # Retain the source paths for each sample.
    out["result_dir"] = " | ".join(_unique_nonempty([row.get("result_dir") for _, row in valid]))
    out["pred_cif"] = " | ".join(
        f"model_idx_{idx}:{row.get('pred_cif', '')}"
        for idx, row in valid
        if row.get("pred_cif")
    )
    out["score_npz"] = " | ".join(
        f"model_idx_{idx}:{row.get('score_npz', '')}"
        for idx, row in valid
        if row.get("score_npz")
    )

    missing = [str(idx) for idx, row in sample_rows if int(row.get("chai_sample_count_used", 0) or 0) == 0]
    warnings = []
    if missing:
        warnings.append("missing_model_idx_" + "_".join(missing))
    for idx, row in valid:
        status = str(row.get("result_status", ""))
        if status != "OK":
            warnings.append(f"model_idx_{idx}:{status}")
        if row.get("cif_parse_error"):
            warnings.append(f"model_idx_{idx}:cif_parse_error={row['cif_parse_error']}")

    if len(valid) == len(MEAN5_MODEL_INDICES) and not warnings:
        out["result_status"] = "OK_MEAN5"
    else:
        out["result_status"] = f"PARTIAL_{len(valid)}_OF_{len(MEAN5_MODEL_INDICES)}"
        out["chai_mean5_error"] = " | ".join(warnings)

    return out


def find_result_rows(
    result_dir: Path,
    seed: int = 1,
    mode: str = "single",
    sample_idx: int = 0,
    mean5_indices: tuple[int, ...] = MEAN5_MODEL_INDICES,
    max_array_values: int = 20,
    decimals: int = DEFAULT_DECIMALS,
    pocket_cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
):
    """Scan Chai-1 task directories and produce one row per task."""
    rows = []
    task_dirs = sorted(path for path in result_dir.iterdir() if path.is_dir())

    for task_dir in task_dirs:
        if mode == "single":
            row = collect_one_sample(
                task_dir=task_dir,
                seed=seed,
                sample_idx=sample_idx,
                max_array_values=max_array_values,
                decimals=decimals,
                pocket_cutoff_a=pocket_cutoff_a,
            )
        elif mode == "mean5":
            sample_rows = [
                (
                    idx,
                    collect_one_sample(
                        task_dir=task_dir,
                        seed=seed,
                        sample_idx=idx,
                        max_array_values=max_array_values,
                        decimals=decimals,
                        pocket_cutoff_a=pocket_cutoff_a,
                    ),
                )
                for idx in mean5_indices
            ]
            row = aggregate_mean5_rows(
                task_dir=task_dir,
                seed=seed,
                sample_rows=sample_rows,
                decimals=decimals,
                pocket_cutoff_a=pocket_cutoff_a,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Chai-1 scores from one model_idx or the mean of model_idx 0-4, "
            "calculate 5 A residue-weighted pocket pLDDT, and merge into an original CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--result-dir",
        required=True,
        help=(
            "Chai-1 output directory containing task directories such as bind_0001 and nobind_0001. "
            "Each task directory should contain a seed_1 subdirectory."
        ),
    )
    parser.add_argument("-o", "--output-csv", required=True, help="Path to the output CSV.")
    parser.add_argument(
        "--merge-csv",
        required=True,
        help="Path to the input CSV to which Chai-1 scores will be appended.",
    )
    parser.add_argument(
        "--merge-key",
        required=True,
        help="Column in the input CSV that matches task directory names, for example task_id.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "mean5"],
        default="single",
        help="single reads the selected model_idx. mean5 averages numerical fields across model_idx 0–4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed directory number. Use 1 for seed_1 (default: 1).",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="model_idx to read in single mode (default: 0). Ignored in mean5 mode.",
    )
    parser.add_argument(
        "--mean5-indices",
        type=int,
        nargs=5,
        metavar=("M0", "M1", "M2", "M3", "M4"),
        default=list(MEAN5_MODEL_INDICES),
        help="Five model_idx values to use in mean5 mode (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        default=DEFAULT_POCKET_CUTOFF_A,
        help=(
            "Pocket distance cutoff between protein and ligand heavy atoms, in Å (default: 5.0). "
            "Each pocket residue has equal weight in the mean pLDDT."
        ),
    )
    parser.add_argument(
        "--how",
        default="left",
        choices=["left", "inner", "outer", "right"],
        help="CSV merge mode (default: left, which retains every input CSV row).",
    )
    parser.add_argument(
        "--max-array-values",
        type=int,
        default=20,
        help="Store NPZ arrays at or below this length as full JSON values (default: 20).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=DEFAULT_DECIMALS,
        help="Decimal places for scores (default: 6).",
    )

    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    merge_csv = Path(args.merge_csv).resolve()

    if args.seed < 0:
        print("[ERROR] --seed must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.sample_idx < 0 or any(idx < 0 for idx in args.mean5_indices):
        print("[ERROR] model_idx must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.pocket_cutoff <= 0:
        print("[ERROR] --pocket-cutoff must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.decimals < 0:
        print("[ERROR] --decimals must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not result_dir.is_dir():
        print(f"[ERROR] Result directory does not exist: {result_dir}", file=sys.stderr)
        sys.exit(1)
    if not merge_csv.is_file():
        print(f"[ERROR] Original CSV does not exist: {merge_csv}", file=sys.stderr)
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Chai-1 result root: {result_dir}")
    print(f"[INFO] Mode: {args.mode}; seed_{args.seed}")
    if args.mode == "single":
        print(f"[INFO] Fixed model_idx: {args.sample_idx}")
    else:
        print("[INFO] Mean model_idx: " + ",".join(map(str, args.mean5_indices)))
    print(f"[INFO] Pocket cutoff: {args.pocket_cutoff:.2f} A; residue-equal-weighted pLDDT")

    result_rows = find_result_rows(
        result_dir=result_dir,
        seed=args.seed,
        mode=args.mode,
        sample_idx=args.sample_idx,
        mean5_indices=tuple(args.mean5_indices),
        max_array_values=args.max_array_values,
        decimals=args.decimals,
        pocket_cutoff_a=args.pocket_cutoff,
    )
    if not result_rows:
        print("[ERROR] No Chai-1 task folder was found.", file=sys.stderr)
        sys.exit(1)

    result_df = pd.DataFrame(result_rows)
    if args.merge_key not in result_df.columns:
        result_df[args.merge_key] = result_df["task_id"]

    print(f"[INFO] Prediction task rows: {len(result_df)}")
    print(f"[INFO] Reading original CSV: {merge_csv}")
    original_df = pd.read_csv(merge_csv, encoding="utf-8-sig")

    if args.merge_key not in original_df.columns:
        print(
            f"[ERROR] merge-key column not found in original CSV: {args.merge_key}",
            file=sys.stderr,
        )
        print("[INFO] Original CSV columns:", file=sys.stderr)
        for column in original_df.columns:
            print(f"  - {column}", file=sys.stderr)
        sys.exit(1)

    original_df[args.merge_key] = original_df[args.merge_key].astype(str).str.strip()
    result_df[args.merge_key] = result_df[args.merge_key].astype(str).str.strip()

    duplicate_count = int(result_df[args.merge_key].duplicated(keep=False).sum())
    if duplicate_count:
        print(
            "[ERROR] Multiple Chai-1 rows share the same merge key; this script should "
            "produce exactly one row per task. Please inspect duplicated task folders.",
            file=sys.stderr,
        )
        duplicated = result_df.loc[
            result_df[args.merge_key].duplicated(keep=False), args.merge_key
        ].tolist()
        print("[ERROR] Duplicated keys: " + ", ".join(map(str, duplicated)), file=sys.stderr)
        sys.exit(1)

    merged_df = original_df.merge(
        result_df,
        on=args.merge_key,
        how=args.how,
        suffixes=("", "_chai"),
        sort=False,
    )
    merged_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=f"%.{args.decimals}f",
    )

    print("[DONE] Chai-1 score collection and CSV merge completed")
    print(f"[INFO] Original CSV rows: {len(original_df)}")
    print(f"[INFO] Prediction task rows: {len(result_df)}")
    print(f"[INFO] Merged rows: {len(merged_df)}")
    print(f"[INFO] Output: {output_csv}")

    if args.how == "left":
        no_result = int(merged_df["result_status"].isna().sum())
        print(f"[INFO] Original rows without matched Chai-1 results: {no_result}")


if __name__ == "__main__":
    main()

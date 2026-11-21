#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect AlphaFold 3 confidence scores and optionally merge them into the input CSV by task directory name.

Scoring conventions
1. --mode single reads the requested seed and sample (by default, seed 1 and sample 0), regardless of rank.
2. --mode mean5 reads samples 0 through 4 for one seed and averages each numerical score.
3. For protein pLDDT, use the official per-residue values in the AF3 mmCIF _ma_qa_metric_local table.
   For older mmCIF files, first average atomic pLDDT within each residue. In both cases, give each
   protein residue equal weight. protein_plddt_count is the number of protein residues.
4. pocket_mean_plddt_5A includes protein residues with a heavy atom within 5.0 Å of any ligand heavy atom.
   Each pocket residue contributes one pLDDT value and receives equal weight in the mean.
5. Ligand pLDDT is the equally weighted mean of ligand atom_plddts from AF3 confidences.json.
   ligand_plddt_count is the number of ligand atoms reported by the model.
6. Ligand_protein_chain_pair_iPTM uses the chain_pair_iptm matrix in summary_confidences.json.
   Read the ligand row and protein column for a single protein chain. For multiple protein chains, average all ligand-to-protein entries.
7. Floating-point scores are written to six decimal places by default.

Example directory layout
<input_dir>/bind_0001/seed-1_sample-0/
    bind_0001_seed-1_sample-0_confidences.json
    bind_0001_seed-1_sample-0_model.cif
    bind_0001_seed-1_sample-0_summary_confidences.json
"""

import argparse
import csv
import json
import shlex
from collections import defaultdict
from pathlib import Path


POCKET_CUTOFF_A = 5.0
MEAN5_SAMPLES = (0, 1, 2, 3, 4)


SCORE_COLUMNS = [
    "af3_folder_name",
    "af3_status",
    "score_mode",
    "sample_count_used",
    "samples_used",

    "ranking_score",
    "ranking_score_seed",
    "ranking_score_sample",
    "ranking_score_source",

    "iptm",
    "Ligand_protein_chain_pair_iPTM",
    "ptm",
    "fraction_disordered",
    "has_clash",

    "protein_mean_plddt",
    "protein_min_plddt",
    "protein_max_plddt",
    "protein_plddt_count",

    "pocket_mean_plddt_5A",
    "pocket_residue_count_5A",

    "ligand_mean_plddt",
    "ligand_min_plddt",
    "ligand_max_plddt",
    "ligand_plddt_count",

    "chain_ptm_A",
    "chain_ptm_B",
    "chain_iptm_A",
    "chain_iptm_B",

    "chain_pair_iptm_A_B",
    "chain_pair_iptm_B_A",
    "chain_pair_pae_min_A_B",
    "chain_pair_pae_min_B_A",

    "chain_pair_pde_mean_A_B",
    "chain_pair_pde_mean_B_A",
    "chain_pair_pde_min_A_B",
    "chain_pair_pde_min_B_A",

    "summary_json_path",
    "confidences_json_path",
    "ranking_scores_csv_path",
    "model_cif_path",
]


FLOAT_COLUMNS = {
    "ranking_score",
    "iptm",
    "Ligand_protein_chain_pair_iPTM",
    "ptm",
    "fraction_disordered",

    "protein_mean_plddt",
    "protein_min_plddt",
    "protein_max_plddt",

    "pocket_mean_plddt_5A",
    "pocket_residue_count_5A",

    "ligand_mean_plddt",
    "ligand_min_plddt",
    "ligand_max_plddt",

    "chain_ptm_A",
    "chain_ptm_B",
    "chain_iptm_A",
    "chain_iptm_B",

    "chain_pair_iptm_A_B",
    "chain_pair_iptm_B_A",
    "chain_pair_pae_min_A_B",
    "chain_pair_pae_min_B_A",

    "chain_pair_pde_mean_A_B",
    "chain_pair_pde_mean_B_A",
    "chain_pair_pde_min_A_B",
    "chain_pair_pde_min_B_A",
}


INT_COLUMNS = {
    "sample_count_used",
    "ranking_score_seed",
    "ranking_score_sample",
    "protein_plddt_count",
    "ligand_plddt_count",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value, digits):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_int(value):
    if value is None or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def format_score_row(row, float_digits):
    out = {}
    for col in SCORE_COLUMNS:
        value = row.get(col, "")
        if col in FLOAT_COLUMNS:
            out[col] = format_float(value, float_digits)
        elif col in INT_COLUMNS:
            out[col] = format_int(value)
        else:
            out[col] = "" if value is None else str(value)
    return out


def find_direct_file(directory, patterns, exclude_keywords=None):
    """Search only this directory to avoid reading files from another sample."""
    if directory is None or not directory.exists():
        return None
    if exclude_keywords is None:
        exclude_keywords = []

    candidates = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))

    clean = []
    for path in candidates:
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(keyword.lower() in lower_name for keyword in exclude_keywords):
            continue
        clean.append(path)

    return sorted(clean, key=lambda p: str(p))[0] if clean else None


def safe_get(mapping, key, default=""):
    value = mapping.get(key, default)
    return "" if value is None else value


def get_list_value(mapping, key, index):
    value = mapping.get(key)
    if not isinstance(value, list) or index is None or index < 0 or index >= len(value):
        return ""
    item = value[index]
    return "" if item is None else item


def get_matrix_value(mapping, key, row_index, col_index):
    matrix = mapping.get(key)
    if not isinstance(matrix, list):
        return ""
    if row_index is None or col_index is None:
        return ""
    try:
        item = matrix[row_index][col_index]
    except (IndexError, TypeError):
        return ""
    return "" if item is None else item


def numeric_mean(values):
    return "" if not values else sum(values) / len(values)


def make_stats(values):
    if not values:
        return {"mean": "", "min": "", "max": "", "count": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _tokenize_cif_row(line):
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def parse_atom_site(cif_path):
    """
    Read the _atom_site loop from an AF3 mmCIF file.

    In addition to per-atom pLDDT, read Cartesian coordinates and elements
    to identify the 5 Å pocket from protein–ligand heavy-atom distances.
    """
    if cif_path is None or not cif_path.exists():
        return []

    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith("_atom_site."):
            headers.append(lines[i].strip())
            i += 1

        if not headers:
            continue

        required = {
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        }
        if not required.issubset(headers):
            continue

        header_index = {name: idx for idx, name in enumerate(headers)}
        pending_tokens = []
        records = []

        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped == "#":
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break

            pending_tokens.extend(_tokenize_cif_row(stripped))
            while len(pending_tokens) >= len(headers):
                fields = pending_tokens[:len(headers)]
                pending_tokens = pending_tokens[len(headers):]

                def field(name, default=""):
                    idx = header_index.get(name)
                    return default if idx is None else fields[idx]

                try:
                    plddt = float(field("_atom_site.B_iso_or_equiv"))
                    x = float(field("_atom_site.Cartn_x"))
                    y = float(field("_atom_site.Cartn_y"))
                    z = float(field("_atom_site.Cartn_z"))
                except (TypeError, ValueError):
                    continue

                records.append({
                    "group_pdb": field("_atom_site.group_PDB").upper(),
                    "atom_id": field("_atom_site.label_atom_id"),
                    "element": field("_atom_site.type_symbol"),
                    "comp_id": field("_atom_site.label_comp_id"),
                    "chain_id": field("_atom_site.label_asym_id"),
                    "entity_id": field("_atom_site.label_entity_id"),
                    "label_seq_id": field("_atom_site.label_seq_id"),
                    "auth_seq_id": field("_atom_site.auth_seq_id"),
                    "insertion_code": field("_atom_site.pdbx_PDB_ins_code"),
                    "x": x,
                    "y": y,
                    "z": z,
                    "plddt": plddt,
                })

            i += 1

        return records

    return []


def _read_cif_loop(lines, loop_prefix):
    """Return (headers, rows) for the requested mmCIF loop in standard AF3 output."""
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1

        if not any(header.startswith(loop_prefix) for header in headers):
            continue

        pending_tokens = []
        rows = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped == "#":
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break

            pending_tokens.extend(_tokenize_cif_row(stripped))
            while len(pending_tokens) >= len(headers):
                rows.append(pending_tokens[:len(headers)])
                pending_tokens = pending_tokens[len(headers):]
            i += 1

        return headers, rows

    return [], []


def parse_native_local_plddt_by_chain(cif_path):
    """
    Read official per-residue or per-component local pLDDT from AF3 v3.0.1+ mmCIF files.
    Values come from _ma_qa_metric_local.metric_value.

    Return {chain_id: [local_plddt, ...]}, with one value per protein residue.
    A single-component ligand usually has one local value. Ligand summaries use per-atom pLDDT.
    """
    if cif_path is None or not cif_path.exists():
        return {}

    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    metric_headers, metric_rows = _read_cif_loop(lines, "_ma_qa_metric.")
    local_plddt_metric_ids = set()
    if metric_headers:
        metric_index = {name: idx for idx, name in enumerate(metric_headers)}
        id_col = metric_index.get("_ma_qa_metric.id")
        mode_col = metric_index.get("_ma_qa_metric.mode")
        name_col = metric_index.get("_ma_qa_metric.name")
        type_col = metric_index.get("_ma_qa_metric.type")

        for fields in metric_rows:
            try:
                metric_id = fields[id_col] if id_col is not None else ""
                mode = fields[mode_col] if mode_col is not None else ""
                name = fields[name_col] if name_col is not None else ""
                metric_type = fields[type_col] if type_col is not None else ""
            except IndexError:
                continue

            descriptor = f"{name} {metric_type}".lower()
            if str(mode).lower() == "local" and "plddt" in descriptor:
                local_plddt_metric_ids.add(str(metric_id))

    local_headers, local_rows = _read_cif_loop(lines, "_ma_qa_metric_local.")
    if not local_headers:
        return {}

    local_index = {name: idx for idx, name in enumerate(local_headers)}
    chain_col = local_index.get("_ma_qa_metric_local.label_asym_id")
    metric_id_col = local_index.get("_ma_qa_metric_local.metric_id")
    value_col = local_index.get("_ma_qa_metric_local.metric_value")
    if chain_col is None or value_col is None:
        return {}

    chain_to_values = defaultdict(list)
    for fields in local_rows:
        try:
            chain_id = str(fields[chain_col])
            metric_id = str(fields[metric_id_col]) if metric_id_col is not None else ""
            value = float(fields[value_col])
        except (IndexError, TypeError, ValueError):
            continue

        # When metric definitions are available, keep only the metric_id for local pLDDT.
        if local_plddt_metric_ids and metric_id not in local_plddt_metric_ids:
            continue
        chain_to_values[chain_id].append(value)

    return dict(chain_to_values)


def protein_native_residue_plddt_stats(native_local_plddt, protein_chains):
    """Summarize official AF3 per-residue pLDDT, giving each residue equal weight."""
    values = []
    for chain_id in protein_chains:
        values.extend(native_local_plddt.get(chain_id, []))
    return make_stats(values)


def parse_native_local_plddt_by_residue(cif_path):
    """Return a mapping of {(chain_id, label_seq_id): official_residue_plddt}."""
    if cif_path is None or not cif_path.exists():
        return {}

    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    metric_headers, metric_rows = _read_cif_loop(lines, "_ma_qa_metric.")
    local_plddt_metric_ids = set()
    if metric_headers:
        metric_index = {name: idx for idx, name in enumerate(metric_headers)}
        id_col = metric_index.get("_ma_qa_metric.id")
        mode_col = metric_index.get("_ma_qa_metric.mode")
        name_col = metric_index.get("_ma_qa_metric.name")
        type_col = metric_index.get("_ma_qa_metric.type")

        for fields in metric_rows:
            try:
                metric_id = fields[id_col] if id_col is not None else ""
                mode = fields[mode_col] if mode_col is not None else ""
                name = fields[name_col] if name_col is not None else ""
                metric_type = fields[type_col] if type_col is not None else ""
            except IndexError:
                continue

            descriptor = f"{name} {metric_type}".lower()
            if str(mode).lower() == "local" and "plddt" in descriptor:
                local_plddt_metric_ids.add(str(metric_id))

    local_headers, local_rows = _read_cif_loop(lines, "_ma_qa_metric_local.")
    if not local_headers:
        return {}

    local_index = {name: idx for idx, name in enumerate(local_headers)}
    chain_col = local_index.get("_ma_qa_metric_local.label_asym_id")
    seq_col = local_index.get("_ma_qa_metric_local.label_seq_id")
    metric_id_col = local_index.get("_ma_qa_metric_local.metric_id")
    value_col = local_index.get("_ma_qa_metric_local.metric_value")
    if chain_col is None or seq_col is None or value_col is None:
        return {}

    result = {}
    for fields in local_rows:
        try:
            chain_id = str(fields[chain_col])
            seq_id = str(fields[seq_col])
            metric_id = str(fields[metric_id_col]) if metric_id_col is not None else ""
            value = float(fields[value_col])
        except (IndexError, TypeError, ValueError):
            continue

        if seq_id in {"", ".", "?"}:
            continue
        if local_plddt_metric_ids and metric_id not in local_plddt_metric_ids:
            continue
        result[(chain_id, seq_id)] = value

    return result


def atom_residue_key(atom):
    label_seq_id = str(atom.get("label_seq_id", ""))
    auth_seq_id = str(atom.get("auth_seq_id", ""))
    seq_id = label_seq_id if label_seq_id not in {"", ".", "?"} else auth_seq_id
    if seq_id in {"", ".", "?"}:
        return None
    return (
        str(atom.get("chain_id", "")),
        seq_id,
        str(atom.get("insertion_code", "")),
        str(atom.get("comp_id", "")),
    )


def is_hydrogen_atom_record(atom):
    element = str(atom.get("element", "")).strip().upper()
    if element:
        return element in {"H", "D"}
    atom_name = str(atom.get("atom_id", "")).strip().upper()
    return atom_name.startswith("H") or atom_name.startswith("D")


def protein_residue_atom_plddt_map(atom_records, protein_chains):
    """For older CIF files, average atomic pLDDT within each residue."""
    protein_set = set(protein_chains)
    residue_to_atom_plddt = defaultdict(list)
    for atom in atom_records:
        if str(atom.get("chain_id", "")) not in protein_set:
            continue
        residue_key = atom_residue_key(atom)
        if residue_key is None:
            continue
        residue_to_atom_plddt[residue_key].append(float(atom["plddt"]))

    return {
        residue_key: sum(values) / len(values)
        for residue_key, values in residue_to_atom_plddt.items()
        if values
    }


def pocket_residue_plddt_stats(
    atom_records,
    protein_chains,
    ligand_chains,
    native_local_plddt_by_residue,
    cutoff_a=POCKET_CUTOFF_A,
):
    """
    Select protein residues with a heavy atom within cutoff_a of any ligand heavy atom.

    Assign one pLDDT value to each pocket residue and weight all residues equally.
    The number of side-chain atoms therefore does not affect a residue's weight.
    """
    protein_set = set(protein_chains)
    ligand_set = set(ligand_chains)

    ligand_coords = [
        (float(atom["x"]), float(atom["y"]), float(atom["z"]))
        for atom in atom_records
        if str(atom.get("chain_id", "")) in ligand_set
        and not is_hydrogen_atom_record(atom)
    ]
    if not ligand_coords:
        return make_stats([]), "unavailable"

    residue_to_heavy_coords = defaultdict(list)
    for atom in atom_records:
        if str(atom.get("chain_id", "")) not in protein_set:
            continue
        if is_hydrogen_atom_record(atom):
            continue
        residue_key = atom_residue_key(atom)
        if residue_key is None:
            continue
        residue_to_heavy_coords[residue_key].append(
            (float(atom["x"]), float(atom["y"]), float(atom["z"]))
        )

    cutoff2 = float(cutoff_a) ** 2
    pocket_residue_keys = []
    for residue_key, protein_coords in residue_to_heavy_coords.items():
        is_pocket = False
        for px, py, pz in protein_coords:
            for lx, ly, lz in ligand_coords:
                dx = px - lx
                dy = py - ly
                dz = pz - lz
                if dx * dx + dy * dy + dz * dz <= cutoff2:
                    is_pocket = True
                    break
            if is_pocket:
                break
        if is_pocket:
            pocket_residue_keys.append(residue_key)

    atom_fallback_map = protein_residue_atom_plddt_map(atom_records, protein_chains)
    values = []
    native_count = 0
    fallback_count = 0
    for residue_key in pocket_residue_keys:
        chain_id, seq_id, _, _ = residue_key
        value = native_local_plddt_by_residue.get((chain_id, seq_id))
        if value is not None:
            values.append(float(value))
            native_count += 1
            continue

        value = atom_fallback_map.get(residue_key)
        if value is not None:
            values.append(float(value))
            fallback_count += 1

    if not values:
        source = "unavailable"
    elif fallback_count == 0:
        source = "native_residue_plddt"
    elif native_count == 0:
        source = "atom_to_residue_fallback"
    else:
        source = "mixed_native_and_fallback"

    return make_stats(values), source


def ordered_chain_ids_from_atoms(atom_records):
    seen = set()
    ordered = []
    for atom in atom_records:
        chain_id = str(atom.get("chain_id", ""))
        if chain_id and chain_id not in seen:
            seen.add(chain_id)
            ordered.append(chain_id)
    return ordered


def identify_protein_and_ligand_chains(atom_records, chain_ids):
    """
    AF3 typically writes protein and peptide polymer atoms as ATOM and small-molecule ligands as HETATM.
    Classify chains containing ATOM records as protein and chains containing only HETATM records as ligand.
    """
    atom_count = defaultdict(int)
    hetatm_count = defaultdict(int)

    for atom in atom_records:
        chain_id = str(atom.get("chain_id", ""))
        if atom.get("group_pdb") == "ATOM":
            atom_count[chain_id] += 1
        elif atom.get("group_pdb") == "HETATM":
            hetatm_count[chain_id] += 1

    protein_chains = [cid for cid in chain_ids if atom_count[cid] > 0]
    ligand_chains = [cid for cid in chain_ids if atom_count[cid] == 0 and hetatm_count[cid] > 0]

    # Use a conservative fallback for atypical or simplified CIF files.
    if not ligand_chains and len(chain_ids) >= 2:
        ligand_chains = [chain_ids[-1]]
    if not protein_chains:
        protein_chains = [cid for cid in chain_ids if cid not in ligand_chains]

    return protein_chains, ligand_chains


def protein_residue_plddt_stats(atom_records, protein_chains):
    """
    For older mmCIF files, average atomic pLDDT within each residue, then give all residues equal weight.
    The minimum and maximum are residue-level pLDDT values, and the count is the number of residues.
    """
    protein_set = set(protein_chains)
    residue_to_atom_plddt = defaultdict(list)

    for atom in atom_records:
        chain_id = str(atom.get("chain_id", ""))
        if chain_id not in protein_set:
            continue

        label_seq_id = str(atom.get("label_seq_id", ""))
        auth_seq_id = str(atom.get("auth_seq_id", ""))
        seq_id = label_seq_id if label_seq_id not in {"", ".", "?"} else auth_seq_id
        if seq_id in {"", ".", "?"}:
            continue

        residue_key = (
            chain_id,
            seq_id,
            str(atom.get("insertion_code", "")),
            str(atom.get("comp_id", "")),
        )
        residue_to_atom_plddt[residue_key].append(float(atom["plddt"]))

    residue_plddts = [
        sum(atom_values) / len(atom_values)
        for atom_values in residue_to_atom_plddt.values()
        if atom_values
    ]
    return make_stats(residue_plddts)


def ligand_atom_plddt_stats(atom_records, ligand_chains):
    ligand_set = set(ligand_chains)
    values = [
        float(atom["plddt"])
        for atom in atom_records
        if str(atom.get("chain_id", "")) in ligand_set
    ]
    return make_stats(values)


def ligand_atom_plddt_stats_from_conf_json(conf_json, ligand_chains):
    """Use per-atom ligand pLDDT from confidence JSON as a fallback when the CIF is missing."""
    chain_ids = conf_json.get("atom_chain_ids")
    plddts = conf_json.get("atom_plddts")
    if not isinstance(chain_ids, list) or not isinstance(plddts, list):
        return make_stats([])
    if len(chain_ids) != len(plddts):
        return make_stats([])

    ligand_set = set(ligand_chains)
    values = []
    for chain_id, score in zip(chain_ids, plddts):
        if str(chain_id) not in ligand_set or score is None:
            continue
        try:
            values.append(float(score))
        except (TypeError, ValueError):
            pass
    return make_stats(values)


def parse_int_like(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def read_selected_ranking_score_from_csv(path, seed, sample):
    """Read only the requested seed and sample from ranking_scores.csv in the task directory."""
    blank = {"ranking_score": "", "seed": "", "sample": ""}
    if path is None or not path.exists():
        return blank

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return blank

    score_key = next((key for key in ["ranking_score", "score", "rank_score"] if key in rows[0]), None)
    seed_key = next((key for key in ["seed", "seed_id", "random_seed"] if key in rows[0]), None)
    sample_key = next((key for key in ["sample", "sample_id"] if key in rows[0]), None)
    if score_key is None or seed_key is None or sample_key is None:
        return blank

    for row in rows:
        if parse_int_like(row.get(seed_key)) != seed:
            continue
        if parse_int_like(row.get(sample_key)) != sample:
            continue
        score = row.get(score_key, "")
        if score == "":
            return blank
        return {
            "ranking_score": score,
            "seed": row.get(seed_key, seed),
            "sample": row.get(sample_key, sample),
        }

    return blank


def index_of_chain(chain_ids, chain_id):
    try:
        return chain_ids.index(chain_id)
    except ValueError:
        return None


def calculate_ligand_protein_chain_pair_iptm(summary, chain_ids, protein_chains, ligand_chains):
    """
    Read chain_pair_iptm with ligand chains as rows and protein chains as columns.
    For multiple protein chains, give each ligand-to-protein chain pair equal weight.
    """
    matrix = summary.get("chain_pair_iptm")
    if not isinstance(matrix, list):
        return ""

    chain_to_index = {chain_id: idx for idx, chain_id in enumerate(chain_ids)}
    values = []

    for ligand_chain in ligand_chains:
        ligand_index = chain_to_index.get(ligand_chain)
        if ligand_index is None:
            continue
        for protein_chain in protein_chains:
            protein_index = chain_to_index.get(protein_chain)
            if protein_index is None:
                continue
            try:
                value = matrix[ligand_index][protein_index]
            except (IndexError, TypeError):
                continue
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass

    return numeric_mean(values)


def collect_one_job(job_dir, seed, sample):
    """Read one requested sample and return its unformatted numerical values."""
    folder_name = job_dir.name
    sample_dir_name = f"seed-{seed}_sample-{sample}"
    sample_dir = job_dir / sample_dir_name

    row = {col: "" for col in SCORE_COLUMNS}
    row["af3_folder_name"] = folder_name
    row["score_mode"] = "single"
    row["sample_count_used"] = 0
    row["samples_used"] = ""
    row["ranking_score_seed"] = seed
    row["ranking_score_sample"] = sample

    if not sample_dir.is_dir():
        row["af3_status"] = f"missing_{sample_dir_name}"
        return row

    summary_path = find_direct_file(
        sample_dir,
        patterns=["*summary_confidences*.json", "*summary*.json"],
    )
    confidences_path = find_direct_file(
        sample_dir,
        patterns=["*confidences*.json"],
        exclude_keywords=["summary"],
    )
    model_cif_path = find_direct_file(
        sample_dir,
        patterns=["*model.cif", "*.cif"],
    )
    ranking_csv_path = find_direct_file(
        job_dir,
        patterns=["*ranking_scores*.csv", "*ranking*.csv"],
    )

    row["summary_json_path"] = str(summary_path) if summary_path else ""
    row["confidences_json_path"] = str(confidences_path) if confidences_path else ""
    row["ranking_scores_csv_path"] = str(ranking_csv_path) if ranking_csv_path else ""
    row["model_cif_path"] = str(model_cif_path) if model_cif_path else ""

    warnings = []
    if summary_path is None:
        row["af3_status"] = "missing_summary_confidences_json"
        return row

    # A summary for the requested sample is sufficient to include its available scores in single or mean5 mode.
    row["sample_count_used"] = 1
    row["samples_used"] = str(sample)
    summary = load_json(summary_path)

    ranking_info = read_selected_ranking_score_from_csv(
        ranking_csv_path,
        seed=seed,
        sample=sample,
    )
    if ranking_info["ranking_score"] != "":
        row["ranking_score"] = ranking_info["ranking_score"]
        row["ranking_score_seed"] = ranking_info["seed"]
        row["ranking_score_sample"] = ranking_info["sample"]
        row["ranking_score_source"] = "ranking_scores_csv_selected_sample"
    else:
        row["ranking_score"] = safe_get(summary, "ranking_score")
        row["ranking_score_source"] = "selected_sample_summary_confidences_json"

    row["iptm"] = safe_get(summary, "iptm")
    row["ptm"] = safe_get(summary, "ptm")
    row["fraction_disordered"] = safe_get(summary, "fraction_disordered")
    row["has_clash"] = safe_get(summary, "has_clash")

    atom_records = parse_atom_site(model_cif_path)
    if not atom_records:
        warnings.append("missing_or_unreadable_model_cif")

    summary_chain_ids = summary.get("chain_ids")
    if isinstance(summary_chain_ids, list) and summary_chain_ids:
        chain_ids = [str(chain_id) for chain_id in summary_chain_ids]
    else:
        chain_ids = ordered_chain_ids_from_atoms(atom_records)

    protein_chains, ligand_chains = identify_protein_and_ligand_chains(atom_records, chain_ids)
    row["Ligand_protein_chain_pair_iPTM"] = calculate_ligand_protein_chain_pair_iptm(
        summary=summary,
        chain_ids=chain_ids,
        protein_chains=protein_chains,
        ligand_chains=ligand_chains,
    )

    # Locate chains A and B in chain_ids to retain the original A/B score components.
    index_a = index_of_chain(chain_ids, "A")
    index_b = index_of_chain(chain_ids, "B")

    row["chain_ptm_A"] = get_list_value(summary, "chain_ptm", index_a)
    row["chain_ptm_B"] = get_list_value(summary, "chain_ptm", index_b)
    row["chain_iptm_A"] = get_list_value(summary, "chain_iptm", index_a)
    row["chain_iptm_B"] = get_list_value(summary, "chain_iptm", index_b)

    row["chain_pair_iptm_A_B"] = get_matrix_value(summary, "chain_pair_iptm", index_a, index_b)
    row["chain_pair_iptm_B_A"] = get_matrix_value(summary, "chain_pair_iptm", index_b, index_a)
    row["chain_pair_pae_min_A_B"] = get_matrix_value(summary, "chain_pair_pae_min", index_a, index_b)
    row["chain_pair_pae_min_B_A"] = get_matrix_value(summary, "chain_pair_pae_min", index_b, index_a)
    row["chain_pair_pde_mean_A_B"] = get_matrix_value(summary, "chain_pair_pde_mean", index_a, index_b)
    row["chain_pair_pde_mean_B_A"] = get_matrix_value(summary, "chain_pair_pde_mean", index_b, index_a)
    row["chain_pair_pde_min_A_B"] = get_matrix_value(summary, "chain_pair_pde_min", index_a, index_b)
    row["chain_pair_pde_min_B_A"] = get_matrix_value(summary, "chain_pair_pde_min", index_b, index_a)

    # Prefer official per-residue local pLDDT for proteins, with a fallback for older files.
    native_local_plddt = parse_native_local_plddt_by_chain(model_cif_path)
    protein_stats = protein_native_residue_plddt_stats(
        native_local_plddt=native_local_plddt,
        protein_chains=protein_chains,
    )
    if protein_stats["count"] == 0:
        protein_stats = protein_residue_plddt_stats(atom_records, protein_chains)
        if protein_stats["count"] > 0:
            warnings.append("protein_plddt_fallback_atom_to_residue")
        else:
            warnings.append("protein_residue_plddt_unavailable")

    # Average the 5 Å pocket confidence using one pLDDT value per residue.
    native_local_plddt_by_residue = parse_native_local_plddt_by_residue(model_cif_path)
    pocket_stats, pocket_source = pocket_residue_plddt_stats(
        atom_records=atom_records,
        protein_chains=protein_chains,
        ligand_chains=ligand_chains,
        native_local_plddt_by_residue=native_local_plddt_by_residue,
        cutoff_a=POCKET_CUTOFF_A,
    )
    if pocket_stats["count"] == 0:
        warnings.append("pocket_residue_plddt_unavailable")
    elif pocket_source == "atom_to_residue_fallback":
        warnings.append("pocket_plddt_fallback_atom_to_residue")
    elif pocket_source == "mixed_native_and_fallback":
        warnings.append("pocket_plddt_mixed_native_and_fallback")

    # Give each ligand atom equal weight in the mean pLDDT.
    if confidences_path is not None:
        conf = load_json(confidences_path)
        ligand_stats = ligand_atom_plddt_stats_from_conf_json(conf, ligand_chains)
    else:
        ligand_stats = make_stats([])
        warnings.append("missing_confidences_json")

    if ligand_stats["count"] == 0:
        ligand_stats = ligand_atom_plddt_stats(atom_records, ligand_chains)

    if ligand_stats["count"] == 0:
        warnings.append("ligand_atom_plddt_unavailable")

    row["protein_mean_plddt"] = protein_stats["mean"]
    row["protein_min_plddt"] = protein_stats["min"]
    row["protein_max_plddt"] = protein_stats["max"]
    row["protein_plddt_count"] = protein_stats["count"]

    row["pocket_mean_plddt_5A"] = pocket_stats["mean"]
    row["pocket_residue_count_5A"] = pocket_stats["count"]

    row["ligand_mean_plddt"] = ligand_stats["mean"]
    row["ligand_min_plddt"] = ligand_stats["min"]
    row["ligand_max_plddt"] = ligand_stats["max"]
    row["ligand_plddt_count"] = ligand_stats["count"]

    row["af3_status"] = "ok" if not warnings else ";".join(warnings)
    return row


def numeric_value(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_mean5_rows(job_dir, seed, sample_rows):
    """Average each score column across samples 0–4 after scoring each sample independently."""
    out = {col: "" for col in SCORE_COLUMNS}
    out["af3_folder_name"] = job_dir.name
    out["score_mode"] = "mean5"
    out["ranking_score_seed"] = seed
    out["ranking_score_sample"] = ""
    out["ranking_score_source"] = "mean_of_available_samples_0_1_2_3_4"

    valid = [
        (sample_id, row)
        for sample_id, row in sample_rows
        if int(row.get("sample_count_used", 0) or 0) == 1
    ]
    out["sample_count_used"] = len(valid)
    out["samples_used"] = ",".join(str(sample_id) for sample_id, _ in valid)

    if not valid:
        out["af3_status"] = "missing_all_mean5_samples"
        return out

    # Average each numerical score, including pocket_mean_plddt_5A, over the available samples.
    for col in FLOAT_COLUMNS:
        values = []
        for _, row in valid:
            value = numeric_value(row.get(col, ""))
            if value is not None:
                values.append(value)
        out[col] = numeric_mean(values)

    # Counts usually agree across samples. If they differ, report the rounded mean and flag the discrepancy.
    count_warnings = []
    for col in ("protein_plddt_count", "ligand_plddt_count"):
        values = []
        for _, row in valid:
            value = numeric_value(row.get(col, ""))
            if value is not None:
                values.append(value)
        if values:
            if len(set(values)) > 1:
                count_warnings.append(f"{col}_varied")
            out[col] = int(round(sum(values) / len(values)))

    # Keep a shared clash flag when all samples agree, and report mixed flags explicitly.
    clash_values = [str(row.get("has_clash", "")).strip() for _, row in valid]
    clash_values = [value for value in clash_values if value != ""]
    if clash_values:
        normalized = [value.lower() in {"true", "1", "yes"} for value in clash_values]
        if all(value == normalized[0] for value in normalized):
            out["has_clash"] = str(normalized[0])
        else:
            out["has_clash"] = f"mixed_true_{sum(normalized)}_of_{len(normalized)}"

    # Keep the source paths for all five samples for traceability.
    for col in (
        "summary_json_path",
        "confidences_json_path",
        "ranking_scores_csv_path",
        "model_cif_path",
    ):
        paths = []
        for _, row in valid:
            value = str(row.get(col, "")).strip()
            if value and value not in paths:
                paths.append(value)
        out[col] = " | ".join(paths)

    sample_warnings = []
    missing_samples = [
        str(sample_id)
        for sample_id, row in sample_rows
        if int(row.get("sample_count_used", 0) or 0) == 0
    ]
    if missing_samples:
        sample_warnings.append("missing_samples_" + "_".join(missing_samples))

    for sample_id, row in valid:
        status = str(row.get("af3_status", ""))
        if status and status != "ok":
            sample_warnings.append(f"sample{sample_id}:{status}")

    sample_warnings.extend(count_warnings)
    if len(valid) == len(MEAN5_SAMPLES) and not sample_warnings:
        out["af3_status"] = "ok_mean5"
    else:
        prefix = f"mean5_used_{len(valid)}"
        out["af3_status"] = prefix if not sample_warnings else prefix + ";" + ";".join(sample_warnings)

    return out


def collect_all_scores(input_dir, seed, sample, mode, float_digits):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"AF3 output directory not found: {input_dir}")

    job_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    score_by_folder_name = {}

    for job_dir in job_dirs:
        if mode == "single":
            raw_row = collect_one_job(job_dir=job_dir, seed=seed, sample=sample)
        elif mode == "mean5":
            sample_rows = [
                (sample_id, collect_one_job(job_dir=job_dir, seed=seed, sample=sample_id))
                for sample_id in MEAN5_SAMPLES
            ]
            raw_row = aggregate_mean5_rows(job_dir=job_dir, seed=seed, sample_rows=sample_rows)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        score_row = format_score_row(raw_row, float_digits)
        score_by_folder_name[score_row["af3_folder_name"]] = score_row

    return score_by_folder_name, len(job_dirs)


def read_merge_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV for the merge not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)

    return rows, fieldnames


def write_score_only_csv(out_csv, score_by_folder_name):
    rows = [score_by_folder_name[key] for key in sorted(score_by_folder_name)]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_merged_csv(out_csv, merge_csv, merge_key, score_by_folder_name):
    original_rows, original_cols = read_merge_csv(merge_csv)
    if merge_key not in original_cols:
        raise ValueError(
            f"Merge column not found: {merge_key}\n"
            f"Available columns: {','.join(original_cols)}"
        )

    # Avoid duplicate headers if the input already contains AF3 score columns.
    final_cols = list(original_cols) + [col for col in SCORE_COLUMNS if col not in original_cols]

    merged_rows = []
    matched = 0
    missing = 0

    for original_row in original_rows:
        key_value = str(original_row.get(merge_key, "")).strip()
        score_row = score_by_folder_name.get(key_value)
        new_row = dict(original_row)

        if score_row is None:
            blank_score = {col: "" for col in SCORE_COLUMNS}
            blank_score["af3_folder_name"] = key_value
            blank_score["af3_status"] = "missing_af3_output"
            new_row.update(blank_score)
            missing += 1
        else:
            new_row.update(score_row)
            matched += 1

        merged_rows.append(new_row)

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    return matched, missing, len(merged_rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect AlphaFold 3 scores in single-sample mode or as the mean of samples 0-4, "
            "and optionally merge them with the original CSV."
        )
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help=(
            "AF3 output directory containing task directories such as bind_0001 and nobind_0001, "
            "for example /path/to/project/af3/output."
        ),
    )
    parser.add_argument("-o", "--out_csv", required=True, help="Path to the output CSV.")
    parser.add_argument(
        "--merge_csv",
        default=None,
        help="Optional input CSV to which AF3 scores will be appended.",
    )
    parser.add_argument(
        "--merge_key",
        default=None,
        help="Column in the input CSV that matches AF3 task directory names, for example task_id.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "mean5"],
        default="single",
        help=(
            "single reads one requested sample. mean5 averages each score column across samples 0–4 for the same seed. "
            "The default is single."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to read (default: 1).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample to read in single mode (default: 0). Ignored in mean5 mode.",
    )
    parser.add_argument(
        "--float_digits",
        type=int,
        default=6,
        help="Decimal places for numerical AF3 scores (default: 6).",
    )

    args = parser.parse_args()

    score_by_folder_name, job_count = collect_all_scores(
        input_dir=args.input_dir,
        seed=args.seed,
        sample=args.sample,
        mode=args.mode,
        float_digits=args.float_digits,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_csv is not None:
        if args.merge_key is None:
            raise ValueError("Specify --merge_key when using --merge_csv.")

        matched, missing, total = write_merged_csv(
            out_csv=out_csv,
            merge_csv=args.merge_csv,
            merge_key=args.merge_key,
            score_by_folder_name=score_by_folder_name,
        )

        print("完成：已合并原始 CSV 和 AF3 分数")
        print(f"AF3 输出结果目录：{args.input_dir}")
        print(f"统计模式：{args.mode}")
        if args.mode == "single":
            print(f"固定读取：seed-{args.seed}_sample-{args.sample}")
        else:
            print(f"固定读取：seed-{args.seed} 的 sample 0-4，并对数值列取平均")
        print(f"口袋定义：配体重原子 5.0 Å 内的蛋白残基；按残基等权平均 pLDDT")
        print(f"AF3 任务文件夹数量：{job_count}")
        print(f"原始 CSV：{args.merge_csv}")
        print(f"合并键：{args.merge_key}")
        print(f"原始 CSV 行数：{total}")
        print(f"成功匹配行数：{matched}")
        print(f"未匹配行数：{missing}")
        print(f"输出 CSV：{out_csv}")
        print(f"数值小数位数：{args.float_digits}")
    else:
        written = write_score_only_csv(
            out_csv=out_csv,
            score_by_folder_name=score_by_folder_name,
        )

        print("完成：只汇总 AF3 分数")
        print(f"AF3 输出结果目录：{args.input_dir}")
        print(f"统计模式：{args.mode}")
        if args.mode == "single":
            print(f"固定读取：seed-{args.seed}_sample-{args.sample}")
        else:
            print(f"固定读取：seed-{args.seed} 的 sample 0-4，并对数值列取平均")
        print(f"口袋定义：配体重原子 5.0 Å 内的蛋白残基；按残基等权平均 pLDDT")
        print(f"AF3 任务文件夹数量：{job_count}")
        print(f"写入行数：{written}")
        print(f"输出 CSV：{out_csv}")
        print(f"数值小数位数：{args.float_digits}")


if __name__ == "__main__":
    main()

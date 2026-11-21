#!/usr/bin/env python3
"""DPL Screening Skill CLI, one target, fixed paper Strategy I.

Original score readers, input converters and screening mathematics are preserved.
This file is the input adapter, reproducible controller, and sequence exporter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import math
import json
import os
from pathlib import Path
import pickletools
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import uuid
from typing import Any

HERE = Path(__file__).resolve().parent
LEGACY = HERE / 'legacy'
SCREEN = HERE / 'screen.py'
MODELS = ('af3', 'boltz2', 'chai1')
REQUIRED = ('task_id', 'Protein_seq', 'ligand_smiles')
AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
SAFE = re.compile(r'[^A-Za-z0-9_.-]+')
# Pandas default NA tokens and numeric IDs can be corrupted in Chai-1's legacy CSV merge.
PANDAS_NA_IDS = {'', 'NA', 'N/A', 'NULL', 'None', 'NaN', 'nan', 'null', '<NA>', 'n/a', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN'}
# pandas may infer numeric-looking task IDs as ints or floats (e.g., 0001, 1e3).
PANDAS_NUMBER = re.compile(r'^(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$')
EXTS = {'af3': '.json', 'boltz2': '.yaml', 'chai1': '.fasta'}
RESERVED_OUTPUT_NAMES = {'af3_scores.csv', 'boltz2_scores.csv', 'chai1_scores.csv',
                         'batch_run_summary.tsv', 'boltz2_runtime_verification.txt'}
FORMAT_VERSION = 5
RELEASE_VERSION = '4.0'
_SCORE_READERS = {}
_SELECTOR = None
_RUNTIME = None
_WORK = None
MODULES = ('rdkit.Chem', 'yaml', 'numpy', 'pandas')
STEPS = {
    'normalize': ('rdkit.Chem',), 'convert_af3': (),
    'convert_boltz2': ('rdkit.Chem', 'yaml'), 'convert_chai1': (),
    'reader_af3': (), 'reader_boltz2': ('numpy',),
    'reader_chai1': ('numpy', 'pandas'), 'npz': ('numpy',),
}
SYNONYMS = {
    'task_id': ('task_id', 'protein_id', 'protein_name', 'name', 'id', 'identifier', 'design_id'),
    'Protein_seq': ('protein_seq', 'sequence', 'seq', 'protein_sequence', 'amino_acid_sequence', 'aa_sequence'),
    'ligand_smiles': ('ligand_smiles', 'smiles', 'ligand', 'molecule_smiles'),
}


class EnvironmentApprovalRequired(RuntimeError):
    pass


def info(work: Path) -> Path:
    return work / 'run_info'


def model_input(work: Path, model: str) -> Path:
    return work / model / 'input'


def model_output(work: Path, model: str) -> Path:
    return work / model / 'output'


def score_path(work: Path, model: str) -> Path:
    return model_output(work, model) / f'{model}_scores.csv'


def reports(work: Path) -> Path:
    return info(work) / 'reports'


def init_workdir(work: Path):
    marker = info(work) / 'layout.json'
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved != {'release': RELEASE_VERSION, 'layout': 'model-input-output-v1'}:
            raise ValueError('Incompatible workdir layout. Use a new, empty workdir.')
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError('Unrecognized nonempty workdir. Use a new, empty workdir. Automatic migration is unavailable.')
        atomic_json(marker, {'release': RELEASE_VERSION, 'layout': 'model-input-output-v1'})
    for name in ('logs', 'tmp', 'provenance', 'backups'):
        (info(work) / name).mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        model_input(work, model).mkdir(parents=True, exist_ok=True)
        model_output(work, model).mkdir(parents=True, exist_ok=True)


def task_env(work=None):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    if work is not None:
        temporary = info(work) / 'tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    return env


def selector():
    global _SELECTOR
    if _SELECTOR is None:
        spec = importlib.util.spec_from_file_location('_dpl_selection_v4', SCREEN)
        _SELECTOR = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_SELECTOR)
    return _SELECTOR


def probe_python(python: str) -> dict:
    """A fixed, CPU-only import probe; no shell interpolation or environment installs."""
    code = '''import importlib,json,sys,contextlib
result={'python':sys.executable,'version':list(sys.version_info[:3]),'modules':{}}
for name in ('rdkit.Chem','yaml','numpy','pandas'):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            importlib.import_module(name)
        result['modules'][name]={'available':True}
    except Exception as exc:
        result['modules'][name]={'available':False,'error':str(exc)}
print(json.dumps(result))
'''
    try:
        result = subprocess.run([str(Path(python).expanduser()), '-B', '-c', code],
                                capture_output=True, text=True, encoding='utf-8',
                                env=task_env(_WORK), timeout=90)
        if result.returncode:
            raise ValueError(result.stderr.strip() or f'exit {result.returncode}')
        report = json.loads(result.stdout)
        report['requested_python'] = str(Path(python).expanduser())
        report['usable'] = tuple(report['version']) >= (3, 10)
        return report
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {'requested_python': str(python), 'usable': False, 'modules': {}, 'error': str(exc)}


def supports(report, modules):
    return report.get('usable', False) and all(
        report.get('modules', {}).get(name, {}).get('available', False) for name in modules)


def environment_report(data: dict) -> dict:
    cache = {}
    candidates = []
    for model in MODELS:
        path = str(Path(data[model]['python']).expanduser().absolute())
        if path not in cache:
            cache[path] = probe_python(path)
        candidates.append({'model': model, **cache[path]})
    current = os.path.normcase(os.path.abspath(sys.executable))
    candidates.sort(key=lambda r: os.path.normcase(os.path.abspath(r['requested_python'])) != current)
    return {'release': RELEASE_VERSION, 'candidates': candidates,
            'status': 'ready' if any(supports(r, MODULES) for r in candidates) else 'approval_required',
            'required_modules': list(MODULES)}


def make_processing_environment(work: Path, candidates: list[dict]):
    """Only called for explicit --env-policy create AFTER human authorization."""
    target = info(work) / 'processing_env'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    base = next((r['requested_python'] for r in candidates if r.get('usable')), None)
    if base is None:
        raise RuntimeError('No usable Python >=3.10 for an approved processing venv')
    if not python.exists():
        if target.exists() and any(target.iterdir()):
            raise ValueError('Processing environment is incomplete; inspect it before retrying (no overwrite)')
        run_cmd([base, '-m', 'venv', str(target)], info(work)/'logs/processing_env.log')
    requirements = HERE.parent / 'assets/requirements_prepare.txt'
    # Installation is constrained to the approved task-local interpreter.
    run_cmd([str(python), '-m', 'pip', 'install', '-r', str(requirements),
             '--cache-dir', str(info(work)/'cache/pip')], info(work)/'logs/processing_env.log')
    report = probe_python(str(python))
    if not supports(report, MODULES):
        raise RuntimeError('Approved processing venv failed dependency checks; model environments were not modified')
    return report


def select_environment(data: dict, work: Path, policy='auto'):
    global _RUNTIME
    report = environment_report(data)
    atomic_json(info(work)/'environment_check.json', report)
    candidates = report['candidates']
    chosen = next((r for r in candidates if supports(r, MODULES)), None)
    saved_path = info(work)/'environment_plan.json'
    saved = json.loads(saved_path.read_text(encoding='utf-8')) if saved_path.is_file() else {}
    if not isinstance(saved, dict):
        raise ValueError('Invalid environment plan; inspect task records rather than adopting it')
    signature = {m: str(Path(data[m]['python']).expanduser().absolute()) for m in MODELS}
    if chosen is None and saved.get('model_pythons') == signature and saved.get('mode') == 'processing_env':
        previous = probe_python(saved['steps']['normalize'])
        if supports(previous, MODULES):
            chosen = previous
    mode = 'unified'
    if chosen is not None:
        mode = 'processing_env' if Path(chosen['requested_python']).parent.parent == info(work)/'processing_env' else 'unified'
        steps = {step: chosen['requested_python'] for step in STEPS}
    else:
        if policy == 'auto' and saved.get('model_pythons') == signature and saved.get('mode') == 'split':
            policy = 'reuse'  # A prior explicit refusal remains valid for this task.
        if policy == 'auto':
            raise EnvironmentApprovalRequired(
                'No model environment has all processing tools. Ask the user whether to create '
                f'{info(work)/"processing_env"}. If approved use --env-policy create; '
                'only after refusal use --env-policy reuse. No installation has occurred.')
        if policy == 'create':
            chosen = make_processing_environment(work, candidates)
            steps = {step: chosen['requested_python'] for step in STEPS}
            mode = 'processing_env'
        else:
            steps = {}
            for step, modules in STEPS.items():
                match = next((r for r in candidates if supports(r, modules)), None)
                if match is None:
                    raise RuntimeError(f'No existing environment can run {step}; missing compatible tools: {list(modules)}. No installation performed.')
                steps[step] = match['requested_python']
            mode = 'split'
    _RUNTIME = {'release': RELEASE_VERSION, 'mode': mode, 'steps': steps, 'model_pythons': signature}
    atomic_json(saved_path, _RUNTIME)
    print(f'Processing environments: {mode}; normalization Python: {steps["normalize"]}')
    return _RUNTIME


def load_environment(work):
    global _RUNTIME
    path = info(work)/'environment_plan.json'
    if not path.is_file():
        raise ValueError('No processing environment plan; run normalize with --config first')
    _RUNTIME = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(_RUNTIME, dict) or _RUNTIME.get('release') != RELEASE_VERSION or not isinstance(_RUNTIME.get('steps'), dict) or set(_RUNTIME['steps']) != set(STEPS) or any(
            not isinstance(value, str) or not value for value in _RUNTIME['steps'].values()):
        raise ValueError('Invalid processing environment plan; run normalize with --config')


def processing_python(step):
    if _RUNTIME is None:
        raise RuntimeError('Processing environment has not been checked')
    return _RUNTIME['steps'][step]


def cpu_call(operation: str, payload: dict, step: str):
    if os.path.normcase(os.path.abspath(processing_python(step))) == os.path.normcase(os.path.abspath(sys.executable)):
        if operation == 'smiles':
            from rdkit import Chem
            return Chem.MolFromSmiles(payload['smiles']) is not None
        if operation == 'npz':
            return _valid_npz(Path(payload['path']), profile=payload.get('profile', 'numeric'))
        if operation == 'arrays':
            return check_arrays(payload['model'], Path(payload['directory']), payload['task'])
    result = subprocess.run([processing_python(step), '-B', str(HERE/'dpl.py'), '_cpu', operation],
                            input=json.dumps(payload), capture_output=True, text=True, encoding='utf-8',
                            env=task_env(_WORK), timeout=180)
    if result.returncode:
        raise RuntimeError(f'{operation} failed in selected environment: {result.stderr.strip()}')
    return json.loads(result.stdout)


def cpu_worker(operation):
    payload = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        if operation == 'smiles':
            from rdkit import Chem
            result = Chem.MolFromSmiles(payload['smiles']) is not None
        elif operation == 'npz':
            result = _valid_npz(Path(payload['path']), profile=payload.get('profile', 'numeric'))
        elif operation == 'arrays':
            result = check_arrays(payload['model'], Path(payload['directory']), payload['task'])
        else:
            raise ValueError('Unknown CPU worker operation')
    def scalar(value):
        if hasattr(value, 'item'):
            return value.item()
        raise TypeError(type(value).__name__)
    print(json.dumps(result, default=scalar))


def archive_current_results(work):
    archive(reports(work), work, 'results')
    for name in ('selected_sequences.csv', 'selected_sequences.fasta'):
        archive(work/name, work, 'published_sequences')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def archive(path: Path, work: Path, category: str):
    if not path.exists():
        return None
    base = info(work) / 'backups' / category
    base.mkdir(parents=True, exist_ok=True)
    dest = base / (path.name + '_' + timestamp())
    shutil.move(str(path), str(dest))
    print(f'[ARCHIVED] {path} -> {dest}', flush=True)
    return dest


def mark_phase(work: Path, phase: str, status: str, **extra):
    state = {'phase': phase, 'status': status, 'utc': datetime.now(timezone.utc).isoformat(), **extra}
    atomic_json(info(work) / 'workflow_status.json', state)


@contextmanager
def phase(work: Path, name: str, *, dry: bool = False):
    """Keep a failed standalone stage from leaving a previous current success."""
    if not dry:
        archive_current_results(work)
        mark_phase(work, name, 'running')
    try:
        yield
    except BaseException as exc:
        if not dry:
            mark_phase(work, name, 'failed', error=str(exc))
        raise
    else:
        if not dry:
            mark_phase(work, name, 'success')


def safe_id(value: str) -> str:
    return SAFE.sub('_', value).strip('._-')


def chai_task_ids(rows: list[dict]) -> dict[str, str]:
    """Isolate legacy filename normalization without changing the public IDs."""
    reserved = {r['task_id'] for r in rows}
    used, mapping = set(), {}
    for task in sorted(reserved):
        alias = task
        if '__' in task or task in {'input', 'chai_input'}:
            salt = 0
            while True:
                alias = 'dplid_' + hashlib.sha256(f'{task}:{salt}'.encode()).hexdigest()[:32]
                if alias not in reserved and alias not in used:
                    break
                salt += 1
        mapping[task] = alias
        used.add(alias)
    return mapping


def load_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols or len(cols) != len(set(cols)):
            raise ValueError(f'CSV header missing or duplicated: {path}')
        rows = list(reader)
    if any(None in r for r in rows):
        raise ValueError('A CSV row has more columns than its header')
    return rows, cols


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_fasta(text: str):
    """Parse all FASTA records; never silently discard a record with an empty header."""
    records, ident, chunks, saw_header = [], None, [], False
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('>'):
            if saw_header:
                if not chunks:
                    raise ValueError(f'FASTA record {ident or "<unnamed>"} has no sequence')
                records.append({'task_id': ident, 'Protein_seq': ''.join(chunks)})
            saw_header = True
            ident = line[1:].strip().split()[0] if line[1:].strip() else ''
            chunks = []
        else:
            if not saw_header:
                raise ValueError(f'FASTA sequence before header at line {line_no}')
            chunks.append(line)
    if saw_header:
        if not chunks:
            raise ValueError(f'FASTA record {ident or "<unnamed>"} has no sequence')
        records.append({'task_id': ident, 'Protein_seq': ''.join(chunks)})
    return records, ['task_id', 'Protein_seq']


def table_from_file(src: Path):
    ext = src.suffix.lower()
    if ext in {'.fa', '.faa', '.fasta', '.fna'}:
        return parse_fasta(src.read_text(encoding='utf-8-sig'))
    if ext in {'.csv', '.tsv', '.txt'}:
        if ext == '.txt':
            txt = src.read_text(encoding='utf-8-sig').strip()
            if txt.startswith('>'):
                return parse_fasta(txt)
            lines = [line.strip() for line in txt.splitlines() if line.strip()]
            if lines and all(len(line.split()) == 1 and AA.fullmatch(line.upper()) for line in lines):
                return [{'task_id': '', 'Protein_seq': line} for line in lines], ['task_id', 'Protein_seq']
            if lines and all(len(line.split()) == 2 for line in lines):
                if lines[0].split()[0].lower() not in {'id', 'task_id', 'protein_id', 'name'}:
                    return [dict(task_id=a, Protein_seq=b) for a, b in (line.split() for line in lines)], ['task_id', 'Protein_seq']
        delim = '\t' if ext == '.tsv' else ','
        with src.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            cols, rows = reader.fieldnames or [], list(reader)
        if not cols or len(cols) != len(set(cols)) or any(None in r for r in rows):
            raise ValueError('Tabular input has missing/duplicate headers or malformed rows')
        return rows, cols
    raise ValueError(f'Unsupported directly: {ext}. Have Codex convert Excel/other formats to CSV first.')


def find_column(cols, field, explicit=None, optional=False):
    if explicit:
        if explicit not in cols: raise ValueError(f'Column {explicit!r} absent from file')
        return explicit
    found = [col for col in cols if col.lower().strip() in SYNONYMS[field]]
    if len(found) > 1:
        raise ValueError(f'Ambiguous {field}: {found}; specify --{field.replace("_", "-")}-col')
    if not found and not optional:
        raise ValueError(f'Cannot recognize {field} column among {cols}')
    return found[0] if found else None


def normalize(args):
    src, work = Path(args.input).expanduser().resolve(), Path(args.workdir).expanduser().resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    smiles = args.smiles
    if args.smiles_file:
        if smiles: raise ValueError('Provide --smiles OR --smiles-file, not both')
        smiles = Path(args.smiles_file).read_text(encoding='utf-8').strip()
    rows, cols = table_from_file(src)
    if not rows: raise ValueError('No protein sequences provided')
    idcol = find_column(cols, 'task_id', args.id_col, optional=True)
    seqcol = find_column(cols, 'Protein_seq', args.seq_col)
    smcol = find_column(cols, 'ligand_smiles', args.smiles_col, optional=True)
    if not smiles and not smcol: raise ValueError('Supply one --smiles/--smiles-file or a SMILES column')
    smiles = str(smiles or '').strip()
    out, ids = [], set()
    for i, row in enumerate(rows, 1):
        original_id = (row.get(idcol) or '').strip() if idcol else ''
        raw_id = original_id or f'design_{i:04d}'
        task_id = safe_id(raw_id)
        # Legacy Chai pandas inference changes purely numeric IDs (0001 -> 1)
        # and reads NA/NULL as missing.  Use a reversible safe alias instead.
        if task_id.lower() in RESERVED_OUTPUT_NAMES:
            task_id = 'protein_' + hashlib.sha256(raw_id.encode()).hexdigest()[:12]
        elif PANDAS_NUMBER.fullmatch(task_id) or task_id in PANDAS_NA_IDS:
            task_id = 'protein_' + task_id
        if not task_id: raise ValueError(f'Row {i}: invalid protein ID: {raw_id!r}')
        if task_id in ids: raise ValueError(f'Row {i}: duplicate/colliding task_id {task_id!r}; rename conflicting IDs')
        ids.add(task_id)
        seq = re.sub(r'\s+', '', str(row.get(seqcol) or '')).upper()
        if not AA.fullmatch(seq):
            raise ValueError(f'Row {i} ({raw_id}): empty/unsupported amino acid sequence; standard 20 residues only')
        ligand = str(smiles or row.get(smcol) or '').strip()
        if not ligand or ('\n' in ligand): raise ValueError(f'Row {i}: empty/invalid SMILES')
        if smiles and smcol and row.get(smcol) and str(row[smcol]).strip() != smiles:
            raise ValueError(f'Row {i}: supplied SMILES disagrees with source file')
        out.append({'task_id': task_id, 'Protein_seq': seq, 'ligand_smiles': ligand, 'original_protein_id': (str(row.get('original_protein_id') or '').strip() if 'original_protein_id' in row and row.get('original_protein_id') is not None else raw_id)})
    if len({r['ligand_smiles'] for r in out}) != 1: raise ValueError('This skill supports exactly one ligand SMILES per run')
    if not cpu_call('smiles', {'smiles': out[0]['ligand_smiles']}, 'normalize'):
        raise ValueError('RDKit could not parse the supplied SMILES')
    csvpath = info(work) / 'input.csv'
    if csvpath.exists():
        old, _ = load_csv(csvpath)
        if old != out: raise ValueError(f'{csvpath} exists with DIFFERENT input; use a new workdir')
    else:
        write_csv(csvpath, out, [*REQUIRED, 'original_protein_id'])
    origin = {'source': str(src), 'source_sha256': digest(src), 'normalized_sha256': digest(csvpath),
              'target_smiles': out[0]['ligand_smiles'], 'protein_count': len(out)}
    (info(work) / 'input_manifest.json').write_text(json.dumps(origin, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Validated {len(out)} sequences: {csvpath}')
    return csvpath


def source_csv(work: Path):
    csvpath = info(work) / 'input.csv'
    rows, cols = load_csv(csvpath)
    if any(c not in cols for c in REQUIRED): raise ValueError(f'input.csv must contain {REQUIRED}')
    if not rows or any(not (r['task_id'] or '').strip() for r in rows) or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Missing/duplicate task_id')
    if len({r['ligand_smiles'] for r in rows}) != 1: raise ValueError('Only one ligand per workdir')
    for row in rows:
        tid = row['task_id']
        if tid != safe_id(tid) or PANDAS_NUMBER.fullmatch(tid) or tid in PANDAS_NA_IDS or tid.lower() in RESERVED_OUTPUT_NAMES:
            raise ValueError(f'Unsafe task_id {tid!r}; normalize the original file first')
        if not AA.fullmatch(row['Protein_seq']):
            raise ValueError(f'Invalid sequence for task {tid!r}')
    manifest = info(work) / 'input_manifest.json'
    if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')).get('normalized_sha256') != digest(csvpath):
        raise ValueError('input.csv changed since normalization; use a new workdir')
    return rows, csvpath


def run_cmd(cmd: list[str], log: Path | None = None, env: dict | None = None, dry=False):
    print('COMMAND:', ' '.join(__import__('shlex').quote(str(v)) for v in cmd), flush=True)
    if dry: return
    env = task_env(_WORK) if env is None else {**env, **{k: v for k, v in task_env(_WORK).items()
                                                     if k in ('TEMP', 'TMP', 'TMPDIR', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUTF8')}}
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write('\nCOMMAND: ' + repr(cmd) + '\n')
            handle.flush()
            subprocess.run(cmd, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=env)


def converter_commands(csvpath: Path, dst: Path):
    return {
        'af3': [processing_python('convert_af3'), str(LEGACY / 'af3/csv_to_json.py'), '-i', str(csvpath),
                '-o', str(dst / 'af3'), '--protein-name-col', 'task_id', '--seeds', '1'],
        'boltz2': [processing_python('convert_boltz2'), str(LEGACY / 'boltz2/csv_to_yaml.py'), '-i', str(csvpath),
                   '-o', str(dst / 'boltz2'), '--no-msa'],
        'chai1': [processing_python('convert_chai1'), str(LEGACY / 'chai1/csv_to_fasta.py'), '--csv', str(csvpath),
                  '--out-dir', str(dst / 'chai1'), '--name-col', 'task_id',
                  '--protein-col', 'Protein_seq', '--smiles-col', 'ligand_smiles'],
    }


def prepared_snapshot(work: Path) -> dict:
    """Fail closed unless model inputs match the previously generated, checked snapshot."""
    rows, csvpath = source_csv(work)
    manifest_path = info(work) / 'prepared_inputs.json'
    if not manifest_path.is_file():
        raise ValueError('No prepared input provenance; run prepare first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('source_sha256') != digest(csvpath):
        raise ValueError('Prepared inputs do not match input.csv; prepare again')
    if manifest.get('converter_hashes') != {
        model: digest(LEGACY / model / ('csv_to_json.py' if model == 'af3' else 'csv_to_yaml.py' if model == 'boltz2' else 'csv_to_fasta.py'))
        for model in MODELS
    }:
        raise ValueError('Input converter scripts changed; re-run prepare')
    expected = {r['task_id'] for r in rows}
    recorded = manifest.get('model_files', {})
    for model in MODELS:
        files = list(model_input(work, model).glob('*' + EXTS[model]))
        actual = {p.stem: digest(p) for p in files}
        if set(actual) != expected or actual != recorded.get(model):
            raise ValueError(f'{model}: prepared model input contents/names changed; run prepare to inspect')
    return manifest


def prepare(work):
    """Idempotent and non-overwriting: compare expected content, then install missing inputs."""
    work = Path(work).expanduser().resolve()
    rows, csvpath = source_csv(work)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=info(work)/'tmp') as tmp:
        temp_dst = Path(tmp) / 'inputs'
        chai_names = chai_task_ids(rows)
        for model, cmd in converter_commands(csvpath, temp_dst).items():
            if model == 'chai1' and any(k != v for k, v in chai_names.items()):
                adapter_csv = Path(tmp) / 'chai_source.csv'
                write_csv(adapter_csv, [{**r, 'task_id': chai_names[r['task_id']]} for r in rows], list(rows[0]))
                cmd[cmd.index('--csv') + 1] = str(adapter_csv)
            run_cmd(cmd)
            if model == 'chai1':
                for task, alias in chai_names.items():
                    if task != alias:
                        (temp_dst / model / (alias + '.fasta')).rename(temp_dst / model / (task + '.fasta'))
        expected = {r['task_id'] for r in rows}
        snapshots = {}
        for model in MODELS:
            generated = {p.stem: p for p in (temp_dst / model).glob('*' + EXTS[model])}
            if set(generated) != expected:
                raise ValueError(f'{model}: converted task IDs mismatch: {set(generated)^expected}')
            current_dir = model_input(work, model)
            existing = {p.stem: p for p in current_dir.glob('*' + EXTS[model])}
            if set(existing) - expected:
                raise ValueError(f'{model}: extra model inputs exist: {set(existing)-expected}')
            for key, file in generated.items():
                if key in existing and digest(file) != digest(existing[key]):
                    raise ValueError(f'{model}/{key}: existing model input differs from current sequence, ligand or parameters. Use a NEW workdir.')
            snapshots[model] = {key: digest(file) for key, file in generated.items()}
        # No mutation of model inputs occurred until *all three* conversions were checked.
        for model in MODELS:
            model_input(work, model).mkdir(parents=True, exist_ok=True)
            for file in (temp_dst / model).glob('*' + EXTS[model]):
                dest = model_input(work, model) / file.name
                if not dest.exists():
                    shutil.copy2(file, dest)
        snapshot = {
            'format_version': FORMAT_VERSION,
            'source_sha256': digest(csvpath),
            'converter_hashes': {
                model: digest(LEGACY / model / ('csv_to_json.py' if model == 'af3' else 'csv_to_yaml.py' if model == 'boltz2' else 'csv_to_fasta.py'))
                for model in MODELS
            },
            'model_files': snapshots,
        }
        manifest_path = info(work) / 'prepared_inputs.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding='utf-8')) != snapshot:
            # Consistent model inputs may have been regenerated by an updated converter;
            # prior predictions must *not* be reused with changed source data.
            if any(any(model_output(work, model).iterdir()) for model in MODELS):
                raise ValueError('Prepared-input recipe changed with existing predictions; use a NEW workdir')
        atomic_json(manifest_path, snapshot)
    prepared_snapshot(work)
    print(f'Prepared {len(rows)} tasks per model under {work}/MODEL/input (safe to repeat)')


def config_load(path: str):
    configpath = Path(path).expanduser().resolve()
    data = json.loads(configpath.read_text(encoding='utf-8'))
    for key in MODELS:
        if key not in data or not isinstance(data[key], dict): raise ValueError(f'Configuration missing {key}')
        for required in (('python', 'model_dir') if key == 'af3' else ('python', 'checkpoint_path', 'affinity_checkpoint_path', 'cache_dir') if key == 'boltz2' else ('python', 'weights_dir')):
            if not data[key].get(required): raise ValueError(f'Configuration missing {key}.{required}')
    mode = data.get('execution', {}).get('mode', 'sequential')
    if mode != 'sequential':
        raise ValueError('DPL Screening Skill runs AF3, Boltz-2 and Chai-1 sequentially. Set execution.mode=sequential.')
    if any(int(data[m].get('gpu', 0)) < 0 for m in MODELS):
        raise ValueError('GPU indices must be nonnegative')
    execution = data.get('execution', {})
    if execution.get('history_file'):
        history_path = Path(execution['history_file']).expanduser()
        if not history_path.is_absolute():
            history_path = configpath.parent / history_path
        execution['history_file'] = str(history_path)
    rates = execution.get('seconds_per_sample', {})
    if not isinstance(rates, dict) or set(rates) - set(MODELS) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError('execution.seconds_per_sample must map model names to positive finite seconds')
    return data, configpath


def frozen_config(work: Path, data: dict):
    lock = info(work) / 'run_settings.json'
    # Preserve stable settings for reproducibility and avoid reusing stale scored structures.
    frozen = {m: {k: v for k, v in data[m].items() if k != 'gpu'} for m in MODELS}
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8')) != frozen:
            raise ValueError('Model configuration differs from this run; use a NEW workdir to prevent stale reuse')
    else:
        lock.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def valid_json(path: Path, required_key: str | None = None) -> bool:
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        return isinstance(obj, dict) and bool(obj) and (required_key is None or obj.get(required_key) is not None)
    except (OSError, UnicodeError, ValueError):
        return False


def valid_npz(path: Path, *, profile: str = 'numeric') -> bool:
    try:
        return bool(cpu_call('npz', {'path': str(path), 'profile': profile}, 'npz'))
    except (OSError, ValueError, RuntimeError):
        return False


def _valid_npz(path: Path, *, profile: str = 'numeric') -> bool:
    if profile == 'boltz_pre_affinity':
        return _valid_pre_affinity_npz(path)
    if profile != 'numeric':
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as arrays:
            if not arrays.files:
                return False
            for key in arrays.files:
                # NumPy can return raw bytes for a ZIP member with an invalid
                # NPY header, so successful ZIP access alone is insufficient.
                if not isinstance(arrays[key], np.ndarray):
                    return False
            return True
    except (OSError, ValueError, TypeError, zipfile.BadZipFile, EOFError):
        return False


def _valid_pre_affinity_npz(path: Path) -> bool:
    """Validate Boltz metadata without unpickling its unused optional pocket=None.

    Required chains/residues and all numeric members still use allow_pickle=False.
    Only scalar object pocket is opaque: verify its NPY header and pickle syntax,
    never execute it. ZIP CRC covers every member, including that optional payload.
    Scientific dimensions and required scores remain checked by the original reader.
    """
    try:
        import numpy as np
        with zipfile.ZipFile(path) as container:
            names = container.namelist()
            if (not names or len(names) != len(set(names)) or
                    any(not name.endswith('.npy') or '/' in name or '\\' in name for name in names) or
                    container.testzip() is not None):
                return False
            with np.load(path, allow_pickle=False) as arrays:
                if not {'chains', 'residues'}.issubset(arrays.files):
                    return False
                required = {key: arrays[key] for key in ('chains', 'residues')}
                fields = {'chains': {'name', 'mol_type', 'atom_num', 'res_idx', 'res_num'},
                          'residues': {'atom_num'}}
                for key, array in required.items():
                    if (not isinstance(array, np.ndarray) or array.ndim != 1 or array.size == 0 or
                            not fields[key].issubset(array.dtype.names or ())):
                        return False
                for key in arrays.files:
                    if key == 'pocket':
                        with container.open('pocket.npy') as member:
                            version = np.lib.format.read_magic(member)
                            if version == (1, 0):
                                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                            elif version in {(2, 0), (3, 0)}:
                                # A plain scalar object header is ASCII in both formats.
                                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                            else:
                                return False
                            if dtype.hasobject:
                                if shape != () or dtype != np.dtype('O'):
                                    return False
                                payload = io.BytesIO(member.read())
                                # dis checks opcodes, stack/memo consistency and STOP,
                                # without executing GLOBAL/REDUCE or constructing objects.
                                pickletools.dis(payload, out=io.StringIO())
                                if payload.tell() != len(payload.getbuffer()):
                                    return False
                                continue
                    if not isinstance(arrays[key], np.ndarray):
                        return False
                return True
    except (OSError, ValueError, TypeError, KeyError, IndexError, zipfile.BadZipFile, EOFError):
        return False


def score_reader(model: str):
    """Load only the unmodified CPU score reader, never a predictor/framework."""
    if model not in _SCORE_READERS:
        path = LEGACY / model / 'score_to_csv.py'
        name = '_dpl_v4_score_' + model + '_' + digest(path)[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        reader = importlib.util.module_from_spec(spec)
        sys.modules[name] = reader
        spec.loader.exec_module(reader)
        _SCORE_READERS[model] = reader
    return _SCORE_READERS[model]


def valid_structure(path: Path) -> bool:
    try:
        if path.stat().st_size < 100:
            return False
        with path.open(encoding='utf-8', errors='replace') as handle:
            start = handle.read(16384)
        if path.suffix.lower() == '.pdb':
            return any(line.startswith(('ATOM  ', 'HETATM')) for line in start.splitlines())
        return ('data_' in start) and bool(score_reader('af3').parse_atom_site(path))
    except OSError:
        return False


def check_arrays(model, directory, task):
    """Inspect the on-disk fields without computing pocket scores or ipTM."""
    import numpy as np
    if model == 'chai1':
        path = directory / 'scores.model_idx_0.npz'
        with np.load(path, allow_pickle=False) as arrays:
            values = {k: arrays[k] for k in arrays.files}
            if any(not isinstance(value, np.ndarray) for value in values.values()):
                return False
            matrix = values['per_chain_pair_iptm']
            while matrix.ndim > 2 and matrix.shape[0] == 1:
                matrix = matrix[0]
            return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2

    shapes = {}
    for field in ('plddt', 'pae', 'pde'):
        path = directory / f'{field}_{task}_model_0.npz'
        with np.load(path, allow_pickle=False) as arrays:
            # Same single-array fallback as the original Boltz reader.
            name = field if field in arrays.files else (arrays.files[0] if len(arrays.files) == 1 else field)
            values = {k: arrays[k] for k in arrays.files}
            if any(not isinstance(value, np.ndarray) for value in values.values()):
                return False
            array = values[name]
            if not np.issubdtype(array.dtype, np.number):
                return False
            shapes[field] = array.size if field == 'plddt' else array.shape
    n = shapes['plddt']
    return n > 0 and shapes['pae'] == shapes['pde'] == (n, n)


def check_output(model, root, task):
    directory = root / task
    if model == 'af3':
        reader = score_reader('af3')
        for idx in range(5):
            sample = directory / f'seed-1_sample-{idx}'
            structures = list(sample.glob('*model.cif'))
            summary = list(sample.glob('*summary_confidences*.json'))
            conf = [p for p in sample.glob('*confidences*.json') if 'summary' not in p.name]
            if len(structures) != 1 or not valid_structure(structures[0]):
                raise ValueError(f'{sample.name}: missing, ambiguous or invalid model CIF')
            if len(conf) != 1 or not valid_json(conf[0], 'atom_plddts'):
                raise ValueError(f'{sample.name}: missing or invalid confidences/atom_plddts')
            if len(summary) != 1:
                raise ValueError(f'{sample.name}: expected one summary JSON')
            fields = json.loads(summary[0].read_text(encoding='utf-8'))
            if not isinstance(fields, dict) or not fields:
                raise ValueError(f'{sample.name}: invalid summary JSON')
            if idx == 0:
                csv_file = reader.find_direct_file(directory, patterns=['*ranking_scores*.csv', '*ranking*.csv'])
                selected = reader.read_selected_ranking_score_from_csv(csv_file, 1, 0)['ranking_score']
                if not math.isfinite(float(fields.get('ranking_score') if selected == '' else selected)):
                    raise ValueError('AF3 seed 1/sample 0 has no finite ranking score')
        return
    if model == 'boltz2':
        directory = directory / 'predictions' / task
        if not valid_structure(directory / f'{task}_model_0.pdb'):
            raise ValueError('Missing or invalid model 0 PDB')
        required = {
            f'confidence_{task}_model_0.json': ('confidence_score', 'ptm', 'iptm', 'ligand_iptm',
                'protein_iptm', 'complex_plddt', 'complex_iplddt', 'complex_pde', 'complex_ipde',
                'chains_ptm', 'pair_chains_iptm'),
            f'affinity_{task}.json': tuple(prefix + suffix for prefix in ('affinity_pred_value',
                'affinity_probability_binary') for suffix in ('', '1', '2')),
        }
        for filename, keys in required.items():
            fields = json.loads((directory / filename).read_text(encoding='utf-8'))
            if not isinstance(fields, dict) or not set(keys).issubset(fields):
                raise ValueError(f'{filename}: missing required JSON fields')
        pre = list(directory.glob('*pre_affinity*.npz'))
        if len(pre) != 1 or not valid_npz(pre[0], profile='boltz_pre_affinity'):
            raise ValueError('Missing, ambiguous or invalid Boltz pre_affinity metadata')
    else:
        directory = directory / 'seed_1'
        if not valid_structure(directory / 'pred.model_idx_0.cif'):
            raise ValueError('Missing or invalid Chai seed 1/model 0 CIF')
    if not cpu_call('arrays', {'model': model, 'directory': str(directory), 'task': task}, 'npz'):
        raise ValueError('Required NPZ arrays or dimensions are invalid')


def completed_files(model, root, task):
    try:
        check_output(model, root, task)
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, RuntimeError, zipfile.BadZipFile):
        return False


def runner_hashes(model):
    runner = {'af3': 'run_alphafold.py', 'boltz2': 'run_boltz2_batch.py', 'chai1': 'run_chai_batch.py'}[model]
    return {name: digest(LEGACY / model / name) for name in (runner, 'score_to_csv.py')}


def recipe_context(work, rows):
    return ({row['task_id']: row for row in rows}, prepared_snapshot(work), {m: runner_hashes(m) for m in MODELS})


def task_recipe(m: str, work: Path, task: str, data: dict, context=None) -> dict:
    if context is None:
        context = ({r['task_id']: r for r in source_csv(work)[0]}, prepared_snapshot(work))
    by_id, snapshot = context[:2]
    record = by_id.get(task)
    if record is None: raise ValueError(f'Unknown task_id: {task}')
    model = {k: v for k, v in data[m].items() if k != 'gpu'}
    scripts = context[2][m] if len(context) > 2 else runner_hashes(m)
    return {'v': FORMAT_VERSION, 'model': m, 'task_id': task,
            'protein_sequence': record['Protein_seq'], 'ligand_smiles': record['ligand_smiles'],
            'prepared_input_sha256': snapshot['model_files'][m][task],
            'model_settings': model, 'scripts_sha256': scripts,
            'selection': {'seed': 1, 'sample': 0, 'af3_samples': 5,
                          'boltz2_structure_samples': 1, 'boltz2_affinity_samples': 1,
                          'chai_results': 1, 'msa': 'off'}}


def receipt_path(work: Path, m: str, task: str) -> Path:
    return info(work) / 'provenance' / m / (task + '.json')


def output_signature(m: str, work: Path, task: str) -> dict:
    """Record file-level output identity; never trust a prediction based on filenames alone."""
    taskdir = model_output(work, m) / task
    result = {}
    for file in taskdir.rglob('*'):
        if file.is_file() and file.suffix.lower() in {'.json', '.cif', '.pdb', '.npz', '.csv', '.yaml'}:
            result[str(file.relative_to(taskdir))] = digest(file)
    return result


def read_record(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def save_receipt(work, model, task, value):
    path = receipt_path(work, model, task)
    archive(path, work, f'receipts/{model}')
    atomic_json(path, value)


def trusted_complete(m, work, data, task, recipe=None):
    saved = read_record(receipt_path(work, m, task))
    expected = recipe if recipe is not None else task_recipe(m, work, task, data)
    return (saved.get('status') == 'complete' and saved.get('recipe') == expected
            and bool(saved.get('outputs_sha256')) and saved['outputs_sha256'] == output_signature(m, work, task))


def complete(model, root, task):
    return completed_files(model, root, task)


def batch_command(m, data, stage, outdir, errors):
    py = str(data[m]['python'])
    gpu = int(data[m].get('gpu', 0))
    env = {**os.environ, 'DPL_BATCH_ERRORS': str(errors)}
    if m == 'af3':
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        cmd = [py, str(LEGACY / 'af3/run_alphafold.py'), '--input_dir', str(stage),
               '--output_dir', str(outdir), '--model_dir', str(data[m]['model_dir']),
               '--run_data_pipeline=false', '--run_inference=true', '--num_diffusion_samples=5', '--gpu_device=0']
    elif m == 'boltz2':
        cmd = [py, str(LEGACY / 'boltz2/run_boltz2_batch.py'), '-i', str(stage), '-o', str(outdir),
               '--checkpoint-path', str(data[m]['checkpoint_path']),
               '--affinity-checkpoint-path', str(data[m]['affinity_checkpoint_path']),
               '--cache-dir', str(data[m]['cache_dir']), '--gpu', str(gpu), '--seed', '1',
               '--diffusion-samples', '1', '--diffusion-samples-affinity', '1', '--keep-tmp']
    else:
        cmd = [py, str(LEGACY / 'chai1/run_chai_batch.py'), '-i', str(stage), '-o', str(outdir),
               '--weights-dir', str(data[m]['weights_dir']), '--gpu', str(gpu),
               '--seeds', '1', '--num-results', '1']
    return cmd, env


def batch_record(work, model, receipt, cache):
    stamp = receipt.get('batch', '')
    if not isinstance(stamp, str) or not re.fullmatch(model + r'-[0-9a-f]{12}', stamp):
        return {}
    if stamp not in cache:
        try:
            raw = (info(work) / 'batches' / stamp / 'execution.json').read_bytes()
            record = json.loads(raw)
            if not isinstance(record, dict) or record.get('batch') != stamp:
                raise ValueError('Wrong batch identity')
            cache[stamp] = {**record, '_sha256': hashlib.sha256(raw).hexdigest()}
        except (ValueError, OSError):
            cache[stamp] = {}
    return cache[stamp]


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def recorded_attempt(receipt, batch, task, recipe):
    return (batch.get('state') == 'ended' and batch.get('model') == recipe['model']
            and isinstance(batch.get('recipes'), dict) and batch['recipes'].get(task) == recipe_hash(recipe)
            and isinstance(batch.get('names'), dict) and task in batch['names']
            and isinstance(batch.get('errors'), dict)
            and (receipt.get('status') != 'complete' or receipt.get('batch_sha256') == batch.get('_sha256')))


def needs_submission(work, model, task, recipe, cache):
    saved = read_record(receipt_path(work, model, task))
    if saved.get('recipe') != recipe:
        return True
    batch = batch_record(work, model, saved, cache)
    if not recorded_attempt(saved, batch, task, recipe) or task in batch['errors']:
        return True
    # Defer every prediction-content check (including reuse hashes) to the central gate.
    return not (saved.get('status') == 'staged' or
                (saved.get('status') == 'complete' and saved.get('outputs_sha256')))


def per_model(m, work, data, recipes, aliases, pending, dry):
    if not pending:
        print(f'{m}: 0 pending; existing results queued for the central check', flush=True)
        return None
    batch_id = f'{m}-{uuid.uuid4().hex[:12]}'
    attempt = info(work) / 'batches' / batch_id
    stage, outdir = attempt / 'i', attempt / 'o'
    error_path = attempt / 'runner_errors.json'
    cmd, env = batch_command(m, data, stage, outdir, error_path)
    hardware = recipe_hash(data[m])
    history = read_record(info(work) / 'timing_history.json').get(m, {})
    rate = (history.get('seconds_per_sample') if isinstance(history, dict) and history.get('hardware') == hardware else None)
    execution = data.get('execution', {})
    if not rate and execution.get('history_file'):
        previous = read_record(Path(execution['history_file'])).get(m, {})
        if isinstance(previous, dict) and previous.get('hardware') == hardware:
            rate = previous.get('seconds_per_sample')
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        rate = execution.get('seconds_per_sample', {}).get(m)
    estimate = len(pending) * rate if rate else None
    label = f'{estimate / 60:.1f} min' if estimate is not None else 'unknown (no timing history)'
    print(f'{m}: {len(pending)} candidates in one batch; estimated duration: {label}', flush=True)
    if dry:
        run_cmd(cmd, env=env, dry=True)
        return None
    attempt.mkdir(parents=True)
    stage.mkdir()
    outdir.mkdir()
    record = {'model': m, 'batch': batch_id, 'state': 'running', 'started_utc': timestamp(), 'hardware': hardware,
              'estimate_seconds': estimate, 'names': {key: aliases[key] for key in pending},
              'recipes': {key: recipe_hash(recipes[key]) for key in pending}}
    for key in pending:
        shutil.copy2(model_input(work, m) / (key + EXTS[m]), stage / (aliases[key] + EXTS[m]))
    atomic_json(attempt / 'execution.json', record)
    for key in pending:
        save_receipt(work, m, key, {'status': 'staged', 'recipe': recipes[key], 'batch': batch_id})
    started = time.monotonic()
    exit_code, message = 0, ''
    try:
        run_cmd(cmd, info(work) / 'logs' / f'{m}.log', env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        exit_code, message = getattr(exc, 'returncode', 1), str(exc)
    elapsed = time.monotonic() - started
    # Runners report caught per-task exceptions once, on exit. A fatal native
    # exit without attribution invalidates this attempt; retain all its artifacts.
    failures = read_record(error_path)
    alias_set = {aliases[key] for key in pending}
    native_error = message if m == 'boltz2' and exit_code and not failures else ''
    if set(failures) - alias_set or (exit_code and not failures and not native_error):
        failures = {aliases[key]: message or 'invalid runner status' for key in pending}
    record.update(state='ended', ended_utc=timestamp(), exit_code=exit_code, seconds=elapsed, native_error=native_error,
                  errors={key: str(failures[aliases[key]]) for key in pending if aliases[key] in failures})
    atomic_json(attempt / 'execution.json', record)
    return record


def preflight(data: dict, *, dry: bool = False):
    """Validate all three model environments BEFORE expensive work begins."""
    if dry:
        print('[DRY RUN] no environment, checkpoint or GPU runtime validation performed')
        return
    errors = []
    if _RUNTIME is None:
        errors.append('No checked processing environment; normalize with --config first')
    for model in MODELS:
        py = Path(data[model]['python']).expanduser()
        if not py.is_file() or not os.access(py, os.X_OK):
            errors.append(f'{model}: Python executable is missing/not executable: {py}')
        elif subprocess.run([str(py), '-c', 'import sys; assert sys.version_info >= (3,10)'],
                            capture_output=True).returncode != 0:
            errors.append(f'{model}: Python interpreter failed self-check: {py}')
        names = ('model_dir',) if model == 'af3' else (
            'checkpoint_path', 'affinity_checkpoint_path', 'cache_dir') if model == 'boltz2' else ('weights_dir',)
        for name in names:
            path = Path(data[model][name]).expanduser()
            needs_file = name in ('checkpoint_path', 'affinity_checkpoint_path')
            valid_kind = path.is_file() if needs_file else path.is_dir()
            if not valid_kind:
                errors.append(f'{model}.{name}: expected a {"file" if needs_file else "directory"}: {path}')
            elif not os.access(path, os.R_OK):
                errors.append(f'{model}.{name}: path is not readable: {path}')
        # Do not import GPU frameworks here; they may initialize devices before scheduling.
    if errors:
        raise ValueError('Server preflight failed; no GPU tasks were launched:\n- ' + '\n- '.join(errors))
    print('Server preflight passed. Model Python interpreters and configured weight and cache paths are accessible.')


def predict(work, cfg, mode_override=None, dry=False):
    with phase(work, 'predict', dry=dry):
        return _predict(work, cfg, mode_override, dry)


def _predict(work, cfg, mode_override=None, dry=False):
    data, _ = config_load(cfg)
    if mode_override not in (None, 'sequential'):
        raise ValueError('Model parallelism is disabled')
    rows, _ = source_csv(work)
    context = recipe_context(work, rows)
    if not dry:
        frozen_config(work, data)
    preflight(data, dry=dry)
    cache, plans = {}, {}
    for model in MODELS:
        recipes = {key: task_recipe(model, work, key, data, context) for key in context[0]}
        aliases = chai_task_ids(rows) if model == 'chai1' else {key: key for key in recipes}
        pending = [key for key in recipes if needs_submission(work, model, key, recipes[key], cache)]
        plans[model] = (recipes, aliases, pending)
    wave = info(work) / 'prediction_wave.json'
    if not dry:
        archive(wave, work, 'waves')
        atomic_json(wave, {'state': 'running'})
    failures = {}
    batches = {}
    for model in MODELS:
        record = per_model(model, work, data, *plans[model], dry)
        if record:
            batches[model] = record['batch']
            if record['native_error']:
                failures[model] = record['native_error']
            failures.update({f'{model}/{key}': error for key, error in record['errors'].items()})
    if not dry:
        atomic_json(wave, {'state': 'ended', 'batches': batches, 'runner_errors': failures})
    print('Prediction stage:', 'DRY RUN' if dry else 'all model processes ended')
    return failures


SCORE_RULES = {
    'af3': {'mode': 'score_mode', 'seed': 'ranking_score_seed', 'sample': 'ranking_score_sample', 'status': 'af3_status', 'score': 'ranking_score'},
    'boltz2': {'mode': 'boltz2_score_mode', 'sample': 'boltz2_samples_used', 'status': 'boltz2_status', 'score': 'boltz2_pocket_mean_plddt_5A'},
    'chai1': {'mode': 'chai_score_mode', 'seed': 'seed', 'sample': 'model_idx', 'status': 'result_status', 'score': 'chai_ligand_protein_chain_pair_iptm_B'},
}


def validate_score_tables(work: Path, *, record_failures=False):
    failures = {}
    rows, _ = source_csv(work)
    input_ids = {r['task_id'] for r in rows}
    for model in MODELS:
        path = score_path(work, model)
        scores, cols = load_csv(path)
        rule = SCORE_RULES[model]
        mandatory = {'task_id', *rule.values()}
        if not mandatory.issubset(set(cols)):
            raise ValueError(f'{model}: scoring CSV lacks required single-mode metadata {sorted(mandatory-set(cols))}')
        if len(scores) != len(input_ids) or {r['task_id'] for r in scores} != input_ids:
            raise ValueError(f'{model}: scored task IDs are missing, extra or duplicated')
        input_by_id = {r['task_id']: r for r in rows}
        for row in scores:
            key = row['task_id']
            for source_field in ('Protein_seq', 'ligand_smiles'):
                if source_field in cols and str(row.get(source_field, '')).strip() != input_by_id[key][source_field]:
                    raise ValueError(f'{model} {key}: {source_field} differs from normalized input')
            try:
                _, warnings = selector().validate_model_score(model, row)
            except ValueError as exc:
                failures[f'{model}/{key}'] = str(exc)
                if record_failures:
                    saved = read_record(receipt_path(work, model, key))
                    save_receipt(work, model, key, {**saved, 'status': 'failed', 'error': str(exc)})
                continue
            for warning in warnings:
                print(f'[AUXILIARY WARNING] {model}/{key}: {warning}; required score remains valid')
    if failures:
        if record_failures:
            atomic_json(info(work) / 'score_errors.json', failures)
        raise ValueError(f'Invalid required scores: {failures}')


def install_batch_output(work, model, task, batch, alias):
    attempt = info(work) / 'batches' / batch['batch']
    out = attempt / 'o'
    if model == 'boltz2':
        native = out / 'boltz_results_i' / 'predictions' / alias
        if not native.is_dir():
            raise ValueError(f'Missing native Boltz prediction directory: {native}')
        source = attempt / 'organized' / task
        dest = source / 'predictions'
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(attempt / 'i' / (alias + '.yaml'), source / 'input_used.yaml')
        native.rename(dest / task)
    else:
        source = out / alias
    if not source.is_dir():
        raise ValueError(f'Missing prediction directory: {source}')
    # Validate the new attempt before moving any current result into backups.
    check_output(model, source.parent, source.name)
    target = model_output(work, model) / task
    archive(target, work, f'predictions/{model}')
    source.rename(target)


def audit_predictions(work, data, rows):
    wave = read_record(info(work) / 'prediction_wave.json')
    if wave.get('state') != 'ended':
        raise ValueError('No finished prediction wave; finish predict before collect')
    context = recipe_context(work, rows)
    chai_names = chai_task_ids(rows)
    failures, cache = {}, {}
    for model in MODELS:
        for task in context[0]:
            saved = read_record(receipt_path(work, model, task))
            recipe = task_recipe(model, work, task, data, context)
            batch = batch_record(work, model, saved, cache)
            alias = chai_names[task] if model == 'chai1' else task
            try:
                if (saved.get('recipe') != recipe or not recorded_attempt(saved, batch, task, recipe)
                        or batch.get('batch') != saved['batch'] or batch['names'][task] != alias):
                    raise ValueError('Missing or conflicting input, configuration or batch provenance')
                if task in batch['errors']:
                    raise ValueError('Runner failure: ' + batch['errors'][task])
                if saved.get('status') == 'complete':
                    if not trusted_complete(model, work, data, task, recipe):
                        raise ValueError('Prediction bytes changed after verification')
                    continue
                if saved.get('status') != 'staged':
                    raise ValueError(saved.get('error', 'Missing fresh prediction'))
                staged_input = info(work) / 'batches' / saved['batch'] / 'i' / (alias + EXTS[model])
                if digest(staged_input) != recipe['prepared_input_sha256']:
                    raise ValueError('Staged input changed since submission')
                install_batch_output(work, model, task, batch, alias)
                save_receipt(work, model, task, {**saved, 'status': 'complete', 'batch_sha256': batch['_sha256'],
                    'outputs_sha256': output_signature(model, work, task)})
            except (ValueError, OSError, TypeError, KeyError, IndexError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
                failures[f'{model}/{task}'] = str(exc)
                if saved.get('status') in ('complete', 'staged'):
                    save_receipt(work, model, task, {**saved, 'status': 'failed', 'error': str(exc)})
    for stamp, batch in cache.items():
        if not batch.get('native_error'):
            continue
        # Native Boltz can terminate after writing a subset. Certify intact peers
        # only when the failure can be localized to unfinished/invalid candidates.
        model = batch['model']
        members = [key for key in batch['names']
                   if read_record(receipt_path(work, model, key)).get('batch') == stamp]
        if len(members) != len(batch['names']) or any(f'{model}/{key}' in failures for key in members):
            continue
        for key in members:
            msg = 'Unattributed nonzero native exit: ' + batch['native_error']
            failures[f'{model}/{key}'] = msg
            saved = read_record(receipt_path(work, model, key))
            save_receipt(work, model, key, {**saved, 'status': 'failed', 'error': msg})
    path = info(work) / 'prediction_audit.json'
    archive(path, work, 'audits')
    atomic_json(path, {'checked': len(rows) * len(MODELS), 'errors': failures})
    history_path = info(work) / 'timing_history.json'
    history = read_record(history_path)
    for model, stamp in wave.get('batches', {}).items():
        batch = cache.get(stamp, {})
        if (batch.get('names') and not batch.get('exit_code') and
                not any(f'{model}/{task}' in failures for task in batch['names'])):
            history[model] = {'batch': stamp, 'hardware': batch['hardware'],
                              'seconds_per_sample': batch['seconds'] / len(batch['names'])}
    atomic_json(history_path, history)
    if failures:
        raise ValueError(f'{len(failures)} invalid predictions; see {path}: {list(failures)[:10]}')


def collect(work, cfg, dry=False):
    if dry:
        return _collect(work, cfg, dry=True)
    with phase(work, 'collect'):
        return _collect(work, cfg)


def _collect(work, cfg, dry=False):
    data, _ = config_load(cfg)
    rows, csvpath = source_csv(work)
    if not dry:
        frozen_config(work, data)
        for model in MODELS:
            archive(score_path(work, model), work, f'scores/{model}')
        archive(info(work) / 'scores_manifest.json', work, 'scores')
        archive(info(work) / 'score_errors.json', work, 'scores')
        audit_predictions(work, data, rows)
    commands = [
        ('af3', [processing_python('reader_af3'), str(LEGACY/'af3/score_to_csv.py'),
                 '-i', str(model_output(work,'af3')), '-o', str(score_path(work,'af3')),
                 '--merge_csv', str(csvpath), '--merge_key', 'task_id',
                 '--mode', 'single', '--seed', '1', '--sample', '0']),
        ('boltz2', [processing_python('reader_boltz2'), str(LEGACY/'boltz2/score_to_csv.py'),
                    '-i', str(model_output(work,'boltz2')), '-o', str(score_path(work,'boltz2')),
                    '--merge-csv', str(csvpath), '--merge-key', 'task_id',
                    '--mode', 'single', '--sample-index', '0']),
        ('chai1', [processing_python('reader_chai1'), str(LEGACY/'chai1/score_to_csv.py'),
                   '-i', str(model_output(work,'chai1')), '-o', str(score_path(work,'chai1')),
                   '--merge-csv', str(csvpath), '--merge-key', 'task_id',
                   '--mode', 'single', '--seed', '1', '--sample-idx', '0']),
    ]
    for m, cmd in commands: run_cmd(cmd, info(work)/'logs'/f'{m}_score.log', dry=dry)
    if not dry:
        validate_score_tables(work, record_failures=True)
        atomic_json(info(work) / 'scores_manifest.json', {
            'format_version': FORMAT_VERSION,
            'input_sha256': digest(csvpath),
            'files': {m: digest(score_path(work, m)) for m in MODELS},
            'receipt_sha256': {m: {r['task_id']: digest(receipt_path(work,m,r['task_id'])) for r in rows}
                               for m in MODELS},
        })
    print('Score collection:', 'DRY RUN' if dry else 'completed')


def verify_score_provenance(work: Path, rows: list[dict], csvpath: Path):
    """Require a complete chain of score provenance records."""
    provenance = json.loads((info(work) / 'scores_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(provenance, dict) or provenance.get('format_version') != FORMAT_VERSION:
        raise ValueError('Invalid score provenance format. Run collect in a new workdir after prediction.')
    ids = {r['task_id'] for r in rows}
    files, receipts = provenance.get('files'), provenance.get('receipt_sha256')
    if not isinstance(files, dict) or set(files) != set(MODELS):
        raise ValueError('Score manifest must name all three model score files')
    if not isinstance(receipts, dict) or set(receipts) != set(MODELS):
        raise ValueError('Score manifest must contain receipt_sha256 for all three models')
    if any(not isinstance(receipts[m], dict) or set(receipts[m]) != ids for m in MODELS):
        raise ValueError('Score manifest must contain exactly every task receipt for every model')
    if provenance.get('input_sha256') != digest(csvpath) or files != {
            m: digest(score_path(work, m)) for m in MODELS}:
        raise ValueError('Score files/input changed after collection; recollect before screening')
    settings = json.loads((info(work) / 'run_settings.json').read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or set(settings) != set(MODELS) or any(
            not isinstance(settings[m], dict) for m in MODELS):
        raise ValueError('Missing/invalid frozen model configuration')
    context = recipe_context(work, rows)
    batch_cache = {}
    for model in MODELS:
        for row in rows:
            task = row['task_id']
            path = receipt_path(work, model, task)
            if not path.is_file() or receipts[model][task] != digest(path):
                raise ValueError(f'{model}/{task}: prediction receipt changed/missing since collect')
            expected = task_recipe(model, work, task, settings, context)
            saved = read_record(path)
            batch = batch_record(work, model, saved, batch_cache)
            if not recorded_attempt(saved, batch, task, expected) or not trusted_complete(model, work, settings, task, expected):
                raise ValueError(f'{model}/{task}: prediction provenance or score readiness is invalid')


def final_screen(work: Path, *, allow_unverified_scores: bool = False, just_collected=False):
    """Safely publish only a completely successful new result; archive previous outputs."""
    archive_current_results(work)
    mark_phase(work, 'screen', 'running')
    stage = Path(tempfile.mkdtemp(prefix='screen-stage-', dir=info(work)/'tmp'))
    try:
        rows, csvpath = source_csv(work)
        for m in MODELS:
            if not score_path(work, m).is_file():
                raise FileNotFoundError(f'Missing {m} score CSV')
        sig = info(work) / 'scores_manifest.json'
        if sig.is_file():
            if not just_collected:
                verify_score_provenance(work, rows, csvpath)
        elif not allow_unverified_scores:
            raise ValueError('No scores_manifest.json: run collect first. For independently sourced tables, pass --allow-unverified-scores explicitly.')
        else:
            print('[WARNING] Screening external scores without prediction provenance; verify source IDs and sample metadata.', file=sys.stderr)
        if not just_collected:
            validate_score_tables(work)
        run_cmd([sys.executable, str(SCREEN), '--input-csv', str(csvpath),
                 '--af3-scores', str(score_path(work,'af3')),
                 '--boltz2-scores', str(score_path(work,'boltz2')),
                 '--chai1-scores', str(score_path(work,'chai1')), '--output-dir', str(stage)])
        selected, _ = load_csv(stage / 'selected_candidates.csv')
        write_csv(stage/'selected_sequences.csv',
                  [{'task_id': r['task_id'], 'original_protein_id': r.get('original_protein_id', r['task_id']),
                    'Protein_seq': r['Protein_seq']} for r in selected],
                  ['task_id','original_protein_id','Protein_seq'])
        with (stage / 'selected_sequences.fasta').open('w', encoding='utf-8') as handle:
            for r in selected:
                handle.write(f'>{r["task_id"]}\n')
                seq = r['Protein_seq']
                for i in range(0, len(seq), 80): handle.write(seq[i:i+80]+'\n')
        summary = json.loads((stage/'screening_summary.json').read_text(encoding='utf-8'))
        text = (f'DPL Screening Skill | Strategy I | one ligand\n'
                f'Input: {summary["input_count"]}\nSelected: {summary["selected_count"]}\n'
                f'Excluded: {summary["excluded_count"]}\n'
                f'Reduction: {100*summary["reduction_fraction"]:.1f}%\n\n'
                f'CSV: {work/"selected_sequences.csv"}\n'
                f'FASTA: {work/"selected_sequences.fasta"}\n\n'
                'Retained sequences (full length):\n')
        text += 'Original protein ID\tTask ID\tProtein sequence\n'
        text += ''.join(
            f'{str(r.get("original_protein_id") or r["task_id"]).replace(chr(10), " ").replace(chr(9), " ")}\t'
            f'{r["task_id"]}\t{r["Protein_seq"]}\n' for r in selected
        )
        (stage/'screening_report.txt').write_text(text, encoding='utf-8')
        result_status = {
            'status': 'publishing', 'input_sha256': digest(csvpath),
            'release_version': RELEASE_VERSION,
            'prediction_provenance_verified': sig.is_file(),
            'scores_sha256': {m: digest(score_path(work, m)) for m in MODELS},
            'exports_sha256': {name: digest(stage/name) for name in ('selected_sequences.csv', 'selected_sequences.fasta')},
        }
        atomic_json(stage/'result_status.json', result_status)
        # Stage both public exports before publishing. Any failure removes all current successes.
        export_stage = Path(tempfile.mkdtemp(prefix='exports-', dir=info(work)/'tmp'))
        try:
            for name in ('selected_sequences.csv', 'selected_sequences.fasta'):
                shutil.copy2(stage/name, export_stage/name)
            os.replace(stage, reports(work))
            for name in ('selected_sequences.csv', 'selected_sequences.fasta'):
                os.replace(export_stage/name, work/name)
            result_status['status'] = 'success'
            atomic_json(reports(work)/'result_status.json', result_status)
        finally:
            shutil.rmtree(export_stage)
        mark_phase(work, 'screen', 'success', selected=summary['selected_count'])
        print(text)
    except BaseException as exc:
        archive(stage, work, 'failed_screen_attempts')
        archive_current_results(work)
        mark_phase(work, 'screen', 'failed', error=str(exc))
        raise


@contextmanager
def workflow_lock(work):
    # OS releases the advisory lock if the controller exits unexpectedly.
    with (info(work) / 'controller.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    global _WORK, _RUNTIME
    sys.dont_write_bytecode = True
    _RUNTIME = None
    if len(sys.argv) == 3 and sys.argv[1] == '_cpu':
        try:
            cpu_worker(sys.argv[2])
            return 0
        except Exception as exc:
            print(f'[CPU ERROR] {exc}', file=sys.stderr)
            return 1
    p = argparse.ArgumentParser(description='DPL screening: normalize -> prepare -> predict -> collect -> screen')
    subs = p.add_subparsers(dest='command', required=True)
    check = subs.add_parser('check-env', help='CPU-only, read-only check of the existing three model environments')
    check.add_argument('--config', required=True)
    n = subs.add_parser('normalize', help='accept FASTA/CSV/TSV/TXT and a ligand SMILES')
    n.add_argument('--input', required=True)
    n.add_argument('--smiles')
    n.add_argument('--smiles-file')
    n.add_argument('--id-col')
    n.add_argument('--seq-col')
    n.add_argument('--smiles-col')
    for sub in (n, *[subs.add_parser(cmd) for cmd in ('prepare', 'predict', 'collect', 'screen', 'all')]):
        sub.add_argument('--workdir', required=sub is not n and not sub.prog.endswith(' all'),
                         help='Default for normalize/all: HOME/<input-stem>_screening')
        if sub.prog.endswith(' all'):
            sub.add_argument('--input', required=True)
            sub.add_argument('--smiles')
            sub.add_argument('--smiles-file')
            sub.add_argument('--id-col')
            sub.add_argument('--seq-col')
            sub.add_argument('--smiles-col')
    subs.choices['screen'].add_argument('--allow-unverified-scores', action='store_true',
                                    help='Explicitly allow external CSV scores without a provenance manifest')
    for cmd in ('predict', 'collect', 'all'):
        sub = subs.choices[cmd]
        sub.add_argument('--config', required=True)
        if cmd != 'collect': sub.add_argument('--mode', choices=('sequential',))
        sub.add_argument('--dry-run', action='store_true')
    n.add_argument('--config', required=True)
    for cmd in ('normalize', 'all', 'predict', 'collect'):
        subs.choices[cmd].add_argument('--env-policy', choices=('auto', 'create', 'reuse'), default='auto',
            help='auto: ask if no complete environment; create: explicit human approval; reuse: after refusal only')
    args = p.parse_args()
    if args.command == 'check-env':
        _WORK = None
        try:
            data, _ = config_load(args.config)
            report = environment_report(data)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report['status'] == 'ready' else 2
        except (OSError, ValueError, RuntimeError) as exc:
            print(f'[ERROR] {exc}', file=sys.stderr)
            return 1
    work = Path(args.workdir).expanduser().resolve() if args.workdir else Path.home()/(Path(args.input).stem+'_screening')
    work = work.resolve()
    args.workdir = str(work)
    _WORK = work
    initialized = False
    locks = ExitStack()
    try:
        init_workdir(work)
        locks.enter_context(workflow_lock(work))
        initialized = True
        if args.command == 'all' and not getattr(args, 'dry_run', False):
            archive_current_results(work)
            mark_phase(work, 'all', 'running')
        if hasattr(args, 'config'):
            data, _ = config_load(args.config)
            if getattr(args, 'dry_run', False) and args.env_policy == 'create':
                raise ValueError('Dry-run never creates or installs an environment; use auto for a preview')
            select_environment(data, work, args.env_policy)
        else:
            load_environment(work)
        if args.command in ('normalize','all'): normalize(args)
        if args.command in ('prepare','all'): prepare(work)
        if args.command in ('predict','all'):
            errors = predict(work, args.config, getattr(args,'mode',None), getattr(args,'dry_run',False))
            if errors and args.command == 'predict':
                raise RuntimeError(f'All models ended with runner errors; run collect for the full audit: {list(errors)[:10]}')
        if args.command in ('collect','all') and not (getattr(args,'dry_run',False) and args.command=='all'):
            collect(work, args.config, getattr(args,'dry_run',False))
        if args.command in ('screen','all') and not (getattr(args,'dry_run',False) and args.command=='all'):
            final_screen(work, allow_unverified_scores=getattr(args,'allow_unverified_scores',False), just_collected=args.command=='all')
        if args.command == 'all' and getattr(args,'dry_run',False):
            mark_phase(work, 'all', 'dry_run_complete')
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as e:
        if initialized:
            if not getattr(args, 'dry_run', False):
                archive_current_results(work)
                mark_phase(work, args.command, 'approval_required' if isinstance(e, EnvironmentApprovalRequired) else 'failed', error=str(e))
            with (info(work)/'logs/errors.log').open('a', encoding='utf-8') as handle:
                handle.write(f'{timestamp()} {args.command}: {e}\n')
        print(f'[ERROR] {e}', file=sys.stderr)
        return 2 if isinstance(e, EnvironmentApprovalRequired) else 1
    finally:
        locks.close()

if __name__ == '__main__':
    sys.exit(main())

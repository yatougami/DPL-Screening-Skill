#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
import gc
import json
from datetime import datetime
from pathlib import Path


def parse_seeds(seed_items):
    seeds = []
    for item in seed_items:
        for part in str(item).split(','):
            part = part.strip()
            if not part:
                continue
            try:
                seeds.append(int(part))
            except ValueError:
                raise ValueError(f'Expected an integer seed, got {part}')
    if not seeds:
        raise ValueError('Specify at least one seed')
    return seeds


def safe_name(name):
    name = str(name).strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('._-')
    return name


def find_fasta_files(input_dir, recursive=False):
    input_dir = Path(input_dir)
    if recursive:
        files = (
            list(input_dir.rglob('*.fasta'))
            + list(input_dir.rglob('*.fa'))
            + list(input_dir.rglob('*.faa'))
        )
    else:
        files = (
            list(input_dir.glob('*.fasta'))
            + list(input_dir.glob('*.fa'))
            + list(input_dir.glob('*.faa'))
        )
    return sorted(files)


def get_task_name(fasta_path):
    fasta_path = Path(fasta_path)
    if fasta_path.name in {'chai_input.fasta', 'input.fasta'}:
        return safe_name(fasta_path.parent.name)
    return safe_name(fasta_path.stem)


def output_has_result(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False
    if list(out_dir.glob('pred.model_idx_*.cif')):
        return True
    if list(out_dir.glob('scores.model_idx_*.npz')):
        return True
    return False


def log(text=''):
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Batch run Chai-1 in one Python process. Logs are printed to stdout/stderr; use nohup redirection for background logs.'
    )

    parser.add_argument('-i', '--input-dir', required=True, help='Directory containing the Chai-1 input FASTA files')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory for Chai-1 predictions')
    parser.add_argument('--weights-dir', required=True, help='Chai-1 weights directory, used to set CHAI_DOWNLOADS_DIR')
    parser.add_argument('--gpu', default=None, help='GPU index to use, for example 2')
    parser.add_argument('--num-trunk-recycles', type=int, default=3, help='Number of trunk recycles (default: 3). Use 10 for more recycling iterations.')
    parser.add_argument('--num-diffn-timesteps', type=int, default=200, help='Number of diffusion sampling steps (default: 200)')
    parser.add_argument('--num-results', type=int, default=5, help='Number of structures to generate per seed, passed to Chai-1 as num_diffn_samples (default: 5)')
    parser.add_argument('--seeds', nargs='+', default=['1'], help='Random seeds, for example --seeds 1 or --seeds 1 2 3 or --seeds 1,2,3')
    parser.add_argument('--recursive', action='store_true', help='Search the input directory recursively for FASTA files')
    parser.add_argument('--skip-existing', action='store_true', help='Skip tasks whose output directory already contains results')
    parser.add_argument('--overwrite', action='store_true', help='Delete existing output and rerun the task')
    parser.add_argument('--dry-run', action='store_true', help='Print the inference command without running it')
    parser.add_argument('--stop-on-error', action='store_true', help='Stop at the first error. By default, processing continues with the remaining tasks.')
    parser.add_argument('--chai-bin', default='chai-lab', help='Accepted for compatibility with older commands. This runner calls the Chai-1 Python API directly and ignores this option.')

    args = parser.parse_args()

    if args.skip_existing and args.overwrite:
        print('[ERROR] --skip-existing 和 --overwrite 不能同时使用', file=sys.stderr, flush=True)
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve()

    if not input_dir.exists():
        print(f'[ERROR] 输入目录不存在：{input_dir}', file=sys.stderr, flush=True)
        sys.exit(1)

    if not weights_dir.exists():
        print(f'[ERROR] 权重目录不存在：{weights_dir}', file=sys.stderr, flush=True)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    fasta_files = find_fasta_files(input_dir, recursive=args.recursive)

    if not fasta_files:
        print(f'[ERROR] 没有找到 fasta 文件：{input_dir}', file=sys.stderr, flush=True)
        sys.exit(1)

    os.environ['CHAI_DOWNLOADS_DIR'] = str(weights_dir)
    os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'

    if args.gpu is not None:
        # Set this variable before importing chai_lab or torch and initializing CUDA.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Import Chai-1 after setting CUDA_VISIBLE_DEVICES and CHAI_DOWNLOADS_DIR.
    # One Python process handles all tasks in the batch.
    from chai_lab.chai1 import run_inference

    try:
        import torch
    except Exception:
        torch = None

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log('========== Chai-1 batch run ==========')
    log(f'[START] {start_time}')
    log(f'[INFO] input_dir           : {input_dir}')
    log(f'[INFO] output_dir          : {output_root}')
    log(f'[INFO] weights_dir         : {weights_dir}')
    log(f'[INFO] gpu                 : {args.gpu}')
    log(f'[INFO] num_trunk_recycles  : {args.num_trunk_recycles}')
    log(f'[INFO] num_diffn_timesteps : {args.num_diffn_timesteps}')
    log(f'[INFO] num_results         : {args.num_results}')
    log(f'[INFO] seeds               : {seeds}')
    log(f'[INFO] fasta files         : {len(fasta_files)}')
    log('======================================')
    log()

    n_total = 0
    n_done = 0
    n_skip = 0
    failures = {}
    n_fail = 0

    for fasta in fasta_files:
        task_name = get_task_name(fasta)

        for seed in seeds:
            n_total += 1
            out_dir = output_root / task_name / f'seed_{seed}'
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log()
            log(f'========== {task_name} | seed {seed} ==========')
            log(f'[TIME] {now}')
            log(f'[INPUT] {fasta}')
            log(f'[OUTPUT] {out_dir}')

            if not os.environ.get("DPL_BATCH_ERRORS") and out_dir.exists() and output_has_result(out_dir):
                if args.skip_existing:
                    log(f'[SKIP] 已有结果，跳过：{out_dir}')
                    n_skip += 1
                    continue
                if args.overwrite:
                    log(f'[OVERWRITE] 删除旧结果：{out_dir}')
                    shutil.rmtree(out_dir)
                else:
                    log('[FAIL] 输出目录已经有结果。加 --skip-existing 跳过，或加 --overwrite 重跑。')
                    n_fail += 1
                    if args.stop_on_error:
                        sys.exit(1)
                    continue

            out_dir.mkdir(parents=True, exist_ok=True)

            cmd_text = (
                f'run_inference('
                f'fasta_file={fasta}, '
                f'output_dir={out_dir}, '
                f'num_trunk_recycles={args.num_trunk_recycles}, '
                f'num_diffn_timesteps={args.num_diffn_timesteps}, '
                f'num_diffn_samples={args.num_results}, '
                f'seed={seed}, '
                f'device=cuda:0'
                f')'
            )

            log('[CMD] ' + cmd_text)
            log('[ENV] CHAI_DOWNLOADS_DIR=' + os.environ['CHAI_DOWNLOADS_DIR'])
            log('[ENV] CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', ''))

            if args.dry_run:
                log('[DRY-RUN] 只打印命令，不实际运行。')
                continue

            log('---------- Chai-1 output start ----------')

            try:
                run_inference(
                    fasta_file=fasta,
                    output_dir=out_dir,
                    num_trunk_recycles=args.num_trunk_recycles,
                    num_diffn_timesteps=args.num_diffn_timesteps,
                    num_diffn_samples=args.num_results,
                    seed=seed,
                    device='cuda:0',
                    use_esm_embeddings=True,
                )
                return_code = 0

            except Exception as e:
                failures[task_name] = repr(e)
                return_code = 1
                print(f'[ERROR] {task_name} seed={seed} 预测失败：{repr(e)}', file=sys.stderr, flush=True)

            finally:
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            log('---------- Chai-1 output end ------------')

            if return_code == 0:
                log(f'[DONE] {task_name} seed={seed}')
                n_done += 1
            else:
                log(f'[FAIL] {task_name} seed={seed}, return_code={return_code}')
                n_fail += 1
                if args.stop_on_error:
                    log('[STOP] 因为设置了 --stop-on-error，程序停止。')
                    sys.exit(return_code)

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log()
    log('========== Summary ==========')
    log(f'[END] {end_time}')
    log(f'[TOTAL] {n_total}')
    log(f'[DONE]  {n_done}')
    log(f'[SKIP]  {n_skip}')
    log(f'[FAIL]  {n_fail}')
    log('=============================')

    if os.environ.get("DPL_BATCH_ERRORS"):
        Path(os.environ["DPL_BATCH_ERRORS"]).write_text(json.dumps(failures), encoding="utf-8")
    if n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

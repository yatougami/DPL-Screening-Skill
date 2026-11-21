#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Submit a prepared YAML directory in one native Boltz-2 predict call."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import shlex
import subprocess
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Boltz-2 predict process for the prepared input directory."
    )

    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing Boltz-2 YAML files")
    parser.add_argument("-o", "--out-dir", required=True, help="Output directory for prediction results")

    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Standard Boltz-2 structure checkpoint (boltz2_conf.ckpt)",
    )
    parser.add_argument(
        "--affinity-checkpoint-path",
        required=True,
        help="Standard Boltz-2 affinity checkpoint (boltz2_aff.ckpt)",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Boltz-2 cache directory containing mols.tar or the extracted mols directory",
    )

    parser.add_argument("--gpu", type=int, default=2, help="Physical GPU index (default: 2)")
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=10,
        help="Number of structure prediction recycling steps (default: 10)",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=200,
        help="Number of structure diffusion sampling steps (default: 200)",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=5,
        help="Number of structures to generate per input (default: 5)",
    )
    parser.add_argument(
        "--max-parallel-samples",
        type=int,
        default=1,
        help="Number of structures to generate concurrently (default: 1 to reduce peak GPU memory use)",
    )
    parser.add_argument(
        "--sampling-steps-affinity",
        type=int,
        default=200,
        help="Number of affinity sampling steps (default: 200)",
    )
    parser.add_argument(
        "--diffusion-samples-affinity",
        type=int,
        default=5,
        help="Number of samples for the affinity ensemble (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (default: 2)",
    )
    parser.add_argument(
        "--preprocessing-threads",
        type=int,
        default=1,
        help="Number of input preprocessing threads (default: 1)",
    )
    parser.add_argument(
        "--output-format",
        choices=["pdb", "mmcif"],
        default="pdb",
        help="Structure output format (default: pdb)",
    )

    parser.add_argument("--keep-tmp", action="store_true", help="Retain the native batch output.")
    parser.add_argument(
        "--no-write-full-pae",
        action="store_true",
        help="Omit the full PAE matrix. Keep it for benchmark runs.",
    )
    parser.add_argument(
        "--no-write-full-pde",
        action="store_true",
        help="Omit the full PDE matrix. Keep it for benchmark runs.",
    )
    parser.add_argument(
        "--affinity-mw-correction",
        action="store_true",
        help="Enable molecular-weight correction for Boltz-2 affinity predictions (disabled by default)",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Disable Boltz CUDA kernels if a kernel compatibility error occurs",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional arguments to pass to boltz predict",
    )

    return parser.parse_args()


def require_boltz2() -> tuple[str, Path]:
    try:
        version = metadata.version("boltz")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("The boltz package is not installed in the current Python environment") from exc

    major_text = version.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError as exc:
        raise RuntimeError(f"Cannot determine the boltz version from {version}") from exc

    if major < 2:
        raise RuntimeError(
            "Boltz-2 is required. This environment has Boltz-1 installed.\n"
            f"Installed boltz version: {version}\n"
            f"Python executable: {sys.executable}"
        )

    boltz_executable = Path(sys.executable).resolve().parent / "boltz"
    if not boltz_executable.is_file():
        raise FileNotFoundError(
            "Cannot find the boltz executable in the current Python environment: "
            f"{boltz_executable}"
        )

    return version, boltz_executable


def build_command(
    boltz_executable: Path,
    input_dir: Path,
    tmp_predict_dir: Path,
    checkpoint_path: Path,
    affinity_checkpoint_path: Path,
    cache_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(boltz_executable),
        "predict",
        str(input_dir),
        "--out_dir",
        str(tmp_predict_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--affinity_checkpoint",
        str(affinity_checkpoint_path),
        "--cache",
        str(cache_dir),
        "--devices",
        "1",
        "--accelerator",
        "gpu",
        "--recycling_steps",
        str(args.recycling_steps),
        "--sampling_steps",
        str(args.sampling_steps),
        "--diffusion_samples",
        str(args.diffusion_samples),
        "--max_parallel_samples",
        str(args.max_parallel_samples),
        "--sampling_steps_affinity",
        str(args.sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(args.diffusion_samples_affinity),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--preprocessing-threads",
        str(args.preprocessing_threads),
        "--output_format",
        args.output_format,
        "--override",
    ]

    if not args.no_write_full_pae:
        command.append("--write_full_pae")

    if not args.no_write_full_pde:
        command.append("--write_full_pde")

    if args.affinity_mw_correction:
        command.append("--affinity_mw_correction")

    if args.no_kernels:
        command.append("--no_kernels")

    if args.extra_args.strip():
        command.extend(shlex.split(args.extra_args))

    return command


def main() -> int:
    args = parse_args()
    inputs = Path(args.input_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Native batch output must be fresh: {output}')
    version, executable = require_boltz2()
    print(f'Boltz {version}: submitting the YAML directory {inputs}', flush=True)
    command = build_command(executable, inputs, output, Path(args.checkpoint_path),
                            Path(args.affinity_checkpoint_path), Path(args.cache_dir), args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    # The native directory interface owns preprocessing, structure and affinity inference.
    # Leave partial native files in the isolated attempt for the controller's central audit.
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())

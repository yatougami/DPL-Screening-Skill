# DPL Screening Skill

DPL Screening Skill helps researchers choose which protein candidates to test against a single small molecule. Give Codex a file of full-length protein sequences and one ligand SMILES. On a GPU server, the skill runs AlphaFold 3 (AF3), Boltz-2 and Chai-1, ranks the candidates independently by one score from each model, and returns the union of the three shortlists as CSV and FASTA.

## Prerequisites

- **Workstation:** Codex with Remote SSH access to the server.
- **Server:** A Linux GPU host or GPU allocation, Python 3.10 or later, and working installations of AF3, Boltz-2 and Chai-1. Each model uses its configured Python environment, weights and caches.
- **Input and scoring tools:** Input preparation and scoring require RDKit, PyYAML, NumPy and pandas. The controller first checks the existing model Python environments. If none provides all four packages, Codex asks for your approval before creating a processing environment for the task.

## Installation

**1. Install the skill locally.** From the repository root, copy the complete `dpl-screening` directory into your Codex user skills directory. The folder includes the [skill instructions](.agents/skills/dpl-screening/SKILL.md), controller and model runners. The repository also provides a [server configuration template](config/server.example.json).

Windows PowerShell:

```powershell
$skillsDir = Join-Path $HOME '.agents/skills'
New-Item -ItemType Directory -Force -Path $skillsDir | Out-Null
Copy-Item -Recurse -Path .agents/skills/dpl-screening -Destination $skillsDir
```

Linux or macOS:

```bash
mkdir -p "$HOME/.agents/skills"
cp -R .agents/skills/dpl-screening "$HOME/.agents/skills/"
```

**2. Deploy to the server project.** Run these commands from the same local repository root. Replace `gpu-server` with your SSH host alias or `user@hostname`. This example creates `~/dpl-project` on the server and uploads the skill, configuration template and example input.

```sh
ssh gpu-server 'mkdir -p dpl-project/.agents/skills dpl-project/config dpl-project/examples'
scp -r .agents/skills/dpl-screening gpu-server:dpl-project/.agents/skills/
scp config/server.example.json gpu-server:dpl-project/config/
scp examples/proteins.fasta gpu-server:dpl-project/examples/
```

**3. Open the remote project in Codex.** Connect to the same host using Codex Remote SSH and open `~/dpl-project`. Codex handles the SSH connection. The deployed skill runs commands inside that server project using the server's files, model environments and GPU allocation.

## Quick Start

Open `~/dpl-project` on the GPU server through Codex Remote SSH. From the remote project root, create the configuration from the deployed template:

```bash
cp -n config/server.example.json config/server.json
```

Edit `config/server.json` and replace the example paths with **absolute paths on the server**. For each model, set `python` to the full path of the Python executable inside its Conda environment. Set each `gpu` to the device index for your allocation. Keep `execution.mode` set to `sequential`. [Configuration template](config/server.example.json)

| Section | `python` | Other server resources |
| --- | --- | --- |
| `af3` | Python interpreter in the Conda environment configured for AF3 | `model_dir` for the AF3 parameters |
| `boltz2` | Python interpreter in the Conda environment configured for Boltz-2 | `boltz` executable beside the interpreter, plus `checkpoint_path`, `affinity_checkpoint_path` and `cache_dir` |
| `chai1` | Python interpreter in the Conda environment configured for Chai-1 | `weights_dir` for the Chai-1 weights |

**In the remote Codex chat**, try the bundled [six-sequence FASTA](examples/proteins.fasta) with ethanol (`CCO`) as an example ligand:

> Use $dpl-screening to screen examples/proteins.fasta against ligand SMILES CCO using config/server.json. Report how many candidates were retained, the percentage reduction, and the absolute server paths to the selected CSV and FASTA files.

## Input

Each screen pairs one small molecule with a set of candidate proteins. Provide a file containing the full-length sequences of the candidate proteins. Include the molecule’s SMILES string in your Codex request, or give Codex the path to a file containing the string.

## Workflow

1. Normalize the candidate records and write AF3 JSON, Boltz-2 YAML and Chai-1 FASTA inputs.
2. Run one batch per model, in order: **AF3 → Boltz-2 → Chai-1**. The controller waits for each batch to exit. AF3 and Boltz-2 accept input directories. One Chai-1 process calls its single-FASTA inference interface for each candidate.
3. After all batches finish, audit the predictions together. Check the required files, fields and provenance before extracting the three score tables. For a failed audit, inspect `run_info/prediction_audit.json` for the affected model and candidate.
4. Apply Strategy I and export the selected full-length sequences. A run can resume in its recognized work directory when its inputs and model settings match. [Workflow controller](.agents/skills/dpl-screening/scripts/dpl.py)

## Screening Strategy

Strategy I ranks all **N** input candidates from highest to lowest on each of three scores. The workflow uses seed 1 and reads sample or model 0.

| Model | Score column | Quantity used for ranking |
| --- | --- | --- |
| AF3 | `ranking_score` | AF3 ranking score for seed 1, sample 0 |
| Boltz-2 | `boltz2_pocket_mean_plddt_5A` | Mean pocket pLDDT within 5 Å of the ligand |
| Chai-1 | `chai_ligand_protein_chain_pair_iptm_B` | Ligand–protein chain-pair ipTM for ligand chain B |

For each score, retain the top `ceil(0.30 × N)` candidates and every candidate tied at the cutoff. A protein is selected if it appears on **any** of the three lists. For the six-sequence example, each list therefore starts with two places. Ties can expand each list, and overlap determines the final count. The reported reduction is `100 × (N − number selected) / N`. [Selection implementation](.agents/skills/dpl-screening/scripts/screen.py)

## Output

For `examples/proteins.fasta`, the default server work directory is **`~/proteins_screening/`**. Set `--workdir` on `all` to choose another location.

You can also name an output directory in your `$dpl-screening` request. Codex passes the requested server path to `--workdir`, so model outputs, run records and the final sequence files are written there. For a new screen, choose a new or empty directory.

> Use $dpl-screening to screen examples/proteins.fasta against ligand SMILES CCO using config/server.json. Save all screening outputs to /data/projectA/results/. Report how many candidates were retained, the percentage reduction, and the absolute server paths to the selected CSV and FASTA files.

| Path inside the work directory | Contents |
| --- | --- |
| `selected_sequences.csv` | Selected candidates with `task_id`, `original_protein_id` and full `Protein_seq` |
| `selected_sequences.fasta` | The same full sequences, with task IDs as FASTA headers |
| `af3/`, `boltz2/`, `chai1/` | Model-specific `input/` and `output/` directories. Each output directory holds candidate predictions and its model score CSV. |
| `run_info/` | Normalized input, logs, prediction audit and provenance. `reports/screening_summary.json` records the counts and reduction, and `reports/result_status.json` records the publication status. |

## License

This repository specifies no project-wide license for the DPL Screening Skill code. The terms below apply to the respective third-party components. All three models’ weights are distributed separately. Configure them on your server under the applicable upstream terms.

| Component | Source code and model weights |
| --- | --- |
| **AlphaFold 3** | DPL Screening Skill modifies the bundled `run_alphafold.py` to isolate recoverable input or inference errors for each complex in a batch. The runner continues with the remaining complexes and records failed task IDs and their errors in JSON. The bundled runner retains its original notice: **Copyright 2024 DeepMind Technologies Limited**, licensed under **CC BY-NC-SA 4.0**. This notice matches the official v3.0.2 source distribution. Upstream changed its source license to Apache 2.0 in v3.0.3 on June 9, 2026. Obtain the model parameters from Google under its separate Model Parameters Terms of Use. [Bundled runner](.agents/skills/dpl-screening/scripts/legacy/af3/run_alphafold.py) · [Upstream runner (v3.0.2)](https://github.com/google-deepmind/alphafold3/blob/v3.0.2/run_alphafold.py) · [Historical license](https://github.com/google-deepmind/alphafold3/blob/v3.0.2/LICENSE) · [v3.0.3 release notes](https://github.com/google-deepmind/alphafold3/releases/tag/v3.0.3) · [Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) |
| **Boltz-2** | The upstream project releases both its code and model weights under the **MIT License**. [Official repository and README](https://github.com/jwohlwend/boltz) · [License](https://github.com/jwohlwend/boltz/blob/main/LICENSE) |
| **Chai-1** | The upstream project releases both its code and model weights under the **Apache License 2.0**. [Official repository and README](https://github.com/chaidiscovery/chai-lab) · [License](https://github.com/chaidiscovery/chai-lab/blob/main/LICENSE) |

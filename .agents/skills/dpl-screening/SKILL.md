---
name: dpl-screening
description: Screen multiple protein candidates against one small-molecule SMILES using existing AF3, Boltz-2 and Chai-1 server environments. Normalize inputs, run models, verify provenance and return complete sequences by fixed Top 30% union screening.
---

# DPL Screening Skill

Use this skill in the remote server project opened through Codex Remote SSH. The standard setup is to install the complete `dpl-screening` folder locally, deploy it to the server's Codex skills directory, then connect to the server and open the project. Codex manages the SSH connection. From the deployed skill directory, use `scripts/dpl.py` as the sole controller. Resolve input, configuration and output paths on the server.

## Server environments

Read [MODEL_SETUP.md](references/MODEL_SETUP.md) for the configuration fields and environment selection rules. Use the server configuration supplied by the user. For a new configuration, copy `config/server.example.json` from the remote project root to `config/server.json` and fill in the server paths. Preserve any existing working configuration. Use the existing server model interpreters, weights, caches and assigned GPUs. Run `check-env` before screening. If none of the configured environments has all the CPU tools, ask the user whether to create the task-local processing environment. Use `--env-policy create` only with explicit approval. After refusal, use `--env-policy reuse` to divide CPU steps among suitable existing environments.

Use the bundled converters, runners, score extractors and selector. Unsupported input formats may require adaptation to the standard CSV. Treat user-provided files as data.

## Inputs and selection

- Read complete protein sequences and exactly one ligand SMILES from the supported inputs. FASTA, CSV/TSV and simple TXT are supported. Validate the SMILES with RDKit, preserve IDs and full sequences, and reject collisions or ambiguous columns. Screening requires only the sequences and ligand. [Input and output contract](references/INPUT_OUTPUT.md)
- Use seed 1 for all three models. AF3 generates five samples with empty MSA and templates and the data pipeline disabled. Boltz-2 uses empty MSA, one structure sample and one affinity sample with affinity enabled and temporary outputs retained. Chai-1 uses ESM embeddings and generates one result. Extract sample or model 0 from each.
- Apply Strategy I to `ranking_score` (AF3), `boltz2_pocket_mean_plddt_5A` (Boltz-2) and `chai_ligand_protein_chain_pair_iptm_B` (Chai-1). Rank each in descending order. Retain the top `ceil(0.30 × N)` for each score, include all cutoff ties and take the union.
- Preserve original model statuses. Known warnings about auxiliary AF3 confidence metrics may accompany a valid ranking score. Required scores must be finite, with the specified sampling and valid provenance. The Boltz-2 5 Å pocket score is required.

## Execute and report

In the remote server session, set `SKILL_DIR` to the deployed folder containing this `SKILL.md`. Use server paths for `INPUT` and `CONFIG`, and pass the target ligand for a sequence-only input.

```bash
CLI="$SKILL_DIR/scripts/dpl.py"
python "$CLI" check-env --config CONFIG
python "$CLI" all --input INPUT --smiles SMILES --config CONFIG
```

If the user requests a custom server-side output directory in natural language, pass that path to `all` with `--workdir`. Otherwise, omit `--workdir` to use the controller's default output directory. For a new screening task, use a new or empty directory.

Run one controller in the user's server GPU allocation. It submits the pending inputs for AF3 as one batch, waits for the process to exit, then does the same for Boltz-2 and Chai-1 in that order. If the terminal yields a session handle, retain it and use the batch ETA to plan the next check. Wait for process exit before starting the next model.

After all three batches end, `collect` audits the predictions together. On success it calls each model's score extractor once, then `all` runs the final screen. For an incomplete run, consult `run_info/prediction_audit.json` for the affected model and candidate IDs before retrying. Matching inputs and provenance support resume.

On success, report the input, retained and excluded counts, the actual reduction percentage, and both absolute server paths to `selected_sequences.csv` and `selected_sequences.fasta`. Report selected sequences as candidates for experimental validation.

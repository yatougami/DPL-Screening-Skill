# DPL Screening Skill inputs and outputs

Provide multiple complete protein sequences and exactly one ligand SMILES. No labels, Kd, reference binders or benchmark are required. Normalize to task_id, Protein_seq, ligand_smiles, original_protein_id. Only standard 20 amino acids are accepted; RDKit validates SMILES in a checked processing environment.

Safe task IDs preserve internal dots/hyphens and repeated underscores. Numeric/NA-like IDs receive reversible aliases; source IDs remain in original_protein_id. Collisions/ambiguous columns fail. Unnamed FASTA records receive design IDs; empty sequences fail. Chai temporary aliases are mapped back without changing sequences or scores.

Paths refer to the server account running the skill in the Codex Remote SSH session. The default work directory is Path.home() / (input_file.stem + '_screening'), normally /home/user_name/ProteinA_screening. --workdir overrides it. Start a new task in an empty directory. To resume an existing task, use its recognized work directory and matching inputs and configuration.

```text
ProteinA_screening/
  af3/input/*.json
  af3/output/<task>/...
  af3/output/af3_scores.csv
  boltz2/input/*.yaml
  boltz2/output/<task>/...
  boltz2/output/boltz2_scores.csv
  chai1/input/*.fasta
  chai1/output/<task>/...
  chai1/output/chai1_scores.csv
  run_info/
    input.csv
    input_manifest.json
    environment_check.json
    environment_plan.json
    prepared_inputs.json
    run_settings.json
    scores_manifest.json
    workflow_status.json
    layout.json
    batches/<batch-id>/    # staged inputs, native outputs and batch exit record
    prediction_wave.json
    prediction_audit.json
    timing_history.json
    logs/
    provenance/<model>/<task>.json
    reports/
    backups/
    tmp/
    processing_env/       # Only if required and explicitly approved
  selected_sequences.csv
  selected_sequences.fasta
```

Standalone `normalize` and `all` accept the default server work directory or `--workdir`. Other task stages require an explicit `--workdir`. `prepare` and `screen` use the recorded processing environment.

Model outputs retain native summary files and failure evidence. Task-generated inputs, logs, receipts, adapters and temporary files remain inside the workdir. Model installations, shared weights and existing caches remain at their configured paths.

All required scores must be finite, single-mode, and from the specified sample. Known auxiliary AF3 warnings are preserved in run_info/reports/score_warnings.csv. AF3 5 Angstrom pocket confidence is diagnostic; Boltz-2 5 Angstrom pocket pLDDT is mandatory. Missing or damaged required structures or confidence files, unknown errors and mismatched provenance block screening. Boltz pre_affinity NPZ validation accepts the unused optional scalar `pocket=None` without unpickling it and checks the required `chains` and `residues` arrays.

Provenance records bind actual inputs, model settings, source script hashes, prediction hashes and score receipts. External score tables require explicit --allow-unverified-scores without a manifest; malformed present manifests still fail.

Successful reports include selection/exclusion tables, all scores, QC/warnings, summary and result_status.json. Root CSV/FASTA exports are current only after successful publication with a matching `run_info/reports/result_status.json`. Re-running archives prior exports before new work; failure does not leave old files as current results. Full sequences are preserved. CSV/text retain original IDs; FASTA uses task IDs. Final reporting prints BOTH absolute server paths.

Standalone `predict` stages each batch under run_info/batches and records process completion. Standalone `collect` requires a finished prediction wave, installs the per-candidate directories into MODEL/output and performs the central audit. Caught sample errors and native partial artifacts remain in the attempt directory or model output. The audit identifies all affected model/ID pairs. A newly discovered invalid receipt is retried by the next predict/all; intact peers keep their output bytes. A nonzero native exit with no identifiable failed candidate blocks publication for the affected batch. Separate `screen` verifies the score manifest and stored hashes; it never calls the model score extractors.

# DPL Screening Skill server model setup

After deploying the complete skill, open the server project through Codex Remote SSH. Run the controller in that remote session on a host with access to the assigned GPU. On a managed cluster, use the site's existing GPU allocation. Codex provides the remote connection. The controller starts each model process on the server and waits for it to exit.

## Configure the installed models

Deploy the repository’s `config/server.example.json` to the same path in the remote project. For initial setup, copy it to `config/server.json` in that project. Supply absolute server paths. The controller requires Python 3.10 or later and can start with the standard library. Each predictor uses the Python interpreter specified in its own section.

| Configuration section | Resources on the server |
|---|---|
| `af3` | Set `python`, `model_dir` and `gpu`. The runner uses the installed AlphaFold 3, JAX, Haiku and tokamax stack and reads the generated AF3 JSON inputs. |
| `boltz2` | Set `python`, `checkpoint_path`, `affinity_checkpoint_path`, `cache_dir` and `gpu`. The runner uses the `boltz` executable beside the configured Python interpreter. |
| `chai1` | Set `python`, `weights_dir` and `gpu`. The runner uses the installed `chai_lab` package and its inference and embedding resources. |

Keep `execution.mode` set to `sequential`. The same assigned GPU can serve all three models. Model installations, checkpoints and shared caches stay at their configured locations. The model frameworks may use their existing shared caches during inference.

## Check CPU processing tools

Run `dpl.py check-env --config CONFIG` in the remote session. It checks Python versions and imports RDKit, PyYAML, NumPy and pandas in the three configured environments. Exit 0 means a model environment supplies all processing tools. Exit 2 calls for an environment choice. Exit 1 indicates a command or configuration error.

The default `--env-policy auto` prefers the current interpreter if it is a complete configured model environment, then checks AF3, Boltz-2 and Chai-1 in order. When none is complete, ask the user whether to create a task-local CPU processing environment.

- With explicit approval, pass `--env-policy create` to `normalize` or `all`. The controller installs the dependencies listed in `assets/requirements_prepare.txt` into `run_info/processing_env` in the task directory.
- If the user declines, pass `--env-policy reuse`. Normalization needs RDKit, Boltz input conversion needs RDKit and PyYAML together, Boltz score reading and NPZ checks need NumPy, and the Chai score reader needs NumPy and pandas. A missing tool stops the affected workflow with an explanation.

The controller records the chosen interpreters in `run_info/environment_plan.json`. An optional `all --dry-run` prepares the inputs and previews the prediction commands. During a prediction run the controller checks all configured interpreter, weight and cache paths before starting the first model.

## Batch timing and prediction

AF3 processes the candidate JSON files with one model runner. Boltz-2 receives the full YAML directory in one native `boltz predict` invocation. Chai-1 runs in one process and calls its single-FASTA inference interface for each candidate. The controller waits for each model batch to exit and writes logs to the server work directory. Sample-level failures are collected for the final audit. [Boltz prediction interface](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)

For a timing estimate, set `execution.seconds_per_sample` to measured seconds per candidate for the relevant model names (`af3`, `boltz2`, `chai1`), or point `execution.history_file` at a previous server task's `run_info/timing_history.json`. Relative history paths resolve beside the configuration file. The controller prefers matching measurements from the current task, followed by the referenced history file, then the configured rates. It multiplies the rate by the number of pending candidates. Without measurements, it reports the duration as unknown.

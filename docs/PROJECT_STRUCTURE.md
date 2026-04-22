# Project Structure

Use this file as the map for the implementation.

## Notebook Workflow

- `notebooks/00_environment_understanding.ipynb`
  - Environment spaces, random rollouts, sparse reward behavior, `info` fields, and
    reward shaping signal.

- `notebooks/01_training_smoke_and_tensorboard.ipynb`
  - 25k PPO smoke training, TensorBoard, `progress.csv`, metadata, and resume check.

- `notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb`
  - Method walkthrough for PPO baseline, shaped PPO, curriculum PPO, self-play fallback,
    optional DQN, and behavior cloning.

- `notebooks/03_full_training_pipeline.ipynb`
  - Full long-running training, plotting, export, imitation learning, and quick evaluation.

- `notebooks/04_submission_and_report.ipynb`
  - Package validation, final `TEAMNAME_AGENT.zip`, final evaluations, and report artifacts.

- `notebooks/CS8803_SoccerTwos_Project.ipynb`
  - Original all-in-one runner retained for convenience.

## Organized Python Package

- `soccer_twos_project/config.py`
  - Hardware profiles.
  - Local artifact directories.
  - JSON/checkpoint serialization helpers.

- `soccer_twos_project/envs.py`
  - Soccer-Twos RLlib environment factory.
  - Reward shaping wrapper.
  - Curriculum sampling helpers.

- `soccer_twos_project/training.py`
  - PPO baseline.
  - PPO reward-shaped training.
  - PPO curriculum training.
  - PPO self-play fallback.
  - Optional DQN baseline.
  - Smoke tests and retry handling.

- `soccer_twos_project/plotting.py`
  - Reads Ray `progress.csv`.
  - Produces per-run plots and overlaid comparison plots.

- `soccer_twos_project/exporting.py`
  - Restores RLlib checkpoints.
  - Exports standalone submission packages.
  - Writes `agent.py`, `model.py`, `checkpoint.pth`, README, metadata, and zip files.

- `soccer_twos_project/evaluation.py`
  - Runs headless agent-vs-agent matches.
  - Saves episode CSV and summary JSON outputs.

- `soccer_twos_project/imitation.py`
  - Downloads/extracts baseline agent when available.
  - Collects expert demonstrations.
  - Trains behavior cloning policy.
  - Packages the imitation agent.

- `soccer_twos_project/notebook_tools.py`
  - Notebook-facing helpers for setup, environment gates, debug rollouts, TensorBoard,
    progress tables, inline plots, training calls, checkpoint lookup, package validation,
    unzip validation, final submission zips, and report artifact checklists.

## Command-Line Entrypoints

Prefer package modules:

- `python -m soccer_twos_project.training ...`
- `python -m soccer_twos_project.plotting ...`
- `python -m soccer_twos_project.exporting ...`
- `python -m soccer_twos_project.evaluation ...`
- `python -m soccer_twos_project.imitation ...`

Thin wrappers also exist in `tools/` for readability.

## Examples And Configs

- `examples/legacy/`: original starter scripts, kept for reference.
- `examples/agents/`: packaged example agents.
- `configs/curriculum.yaml`: curriculum task setup.
- `scripts/`: shell/batch and environment utility scripts.
- `requirements/`: optional requirement overlays.
- `docs/FINAL_PROJECT_TRACKER.md`: assignment checklist from setup through report.

## Outputs

- `artifacts/cs8803_soccer_twos/checkpoints/`: Ray/TensorBoard training logs and checkpoints.
- `artifacts/cs8803_soccer_twos/plots/`: generated PNG learning curves.
- `artifacts/cs8803_soccer_twos/submissions/`: exported agent folders and zip files.
- `artifacts/cs8803_soccer_twos/datasets/`: behavior cloning datasets.
- `artifacts/cs8803_soccer_twos/evals/`: evaluation CSV/JSON outputs.

`artifacts/` is ignored by git.

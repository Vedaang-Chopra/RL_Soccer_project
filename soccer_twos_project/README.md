# Soccer-Twos Project Package

This package gives the project a readable import structure for notebooks and future work.

## Submodules

- `config.py`: hardware profiles, artifact folders, JSON/checkpoint helpers.
- `envs.py`: RLlib environment factory, reward shaping wrapper, curriculum sampling helpers.
- `training.py`: Ray/RLlib training stages, including baseline, shaped, curriculum, self-play fallback, optional DQN, and smoke test.
- `plotting.py`: `progress.csv` learning-curve plots.
- `exporting.py`: RLlib checkpoint export into standalone `AgentInterface` packages.
- `evaluation.py`: headless agent-vs-agent evaluation.
- `imitation.py`: expert data collection and behavior cloning.
- `notebook_tools.py`: high-level notebook helpers for setup, environment inspection, TensorBoard, progress tables, inline plots, training, checkpoint lookup, package validation, final zip creation, and report artifact checks.

Hardware profile selection is automatic by default. `select_profile("auto")`
uses PyTorch CUDA detection to choose a GPU profile when CUDA is available, and
the training config passes that choice to Ray/RLlib through `num_gpus`.

## Command-Line Entrypoints

Use package modules directly:

- `python -m soccer_twos_project.training ...`
- `python -m soccer_twos_project.plotting ...`
- `python -m soccer_twos_project.exporting ...`
- `python -m soccer_twos_project.evaluation ...`
- `python -m soccer_twos_project.imitation ...`

Thin wrappers also exist in `tools/`.

The notebook workflow is:

- `notebooks/00_environment_understanding.ipynb`
- `notebooks/01_training_smoke_and_tensorboard.ipynb`
- `notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb`
- `notebooks/03_full_training_pipeline.ipynb`
- `notebooks/04_submission_and_report.ipynb`
- `notebooks/05_submission_smoke_test.ipynb`

The old all-in-one notebook was removed to avoid duplicate execution paths.

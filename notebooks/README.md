# Notebook Workflow

Run these notebooks in order from the project root or from this `notebooks/` directory.
They assume the existing `soccertwos` environment is active and `requirements.txt` is
already installed.

Training notebooks use `PROFILE_NAME = "auto"` by default. The project detects
CUDA through PyTorch and uses GPU training automatically when CUDA is available;
otherwise it keeps CPU-safe settings.

Each notebook setup cell now detects macOS/local, Colab, and PACE/SLURM-style
runtimes. On Colab it mounts Google Drive if needed, searches common Drive
locations for `soccer-twos-starter`, and sets paths from the discovered project
root. If your Drive folder is unusual, set `SOCCER_TWOS_PROJECT_ROOT` before
running the setup cell.

Unity playback stays headless by default on Linux/PACE-style sessions. Only
enable rendering in a GUI-capable session with a live display; notebook helpers
will otherwise keep Unity in headless mode to avoid startup hangs.

1. `00_environment_understanding.ipynb`
   Understand the SoccerTwos observation/action spaces, sparse rewards, `info` fields,
   and reward shaping signal.

2. `01_training_smoke_and_tensorboard.ipynb`
   Run a short PPO smoke job, watch TensorBoard, inspect `progress.csv`, and test resume.

3. `02_methods_baseline_shaping_curriculum_imitation.ipynb`
   Walk through the algorithms and project criteria: PPO baseline, shaped PPO,
   curriculum PPO, self-play fallback, optional DQN, and behavior cloning.

4. `03_full_training_pipeline.ipynb`
   Execute the full training/export/evaluation pipeline for final artifacts.

5. `04_submission_and_report.ipynb`
   Validate exported packages, create the final `TEAMNAME_AGENT.zip`, and gather
   report-ready plots/tables.

6. `05_submission_smoke_test.ipynb`
   Run a tiny train/export/zip/import smoke test for the submission path. This
   is structural validation only; it is not a performance run.

For a beginner-friendly visual explanation of RL concepts in this environment,
use the separate `rl_learning/` sequence. Those notebooks focus on short
rollouts, plots, value/advantage intuition, tiny PPO training, and behavior
comparison before full-project training.

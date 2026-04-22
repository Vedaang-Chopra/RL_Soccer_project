# Notebook Workflow

Run these notebooks in order from the project root or from this `notebooks/` directory.
They assume the existing `soccertwos` environment is active and `requirements.txt` is
already installed.

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

`CS8803_SoccerTwos_Project.ipynb` is retained as the original all-in-one runner.

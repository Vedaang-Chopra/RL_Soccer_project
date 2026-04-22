# Final Project Tracker

Use this as the execution checklist for the updated SoccerTwos final project.

## Source Of Truth

- Updated assignment PDF: `project/Final Project Instructions Document (Updated).pdf`.
- Agent zip due: April 24, 2026 at 11:59 PM.
- Late agent zip deadline: April 27, 2026 at 11:59 PM.
- Final report due: May 4, 2026 at 11:59 PM.

## Learning And Debugging

- [ ] Run `notebooks/00_environment_understanding.ipynb`.
- [ ] Confirm imports for `soccer_twos`, `ray`, `torch`, `gym`, and `numpy`.
- [ ] Confirm training env observation shape is `(336,)`.
- [ ] Confirm training env action space is `Discrete(27)`.
- [ ] Confirm live match action space is `MultiDiscrete([3 3 3])`.
- [ ] Inspect random rollout rewards and `info["player_info"]` / `info["ball_info"]`.

## Smoke Training

- [ ] Run `notebooks/01_training_smoke_and_tensorboard.ipynb`.
- [ ] Run 25k timestep `ppo_baseline` smoke training.
- [ ] Confirm `progress.csv` exists.
- [ ] Confirm TensorBoard event file exists.
- [ ] Confirm at least one checkpoint exists.
- [ ] Confirm `run_metadata.json` loads and contains `best_checkpoint`.
- [ ] Optionally resume from the smoke checkpoint.

## Full Training

- [ ] Run `ppo_baseline` for a full profile-budget run.
- [ ] Run `ppo_shaped` for a full profile-budget run.
- [ ] Run `ppo_curriculum` for a full profile-budget run.
- [ ] Only if curriculum is weak, run `ppo_selfplay` as a fallback.
- [ ] Generate per-agent plots and `learning_curve_comparison.png`.
- [ ] Save notes on hyperparameters, reward shaping, and curriculum thresholds for the report.

## Export And Imitation

- [ ] Fill author name and email in `03_full_training_pipeline.ipynb`.
- [ ] Export `soccer_ppo_baseline`.
- [ ] Export `soccer_ppo_shaped`.
- [ ] Export `soccer_ppo_curriculum`.
- [ ] Collect behavior-cloning samples from the strongest expert.
- [ ] Train and export `soccer_bc_imitation`.
- [ ] Validate every exported package imports and returns player actions.

## Evaluation

- [ ] Run 5-episode quick checks for all exported agents.
- [ ] Run 100-episode final evaluations against `ceia_baseline_agent`.
- [ ] Run random-agent evaluation if a random package is available.
- [ ] Run TA-agent evaluation if the TA agent is released.
- [ ] Save evaluation CSV/JSON files under `artifacts/cs8803_soccer_twos/evals`.

## Submission

- [ ] Choose strongest validated policy as final package.
- [ ] Create `TEAMNAME_AGENT.zip` from `notebooks/04_submission_and_report.ipynb`.
- [ ] Unzip-test the final zip with `validate_zip_package(final_zip)`.
- [ ] Fresh-clone test with `python -m soccer_twos.watch -m TEAMNAME_AGENT`.
- [ ] Fresh-clone test with `python -m soccer_twos.watch -m1 TEAMNAME_AGENT -m2 ceia_baseline_agent`.
- [ ] Submit exactly the final zip, not the repository.

## Report

- [ ] Use CoRL 2026 template.
- [ ] Include one curve per discussed agent.
- [ ] Include overlaid comparison curve.
- [ ] Include final evaluation table.
- [ ] State PPO/RLlib/PyTorch and final hyperparameters.
- [ ] Explain reward shaping and curriculum motivation.
- [ ] State whether shaped/curriculum improved reward, convergence, or performance.
- [ ] Give a technical explanation for the result.

# SoccerTwos CoRL Report Notes

## What Was Found In The Codebase

- The report template folder contains `example.tex`, `example.bib`, `corl_2026.sty`, and `corlabbrvnat.bst`. There was no pre-existing `figures/` folder.
- The stock CoRL template compiled locally with `latexmk -pdf -interaction=nonstopmode -halt-on-error example.tex`.
- The SoccerTwos project has organized Python modules under `soccer_twos_project/`, workflow notebooks under `notebooks/`, training artifacts under `artifacts/cs8803_soccer_twos/`, and final exported packages including `TEAMNAME_v3_AGENT`.
- The final submitted package evidence is `TEAMNAME_v3_AGENT`, whose metadata identifies `soccer_ppo_curriculum_v3`, PPO, observation size `336`, flat action space size `27`, hidden layers `[256, 256]`, and source checkpoint `checkpoint-360`.

## Identified Modifications

- Reward modification: `soccer_twos_project/envs.py` defines `RewardShapingWrapper`, which adds a clipped distance-progress bonus from player-to-ball and ball-to-goal distance deltas.
- Reward configuration: `soccer_twos_project/training.py` enables shaping for `ppo_shaped` with `player_to_ball_weight=0.01`, `ball_to_goal_weight=0.02`, and `clip=0.05`.
- Observation/action handling: no custom observation wrapper or Unity-side observation rewrite was found. Training uses a configured single-player 336-dimensional observation vector with flattened `Discrete(27)` actions; live matches use `MultiDiscrete([3,3,3])`.
- Architecture/learning modification: baseline and shaped PPO use `[512]`; curriculum PPO uses `[256,256]`. Curriculum learning uses staged start states from `configs/curriculum.yaml` and advances when `episode_reward_mean > 1.5`.

## Metrics Used

- Training metrics came from Ray `progress.csv` files under `artifacts/cs8803_soccer_twos/checkpoints/`.
- Evaluation metrics came from JSON and CSV summaries under `artifacts/cs8803_soccer_twos/evals/`.
- Main training values used in the report:
  - baseline: final reward mean `0.2399`, best `0.2631`, `2.0M` timesteps.
  - shaped: final/best reward mean `0.4036`, `2.0M` timesteps.
  - curriculum: final/best reward mean `1.7869`, `2.06M` timesteps.
  - curriculum-v3: final reward mean `1.9692`, best `1.9705`, `30,011,592` timesteps.
- Main evaluation values used in the report:
  - curriculum-v3 vs baseline: `8/10` wins, curriculum-v3 mean reward `1.1140`, baseline mean reward `-1.2084`.
  - curriculum-v3 vs prior curriculum: `6/10` wins, curriculum-v3 mean reward `0.3407`, prior curriculum mean reward `-0.4258`.

## Notebooks Used As Evidence

- `notebooks/00_environment_understanding.ipynb`: environment spaces, sparse rewards, action decoding, and reward-shaping concept.
- `notebooks/01_training_smoke_and_tensorboard.ipynb`: infrastructure proof for TensorBoard, checkpoints, `progress.csv`, and metadata.
- `notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb`: method roles for baseline, shaped PPO, curriculum PPO, and optional methods.
- `notebooks/03_full_training_pipeline.ipynb`: first baseline, shaped, and curriculum training/export run.
- `notebooks/03_full_training_pipeline_v3.ipynb`: strongest curriculum-v3 training, exports, and quick evaluations.
- `notebooks/04_submission_and_report.ipynb`: package validation, final `TEAMNAME_v3_AGENT.zip`, plot rebuilding, and report metadata.
- `notebooks/05_submission_smoke_test.ipynb`: structural zip/import/action validation for a smoke package only.

## Inferences And Caveats

- The report uses the logged curriculum-v3 run value of about `30M` timesteps. This supersedes stale package text that says `20M` timesteps.
- TensorBoard event files exist, but the report uses `progress.csv` and evaluation summaries because they contain the needed scalar metrics directly.
- The evaluation sample sizes are small: 5 or 10 episodes were found, not a 100-episode final evaluation.
- No screenshots or videos were found by file search. The final report therefore uses plots and tables rather than gameplay media.
- A behavior-cloning dataset and package exist, but one notebook records a failed imitation step with `KeyError: 'action_mode'`; imitation is not emphasized as a successful final method.

## Missing Data And TODOs

- `TODO_ADD_GITHUB_LINK` remains the required GitHub URL placeholder.
- Run larger final evaluations if time allows, especially against the random agent, baseline agent, and TA agent.
- Add gameplay screenshots or videos only if real artifacts are generated later.

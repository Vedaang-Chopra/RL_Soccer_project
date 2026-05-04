# SoccerTwos Project Explanation — Part 3: Results, File Map, Mental Model & Oral Prep

> **This is Part 3 of 3. See also:**
> - `PROJECT_EXPLANATION_1_OVERVIEW.md` — Big Picture, Environment, Baseline, Training Pipeline, Algorithms
> - `PROJECT_EXPLANATION_2_MODIFICATIONS.md` — Reward, Observation & Architecture changes

---

## 9. Training Results

### Learning Curves

Training metrics are logged by Ray into `progress.csv` files under:
```
artifacts/cs8803_soccer_twos/checkpoints/<experiment>/<trial>/progress.csv
```

The `plotting.py` script reads these and generates PNG learning curves saved to `artifacts/cs8803_soccer_twos/plots/`.

**Available plots:**
- Multiple baseline runs: `soccer_ppo_baseline__PPO_Soccer_*.png`
- Shaped: `soccer_ppo_shaped__PPO_Soccer_62ae7_00000_0_2026-04-23_09-00-21.png`
- Curriculum: `soccer_ppo_curriculum__PPO_Soccer_8ce04_00000_0_2026-04-23_09-08-41.png`
- Curriculum v2: `soccer_ppo_curriculum_v2__PPO_Soccer_dca3b_00000_0_2026-04-23_16-13-15.png`
- Curriculum v3: `soccer_ppo_curriculum_v3__PPO_Soccer_0b1b1_00000_0_2026-04-23_18-09-05.png`
- Overlaid comparison: `learning_curve_comparison.png`

### What the Reward Trends Mean

| Agent | Final Reward Mean | Budget |
|---|---|---|
| PPO Baseline | 0.2399 | 2.0M steps |
| PPO Shaped | 0.4036 | 2.0M steps |
| PPO Curriculum (v1) | 1.7869 | 2.06M steps |
| PPO Curriculum v3 | **1.9692** (best: 1.9705) | ~30M steps |

- A reward near **2.0 is close to the theoretical maximum**: scoring every episode.
- The curriculum v3 agent consistently achieved rewards above 1.9, meaning it scored in nearly every episode during training.
- The baseline's 0.24 means it barely learned to score reliably.
- Shaped PPO's improvement (0.24→0.40) shows dense rewards help, but curriculum learning is far more impactful (0.40→1.99).

**What a rising learning curve means:** the policy is improving. If it plateaus, the policy has converged (or gotten stuck in a local optimum). If it drops, the policy destabilized.

### Evaluation Results

Headless evaluations were run using `evaluation.py` and stored in `artifacts/cs8803_soccer_twos/evals/`.

**Curriculum v3 vs. PPO Baseline** (10 episodes):
- Curriculum v3 wins: **8/10** (win rate 0.80)
- Curriculum v3 mean reward: **+1.114**
- Baseline mean reward: **-1.208**
- Average episode length: **47.3 steps** (episodes end quickly — curriculum v3 scores fast)

**Curriculum v3 vs. Curriculum v1** (10 episodes):
- Curriculum v3 wins: **6/10** (win rate 0.60)
- Curriculum v3 mean reward: **+0.341**
- Curriculum v1 mean reward: **-0.426**
- Average episode length: **42.8 steps**

**Caveat:** sample sizes are small (5–10 episodes per matchup), so these numbers should be interpreted as preliminary evidence, not definitive conclusions. The planned 100-episode evaluations were not completed.

### What Statistical Results Mean

- **Win rate** = fraction of episodes where the agent scored more than the opponent. In a single-goal game, this roughly equals "did this agent score the only goal?"
- **Mean reward** above 0 = the agent tends to be the scoring team more often than not.
- **Episode length** dropping = the agent scores quickly, fewer steps per episode.

---

## 10. Final Agent: TEAMNAME_v3_AGENT

### What Is Inside This Directory?

```
TEAMNAME_v3_AGENT/
├── __init__.py         # Exports SoccerTorchAgent
├── agent.py            # The AgentInterface subclass — runs inference
├── model.py            # PolicyNetwork definition (Linear + ReLU MLP)
├── checkpoint.pth      # PyTorch state dict (the trained weights)
├── metadata.json       # Architecture config, training config, author info
├── requirements.txt    # gym, gym-unity, numpy, torch
└── README.md           # Human-readable training summary
```

### How the Final Policy Is Loaded

When the environment calls `SoccerTorchAgent(env)`:

```python
with open(metadata_path) as f:
    self.metadata = json.load(f)
self.model = PolicyNetwork(
    self.metadata["obs_size"],        # 336
    self.metadata["action_size"],     # 27
    self.metadata["hidden_layers"],   # [256, 256]
)
payload = torch.load(checkpoint_path, map_location="cpu")
state_dict = payload.get("state_dict", payload)
self.model.load_state_dict(state_dict)
self.model.eval()
```

The model is loaded in **eval mode** (no gradient tracking, BatchNorm in eval state). Inference runs via `argmax` of the 27 logits — it's greedy (always picks the highest-scored action). No sampling.

### Why This Model Was Chosen

From `REPORT_NOTES.md` and `metadata.json`:
- It was trained for the most steps (~30M, labeled "30M" in the report; metadata says "20M" which is stale)
- It achieved the highest final mean reward (1.9692) across all training runs
- It won 8/10 matches against the baseline agent
- Checkpoint 360 (identified by `analysis.get_best_checkpoint`) was the best-performing checkpoint by `episode_reward_mean`

### What Behavior Did It Likely Learn?

Based on the curriculum stages and the high win rate:
- **Stage 1-2 (Very Easy / Easy Goal):** learned to kick a nearby ball into a nearby goal — basic goal-scoring mechanics.
- **Stage 3-4 (Medium / Hard Goal):** learned to navigate from further away, chase the ball, position, and shoot.
- **Stage 5 (Random Players):** learned to handle opponent interference, though with a random opponent policy (not a strategic one).

The agent likely learned: move toward ball → approach from the side → kick toward goal. Whether it learned true multi-agent coordination (passing, positioning) is uncertain given the `single_player=True` training mode.

---

## 11. Report Summary

**File:** `report/corl_2026_template_cameraready/example.tex`

### Main Claims in the Report

1. **PPO + reward shaping improves early training** over pure sparse PPO (0.24 → 0.40 at 2M steps).
2. **Curriculum PPO is the strongest approach** (1.9692 final reward, 8/10 wins vs baseline).
3. **Curriculum learning reduces the exploration bottleneck** by starting with easy start states.
4. **The final agent (TEAMNAME_v3_AGENT)** is a curriculum PPO policy trained for ~30M steps.
5. **Reward shaping is training-only**: the submitted agent does not use privileged info at inference.

### Which Results Support These Claims

- **Claim 1:** Progress CSV values from `REPORT_NOTES.md` (0.2399 baseline vs 0.4036 shaped).
- **Claim 2:** Evaluation JSONs (`soccer_ppo_baseline_vs_soccer_ppo_curriculum_v3_summary.json`).
- **Claims 3, 4, 5:** Code evidence from `training.py`, `envs.py`, `curriculum.yaml`, `metadata.json`.

### Uncertain or TODO Claims

From `REPORT_NOTES.md`:
- `TODO_ADD_GITHUB_LINK` — GitHub URL is a placeholder in the report.
- Evaluation sample sizes are small (5–10 episodes, not 100). Results are labeled "preliminary."
- No screenshots or videos exist in the artifacts. The report uses only plots and tables.
- Imitation learning was attempted but failed. Not included as a successful method.

---

## 12. File Map

### Important Documentation

| File | What It Does |
|---|---|
| `README.md` | Project overview, setup instructions, agent packaging rules |
| `docs/PROJECT_STRUCTURE.md` | Map of all Python modules and their roles |
| `docs/FINAL_PROJECT_TRACKER.md` | Checklist of all tasks from setup to report |
| `report/corl_2026_template_cameraready/REPORT_NOTES.md` | Evidence, metrics, caveats used to write the report |
| `report/corl_2026_template_cameraready/example.tex` | The actual final report (LaTeX) |

### Important Notebooks

| Notebook | What It Does |
|---|---|
| `notebooks/00_environment_understanding.ipynb` | Inspect obs/action spaces, run random rollouts, see info dict |
| `notebooks/01_training_smoke_and_tensorboard.ipynb` | 25k-step smoke training to verify the full stack |
| `notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb` | Overview of all method options |
| `notebooks/03_full_training_pipeline.ipynb` | First real training run (baseline, shaped, curriculum) |
| `notebooks/03_full_training_pipeline_v3.ipynb` | Final strongest training run (curriculum v3, ~30M steps) |
| `notebooks/04_submission_and_report.ipynb` | Export, evaluation, report artifacts |
| `notebooks/05_submission_smoke_test.ipynb` | Structural validation of the submission zip |

### Important Python Files

| File | What It Does |
|---|---|
| `soccer_twos_project/config.py` | Hardware profiles (cpu_debug, free_gpu, a40_full, etc.), artifact paths |
| `soccer_twos_project/envs.py` | `create_rllib_env` factory, `RewardShapingWrapper`, curriculum sampling helpers |
| `soccer_twos_project/training.py` | Training stages (ppo_baseline, ppo_shaped, ppo_curriculum, etc.), Ray Tune integration |
| `soccer_twos_project/exporting.py` | Extracts policy weights from RLlib checkpoint, writes submission package |
| `soccer_twos_project/evaluation.py` | Headless agent-vs-agent evaluation, writes CSV and JSON summaries |
| `soccer_twos_project/imitation.py` | Behavior cloning — collect expert data, train BC policy |
| `soccer_twos_project/plotting.py` | Reads `progress.csv`, generates PNG learning curves |
| `soccer_twos_project/notebook_tools.py` | Helper wrappers for all notebook cells |

### Result/Log Files

| File | What It Contains |
|---|---|
| `artifacts/cs8803_soccer_twos/checkpoints/<run>/progress.csv` | Per-iteration timesteps and reward during training |
| `artifacts/cs8803_soccer_twos/checkpoints/<run>/run_metadata.json` | Best checkpoint path, hardware info, training config |
| `artifacts/cs8803_soccer_twos/plots/*.png` | Learning curves per run + overlay comparison |
| `artifacts/cs8803_soccer_twos/evals/*_summary.json` | Head-to-head evaluation results (wins, mean reward) |
| `artifacts/cs8803_soccer_twos/evals/*_episodes.csv` | Per-episode detail of evaluation matches |

### Report Files

| File | What It Does |
|---|---|
| `report/corl_2026_template_cameraready/example.tex` | Final report source (LaTeX, CoRL 2026 template) |
| `report/corl_2026_template_cameraready/example.pdf` | Compiled report PDF |
| `report/corl_2026_template_cameraready/example.bib` | Bibliography (Unity ML-Agents, PPO, RLlib, curriculum learning papers) |

### Final Agent Files

| File | What It Does |
|---|---|
| `TEAMNAME_v3_AGENT/agent.py` | `SoccerTorchAgent` — loads model, runs inference |
| `TEAMNAME_v3_AGENT/model.py` | `PolicyNetwork` — the MLP definition |
| `TEAMNAME_v3_AGENT/checkpoint.pth` | Trained weights (PyTorch state dict) |
| `TEAMNAME_v3_AGENT/metadata.json` | obs_size=336, action_size=27, hidden=[256,256], stage=ppo_curriculum_v3 |
| `TEAMNAME_v3_AGENT.zip` | Submission zip |
| `configs/curriculum.yaml` | 5 curriculum stages with ball/player position ranges |

---

## 13. Step-by-Step Mental Model

Here is the complete story, end to end:

1. **The soccer environment starts.** The Unity binary launches. The ball is placed at center field (or a curriculum-specified position in early training). Four players are placed — two blue, two orange.

2. **The agent observes the field.** The Unity binary computes a 336-dimensional observation vector — relative positions and velocities of all entities, from this player's perspective. This vector is sent to the Python process.

3. **The agent's policy network runs.** The 336 floats go through:
   `Linear(336→256) → ReLU → Linear(256→256) → ReLU → Linear(256→27)`
   This produces 27 logits — one score per possible action combination.

4. **An action is selected.** The `argmax` of the 27 logits is chosen. This index maps to one of 27 combinations of (forward/back, rotate, strafe).

5. **The action is sent to Unity.** The Unity simulator steps forward one frame. The player moves. The ball moves. The opponent (fixed policy or curriculum opponent) also acts.

6. **A reward is returned.** If no goal: reward = 0 (plus a tiny shaping bonus during `ppo_shaped` training). If a goal was scored: reward = +1 (blue team) or -1 (orange team).

7. **The experience is stored.** The (observation, action, reward, next observation) tuple is stored in the worker's rollout buffer.

8. **After 2,000 steps, the worker sends its buffer to the driver.** The driver waits for all 40 workers to send their buffers (total: 80,000 steps).

9. **PPO computes advantages.** For each step, it estimates: "was this action better or worse than what the value function expected?" This is the Generalized Advantage Estimate (GAE).

10. **The policy network is updated.** Multiple SGD passes over minibatches of 8,000 steps. The policy parameters move to increase probability of high-advantage actions. The clip ratio prevents any parameter from moving too far.

11. **Updated weights are sent back to workers.** All 40 workers get the new policy. They collect the next 2,000 steps each with the new policy.

12. **This repeats ~15,000 times** (80k steps/iteration × 15,000 iterations ≈ 30M total steps for curriculum v3).

13. **The best checkpoint is identified** by `analysis.get_best_checkpoint` (highest `episode_reward_mean`). Checkpoint 360 was selected.

14. **The policy is exported.** `exporting.py` restores the RLlib checkpoint, extracts the MLP weights, and writes them to `checkpoint.pth` alongside `agent.py` and `model.py`. This forms the standalone `TEAMNAME_v3_AGENT` package.

15. **Better behaviors emerged.** By the end, the agent learned to find the ball, approach it, and kick it into the goal — scoring in ~95%+ of episodes during training.

---

## 14. Common Confusions

### Why does reward shaping matter if it's such a small bonus?

The bonus (max ±0.05) seems tiny compared to a goal reward (±1.0). But during early training, the agent almost never scores — so it sees goal rewards only rarely. The shaping bonus appears every step, giving a consistent gradient signal. Over thousands of steps, even a +0.002 bonus per step adds up and guides the policy toward productive behavior.

Think of it as: the goal reward teaches "what to ultimately achieve," while the shaping reward teaches "how to get started."

### Why might high reward not mean good soccer behavior?

If the curriculum has easy early tasks, the agent can accumulate reward by only scoring near-goal shots, never learning how to attack from midfield. The high reward (1.9+) is in the curriculum context — against a stationary opponent, with the ball starting nearby. Against a skilled opponent, the same policy might struggle.

High reward = good performance within the training distribution. Generalization to real competitive play is a separate question.

### Why can training be unstable?

Several causes:
1. **Large batch sizes** can cause gradient explosions (controlled by `grad_clip=0.5`).
2. **Worker crashes** from Unity environment process failures (handled by retry logic in `training.py` `should_retry`).
3. **Non-stationarity**: if the environment changes (curriculum advancement), the policy may temporarily get worse.
4. **PPO clip mistuning**: if the learning rate is too large, the policy ratio exceeds the clip threshold too often, causing noisy updates.

### Why might multiple algorithms perform differently?

- **PPO vs DQN**: PPO is on-policy (uses only fresh data), DQN is off-policy (reuses old data). For sparse reward, DQN's replay buffer can help, but PPO's stability often wins in practice.
- **Baseline vs Shaped PPO**: same algorithm, different reward signal. Shaped has more gradient signal per step.
- **Baseline vs Curriculum**: same algorithm, different data distribution. Curriculum has better-distributed early experience.

The curriculum's advantage is not about the algorithm — it's about the training data distribution.

### Why are screenshots and videos needed in the report?

The assignment requires evidence of learned behavior, not just numbers. A table of rewards is abstract — it doesn't show whether the agent actually plays meaningful soccer or is exploiting some quirk of the environment. Screenshots/videos provide qualitative evidence that the agent learned real soccer-like behavior. Since no gameplay videos were generated in this project, the report notes this as a limitation.

---

## 15. What I Should Say If Asked

### "What did you modify?"

*"I made three types of modifications. First, I added a reward shaping wrapper that gives small dense bonuses for moving toward the ball and pushing the ball toward the goal — this is implemented in `envs.py`. Second, I implemented a curriculum learning pipeline using `curriculum.yaml`, which starts the agent with easy near-goal scenarios and advances to harder ones as it improves — this is managed by the `CurriculumUpdateCallback` in `training.py`. Third, I changed the network architecture for curriculum PPO from one 512-unit layer to two 256-unit layers, allowing the policy to learn more structured representations."*

### "Why did you modify reward/observation/architecture?"

*"The environment has sparse rewards — the agent only gets feedback when a goal is scored. Early in training, that almost never happens, so the gradient signal is nearly zero. Reward shaping gives the agent intermediate feedback every step, so it can learn to approach the ball before it learns to score. Curriculum learning addresses the same problem differently — instead of changing the reward, it changes the starting states to be easy early on, so the agent sees goal rewards frequently from the beginning. The deeper architecture [256,256] was chosen for the curriculum stage because a more complex training signal can benefit from a more expressive network."*

### "What algorithm did you use?"

*"I used Proximal Policy Optimization (PPO) implemented via Ray RLlib with PyTorch. PPO is an on-policy policy gradient method that clips the probability ratio to prevent large destructive policy updates. I trained three variants: plain PPO baseline, reward-shaped PPO, and curriculum PPO. The final submitted agent comes from the curriculum PPO v3 run, trained for approximately 30 million environment steps."*

### "What were your results?"

*"The baseline PPO reached a training reward of 0.24 after 2 million steps. Reward shaping improved this to 0.40 at the same budget. Curriculum PPO dramatically outperformed both — reaching 1.97 after 30 million steps, which is close to the theoretical maximum of 2.0. In head-to-head evaluation, the final curriculum agent won 8 out of 10 matches against the baseline PPO agent. I should note the evaluation sample size is small — 10 episodes — so these results are preliminary."*

### "What were the limitations?"

*"Several. First, evaluation sample sizes are small — 5 to 10 episodes per matchup rather than the planned 100. Second, training always used a fixed, weak opponent (standing still), so the agent hasn't been tested against adaptive opponents. Third, imitation learning was attempted but encountered an error and was not successfully completed. Fourth, no gameplay screenshots or videos were generated to provide qualitative evidence of learned behavior. Fifth, the GitHub link in the report is still a placeholder."*

---

*End of Part 3. Read all three files together for a complete understanding of the SoccerTwos project.*

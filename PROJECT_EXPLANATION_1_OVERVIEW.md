# SoccerTwos Project Explanation — Part 1: Overview, Environment & Training Pipeline

> **This is Part 1 of 3. See also:**
> - `PROJECT_EXPLANATION_2_MODIFICATIONS.md` — Reward, Observation & Architecture changes
> - `PROJECT_EXPLANATION_3_RESULTS_AND_ORAL.md` — Results, File Map, Mental Model, Oral Prep

---

## 1. Big Picture

### What Is SoccerTwos?

SoccerTwos is a simulated 2-vs-2 soccer game built inside Unity, exposed to Python via the ML-Agents toolkit. There are four players on the field — two on the blue team, two on the orange team. The agent we train controls the blue team. The goal is simple to describe but hard to learn: **score more goals than the opponent**.

The game is "sparse reward" — the agent only receives a meaningful signal (+1 for scoring, -1 for conceding) when a goal is actually scored. This is like teaching a child soccer by only saying "good job" when they score and "bad job" when they concede — with no feedback for 99% of the time.

### What Is the Agent Trying to Learn?

The agent must learn:
- How to find the ball
- How to kick the ball toward the opponent's goal
- How to position itself to defend

None of this is told to the agent. It must discover all of it through trial and error. The training process — reinforcement learning — is what makes this happen.

### What Counts as Good Behavior?

From the environment's perspective, good behavior = maximizing cumulative reward. In practice this means:
- Moving toward the ball efficiently
- Pushing the ball toward the opponent's goal
- Not ignoring the ball (standing still is penalized implicitly because no reward arrives)

### Why Is This a Reinforcement Learning Problem?

Three reasons:
1. **No teacher**: we can't write rules for every possible ball position. There are too many states.
2. **Sequential decisions**: each action affects the next state. You can't optimize each step independently.
3. **Delayed reward**: the signal (a goal) only appears long after the actions that caused it.

This is exactly the setting RL is designed for: an agent taking sequences of actions in an environment, receiving sparse delayed rewards, and learning a policy that maximizes total return.

---

## 2. Environment Setup

### What Does the Agent Observe?

The training observation is a **336-dimensional float vector** (confirmed in `TEAMNAME_v3_AGENT/metadata.json`: `"obs_size": 336`).

This is a single flat array that encodes positions, velocities, and orientations of all players and the ball, from the agent's local perspective. The agent cannot directly see "I am at position (3.2, 1.5)." Instead, it sees relative values — distances and angles — from its own frame of reference.

Why 336 dimensions? The Unity ML-Agents soccer environment concatenates several observation stacks (player state, teammate state, opponent states, ball state) across a few history frames to give the agent some short-term memory.

### What Actions Can the Agent Take?

**During training** (`training.py`, `base_env_config`): the action space is `Discrete(27)` — 27 discrete flat actions. This is a flattened version of a branched action space where the agent independently controls:
- Forward/backward movement (3 values)
- Left/right rotation (3 values)
- Lateral strafe (3 values)

3 × 3 × 3 = 27 total combinations.

**During live matches**: the action space is `MultiDiscrete([3, 3, 3])` — three separate branches. The exported agent handles this conversion automatically using `ActionFlattener` in `TEAMNAME_v3_AGENT/agent.py`.

### What Rewards Does the Agent Receive?

**Sparse base reward:**
- `+1.0` when the blue team scores a goal
- `-1.0` when the orange team scores a goal

**Dense shaping bonus (during `ppo_shaped` training only):**
- A small clipped bonus based on how much closer the player moved to the ball, and how much closer the ball moved to the goal.
- Implemented in `soccer_twos_project/envs.py`, class `RewardShapingWrapper`.
- Weights: `player_to_ball_weight=0.01`, `ball_to_goal_weight=0.02`, clipped to ±0.05.

### What Is an Episode?

An episode starts when the ball is placed at center field (or at a curriculum-defined position). It ends when a goal is scored. The agent interacts with the environment step by step until one team scores. Steps between goals are where all the learning happens.

### What Makes This Hard?

1. **Sparse reward**: the agent can go hundreds of steps with reward = 0, giving it almost no gradient signal.
2. **Multi-agent complexity**: the opponent team also acts, making the environment non-stationary.
3. **Exploration**: a random policy almost never scores, so the agent rarely sees positive reward early in training.
4. **Long episodes**: if neither team scores, the episode can last a very long time with no feedback.

---

## 3. Baseline Approach

### What Did the Starter Code Provide?

The starter kit (from `https://github.com/bryanoliveira/soccer-twos-starter`) provides:
- A Unity binary running the SoccerTwos game
- Python wrappers: `soccer_twos.make(...)` to create the environment
- An `AgentInterface` base class for implementing agents
- A pre-trained `ceia_baseline_agent` to evaluate against

The starter kit does **not** provide a trained RL agent — that is what this project builds.

### What Is the Baseline Algorithm?

**Proximal Policy Optimization (PPO)**, implemented via Ray RLlib.

PPO is a policy gradient method. It:
1. Collects experience rollouts (observations, actions, rewards)
2. Estimates the advantage (how much better was this action than average?)
3. Updates the policy network to increase probability of good actions
4. Clips the update ratio to prevent large destabilizing changes

The baseline PPO configuration (from `training.py`, `build_training_config`):
- Network: single hidden layer of 512 units, ReLU activation
- `vf_share_layers=True` (policy and value networks share the hidden layers)
- `grad_clip=0.5` (gradient clipping for stability)
- No reward shaping — pure sparse reward

### How Does the Baseline Policy Learn?

1. Ray spawns multiple worker processes, each running a copy of the soccer environment.
2. Workers collect `rollout_fragment_length` steps of experience.
3. The driver aggregates `train_batch_size` total steps.
4. PPO computes advantages (using GAE), then performs multiple SGD epochs over minibatches.
5. The updated weights are broadcast back to all workers.
6. Repeat until `timesteps_total` is reached.

### What Were the Baseline's Limitations?

From `REPORT_NOTES.md` and the training logs:
- Baseline PPO reached a final mean reward of only **0.2399** after 2.0M steps.
- Sparse reward meant the agent spent most of early training with near-zero gradient signal.
- A single 512-unit hidden layer may not capture the full complexity of coordinated soccer behavior.
- No curriculum — the agent always started from a random ball placement, even early in training when it had no idea how to score.

---

## 4. Final Training Pipeline

### Which Notebooks/Scripts Train the Agent?

The main training flow runs through:

1. **`notebooks/00_environment_understanding.ipynb`** — environment inspection, spaces, rewards
2. **`notebooks/01_training_smoke_and_tensorboard.ipynb`** — 25k-step smoke training to verify the stack works
3. **`notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb`** — method overview
4. **`notebooks/03_full_training_pipeline.ipynb`** — first full training run (baseline, shaped, curriculum)
5. **`notebooks/03_full_training_pipeline_v3.ipynb`** — the final strongest training run (curriculum v3, ~30M steps)
6. **`notebooks/04_submission_and_report.ipynb`** — export, evaluation, and report artifacts
7. **`notebooks/05_submission_smoke_test.ipynb`** — structural validation of the submission zip

The underlying Python logic lives in `soccer_twos_project/training.py`.

### How Does Training Start?

From `training.py`, function `run_tune`:

```python
ray.init(ignore_reinit_error=True, include_dashboard=False)
tune.registry.register_env("Soccer", create_rllib_env)
analysis = tune.run(spec["algo"], name=spec["experiment"], config=config, stop=stop, ...)
```

Ray is initialized, the environment factory `create_rllib_env` is registered under the name `"Soccer"`, and `tune.run` launches the training experiment.

The hardware profile (`config.py`) determines how many workers and GPUs are used. The v3 run used the `a40_full` profile: 40 workers, 1 GPU, `train_batch_size=80000`, `rollout_fragment_length=2000`.

### How Are Rollouts Collected?

Each of the 40 workers runs a copy of the soccer environment and collects 2,000 steps of experience per iteration. Workers run asynchronously. The driver aggregates 80,000 total steps before performing a PPO update.

In `envs.py`, `create_rllib_env` creates the environment with:
- `variation=EnvType.team_vs_policy` — the agent controls one team
- `single_player=True` — single controlled player
- `flatten_branched=True` — actions are flattened to Discrete(27)
- `opponent_policy=lambda *_: 0` — the opponent always takes action 0 (standing still)

### How Are Rewards Computed?

**For the baseline and curriculum**: the environment returns the sparse ±1 reward directly. No modification.

**For the shaped variant**: the `RewardShapingWrapper` (in `envs.py`) intercepts each `env.step()` call, extracts `player_info` and `ball_info` from the info dict, computes distance deltas, adds a clipped bonus, and returns the modified reward. The curriculum variants do **not** use this wrapper.

### How Do Policy Updates Happen?

PPO in RLlib:
1. Collects a full batch of `train_batch_size=80000` steps across workers.
2. Computes advantages using GAE (Generalized Advantage Estimation).
3. Runs multiple SGD passes over minibatches of size `sgd_minibatch_size`.
4. Applies gradient clipping at 0.5 (`grad_clip=0.5`) to prevent NaN logits from large batch sizes.
5. Uses `batch_mode="complete_episodes"` for curriculum runs — waits for full episodes before updating.

### How Are Checkpoints Saved?

Ray Tune saves checkpoints automatically. From `run_tune`:
```python
analysis = tune.run(
    ...,
    checkpoint_freq=profile.checkpoint_freq,  # every N iterations
    checkpoint_at_end=True,                    # always save at the end
    local_dir=str(dirs["checkpoints"]),        # saved to artifacts/cs8803_soccer_twos/checkpoints/
)
```

After training, the best checkpoint is identified by:
```python
best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
```

Metadata is saved to `run_metadata.json` including the best checkpoint path, hardware info, and training config.

### Which Final Model Was Selected?

The final model is `TEAMNAME_v3_AGENT`, exported from the v3 curriculum run. From `metadata.json`:
- Stage: `ppo_curriculum_v3`
- Checkpoint: `checkpoint-360` (360th checkpoint of the v3 run)
- Trained for ~30M environment steps
- Best logged mean reward: **1.9705**

---

## 5. Algorithm Explanation

### PPO Baseline (`ppo_baseline`)

**What it does:** Standard on-policy policy gradient with clipped surrogate objective. Collects experience, estimates advantages, updates policy toward better-than-average actions, but limits the size of each update.

**Why it might work:** PPO is a reliable workhorse for continuous and discrete control. It's stable and doesn't require a replay buffer.

**Files:** `training.py` `build_training_config` stage `"ppo_baseline"`, network `[512]`, no reward shaping.

**What happened:** Final reward 0.2399 at 2.0M steps. Reasonable but slow due to sparse reward signal.

---

### PPO with Reward Shaping (`ppo_shaped`)

**What it does:** Same PPO, but the environment is wrapped with `RewardShapingWrapper` to add dense feedback during training.

**Why it might work:** Dense rewards give the agent something to optimize before it ever scores a goal. Even small positive signals for moving toward the ball can guide early exploration.

**Files:** `training.py` stage `"ppo_shaped"` enables `reward_shaping` dict in `env_config`; `envs.py` `RewardShapingWrapper` applies the bonus.

**What happened:** Final reward 0.4036 at 2.0M steps — better than baseline. But still not the strongest policy because dense shaping alone doesn't teach full game strategy.

---

### Curriculum PPO (`ppo_curriculum`, `ppo_curriculum_v2`, `ppo_curriculum_v3`)

**What it does:** PPO with a curriculum learning callback. The curriculum starts with easy tasks (ball placed close to the opponent goal, player already in scoring position) and gradually advances to harder configurations as the agent improves.

**Why it might work:** Early easy tasks give the agent frequent reward signals, allowing it to learn the goal-scoring behavior first, then generalize to harder situations. This addresses the exploration bottleneck directly.

**Files:**
- `configs/curriculum.yaml` — defines the 5 curriculum stages
- `training.py` `CurriculumUpdateCallback` — advances stage when `episode_reward_mean > 1.5`
- `training.py` stage `"ppo_curriculum_v3"` — used for the final model

**Curriculum stages** (from `curriculum.yaml`):
1. **Very Easy Goal**: ball is x=[12,14], player is x=[7,11] — already near the goal
2. **Easy Goal**: ball at x=[7,14], any player rotation
3. **Medium Goal**: ball at x=[0,14]
4. **Hard Goal**: ball at x=[-10,14]
5. **Random Players**: full field, with opponent players moving randomly

**What happened:** Curriculum v3 reached final reward **1.9692** at ~30M steps. This is close to the theoretical maximum of 2.0 (scoring every episode). It also won 8/10 matches against the baseline agent.

---

### Self-Play (`ppo_selfplay`) — Fallback

**What it does:** Instead of training against a fixed opponent, the agent trains against past versions of itself. The `SelfPlayUpdateCallback` maintains an archive of past policies (opponent_1, opponent_2, opponent_3) and periodically shifts the current policy into the archive.

**Files:** `training.py` `SelfPlayUpdateCallback`, `selfplay_policy_mapping_fn` — player 0 always uses the current "default" policy; players 1-3 sample from the archive with decreasing probability.

**What happened:** This was configured as a fallback if curriculum was insufficient. It was not used as the primary final agent.

---

### DQN Baseline (`dqn_baseline`) — Optional

**What it does:** Deep Q-Network — an off-policy value-based method. Stores transitions in a replay buffer and trains a Q-function (value of each action in each state) using TD learning.

**Why it might work:** DQN can be more sample-efficient than PPO because it reuses past experience.

**Files:** `training.py` stage `"dqn_baseline"`, network `[512, 256]`.

**What happened:** Included as an optional comparison. Not selected as the final policy.

---

### Behavior Cloning (`bc_imitation`) — Optional

**What it does:** Supervised learning. Collects state-action pairs from an expert agent (the `ceia_baseline_agent`), then trains a neural network to imitate those actions using cross-entropy loss.

**Files:** `soccer_twos_project/imitation.py` — `collect_dataset` gathers samples, `train_bc` trains the `BCPolicyNetwork`.

**What happened:** A dataset was collected and training was attempted, but a `KeyError: 'action_mode'` was logged in a notebook. Imitation learning was not selected as the primary final agent. `REPORT_NOTES.md` notes: "imitation is not emphasized as a successful final method."

---

*Continue reading: `PROJECT_EXPLANATION_2_MODIFICATIONS.md`*

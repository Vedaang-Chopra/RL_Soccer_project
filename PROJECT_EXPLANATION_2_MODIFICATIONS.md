# SoccerTwos Project Explanation — Part 2: Modifications

> **This is Part 2 of 3. See also:**
> - `PROJECT_EXPLANATION_1_OVERVIEW.md` — Big Picture, Environment, Baseline, Training Pipeline, Algorithms
> - `PROJECT_EXPLANATION_3_RESULTS_AND_ORAL.md` — Results, File Map, Mental Model, Oral Prep

---

## 6. Reward Modifications

### Why Reward Shaping Matters

In SoccerTwos, the base environment gives reward only when a goal is scored. This is **extremely sparse**. For most of training — especially early on — the reward is exactly 0 every step. This means the gradient of the loss with respect to policy parameters is also approximately 0. The agent gets almost no signal about which actions were good.

Reward shaping is the practice of adding **auxiliary dense reward terms** during training to give the agent intermediate feedback. The key insight: if you reward the agent for *moving toward* the goal, it will explore more productively, even before it knows how to score.

**The critical rule:** shaping rewards must only be used during training, not at inference time. The submitted agent should not depend on privileged environment info to act.

---

### What Reward Terms Were Added

**File:** `soccer_twos_project/envs.py`, class `RewardShapingWrapper`

Two distance-progress terms were added:

#### Term 1: Player-to-Ball Progress

```python
bonus += self.player_to_ball_weight * (
    self.previous_metrics["player_ball_dist"] - metrics["player_ball_dist"]
)
```

- **What it measures:** how much closer did the player get to the ball this step?
- **Weight:** `player_to_ball_weight = 0.01`
- **Why it helps:** encourages the agent to move toward the ball. Without this, a randomly-initialized policy often stands still or wanders away.
- **Example:** if the player was 5.0 units from the ball last step and is 4.8 units now, the bonus is `0.01 × 0.2 = +0.002`. Small, but it accumulates over hundreds of steps.
- **Possible downside:** the agent might learn to chase the ball obsessively without ever shooting. "Reward hacking" — optimizing the proxy reward instead of actually scoring.

#### Term 2: Ball-to-Goal Progress

```python
bonus += self.ball_to_goal_weight * (
    self.previous_metrics["ball_goal_dist"] - metrics["ball_goal_dist"]
)
```

- **What it measures:** how much closer did the ball get to the opponent's goal this step?
- **Weight:** `ball_to_goal_weight = 0.02` (twice the player-to-ball weight)
- **Goal x-coordinate used:** `goal_x = 14.0` (hardcoded in the wrapper constructor)
- **Why it helps:** teaches the agent that pushing the ball toward the goal is the right direction. Without this, the agent might kick the ball sideways or backward.
- **Example:** if the ball was 6.0 units from the goal last step and is 5.5 units now, the bonus is `0.02 × 0.5 = +0.01`.
- **Possible downside:** the agent might learn to push the ball in a straight line regardless of defenders, missing the strategic dimension of the game.

#### Clipping

```python
return max(-self.clip, min(self.clip, bonus))
```

- **Clip value:** ±0.05
- **Why clip?** Without clipping, a single very large distance delta (e.g., kicking the ball hard toward the goal in one step) could produce a huge reward that dwarfs the sparse goal reward. Clipping keeps the shaping signal small and supplementary, not dominant.

---

### Where Is It Enabled?

In `training.py`, `build_training_config`:

```python
if stage == "ppo_shaped":
    env_config["reward_shaping"] = {
        "player_to_ball_weight": 0.01,
        "ball_to_goal_weight": 0.02,
        "clip": 0.05,
    }
```

The `create_rllib_env` factory in `envs.py` reads this key:

```python
reward_shaping = env_config.pop("reward_shaping", None)
...
if reward_shaping:
    env = RewardShapingWrapper(env, **reward_shaping)
```

**Important:** the curriculum variants (`ppo_curriculum`, `ppo_curriculum_v2`, `ppo_curriculum_v3`) do **not** use `RewardShapingWrapper`. They rely on the curriculum to provide early positive signal instead.

---

### How Reward Is Computed From the Info Dict

The `RewardShapingWrapper` extracts positions from the environment info dict:

```python
player_pos = self._extract_position(entry.get("player_info"))
ball_pos = self._extract_position(entry.get("ball_info"))
```

`player_info` and `ball_info` are fields returned by the `soccer_twos` environment in the `info` dict on each step. These contain 2D positions and are not part of the observation vector — they are privileged information only available during training, not exposed to the deployed agent.

---

## 7. Observation Modifications

### What Was Modified

**Short answer:** No custom observation wrapper or Unity-side observation change was implemented. This is explicitly stated in `REPORT_NOTES.md`:

> "Observation/action handling: no custom observation wrapper or Unity-side observation rewrite was found."

And confirmed in the report (`example.tex`):

> "I did not find a custom observation-space wrapper or Unity-side observation change in the final code."

The 336-dimensional observation vector comes directly from the `soccer_twos` environment's Unity binary. It encodes player positions, velocities, orientations, and ball state — all from the agent's local frame of reference.

---

### What Was Done at the Interface Level

While no custom feature engineering was added, there is a **configuration-level observation/action handling change** that matters:

#### Single-Player Training Mode

In `training.py`, `base_env_config`:

```python
return {
    "variation": EnvType.team_vs_policy,
    "multiagent": False,
    "single_player": True,
    "flatten_branched": True,
    "opponent_policy": lambda *_: 0,
}
```

- `single_player=True`: trains one player at a time rather than jointly training both team members. This simplifies the learning problem.
- `flatten_branched=True`: converts the `MultiDiscrete([3,3,3])` live action space into a flat `Discrete(27)` for training. This makes the action space smaller and simpler for PPO.
- `opponent_policy=lambda *_: 0`: the opposing team always takes action 0 (stand still). This is a fixed weak opponent — easier to beat than the real baseline agent.

#### Action Conversion at Inference

The deployed agent in `TEAMNAME_v3_AGENT/agent.py` handles the action space mismatch:

```python
if self.action_mode == "flat_discrete" and hasattr(env.action_space, "nvec"):
    self.flattener = ActionFlattener(env.action_space.nvec)
```

During `act()`:
```python
action_index = int(torch.argmax(logits, dim=-1).item())
if self.flattener is not None:
    actions[player_id] = self.flattener.lookup_action(action_index)
```

The policy was trained on `Discrete(27)` flat actions. During live matches the environment expects `MultiDiscrete([3,3,3])`. The `ActionFlattener` converts the flat index back to the three-branch format.

---

### Why Relative Observations Matter (Conceptual Note)

Even though we didn't add custom observations, understanding what the 336-dim vector contains is important:

The observation is **egocentric** (from the agent's own frame). This means:
- "The ball is 3 units ahead and 1 unit to the left of me" — not "the ball is at world position (7.3, 2.1)."
- This is good for generalization: the same policy weights work regardless of where on the field the agent is standing.

If someone asks "why 336 dimensions?": ML-Agents stacks several sensor observations and may include a few history frames. `TODO: confirm exact breakdown from ML-Agents soccer environment source if needed.`

---

## 8. Architecture Modifications

### What Network Architectures Were Used?

Two configurations appear in `training.py`:

#### Baseline and Shaped PPO: `[512]`

```python
"model": {
    "vf_share_layers": True,
    "fcnet_hiddens": [512],
    "fcnet_activation": "relu",
},
```

- **One hidden layer of 512 units, ReLU activation.**
- Shared layers between policy head (outputs action logits) and value head (outputs state value estimate).
- `vf_share_layers=True` means the hidden layer(s) are used for both the policy and the value function. Only the output heads differ. This is computationally efficient and reduces the number of parameters.

#### Curriculum PPO: `[256, 256]`

```python
"model": {
    "vf_share_layers": True,
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
},
```

- **Two hidden layers of 256 units each, ReLU activation.**
- More depth allows the network to represent more complex nonlinear functions.
- Why 256×2 instead of 512? With curriculum learning providing a richer training signal, a deeper (but not wider) network can learn more structured representations. Also, two layers of 256 have fewer parameters than one layer of 512 (256×336 + 256×256 ≈ 151k vs 512×336 ≈ 172k for the first layer alone), making training faster per step.

---

### The Final Submitted Architecture

From `TEAMNAME_v3_AGENT/metadata.json`:

```json
"hidden_layers": [256, 256],
"obs_size": 336,
"action_size": 27
```

The `PolicyNetwork` class in `TEAMNAME_v3_AGENT/model.py` (and identical in `exporting.py`'s `MODEL_TEMPLATE`):

```python
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_layers):
        super().__init__()
        layers = []
        last_size = obs_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(last_size, action_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
```

So the full architecture is:
```
Input (336) → Linear(336→256) → ReLU → Linear(256→256) → ReLU → Linear(256→27) → logits
```

The `argmax` of the 27 logits gives the selected action.

---

### Why Architecture Changes Matter

A single hidden layer of 512 units maps directly from the 336-dimensional observation to the 512-dimensional hidden representation. This can capture first-order patterns: "if ball is far, move toward it."

Two layers of 256 allow **compositional representations**: the first layer can detect features (ball direction, distance to goal), the second layer can combine features (ball-is-near AND facing-goal → shoot). This is why deeper networks can represent more complex soccer strategies like "position myself to receive a pass" or "cut off the opponent."

That said, the key improvement from baseline to curriculum v3 is **curriculum learning**, not just the architecture. The architecture change is secondary.

---

### Gradient Clipping

All variants use `"grad_clip": 0.5`. This prevents gradient explosions when using large batch sizes (80,000 steps) with many parallel workers. Without clipping, a single bad minibatch can cause the policy logits to go to ±infinity (NaN), crashing training. This is documented in the code comment in `training.py`:

```python
# Clamp gradients — prevents NaN logits from gradient overflow
# with large batches and many parallel workers.
"grad_clip": 0.5,
```

---

*Continue reading: `PROJECT_EXPLANATION_3_RESULTS_AND_ORAL.md`*

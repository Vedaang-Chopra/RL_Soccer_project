# RL Visual Learning Notebooks

These notebooks are a beginner-focused track for understanding reinforcement
learning in SoccerTwos. They are separate from the final-project workflow
notebooks and are meant for intuition, plots, and tiny experiments.

Run them in order:

1. `00_env_visual_probe.ipynb`
   Inspect rewards, actions, distances, and top-down trajectories.

2. `01_rl_basics_policy_value_advantage.ipynb`
   Learn state, action, reward, return, value, advantage, and PPO intuition.

3. `02_tiny_ppo_training_watch.ipynb`
   Run a tiny PPO smoke job and inspect training diagnostics.

4. `03_behavior_before_after_training.ipynb`
   Compare random or fixed behavior against a trained checkpoint or exported
   package.

Unity rendering cells are optional and disabled by default. Linux/PACE sessions
stay headless unless you are in a GUI-enabled session with a live display. The
default path is still headless plots, so the notebooks are usable on local Mac,
Colab, or PACE.

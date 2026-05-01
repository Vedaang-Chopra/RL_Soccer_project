import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        # Skip RLlib's env validity checks — we only use this Trainer to load
        # checkpoint weights and call compute_single_action, not to roll out.
        config["disable_env_checking"] = True

        # The checkpoint was trained with 4 self-play policies. RLlib requires a
        # MultiAgentEnv when multiagent.policies is non-empty, even for inference.
        # Reduce to just the "default" policy so a plain gym.Env is sufficient.
        multiagent_cfg = config.get("multiagent", {})
        policies = multiagent_cfg.get("policies", {}) if isinstance(multiagent_cfg, dict) else {}

        # Extract obs/action spaces from the saved default policy spec.
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
        act_space = gym.spaces.MultiDiscrete([3, 3, 3])
        default_spec = policies.get(POLICY_NAME)
        if isinstance(default_spec, (tuple, list)) and len(default_spec) >= 3:
            if default_spec[1] is not None:
                obs_space = default_spec[1]
            if default_spec[2] is not None:
                act_space = default_spec[2]

        # Strip down to a single policy so RLlib accepts a plain gym.Env.
        config["multiagent"] = {
            "policies": {POLICY_NAME: default_spec},
            "policy_mapping_fn": lambda agent_id, **kwargs: POLICY_NAME,
            "policies_to_train": [],
        }

        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        # Minimal single-agent MultiAgentEnv that satisfies RLlib's Trainer setup.
        class _DummyEnv(MultiAgentEnv):
            metadata = {"render.modes": []}

            def __init__(self_inner):
                super().__init__()
                self_inner.observation_space = obs_space
                self_inner.action_space = act_space

            def reset(self_inner):
                return {0: self_inner.observation_space.sample()}

            def step(self_inner, action_dict):
                obs = {0: self_inner.observation_space.sample()}
                rews = {0: 0.0}
                dones = {0: True, "__all__": True}
                infos = {0: {}}
                return obs, rews, dones, infos

        tune.registry.register_env("DummyEnv", lambda *_: _DummyEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        
        # load state from checkpoint
        try:
            agent.restore(CHECKPOINT_PATH)
        except KeyError as e:
            # Handle RLLib older version checkpoint issue where 'weights' key is missing
            if "weights" in str(e):
                # load state from checkpoint manually
                with open(CHECKPOINT_PATH, "rb") as f:
                    checkpoint_state = pickle.load(f)
                worker_state = pickle.loads(checkpoint_state["worker"])
                policy_states = worker_state["state"]
                
                # Check if we need to wrap weights in {'weights': ...}
                policy = agent.get_policy(POLICY_NAME)
                old_state = policy_states.get(POLICY_NAME)
                
                if old_state and "weights" not in old_state:
                    print("Adapting old checkpoint format for newer RLLib...")
                    # Wrap it properly, removing incompatible keys like _optimizer_variables
                    filtered_state = {k: v for k, v in old_state.items() if k != "_optimizer_variables"}
                    policy_states[POLICY_NAME] = {"weights": filtered_state}
                    
                    # Instead of calling agent.restore which expects a file, we set the state manually
                    agent.workers.local_worker().set_weights(
                        {POLICY_NAME: policy_states[POLICY_NAME]["weights"]}
                    )
            else:
                raise e
                
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions

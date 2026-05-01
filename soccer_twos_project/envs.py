import math
from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

from soccer_twos_project.mlagents_compat import patch_unity_environment_close


patch_unity_environment_close()


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.core.Wrapper):
    """
    Adds small potential-based reward shaping for single-agent training.

    The Soccer-Twos environment returns ball/player state in the info dict.
    This wrapper uses that signal only during training and leaves packaged
    submission agents independent from the shaping code.
    """

    def __init__(
        self,
        env,
        player_to_ball_weight=0.01,
        ball_to_goal_weight=0.02,
        goal_x=14.0,
        clip=0.05,
    ):
        super().__init__(env)
        self.player_to_ball_weight = float(player_to_ball_weight)
        self.ball_to_goal_weight = float(ball_to_goal_weight)
        self.goal_x = float(goal_x)
        self.clip = float(clip)
        self.previous_metrics = None

    def reset(self, **kwargs):
        self.previous_metrics = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        metrics = self._metrics_from_info(info)
        bonus = self._bonus(metrics)
        self.previous_metrics = metrics
        return obs, self._add_bonus(reward, bonus), done, info

    def _bonus(self, metrics):
        if not metrics or not self.previous_metrics:
            return 0.0
        bonus = 0.0
        if "player_ball_dist" in metrics and "player_ball_dist" in self.previous_metrics:
            bonus += self.player_to_ball_weight * (
                self.previous_metrics["player_ball_dist"] - metrics["player_ball_dist"]
            )
        if "ball_goal_dist" in metrics and "ball_goal_dist" in self.previous_metrics:
            bonus += self.ball_to_goal_weight * (
                self.previous_metrics["ball_goal_dist"] - metrics["ball_goal_dist"]
            )
        return max(-self.clip, min(self.clip, bonus))

    def _add_bonus(self, reward, bonus):
        if bonus == 0:
            return reward
        if isinstance(reward, dict):
            return {agent_id: value + bonus for agent_id, value in reward.items()}
        return reward + bonus

    def _metrics_from_info(self, info):
        entry = self._first_info_entry(info)
        if not entry:
            return {}
        player_pos = self._extract_position(entry.get("player_info"))
        ball_pos = self._extract_position(entry.get("ball_info"))
        if player_pos is None or ball_pos is None:
            return {}
        return {
            "player_ball_dist": self._distance(player_pos, ball_pos),
            "ball_goal_dist": self._distance(ball_pos, [self.goal_x, 0.0]),
        }

    def _first_info_entry(self, info):
        if not isinstance(info, dict):
            return {}
        if "player_info" in info or "ball_info" in info:
            return info
        if 0 in info and isinstance(info[0], dict):
            return info[0]
        for value in info.values():
            if isinstance(value, dict):
                return value
        return {}

    def _extract_position(self, section):
        if isinstance(section, dict) and "position" in section:
            position = section["position"]
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                return [float(position[0]), float(position[1])]
        return None

    def _distance(self, a, b):
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
        )


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    original_env_config = env_config
    env_config = dict(env_config or {})
    reward_shaping = env_config.pop("reward_shaping", None)
    if hasattr(original_env_config, "worker_index"):
        env_config["worker_id"] = (
            original_env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + original_env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if reward_shaping:
        if reward_shaping is True:
            reward_shaping = {}
        env = RewardShapingWrapper(env, **reward_shaping)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s

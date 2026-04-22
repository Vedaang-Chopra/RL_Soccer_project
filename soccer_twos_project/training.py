import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, Optional

import yaml

from soccer_twos_project.config import (
    PROFILES,
    artifact_dirs,
    checkpoint_path,
    ensure_artifact_dirs,
    fallback_profile,
    hardware_report,
    json_safe,
    profile_dict,
    select_profile,
    write_json,
)
from soccer_twos_project.envs import create_rllib_env, sample_player, sample_pos_vel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STAGES = {
    "ppo_baseline": {
        "algo": "PPO",
        "experiment": "soccer_ppo_baseline",
        "criterion": "policy performance baseline",
    },
    "ppo_shaped": {
        "algo": "PPO",
        "experiment": "soccer_ppo_shaped",
        "criterion": "reward modification",
    },
    "ppo_curriculum": {
        "algo": "PPO",
        "experiment": "soccer_ppo_curriculum",
        "criterion": "curriculum policy performance",
    },
    "ppo_selfplay": {
        "algo": "PPO",
        "experiment": "soccer_ppo_selfplay",
        "criterion": "self-play performance fallback",
    },
    "dqn_baseline": {
        "algo": "DQN",
        "experiment": "soccer_dqn_baseline",
        "criterion": "algorithm comparison baseline",
    },
}

DEFAULT_TRAIN_ALL = ("ppo_baseline", "ppo_shaped", "ppo_curriculum")
RETRYABLE_ERROR_TEXT = (
    "out of memory",
    "oom",
    "raylet",
    "worker died",
    "connection refused",
    "address already in use",
    "port",
    "unity",
)


def load_curriculum(path: Optional[str] = None):
    if path is None:
        path = PROJECT_ROOT / "configs" / "curriculum.yaml"
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)["tasks"]


CURRICULUM_TASKS = None
CURRENT_CURRICULUM_TASK = 0


def curriculum_config_fn(name, env):
    if name == "none":
        return
    if name == "random_players":
        env.set_policies(lambda *_: env.action_space.sample())
        return
    raise ValueError("Unknown curriculum config_fn: {}".format(name))


class CurriculumUpdateCallback:
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        global CURRENT_CURRICULUM_TASK, CURRICULUM_TASKS

        if CURRICULUM_TASKS is None:
            CURRICULUM_TASKS = load_curriculum()
        task = CURRICULUM_TASKS[CURRENT_CURRICULUM_TASK]
        for env in base_env.get_unwrapped():
            curriculum_config_fn(task["config_fn"], env)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(task["ranges"]["ball"]),
                players_states={
                    player: sample_player(task["ranges"]["players"][player])
                    for player in task["ranges"]["players"]
                },
            )

    def on_train_result(self, **info):
        global CURRENT_CURRICULUM_TASK, CURRICULUM_TASKS

        if CURRICULUM_TASKS is None:
            CURRICULUM_TASKS = load_curriculum()
        result = info["result"]
        if result.get("episode_reward_mean", float("-inf")) > 1.5:
            if CURRENT_CURRICULUM_TASK < len(CURRICULUM_TASKS) - 1:
                CURRENT_CURRICULUM_TASK += 1
                task = CURRICULUM_TASKS[CURRENT_CURRICULUM_TASK]
                print("---- Updating curriculum task: {} - {} ----".format(
                    CURRENT_CURRICULUM_TASK, task["name"]
                ))


def _import_ray_callback_base():
    from ray.rllib.agents.callbacks import DefaultCallbacks

    return DefaultCallbacks


def curriculum_callback_class():
    base = _import_ray_callback_base()

    class RayCurriculumUpdateCallback(CurriculumUpdateCallback, base):
        pass

    return RayCurriculumUpdateCallback


def selfplay_policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    return random.choices(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        weights=[0.50, 0.25, 0.125, 0.125],
        k=1,
    )[0]


class SelfPlayUpdateCallback:
    def on_train_result(self, **info):
        result = info["result"]
        if result.get("episode_reward_mean", float("-inf")) <= 0.5:
            return
        trainer = info["trainer"]
        print("---- Updating self-play opponent archive ----")
        current = trainer.get_weights(["default"])["default"]
        opponent_1 = trainer.get_weights(["opponent_1"])["opponent_1"]
        opponent_2 = trainer.get_weights(["opponent_2"])["opponent_2"]
        trainer.set_weights(
            {
                "opponent_3": opponent_2,
                "opponent_2": opponent_1,
                "opponent_1": current,
            }
        )


def selfplay_callback_class():
    base = _import_ray_callback_base()

    class RaySelfPlayUpdateCallback(SelfPlayUpdateCallback, base):
        pass

    return RaySelfPlayUpdateCallback


def selfplay_spaces(profile):
    from soccer_twos import EnvType

    env_config = {
        "render": False,
        "num_envs_per_worker": profile.num_envs_per_worker,
        "variation": EnvType.multiagent_player,
    }
    env = create_rllib_env(env_config)
    try:
        return env.observation_space, env.action_space, env_config
    finally:
        env.close()


def base_env_config(profile):
    from soccer_twos import EnvType

    return {
        "render": False,
        "num_envs_per_worker": profile.num_envs_per_worker,
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": True,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }


def build_training_config(stage: str, profile, timesteps: Optional[int] = None) -> Dict:
    if stage not in STAGES:
        raise ValueError("Unknown stage: {}".format(stage))

    env_config = base_env_config(profile)
    config = {
        "num_gpus": profile.num_gpus,
        "num_workers": profile.num_workers,
        "num_envs_per_worker": profile.num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "env": "Soccer",
        "env_config": env_config,
    }

    if stage in ("ppo_baseline", "ppo_shaped"):
        if stage == "ppo_shaped":
            env_config["reward_shaping"] = {
                "player_to_ball_weight": 0.01,
                "ball_to_goal_weight": 0.02,
                "clip": 0.05,
            }
        config.update(
            {
                "model": {
                    "vf_share_layers": True,
                    "fcnet_hiddens": [512],
                    "fcnet_activation": "relu",
                },
                "rollout_fragment_length": profile.rollout_fragment_length,
                "train_batch_size": profile.train_batch_size,
            }
        )
    elif stage == "ppo_curriculum":
        config.update(
            {
                "callbacks": curriculum_callback_class(),
                "model": {
                    "vf_share_layers": True,
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                "rollout_fragment_length": max(profile.rollout_fragment_length, 1000),
                "batch_mode": "complete_episodes",
            }
        )
    elif stage == "ppo_selfplay":
        obs_space, act_space, selfplay_env_config = selfplay_spaces(profile)
        config["env_config"] = selfplay_env_config
        config.update(
            {
                "callbacks": selfplay_callback_class(),
                "multiagent": {
                    "policies": {
                        "default": (None, obs_space, act_space, {}),
                        "opponent_1": (None, obs_space, act_space, {}),
                        "opponent_2": (None, obs_space, act_space, {}),
                        "opponent_3": (None, obs_space, act_space, {}),
                    },
                    "policy_mapping_fn": selfplay_policy_mapping_fn,
                    "policies_to_train": ["default"],
                },
                "model": {
                    "vf_share_layers": True,
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
                "rollout_fragment_length": max(profile.rollout_fragment_length, 2000),
                "batch_mode": "complete_episodes",
            }
        )
    elif stage == "dqn_baseline":
        config.update(
            {
                "model": {
                    "fcnet_hiddens": [512, 256],
                    "fcnet_activation": "relu",
                },
            }
        )

    return config


def build_stop(profile, timesteps: Optional[int] = None, time_total_s: Optional[int] = None):
    stop = {"timesteps_total": timesteps or profile.timesteps_total}
    if time_total_s is not None:
        stop["time_total_s"] = time_total_s
    elif profile.time_total_s:
        stop["time_total_s"] = profile.time_total_s
    return stop


def smoke_test_env(steps: int = 10, base_port: Optional[int] = None):
    import soccer_twos
    from soccer_twos import EnvType

    env = soccer_twos.make(
        render=False,
        variation=EnvType.team_vs_policy,
        flatten_branched=True,
        single_player=True,
        base_port=base_port,
    )
    obs = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done_all(done):
            obs = env.reset()
    env.close()
    print("Headless smoke test passed for {} random steps.".format(steps))


def done_all(done) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(done["__all__"])
        return bool(max(done.values())) if done else False
    return bool(done)


def should_retry(exc: BaseException) -> bool:
    text = "{}\n{}".format(exc, traceback.format_exc()).lower()
    if "json serializable" in text:
        return False
    return any(fragment in text for fragment in RETRYABLE_ERROR_TEXT)


def run_tune(stage, profile, args, retrying=False):
    import ray
    from ray import tune

    spec = STAGES[stage]
    dirs = ensure_artifact_dirs(args.artifact_root)
    local_dir = str(dirs["checkpoints"])
    config = build_training_config(stage, profile, timesteps=args.timesteps)
    stop = build_stop(profile, timesteps=args.timesteps, time_total_s=args.time_total_s)

    if retrying:
        print("Retrying with conservative single-worker settings.")
    print("Hardware:", json.dumps(hardware_report(), indent=2))
    print("Profile:", json.dumps(profile_dict(profile), indent=2))
    print("Stage:", stage)
    print("Stop:", stop)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)
    analysis = tune.run(
        spec["algo"],
        name=spec["experiment"],
        config=config,
        stop=stop,
        checkpoint_freq=args.checkpoint_freq or profile.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=local_dir,
        restore=checkpoint_path(args.restore) if args.restore else None,
        verbose=getattr(args, "verbose", 1),
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = None
    if best_trial is not None:
        raw_best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        best_checkpoint = checkpoint_path(raw_best_checkpoint)
    metadata_path = dirs["checkpoints"] / spec["experiment"] / "run_metadata.json"
    write_json(
        metadata_path,
        {
            "stage": stage,
            "algo": spec["algo"],
            "criterion": spec["criterion"],
            "profile": profile_dict(profile),
            "hardware": hardware_report(),
            "stop": stop,
            "local_dir": local_dir,
            "best_trial": str(best_trial),
            "best_checkpoint": best_checkpoint,
            "config": json_safe(config),
        },
    )
    print("Best trial:", best_trial)
    print("Best checkpoint:", best_checkpoint)
    print("Wrote run metadata:", metadata_path)
    ray.shutdown()
    return best_checkpoint


def train_stage(stage, profile, args):
    try:
        return run_tune(stage, profile, args)
    except Exception as exc:
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass
        if should_retry(exc):
            print("Training failed with retryable error:", exc)
            return run_tune(stage, fallback_profile(), args, retrying=True)
        raise


def print_profiles():
    print("Hardware:")
    print(json.dumps(hardware_report(), indent=2))
    print("Profiles:")
    print(json.dumps({name: profile_dict(p) for name, p in select_profiles().items()}, indent=2))


def select_profiles():
    from soccer_twos_project.config import PROFILES

    return PROFILES


def parse_args():
    parser = argparse.ArgumentParser(description="Soccer-Twos training runner.")
    parser.add_argument(
        "command",
        choices=["profiles", "smoke", "train", "train-all"],
        help="Action to run.",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        default="ppo_baseline",
        help="Training stage for the train command.",
    )
    parser.add_argument(
        "--profile",
        default="auto",
        help="Hardware profile: auto or one of {}.".format(", ".join(sorted(PROFILES))),
    )
    parser.add_argument("--smoke", action="store_true", help="Use tiny CPU-safe training limits.")
    parser.add_argument("--timesteps", type=int, help="Override timesteps_total.")
    parser.add_argument("--time-total-s", type=int, help="Override time_total_s.")
    parser.add_argument("--checkpoint-freq", type=int, help="Override checkpoint frequency.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Ray Tune verbosity. Use 1 or 2 to see live training progress.",
    )
    parser.add_argument("--restore", help="Ray/RLlib checkpoint path to restore from.")
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"),
        help="Artifact root. Defaults to artifacts/cs8803_soccer_twos.",
    )
    parser.add_argument("--base-port", type=int, help="Unity base port for smoke test.")
    parser.add_argument(
        "--include-dqn",
        action="store_true",
        help="Also train dqn_baseline when command is train-all.",
    )
    parser.add_argument(
        "--include-selfplay",
        action="store_true",
        help="Also train ppo_selfplay when command is train-all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "profiles":
        print_profiles()
        return
    if args.command == "smoke":
        smoke_test_env(base_port=args.base_port)
        return

    profile = select_profile(args.profile, smoke=args.smoke)
    if args.command == "train":
        train_stage(args.stage, profile, args)
        return

    stages = list(DEFAULT_TRAIN_ALL)
    if args.include_dqn:
        stages.append("dqn_baseline")
    if getattr(args, "include_selfplay", False):
        stages.append("ppo_selfplay")
    for stage in stages:
        train_stage(stage, profile, args)


if __name__ == "__main__":
    main()

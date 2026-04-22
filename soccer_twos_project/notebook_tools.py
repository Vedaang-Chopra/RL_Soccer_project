import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pandas as pd


def project_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        for candidate in (base, base / "soccer-twos-starter"):
            if (candidate / "soccer_twos_project" / "training.py").exists():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not find soccer_twos_project/training.py. Open this notebook "
        "from the soccer-twos-starter project root, the notebooks/ directory, "
        "or the parent course project directory."
    )


def setup_project(artifact_subdir: str = "artifacts/cs8803_soccer_twos"):
    root = project_root_from_cwd()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    artifact_root = root / artifact_subdir
    os.environ["SOCCER_TWOS_DRIVE_ROOT"] = str(artifact_root)

    from soccer_twos_project.config import ensure_artifact_dirs

    dirs = ensure_artifact_dirs(str(artifact_root))
    submissions_dir = dirs["submissions"]
    if str(submissions_dir) not in sys.path:
        sys.path.insert(0, str(submissions_dir))

    ctx = SimpleNamespace(
        root=root,
        artifact_root=artifact_root,
        dirs=dirs,
        submissions_dir=submissions_dir,
    )
    print("Project root:", ctx.root)
    print("Artifact root:", ctx.artifact_root)
    print("Python:", sys.executable)
    return ctx


def print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


DEFAULT_UNITY_BASE_PORT = int(os.environ.get("SOCCER_TWOS_BASE_PORT", "50039"))
_next_unity_base_port = DEFAULT_UNITY_BASE_PORT


def port_is_available(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def find_free_base_port(start_port: Optional[int] = None, max_tries: int = 200) -> int:
    global _next_unity_base_port
    start = int(start_port or _next_unity_base_port)
    for port in range(start, start + max_tries):
        if port_is_available(port):
            _next_unity_base_port = port + 1
            return port
    raise OSError("Could not find a free Unity base port starting at {}".format(start))


def make_soccer_env(**env_config):
    import soccer_twos

    auto_base_port = env_config.get("base_port") is None
    attempts = 5 if auto_base_port else 1
    for attempt in range(attempts):
        if auto_base_port:
            env_config["base_port"] = find_free_base_port()
            print("Using Unity base_port:", env_config["base_port"])
        try:
            return soccer_twos.make(**env_config)
        except Exception as exc:
            if (
                not auto_base_port
                or exc.__class__.__name__ != "UnityWorkerInUseException"
                or attempt == attempts - 1
            ):
                raise
            env_config["base_port"] = None


def show_hardware():
    import ray
    import soccer_twos
    import torch
    from soccer_twos_project.config import hardware_report

    print("soccer_twos:", soccer_twos.__file__)
    print("ray:", ray.__version__)
    print("torch:", torch.__version__)
    print("python:", sys.executable)
    print_json(hardware_report())


def make_train_args(
    ctx,
    timesteps: Optional[int] = None,
    time_total_s: Optional[int] = None,
    checkpoint_freq: Optional[int] = None,
    restore: Optional[str] = None,
    verbose: int = 1,
):
    return SimpleNamespace(
        artifact_root=str(ctx.artifact_root),
        timesteps=timesteps,
        time_total_s=time_total_s,
        checkpoint_freq=checkpoint_freq,
        restore=restore,
        verbose=verbose,
    )


def progress_files(ctx, stage=None):
    from soccer_twos_project.training import STAGES

    root = ctx.dirs["checkpoints"]
    if stage is not None:
        root = root / STAGES[stage]["experiment"]
    return sorted(root.rglob("progress.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def show_progress(ctx, stage=None, rows=10):
    files = progress_files(ctx, stage)
    if not files:
        print("No progress.csv files found yet.")
        return None
    latest = files[0]
    df = pd.read_csv(latest)
    preferred_cols = [
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "time_total_s",
    ]
    cols = [col for col in preferred_cols if col in df.columns]
    print("Latest progress file:", latest)
    return df[cols].tail(rows) if cols else df.tail(rows)


def plot_progress(ctx, stage=None, rows=None):
    import matplotlib.pyplot as plt

    files = progress_files(ctx, stage)
    if not files:
        print("No progress.csv files found yet.")
        return None
    latest = files[0]
    df = pd.read_csv(latest)
    if rows:
        df = df.tail(rows)
    if "timesteps_total" not in df or "episode_reward_mean" not in df:
        print("progress.csv does not contain expected reward columns:", latest)
        return df
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["timesteps_total"], df["episode_reward_mean"], linewidth=2)
    ax.set_title(stage or latest.parent.name)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward Mean")
    ax.grid(True, alpha=0.3)
    plt.show()
    return df


def run_training(
    ctx,
    stage,
    profile_name="auto",
    timesteps=None,
    smoke=False,
    restore=None,
    verbose=1,
    time_total_s=None,
    checkpoint_freq=None,
):
    from soccer_twos_project.config import select_profile
    from soccer_twos_project.training import train_stage

    profile = select_profile(profile_name, smoke=smoke)
    print("Training stage:", stage)
    print("Profile:", profile)
    args = make_train_args(
        ctx,
        timesteps=timesteps,
        time_total_s=time_total_s,
        checkpoint_freq=checkpoint_freq,
        restore=restore,
        verbose=verbose,
    )
    checkpoint = train_stage(stage, profile, args)
    print("Returned checkpoint:", checkpoint)
    return checkpoint


def dependency_report() -> Dict[str, str]:
    import gym
    import numpy
    import ray
    import soccer_twos
    import torch

    return {
        "python": sys.executable,
        "soccer_twos": soccer_twos.__file__,
        "ray": ray.__version__,
        "torch": torch.__version__,
        "gym": gym.__version__,
        "numpy": numpy.__version__,
    }


def inspect_environment_spaces(base_port=None):
    from soccer_twos import EnvType

    training_env = make_soccer_env(
        render=False,
        variation=EnvType.team_vs_policy,
        flatten_branched=True,
        single_player=True,
        base_port=base_port,
    )
    try:
        training_obs = training_env.reset()
        training_report = {
            "variation": "team_vs_policy + single_player + flatten_branched",
            "observation_space": str(training_env.observation_space),
            "action_space": str(training_env.action_space),
            "reset_type": type(training_obs).__name__,
            "reset_shape": getattr(training_obs, "shape", None),
        }
    finally:
        training_env.close()

    live_env = make_soccer_env(render=False, base_port=base_port)
    try:
        live_obs = live_env.reset()
        first_key = sorted(live_obs)[0]
        live_report = {
            "variation": "default multi-agent match",
            "observation_space": str(live_env.observation_space),
            "action_space": str(live_env.action_space),
            "reset_type": type(live_obs).__name__,
            "agent_ids": sorted(live_obs),
            "single_agent_shape": getattr(live_obs[first_key], "shape", None),
        }
    finally:
        live_env.close()

    report = {"training_env": training_report, "live_match_env": live_report}
    print_json(report)
    return report


def run_environment_gate(steps=10, base_port=None):
    from soccer_twos import EnvType

    print_json(dependency_report())
    inspect_environment_spaces(base_port=base_port)
    env = make_soccer_env(
        render=False,
        variation=EnvType.team_vs_policy,
        flatten_branched=True,
        single_player=True,
        base_port=base_port,
    )
    try:
        obs = env.reset()
        print("Gate reset observation shape:", getattr(obs, "shape", None))
        total_reward = 0.0
        for step in range(steps):
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward += float(reward)
            if step == 0:
                print("First-step info:", compact_info(info))
            if done:
                obs = env.reset()
        print("Environment gate passed. steps={} total_reward={:.3f}".format(steps, total_reward))
    finally:
        env.close()


def compact_info(info):
    if not isinstance(info, dict):
        return info
    compact = {}
    for key, value in info.items():
        if isinstance(value, dict):
            compact[key] = {
                sub_key: (
                    sub_value.tolist()
                    if hasattr(sub_value, "tolist")
                    else float(sub_value)
                    if isinstance(sub_value, (int, float))
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        else:
            compact[key] = value
    return compact


def run_random_debug_episode(max_steps=10, show_info=True, base_port=None):
    from soccer_twos import EnvType

    env = make_soccer_env(
        render=False,
        variation=EnvType.team_vs_policy,
        flatten_branched=True,
        single_player=True,
        base_port=base_port,
    )
    try:
        obs = env.reset()
        print("Observation type:", type(obs))
        print("Observation shape:", getattr(obs, "shape", None))
        print("Action space:", env.action_space)
        total_reward = 0.0
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            info_repr = compact_info(info) if show_info else list(info.keys()) if isinstance(info, dict) else type(info)
            print(
                "step={:02d} action={} reward={} done={} info={}".format(
                    step + 1,
                    action,
                    reward,
                    done,
                    info_repr,
                )
            )
            if done:
                break
        print("Debug episode reward:", total_reward)
    finally:
        env.close()


def load_run_metadata(ctx, stage):
    from soccer_twos_project.training import STAGES

    path = ctx.dirs["checkpoints"] / STAGES[stage]["experiment"] / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def best_checkpoint(ctx, stage):
    metadata = load_run_metadata(ctx, stage)
    checkpoint = metadata.get("best_checkpoint")
    if not checkpoint:
        raise ValueError("No best checkpoint recorded for {}".format(stage))
    return checkpoint


def run_metadata_status(ctx, stage):
    try:
        metadata = load_run_metadata(ctx, stage)
    except Exception as exc:
        print("Metadata invalid for {}: {}".format(stage, exc))
        return None
    keys = ["stage", "algo", "criterion", "best_checkpoint", "stop"]
    print_json({key: metadata.get(key) for key in keys})
    return metadata


def evaluation_summaries(ctx):
    rows = []
    for path in sorted(ctx.dirs["evals"].glob("*_summary.json")):
        with path.open() as f:
            payload = json.load(f)
        payload["file"] = path.name
        rows.append(payload)
    if not rows:
        print("No evaluation summaries found.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    display_cols = [
        "file",
        "agent1",
        "agent2",
        "episodes",
        "agent1_win_rate",
        "agent1_reward_mean",
        "agent2_reward_mean",
        "draws",
    ]
    return df[[col for col in display_cols if col in df.columns]]


def validate_agent_package(ctx, module_name, base_port=None):
    from soccer_twos_project.evaluation import get_agent_class

    if str(ctx.submissions_dir) not in sys.path:
        sys.path.insert(0, str(ctx.submissions_dir))
    clear_imported_package(module_name)
    module = importlib.import_module(module_name)
    env = make_soccer_env(render=False, base_port=base_port)
    try:
        agent = get_agent_class(module)(env)
        obs = env.reset()
        team_obs = {0: obs[0], 1: obs[1]}
        actions = agent.act(team_obs)
        if sorted(actions) != sorted(team_obs):
            raise AssertionError(
                "Action keys {} do not match observation keys {}".format(
                    sorted(actions), sorted(team_obs)
                )
            )
        print("Validated package:", module_name)
        print("Actions:", actions)
        return actions
    finally:
        env.close()


def validate_zip_package(zip_path, base_port=None):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with tempfile.TemporaryDirectory(prefix="soccer_twos_zipcheck_") as tmp:
        tmp_path = Path(tmp)
        shutil.unpack_archive(str(zip_path), str(tmp_path))
        packages = [path for path in tmp_path.iterdir() if path.is_dir()]
        if len(packages) != 1:
            raise AssertionError("Expected one top-level package folder, found {}".format(packages))
        module_name = packages[0].name
        sys.path.insert(0, str(tmp_path))
        try:
            clear_imported_package(module_name)
            ctx = SimpleNamespace(submissions_dir=tmp_path)
            return validate_agent_package(ctx, module_name, base_port=base_port)
        finally:
            clear_imported_package(module_name)
            try:
                sys.path.remove(str(tmp_path))
            except ValueError:
                pass


def clear_imported_package(module_name):
    for name in list(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            sys.modules.pop(name, None)


def make_final_submission(ctx, source_agent, team_name="TEAMNAME_AGENT"):
    source_dir = ctx.dirs["submissions"] / source_agent
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    final_name = team_name if team_name.endswith("_AGENT") else "{}_AGENT".format(team_name)
    final_dir = ctx.dirs["submissions"] / final_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(source_dir, final_dir)

    from soccer_twos_project.exporting import zip_agent_package

    zip_path = zip_agent_package(final_dir)
    print("Final submission folder:", final_dir)
    print("Final submission zip:", zip_path)
    return zip_path


def artifact_checklist(ctx):
    checklist = {}
    for name, path in ctx.dirs.items():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        checklist[name] = {
            "path": str(path),
            "file_count": len(files),
            "examples": [str(item.relative_to(path)) for item in files[:10]],
        }
    print_json(checklist)
    return checklist


def launch_tensorboard(ctx, port=6006):
    logdir = ctx.dirs["checkpoints"]
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--port",
        str(port),
        "--reload_interval",
        "10",
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2)
    print("TensorBoard command:", " ".join(command))
    print("Open: http://localhost:{}".format(port))
    print("PID:", proc.pid)
    return proc


def stop_process(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        print("Stopped process", proc.pid)

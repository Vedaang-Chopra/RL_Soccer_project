import importlib
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import pandas as pd

from soccer_twos_project.mlagents_compat import patch_unity_environment_close


PROJECT_MARKER = Path("soccer_twos_project") / "training.py"


def is_colab_runtime() -> bool:
    if "google.colab" in sys.modules:
        return True
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


def runtime_name() -> str:
    if is_colab_runtime():
        return "colab"
    if os.environ.get("SLURM_JOB_ID"):
        return "pace/slurm"
    if sys.platform == "darwin":
        return "mac"
    return sys.platform


def resolve_unity_render_request(
    requested: bool,
    ctx=None,
    label: str = "Unity rendering",
) -> bool:
    if not requested:
        return False

    runtime = getattr(ctx, "runtime", None) or runtime_name()
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    force_render = os.environ.get("SOCCER_TWOS_FORCE_RENDER") == "1"

    if runtime == "colab":
        print("{} disabled: Colab sessions cannot open the Unity window.".format(label))
        return False

    if not display:
        if force_render:
            print(
                "{} disabled: SOCCER_TWOS_FORCE_RENDER=1 was ignored because no live display was detected.".format(
                    label
                )
            )
        else:
            print("{} disabled: no live display was detected.".format(label))
        return False

    if runtime == "pace/slurm" and not force_render:
        print(
            "{} disabled on PACE/SLURM by default. Set SOCCER_TWOS_FORCE_RENDER=1 only in a GUI-enabled session.".format(
                label
            )
        )
        return False

    return True


def mount_colab_drive_if_needed() -> bool:
    if not is_colab_runtime():
        return False
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    return Path("/content/drive/MyDrive").exists()


def _candidate_project_roots():
    seen = set()

    def add(candidate):
        path = Path(candidate).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            yield path

    for env_name in ("SOCCER_TWOS_PROJECT_ROOT", "PROJECT_ROOT"):
        value = os.environ.get(env_name)
        if value:
            yield from add(value)

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        yield from add(base)
        yield from add(base / "soccer-twos-starter")
        yield from add(base / "project" / "soccer-twos-starter")

    if sys.platform == "darwin":
        yield from add(
            Path.home()
            / "all_data"
            / "Georgia Tech"
            / "Course Content"
            / "CS 8803- DRL"
            / "project"
            / "soccer-twos-starter"
        )

    if is_colab_runtime():
        mount_colab_drive_if_needed()
        drive_roots = [
            Path("/content/drive/MyDrive"),
            Path("/content/drive/Shareddrives"),
            Path("/content"),
        ]
        relative_roots = [
            Path("CS 8803- DRL") / "project" / "soccer-twos-starter",
            Path("project") / "soccer-twos-starter",
            Path("soccer-twos-starter"),
            Path("Colab Notebooks") / "soccer-twos-starter",
        ]
        for drive_root in drive_roots:
            for relative_root in relative_roots:
                yield from add(drive_root / relative_root)


def project_root_from_cwd() -> Path:
    for candidate in _candidate_project_roots():
        if (candidate / PROJECT_MARKER).exists():
            return candidate.resolve()

    if is_colab_runtime():
        for drive_root in (Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")):
            if not drive_root.exists():
                continue
            try:
                matches = drive_root.glob("**/soccer-twos-starter")
                for candidate in matches:
                    if (candidate / PROJECT_MARKER).exists():
                        return candidate.resolve()
            except OSError:
                pass

    raise FileNotFoundError(
        "Could not find soccer_twos_project/training.py. Open this notebook "
        "from the soccer-twos-starter project root, the notebooks/ directory, "
        "or the parent course project directory. On Colab, mount Google Drive "
        "or set SOCCER_TWOS_PROJECT_ROOT to the Drive path."
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
        runtime=runtime_name(),
    )
    print("Runtime:", ctx.runtime)
    print("Project root:", ctx.root)
    print("Artifact root:", ctx.artifact_root)
    print("Python:", sys.executable)
    return ctx


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Enum):
        return "{}.{}".format(obj.__class__.__name__, obj.name)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, SimpleNamespace):
        return vars(obj)
    if callable(obj):
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        module = getattr(obj, "__module__", "")
        return "{}.{}".format(module, name) if module else name
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return repr(obj)


def print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


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
    patch_unity_environment_close()
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


def latest_progress_path(ctx, stage=None):
    files = progress_files(ctx, stage)
    if not files:
        label = " for {}".format(stage) if stage else ""
        print("No progress.csv files found{}.".format(label))
        return None
    return files[0]


def load_progress_table(ctx, stage=None, rows=None, latest: bool = True):
    files = progress_files(ctx, stage)
    if not files:
        label = " for {}".format(stage) if stage else ""
        print("No progress.csv files found{}.".format(label))
        return pd.DataFrame()
    selected = files[0] if latest else files[-1]
    df = pd.read_csv(selected)
    df.attrs["progress_path"] = str(selected)
    if rows:
        return df.tail(rows)
    return df


def progress_status(ctx, stage=None, rows=10):
    df = load_progress_table(ctx, stage=stage, rows=rows)
    if df.empty:
        return df
    preferred_cols = [
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "episode_len_mean",
        "time_total_s",
    ]
    cols = [col for col in preferred_cols if col in df.columns]
    print("Latest progress file:", df.attrs.get("progress_path"))
    return df[cols] if cols else df


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
    import soccer_twos_project.config as config_module
    import soccer_twos_project.training as training_module

    importlib.reload(config_module)
    importlib.reload(training_module)

    profile = config_module.select_profile(profile_name, smoke=smoke)
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
    checkpoint = training_module.train_stage(stage, profile, args)
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


def portable_artifact_path(ctx, value):
    if not value:
        return value
    path = Path(value)
    if path.exists() or not path.is_absolute():
        return str(path)

    parts = path.parts
    marker = ("artifacts", "cs8803_soccer_twos")
    for index in range(len(parts) - len(marker) + 1):
        if parts[index : index + len(marker)] == marker:
            tail = Path(*parts[index + len(marker) :])
            candidate = ctx.artifact_root / tail
            if candidate.exists():
                print("Resolved copied artifact path:", candidate)
                return str(candidate)
            return str(candidate)
    return str(path)


def best_checkpoint(ctx, stage):
    metadata = load_run_metadata(ctx, stage)
    checkpoint = metadata.get("best_checkpoint")
    if not checkpoint:
        raise ValueError("No best checkpoint recorded for {}".format(stage))
    return portable_artifact_path(ctx, checkpoint)


def run_metadata_status(ctx, stage):
    try:
        metadata = load_run_metadata(ctx, stage)
    except Exception as exc:
        print("Metadata invalid for {}: {}".format(stage, exc))
        return None
    keys = ["stage", "algo", "criterion", "best_checkpoint", "stop"]
    print_json({key: metadata.get(key) for key in keys})
    return metadata


def checkpoint_summary(ctx, stage):
    try:
        metadata = load_run_metadata(ctx, stage)
    except Exception as exc:
        return {
            "stage": stage,
            "exists": False,
            "error": str(exc),
        }
    checkpoint = portable_artifact_path(ctx, metadata.get("best_checkpoint"))
    path = Path(checkpoint) if checkpoint else None
    return {
        "stage": stage,
        "exists": bool(path and path.exists()),
        "checkpoint": str(path) if path else None,
        "experiment": metadata.get("config", {}).get("env"),
        "stop": metadata.get("stop"),
        "criterion": metadata.get("criterion"),
    }


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


REQUIRED_AGENT_PACKAGE_FILES = (
    "__init__.py",
)


def _has_agent_implementation_file(file_paths: Iterable[str]) -> bool:
    for rel_path in file_paths:
        path = Path(rel_path)
        if len(path.parts) != 1:
            continue
        name = path.name
        if name == "__init__.py":
            continue
        if name.startswith("agent") and name.endswith(".py"):
            return True
    return False


def package_summary(package_dir):
    package_dir = Path(package_dir)
    required = set(REQUIRED_AGENT_PACKAGE_FILES)
    files = sorted(path.relative_to(package_dir).as_posix() for path in package_dir.rglob("*") if path.is_file())
    has_agent_file = _has_agent_implementation_file(files)
    missing = sorted(required.difference(files))
    if not has_agent_file:
        missing.append("agent*.py")
    return {
        "package_dir": str(package_dir),
        "exists": package_dir.exists(),
        "file_count": len(files),
        "required_files_present": sorted(required.intersection(files)),
        "missing_required_files": missing,
        "has_agent_implementation": has_agent_file,
        "files": files,
    }


def validate_package_folder(package_dir, required_files: Optional[Iterable[str]] = None):
    package_dir = Path(package_dir)
    if not package_dir.exists():
        raise FileNotFoundError(package_dir)
    required = tuple(required_files or REQUIRED_AGENT_PACKAGE_FILES)
    files = sorted(path.relative_to(package_dir).as_posix() for path in package_dir.rglob("*") if path.is_file())
    missing = [name for name in required if not (package_dir / name).is_file()]
    if not _has_agent_implementation_file(files):
        missing.append("agent*.py")
    if missing:
        raise AssertionError("Package {} is missing required files: {}".format(package_dir, missing))
    return package_summary(package_dir)


def validate_agent_package(ctx, module_name, base_port=None, require_package_files: bool = False):
    from soccer_twos_project.evaluation import get_agent_class

    if str(ctx.submissions_dir) not in sys.path:
        sys.path.insert(0, str(ctx.submissions_dir))
    if require_package_files:
        validate_package_folder(ctx.submissions_dir / module_name)
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


def validate_zip_manifest(zip_path, expected_top_level=None, required_files: Optional[Iterable[str]] = None):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    required = tuple(required_files or REQUIRED_AGENT_PACKAGE_FILES)
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
    top_levels = sorted({Path(name).parts[0] for name in names if Path(name).parts})
    if len(top_levels) != 1:
        raise AssertionError("Expected one top-level package folder, found {}".format(top_levels))
    top_level = top_levels[0]
    if expected_top_level and top_level != expected_top_level:
        raise AssertionError(
            "Expected top-level folder '{}', found '{}'.".format(expected_top_level, top_level)
        )
    missing = [
        name for name in required
        if "{}/{}".format(top_level, name) not in names
    ]
    rel_paths = [
        str(Path(name).relative_to(top_level))
        for name in names
        if Path(name).parts and Path(name).parts[0] == top_level
    ]
    if not _has_agent_implementation_file(rel_paths):
        missing.append("agent*.py")
    if missing:
        raise AssertionError("Zip {} is missing required files: {}".format(zip_path, missing))
    manifest = {
        "zip_path": str(zip_path),
        "top_level": top_level,
        "file_count": len(names),
        "required_files": list(required),
        "missing_required_files": missing,
        "files": names,
    }
    print_json({key: manifest[key] for key in ("zip_path", "top_level", "file_count", "missing_required_files")})
    return manifest


def validate_zip_package(zip_path, base_port=None, expected_top_level=None, required_files: Optional[Iterable[str]] = None):
    zip_path = Path(zip_path)
    manifest = validate_zip_manifest(zip_path, expected_top_level=expected_top_level, required_files=required_files)
    with tempfile.TemporaryDirectory(prefix="soccer_twos_zipcheck_") as tmp:
        tmp_path = Path(tmp)
        shutil.unpack_archive(str(zip_path), str(tmp_path))
        packages = [path for path in tmp_path.iterdir() if path.is_dir()]
        if len(packages) != 1:
            raise AssertionError("Expected one top-level package folder, found {}".format(packages))
        module_name = packages[0].name
        if expected_top_level and module_name != expected_top_level:
            raise AssertionError(
                "Expected extracted folder '{}', found '{}'.".format(expected_top_level, module_name)
            )
        sys.path.insert(0, str(tmp_path))
        try:
            clear_imported_package(module_name)
            ctx = SimpleNamespace(submissions_dir=tmp_path)
            actions = validate_agent_package(
                ctx,
                module_name,
                base_port=base_port,
                require_package_files=True,
            )
            return {"manifest": manifest, "actions": actions}
        finally:
            clear_imported_package(module_name)
            try:
                sys.path.remove(str(tmp_path))
            except ValueError:
                pass


def validate_exported_package(ctx, module_name, rollout_steps: int = 0, base_port=None):
    validate_package_folder(ctx.submissions_dir / module_name)
    actions = validate_agent_package(ctx, module_name, base_port=base_port)
    result = {
        "module_name": module_name,
        "actions": actions,
        "package": package_summary(ctx.submissions_dir / module_name),
    }
    if rollout_steps:
        rollout = collect_standalone_agent_rollout(
            ctx,
            module_name,
            steps=rollout_steps,
            render=False,
            base_port=base_port,
            label="{} validation rollout".format(module_name),
        )
        result["rollout_summary"] = rollout_summary(rollout, label=module_name)
    print_json(result)
    return result


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


def submission_readiness_summary(ctx, team_name="TEAMNAME_AGENT", zip_path=None):
    final_name = team_name if team_name.endswith("_AGENT") else "{}_AGENT".format(team_name)
    package_dir = ctx.dirs["submissions"] / final_name
    zip_path = Path(zip_path or package_dir.with_suffix(".zip"))
    summary = {
        "team_agent_name": final_name,
        "package_dir": str(package_dir),
        "package_exists": package_dir.exists(),
        "zip_path": str(zip_path),
        "zip_exists": zip_path.exists(),
        "folder_missing_required": [],
        "zip_manifest_ok": False,
        "ready": False,
    }
    if package_dir.exists():
        folder_summary = package_summary(package_dir)
        summary["folder_missing_required"] = folder_summary["missing_required_files"]
    try:
        manifest = validate_zip_manifest(zip_path, expected_top_level=final_name)
        summary["zip_manifest_ok"] = not manifest["missing_required_files"]
    except Exception as exc:
        summary["zip_manifest_error"] = str(exc)
    summary["ready"] = (
        summary["package_exists"]
        and summary["zip_exists"]
        and not summary["folder_missing_required"]
        and summary["zip_manifest_ok"]
    )
    print_json(summary)
    return summary


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


def learning_artifact_dir(ctx, name: Optional[str] = None) -> Path:
    base = ctx.artifact_root / "learning"
    path = base / name if name else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def _learning_done_all(done) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(done["__all__"])
        return bool(max(done.values())) if done else False
    return bool(done)


def _learning_xy(section):
    if isinstance(section, dict):
        if "position" in section:
            position = section["position"]
            if hasattr(position, "tolist"):
                position = position.tolist()
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                return float(position[0]), float(position[1])
        if "x" in section and "y" in section:
            return float(section["x"]), float(section["y"])
    return None


def _learning_first_info_entry(info):
    if not isinstance(info, dict):
        return {}
    if "player_info" in info or "ball_info" in info:
        return info
    for key in (0, "0"):
        value = info.get(key)
        if isinstance(value, dict):
            return value
    for value in info.values():
        if isinstance(value, dict):
            return value
    return {}


def rollout_metrics_from_info(info, goal_x: float = 14.0) -> Dict[str, float]:
    import math

    entry = _learning_first_info_entry(info)
    player_xy = _learning_xy(entry.get("player_info"))
    ball_xy = _learning_xy(entry.get("ball_info"))
    metrics = {}
    if player_xy is not None:
        metrics["player_x"], metrics["player_y"] = player_xy
    if ball_xy is not None:
        metrics["ball_x"], metrics["ball_y"] = ball_xy
    if player_xy is not None and ball_xy is not None:
        metrics["player_ball_dist"] = math.dist(player_xy, ball_xy)
        metrics["ball_goal_dist"] = math.dist(ball_xy, (float(goal_x), 0.0))
    return metrics


def _action_repr(action) -> str:
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, (list, tuple)):
        return str([int(v) for v in action])
    try:
        return str(int(action))
    except Exception:
        return str(action)


SOCCER_ACTION_BRANCHES = [
    (
        "forward_axis",
        {
            0: "no forward/backward force",
            1: "forward force (W); also gives kick power on ball contact",
            2: "backward force (S)",
        },
    ),
    (
        "right_axis",
        {
            0: "no sideways force",
            1: "strafe right (E)",
            2: "strafe left (Q)",
        },
    ),
    (
        "rotate_axis",
        {
            0: "no rotation",
            1: "rotate A branch",
            2: "rotate D branch",
        },
    ),
]


NAMED_SOCCER_ACTIONS = {
    "noop": [0, 0, 0],
    "no_op": [0, 0, 0],
    "stay": [0, 0, 0],
    "zero": [0, 0, 0],
    "center": [0, 0, 0],
    "forward": [1, 0, 0],
    "backward": [2, 0, 0],
    "right": [0, 1, 0],
    "left": [0, 2, 0],
    "strafe_right": [0, 1, 0],
    "strafe_left": [0, 2, 0],
    "rotate_a": [0, 0, 1],
    "rotate_d": [0, 0, 2],
    "rotate_left": [0, 0, 1],
    "rotate_right": [0, 0, 2],
    "forward_right": [1, 1, 0],
    "forward_left": [1, 2, 0],
}


def soccer_action_index(branches) -> int:
    branch_values = [int(v) for v in branches]
    if len(branch_values) != 3:
        raise ValueError("Expected 3 SoccerTwos action branches, got {}".format(branch_values))
    return branch_values[0] * 9 + branch_values[1] * 3 + branch_values[2]


def soccer_action_branches(action):
    import numpy as np

    if isinstance(action, str):
        if action not in NAMED_SOCCER_ACTIONS:
            raise ValueError(
                "Unknown named SoccerTwos action '{}'. Known names: {}".format(
                    action,
                    ", ".join(sorted(NAMED_SOCCER_ACTIONS)),
                )
            )
        return list(NAMED_SOCCER_ACTIONS[action])
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, np.integer):
        action = int(action)
    if isinstance(action, int):
        if not 0 <= action <= 26:
            raise ValueError("Flat SoccerTwos action must be in [0, 26], got {}".format(action))
        return [action // 9, (action % 9) // 3, action % 3]
    if isinstance(action, (list, tuple)) and len(action) == 3:
        return [int(v) for v in action]
    raise ValueError("Cannot decode SoccerTwos action {!r}".format(action))


def soccer_action_description(action) -> Dict[str, object]:
    branches = soccer_action_branches(action)
    flat_action = soccer_action_index(branches)
    branch_names = [name for name, _ in SOCCER_ACTION_BRANCHES]
    meaning_parts = []
    out = {
        "flat_action": flat_action,
        "branch_action": str(branches),
    }
    for idx, (name, value_map) in enumerate(SOCCER_ACTION_BRANCHES):
        value = branches[idx]
        out[name] = value
        out["{}_meaning".format(name)] = value_map.get(value, "unknown")
        if value != 0:
            meaning_parts.append(value_map.get(value, "{}={}".format(name, value)))
    out["action_meaning"] = " + ".join(meaning_parts) if meaning_parts else "no-op / stay still"
    out["branch_order"] = str(branch_names)
    return out


def soccer_action_glossary():
    rows = []
    for branch_idx, (name, value_map) in enumerate(SOCCER_ACTION_BRANCHES):
        for value, meaning in value_map.items():
            rows.append(
                {
                    "branch_index": branch_idx,
                    "branch_name": name,
                    "value": value,
                    "meaning": meaning,
                }
            )
    return pd.DataFrame(rows)


def soccer_flat_action_table():
    rows = []
    for flat_action in range(27):
        row = soccer_action_description(flat_action)
        rows.append(
            {
                "flat_action": row["flat_action"],
                "branch_action": row["branch_action"],
                "meaning": row["action_meaning"],
            }
        )
    return pd.DataFrame(rows)


def soccer_named_action_table():
    rows = []
    seen = set()
    for name, branches in sorted(NAMED_SOCCER_ACTIONS.items()):
        key = tuple(branches)
        if name in {"no_op", "zero", "center", "stay"}:
            continue
        row = soccer_action_description(branches)
        rows.append(
            {
                "name": name,
                "flat_action": row["flat_action"],
                "branch_action": row["branch_action"],
                "meaning": row["action_meaning"],
            }
        )
        seen.add(key)
    return pd.DataFrame(rows).sort_values(["flat_action", "name"]).reset_index(drop=True)


def print_soccer_action_help():
    print("Live Unity action: MultiDiscrete([3, 3, 3])")
    print("Branch order: [forward_axis, right_axis, rotate_axis]")
    print("Flattened training action: Discrete(27), where index = forward*9 + right*3 + rotate")
    print("No-op is flat action 0 -> [0, 0, 0].")
    print("Random policy samples one of the 27 flat actions at every env.step().")


def _action_for_space(action, action_space):
    branches = soccer_action_branches(action)
    if hasattr(action_space, "n"):
        return soccer_action_index(branches)
    if hasattr(action_space, "nvec"):
        import numpy as np

        return np.asarray(branches, dtype=np.int64)
    return action


def _resolve_learning_action(policy, env, obs, step, rows):
    import numpy as np

    if callable(policy):
        return policy(obs, env, step, rows)
    if policy in (None, "random"):
        return env.action_space.sample()
    if isinstance(policy, str) and policy in NAMED_SOCCER_ACTIONS:
        return _action_for_space(policy, env.action_space)
    if policy in ("middle", "middle_index"):
        if hasattr(env.action_space, "n"):
            return int(env.action_space.n // 2)
        if hasattr(env.action_space, "nvec"):
            return np.asarray(np.asarray(env.action_space.nvec, dtype=np.int64) // 2, dtype=np.int64)
    if isinstance(policy, int):
        return int(policy)
    raise ValueError(
        "Unknown policy {!r}. Use 'random', a named SoccerTwos action, an integer "
        "flat action, or a callable.".format(policy)
    )


def _collect_single_player_rollout_from_env(
    env,
    policy="random",
    steps: int = 250,
    label: str = "rollout",
    reset_on_done: bool = False,
    goal_x: float = 14.0,
):
    import numpy as np

    obs = env.reset()
    rows = []
    cumulative_reward = 0.0
    episode_reward = 0.0
    episode_idx = 0
    for step in range(int(steps)):
        action = _resolve_learning_action(policy, env, obs, step, rows)
        next_obs, reward, done, info = env.step(action)
        reward_value = float(reward)
        cumulative_reward += reward_value
        episode_reward += reward_value
        row = {
            "step": step,
            "episode": episode_idx,
            "action": _action_repr(action),
            "reward": reward_value,
            "cumulative_reward": cumulative_reward,
            "episode_reward": episode_reward,
            "done": _learning_done_all(done),
            "obs_norm": float(np.linalg.norm(np.asarray(obs, dtype=np.float32).reshape(-1))),
            "next_obs_norm": float(
                np.linalg.norm(np.asarray(next_obs, dtype=np.float32).reshape(-1))
            ),
        }
        try:
            row.update(soccer_action_description(action))
        except ValueError:
            pass
        row.update(rollout_metrics_from_info(info, goal_x=goal_x))
        rows.append(row)
        obs = next_obs
        if row["done"]:
            if not reset_on_done:
                break
            obs = env.reset()
            episode_idx += 1
            episode_reward = 0.0

    df = pd.DataFrame(rows)
    df.attrs["label"] = label
    if not df.empty:
        if "player_ball_dist" in df:
            df["player_ball_progress"] = df["player_ball_dist"].shift(1) - df["player_ball_dist"]
        if "ball_goal_dist" in df:
            df["ball_goal_progress"] = df["ball_goal_dist"].shift(1) - df["ball_goal_dist"]
    return df


def collect_single_player_rollout(
    policy="random",
    steps: int = 250,
    render: bool = False,
    base_port: Optional[int] = None,
    label: Optional[str] = None,
    reset_on_done: bool = False,
    env_config: Optional[Dict] = None,
):
    from soccer_twos import EnvType

    config = {
        "render": render,
        "variation": EnvType.team_vs_policy,
        "flatten_branched": True,
        "single_player": True,
        "base_port": base_port,
    }
    if env_config:
        config.update(env_config)
    env = make_soccer_env(**config)
    try:
        return _collect_single_player_rollout_from_env(
            env,
            policy=policy,
            steps=steps,
            label=label or str(policy),
            reset_on_done=reset_on_done,
        )
    finally:
        env.close()


def collect_standalone_agent_rollout(
    ctx,
    module_name: str,
    steps: int = 250,
    render: bool = False,
    base_port: Optional[int] = None,
    label: Optional[str] = None,
):
    from soccer_twos import EnvType
    from soccer_twos_project.evaluation import get_agent_class

    if str(ctx.submissions_dir) not in sys.path:
        sys.path.insert(0, str(ctx.submissions_dir))
    clear_imported_package(module_name)
    module = importlib.import_module(module_name)
    env = make_soccer_env(
        render=render,
        variation=EnvType.team_vs_policy,
        flatten_branched=True,
        single_player=True,
        base_port=base_port,
    )
    try:
        agent = get_agent_class(module)(env)

        def agent_policy(obs, _env, _step, _rows):
            actions = agent.act({0: obs})
            return actions[0]

        return _collect_single_player_rollout_from_env(
            env,
            policy=agent_policy,
            steps=steps,
            label=label or module_name,
            reset_on_done=False,
        )
    finally:
        env.close()


def make_rllib_checkpoint_action_fn(
    checkpoint: str,
    stage: str = "ppo_baseline",
    profile_name: str = "cpu_debug",
    policy_id: str = "default_policy",
):
    from soccer_twos_project.exporting import restore_policy

    ray, trainer, _policy, _config, resolved_policy_id, direct_weights = restore_policy(
        checkpoint,
        stage,
        profile_name,
        policy_id,
    )
    if direct_weights is not None:
        trainer.set_weights({resolved_policy_id: direct_weights})

    def action_fn(obs, _env=None, _step=None, _rows=None):
        try:
            action = trainer.compute_single_action(
                obs,
                policy_id=resolved_policy_id,
                explore=False,
            )
        except (AttributeError, TypeError):
            action = trainer.compute_action(
                obs,
                policy_id=resolved_policy_id,
                explore=False,
            )
        if isinstance(action, tuple):
            return action[0]
        return action

    def close_fn():
        trainer.stop()
        ray.shutdown()

    return action_fn, close_fn


def collect_checkpoint_rollout(
    checkpoint: str,
    stage: str = "ppo_baseline",
    steps: int = 250,
    render: bool = False,
    base_port: Optional[int] = None,
    label: Optional[str] = None,
):
    action_fn, close_fn = make_rllib_checkpoint_action_fn(checkpoint, stage=stage)
    try:
        return collect_single_player_rollout(
            policy=action_fn,
            steps=steps,
            render=render,
            base_port=base_port,
            label=label or "{} checkpoint".format(stage),
        )
    finally:
        close_fn()


def rollout_summary(df, label: Optional[str] = None) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "label": label or "empty",
            "steps": 0,
            "total_reward": 0.0,
        }
    summary = {
        "label": label or df.attrs.get("label", "rollout"),
        "steps": int(len(df)),
        "total_reward": float(df["reward"].sum()),
        "mean_reward": float(df["reward"].mean()),
        "final_cumulative_reward": float(df["cumulative_reward"].iloc[-1]),
        "unique_actions": int(df["action"].nunique()) if "action" in df else 0,
    }
    for col in ("player_ball_dist", "ball_goal_dist"):
        if col in df:
            values = df[col].dropna()
            if not values.empty:
                summary["{}_start".format(col)] = float(values.iloc[0])
                summary["{}_final".format(col)] = float(values.iloc[-1])
                summary["{}_min".format(col)] = float(values.min())
    return summary


def rollout_summary_table(rollouts: Dict[str, pd.DataFrame]):
    return pd.DataFrame(
        [rollout_summary(df, label=label) for label, df in rollouts.items()]
    )


def plot_rollout_overview(df, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Cannot plot an empty rollout.")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(df["step"], df["reward"], label="step reward", linewidth=1.5)
    axes[0].plot(df["step"], df["cumulative_reward"], label="cumulative reward", linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    if "player_ball_dist" in df:
        axes[1].plot(df["step"], df["player_ball_dist"], label="player to ball")
    if "ball_goal_dist" in df:
        axes[1].plot(df["step"], df["ball_goal_dist"], label="ball to goal")
    axes[1].set_ylabel("Distance")
    axes[1].legend()
    df["action"].value_counts().sort_index().plot(kind="bar", ax=axes[2])
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Action")
    fig.suptitle(title or df.attrs.get("label", "Rollout overview"))
    fig.tight_layout()
    return fig


def plot_reward_timeline(df, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Cannot plot an empty rollout.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["step"], df["reward"], linewidth=1.5, label="step reward")
    if "cumulative_reward" in df:
        ax.plot(df["step"], df["cumulative_reward"], linewidth=2, label="cumulative reward")
    nonzero = df[df["reward"] != 0]
    if not nonzero.empty:
        ax.scatter(nonzero["step"], nonzero["reward"], color="tab:red", zorder=3, label="nonzero reward")
    if "done" in df:
        done_steps = df[df["done"] == True]  # noqa: E712
        for step in done_steps["step"].tolist():
            ax.axvline(step, color="0.5", linestyle="--", alpha=0.5)
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_title(title or "{} reward timeline".format(df.attrs.get("label", "Rollout")))
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_down_trajectory(df, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Cannot plot an empty rollout.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-7, 7)
    ax.axvline(0, color="0.75", linewidth=1)
    ax.axvline(14, color="tab:green", linestyle="--", linewidth=1.5, label="target goal")
    ax.axvline(-14, color="tab:red", linestyle="--", linewidth=1.5, label="own goal")
    if {"player_x", "player_y"}.issubset(df.columns):
        ax.plot(df["player_x"], df["player_y"], marker="o", markersize=3, label="player")
        ax.scatter(df["player_x"].iloc[0], df["player_y"].iloc[0], s=80, marker="s")
    if {"ball_x", "ball_y"}.issubset(df.columns):
        ax.plot(df["ball_x"], df["ball_y"], marker="o", markersize=3, label="ball")
        ax.scatter(df["ball_x"].iloc[0], df["ball_y"].iloc[0], s=80, marker="s")
    ax.set_title(title or "{} top-down trajectory".format(df.attrs.get("label", "Rollout")))
    ax.set_xlabel("Arena x")
    ax.set_ylabel("Arena y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_position_density(df, title: Optional[str] = None, entity: str = "player"):
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Cannot plot an empty rollout.")
    x_col = "{}_x".format(entity)
    y_col = "{}_y".format(entity)
    if not {x_col, y_col}.issubset(df.columns):
        print("Rollout does not contain {} position columns.".format(entity))
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist2d(df[x_col], df[y_col], bins=[30, 18], range=[[-15, 15], [-7, 7]], cmap="Blues")
    ax.axvline(14, color="tab:green", linestyle="--", linewidth=1.5, label="target goal")
    ax.axvline(-14, color="tab:red", linestyle="--", linewidth=1.5, label="own goal")
    ax.set_title(title or "{} position density".format(entity.title()))
    ax.set_xlabel("Arena x")
    ax.set_ylabel("Arena y")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_action_distribution(df, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    if df is None or df.empty:
        raise ValueError("Cannot plot an empty rollout.")
    fig, ax = plt.subplots(figsize=(9, 4))
    df["action"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(title or "{} action distribution".format(df.attrs.get("label", "Rollout")))
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_rollout_comparison(rollouts: Dict[str, pd.DataFrame], title: str = "Rollout comparison"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, df in rollouts.items():
        if df is None or df.empty:
            continue
        axes[0].plot(df["step"], df["cumulative_reward"], label=label)
        if "player_ball_dist" in df:
            axes[1].plot(df["step"], df["player_ball_dist"], label=label)
    axes[0].set_ylabel("Cumulative reward")
    axes[1].set_ylabel("Player-ball distance")
    axes[1].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def discounted_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = float(reward) + float(gamma) * running
        returns.append(running)
    return list(reversed(returns))


def value_advantage_table(rewards, values=None, gamma: float = 0.99):
    returns = discounted_returns(rewards, gamma=gamma)
    if values is None:
        values = [0.0 for _ in returns]
    rows = []
    for idx, (reward, ret, value) in enumerate(zip(rewards, returns, values)):
        rows.append(
            {
                "t": idx,
                "reward": float(reward),
                "return_Gt": float(ret),
                "value_Vs": float(value),
                "advantage_Gt_minus_Vs": float(ret) - float(value),
            }
        )
    return pd.DataFrame(rows)


def plot_value_advantage(table, title: str = "Return, value, and advantage"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in ("return_Gt", "value_Vs", "advantage_Gt_minus_Vs"):
        if col in table:
            ax.plot(table["t"], table[col], marker="o", label=col)
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Scalar")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def load_progress_dataframe(progress_path):
    path = Path(progress_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_training_diagnostics(progress_path_or_df, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    df = (
        pd.read_csv(progress_path_or_df)
        if isinstance(progress_path_or_df, (str, os.PathLike, Path))
        else progress_path_or_df.copy()
    )
    if df.empty:
        raise ValueError("Cannot plot empty training progress.")

    x_values = df["timesteps_total"] if "timesteps_total" in df else df.index
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    if "episode_reward_mean" in df:
        axes[0].plot(x_values, df["episode_reward_mean"], label="mean")
    if "episode_reward_min" in df:
        axes[0].plot(x_values, df["episode_reward_min"], label="min", alpha=0.7)
    if "episode_reward_max" in df:
        axes[0].plot(x_values, df["episode_reward_max"], label="max", alpha=0.7)
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()

    if "episode_len_mean" in df:
        axes[1].plot(x_values, df["episode_len_mean"], color="tab:orange")
    axes[1].set_ylabel("Episode length")

    diagnostic_cols = [
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default_policy/learner_stats/vf_loss",
        "info/learner/default_policy/learner_stats/kl",
        "info/learner/default_policy/learner_stats/entropy",
    ]
    for col in diagnostic_cols:
        if col in df:
            axes[2].plot(x_values, df[col], label=col.rsplit("/", 1)[-1])
    axes[2].set_ylabel("PPO diagnostics")
    axes[2].set_xlabel("Timesteps")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title or "Tiny PPO training diagnostics")
    fig.tight_layout()
    return fig


def show_training_snapshot(ctx, stage: str, rows: int = 10, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    snapshot = progress_status(ctx, stage=stage, rows=rows)
    if snapshot is None or snapshot.empty:
        return snapshot
    try:
        from IPython.display import display

        display(snapshot)
    except Exception:
        print(snapshot)

    full_df = load_progress_table(ctx, stage=stage)
    if full_df.empty:
        return full_df
    plot_training_diagnostics(full_df, title=title or "{} training diagnostics".format(stage))
    plt.show()
    return full_df


def compare_training_progress(
    ctx,
    stages=None,
    metric: str = "episode_reward_mean",
    smoothing_window: int = 1,
    title: str = "Training comparison",
):
    import matplotlib.pyplot as plt

    if stages is None:
        from soccer_twos_project.training import STAGES

        stages = list(STAGES)

    fig, ax = plt.subplots(figsize=(10, 6))
    summary_rows = []
    plotted = 0

    for stage in stages:
        df = load_progress_table(ctx, stage=stage)
        if df.empty or metric not in df.columns:
            continue

        x_values = df["timesteps_total"] if "timesteps_total" in df.columns else df.index
        y_values = df[metric].astype(float)
        if smoothing_window and smoothing_window > 1:
            y_values = y_values.rolling(smoothing_window, min_periods=1).mean()

        ax.plot(x_values, y_values, linewidth=2, label=stage)
        plotted += 1

        row = {
            "stage": stage,
            metric: float(df[metric].dropna().iloc[-1]) if df[metric].dropna().size else None,
        }
        for col in (
            "training_iteration",
            "timesteps_total",
            "episodes_total",
            "episode_reward_max",
            "episode_len_mean",
            "time_total_s",
        ):
            if col in df.columns and df[col].dropna().size:
                value = df[col].dropna().iloc[-1]
                row[col] = value.item() if hasattr(value, "item") else value
        summary_rows.append(row)

    if plotted == 0:
        plt.close(fig)
        print("No usable progress.csv files found for inline comparison.")
        return pd.DataFrame()

    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return pd.DataFrame(summary_rows)

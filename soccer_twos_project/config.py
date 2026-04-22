import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_ARTIFACT_ROOT = "artifacts/cs8803_soccer_twos"
# Backwards-compatible name for older Colab-oriented code.
DEFAULT_DRIVE_ROOT = DEFAULT_ARTIFACT_ROOT


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    num_gpus: int
    num_workers: int
    num_envs_per_worker: int
    train_batch_size: int
    rollout_fragment_length: int
    timesteps_total: int
    time_total_s: Optional[int]
    checkpoint_freq: int


PROFILES: Dict[str, HardwareProfile] = {
    "cpu_debug": HardwareProfile(
        name="cpu_debug",
        num_gpus=0,
        num_workers=0,
        num_envs_per_worker=1,
        train_batch_size=1000,
        rollout_fragment_length=200,
        timesteps_total=25000,
        time_total_s=1800,
        checkpoint_freq=1,
    ),
    "free_gpu": HardwareProfile(
        name="free_gpu",
        num_gpus=1,
        num_workers=1,
        num_envs_per_worker=1,
        train_batch_size=4000,
        rollout_fragment_length=500,
        timesteps_total=1500000,
        time_total_s=10800,
        checkpoint_freq=5,
    ),
    "pro_gpu": HardwareProfile(
        name="pro_gpu",
        num_gpus=1,
        num_workers=2,
        num_envs_per_worker=2,
        train_batch_size=10000,
        rollout_fragment_length=1000,
        timesteps_total=5000000,
        time_total_s=21600,
        checkpoint_freq=10,
    ),
    "laptop_cuda": HardwareProfile(
        name="laptop_cuda",
        num_gpus=1,
        num_workers=2,
        num_envs_per_worker=1,
        train_batch_size=8000,
        rollout_fragment_length=1000,
        timesteps_total=2500000,
        time_total_s=None,
        checkpoint_freq=5,
    ),
    "laptop_mps": HardwareProfile(
        name="laptop_mps",
        num_gpus=0,
        num_workers=1,
        num_envs_per_worker=1,
        train_batch_size=4000,
        rollout_fragment_length=500,
        timesteps_total=1000000,
        time_total_s=None,
        checkpoint_freq=5,
    ),
}


def artifact_root(root: Optional[str] = None) -> Path:
    return Path(root or os.environ.get("SOCCER_TWOS_DRIVE_ROOT", DEFAULT_DRIVE_ROOT))


def artifact_dirs(root: Optional[str] = None) -> Dict[str, Path]:
    base = artifact_root(root)
    return {
        "root": base,
        "checkpoints": base / "checkpoints",
        "plots": base / "plots",
        "datasets": base / "datasets",
        "submissions": base / "submissions",
        "evals": base / "evals",
        "logs": base / "logs",
    }


def ensure_artifact_dirs(root: Optional[str] = None) -> Dict[str, Path]:
    dirs = artifact_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def command_output(cmd) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""


def gpu_name() -> str:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    return output.splitlines()[0].strip() if output else ""


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


def total_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    output = command_output(["sysctl", "-n", "hw.memsize"])
    if output:
        try:
            return int(output) / 1024 / 1024 / 1024
        except ValueError:
            pass
    return 0.0


def cpu_count() -> int:
    return os.cpu_count() or 1


def hardware_report() -> Dict[str, object]:
    return {
        "cpu_count": cpu_count(),
        "ram_gb": round(total_ram_gb(), 2),
        "gpu_name": gpu_name(),
        "torch_cuda_available": torch_cuda_available(),
        "torch_mps_available": torch_mps_available(),
        "mlx_available": mlx_available(),
    }


def select_profile(requested: str = "auto", smoke: bool = False) -> HardwareProfile:
    if smoke:
        return PROFILES["cpu_debug"]
    if requested != "auto":
        if requested not in PROFILES:
            raise ValueError(
                "Unknown profile {}. Valid profiles: {}".format(
                    requested, ", ".join(sorted(PROFILES))
                )
            )
        return PROFILES[requested]
    if torch_cuda_available():
        if total_ram_gb() >= 20 and cpu_count() >= 4:
            return PROFILES["laptop_cuda"]
        return PROFILES["free_gpu"]
    if torch_mps_available() or mlx_available():
        return PROFILES["laptop_mps"]
    return PROFILES["cpu_debug"]


def fallback_profile() -> HardwareProfile:
    base = PROFILES["cpu_debug"]
    return HardwareProfile(
        name="fallback_single_worker",
        num_gpus=base.num_gpus,
        num_workers=0,
        num_envs_per_worker=1,
        train_batch_size=base.train_batch_size,
        rollout_fragment_length=base.rollout_fragment_length,
        timesteps_total=base.timesteps_total,
        time_total_s=base.time_total_s,
        checkpoint_freq=1,
    )


def profile_dict(profile: HardwareProfile) -> Dict[str, object]:
    return asdict(profile)


def checkpoint_path(checkpoint) -> Optional[str]:
    if checkpoint is None:
        return None
    for attr in ("local_path", "path", "uri"):
        if hasattr(checkpoint, attr):
            value = getattr(checkpoint, attr)
            value = value() if callable(value) else value
            if value:
                return str(value)
    try:
        value = os.fspath(checkpoint)
        if value:
            return str(value)
    except TypeError:
        pass
    return str(checkpoint)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)


def json_safe(value):
    if hasattr(value, "local_path"):
        return checkpoint_path(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, os.PathLike):
        try:
            return os.fspath(value)
        except TypeError:
            return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

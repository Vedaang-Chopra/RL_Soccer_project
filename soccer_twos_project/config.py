import json
import os
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
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
    "multi_gpu": HardwareProfile(
        name="multi_gpu",
        num_gpus=8,
        num_workers=16,
        num_envs_per_worker=1,
        train_batch_size=32000,
        rollout_fragment_length=2000,
        timesteps_total=50000000,
        time_total_s=None,
        checkpoint_freq=10,
    ),
    # Smoke / development: 1 GPU + 16 workers.  Fast startup, good for
    # validating the full stack before committing to a long run.
    "a40": HardwareProfile(
        name="a40",
        num_gpus=1,
        num_workers=16,
        num_envs_per_worker=1,
        train_batch_size=32000,   # 16 workers × 2000 steps
        rollout_fragment_length=2000,
        timesteps_total=50000000,
        time_total_s=None,
        checkpoint_freq=10,
    ),
    # Production: 1 GPU + 40 CPU workers.  Uses ~80 of the 128 available CPUs
    # (40 workers + Ray driver + PPOTrainer actor + headroom).  Batch size
    # scales with worker count so each PPO iteration still gets a full 80k
    # samples.  On A40 hardware this roughly doubles throughput vs a40.
    "a40_full": HardwareProfile(
        name="a40_full",
        num_gpus=1,
        num_workers=40,
        num_envs_per_worker=1,
        train_batch_size=80000,   # 40 workers × 2000 steps
        rollout_fragment_length=2000,
        timesteps_total=50000000,
        time_total_s=None,
        checkpoint_freq=10,
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


def torch_cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def torch_cuda_device_names():
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def torch_cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "")
    except Exception:
        return ""


def torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return ""


@lru_cache(maxsize=1)
def torch_cuda_runtime_probe() -> Dict[str, object]:
    report = {
        "ok": False,
        "device_name": "",
        "error": "",
    }
    try:
        import torch

        if not torch.cuda.is_available():
            report["error"] = "torch.cuda.is_available() returned False"
            return report
        if torch.cuda.device_count() < 1:
            report["error"] = "torch.cuda.device_count() returned 0"
            return report

        device = torch.device("cuda:0")
        report["device_name"] = str(torch.cuda.get_device_name(device))
        layer = torch.nn.Linear(8, 8).to(device)
        x = torch.randn(4, 8, device=device)
        with torch.no_grad():
            _ = layer(x).sum().item()
        torch.cuda.synchronize(device)
        report["ok"] = True
        return report
    except Exception as exc:
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        return report


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
    cuda_probe = torch_cuda_runtime_probe() if torch_cuda_available() else {
        "ok": False,
        "device_name": "",
        "error": "",
    }
    return {
        "cpu_count": cpu_count(),
        "ram_gb": round(total_ram_gb(), 2),
        "gpu_name": gpu_name(),
        "torch_version": torch_version(),
        "torch_cuda_available": torch_cuda_available(),
        "torch_cuda_device_count": torch_cuda_device_count(),
        "torch_cuda_device_names": torch_cuda_device_names(),
        "torch_cuda_version": torch_cuda_version(),
        "torch_cuda_runtime_ready": cuda_probe["ok"],
        "torch_cuda_runtime_probe_device": cuda_probe["device_name"],
        "torch_cuda_runtime_probe_error": cuda_probe["error"],
        "torch_mps_available": torch_mps_available(),
        "mlx_available": mlx_available(),
    }


def cuda_training_report(profile: Optional[HardwareProfile] = None) -> Dict[str, object]:
    requested_gpus = int(profile.num_gpus) if profile else 0
    device_count = torch_cuda_device_count()
    cuda_available = torch_cuda_available()
    cuda_probe = torch_cuda_runtime_probe() if cuda_available and device_count > 0 else {
        "ok": False,
        "device_name": "",
        "error": "",
    }
    return {
        "profile": profile.name if profile else None,
        "profile_num_gpus": requested_gpus,
        "cuda_required_by_profile": requested_gpus > 0,
        "torch_version": torch_version(),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "torch_cuda_device_names": torch_cuda_device_names(),
        "torch_cuda_version": torch_cuda_version(),
        "torch_cuda_runtime_ready": cuda_probe["ok"],
        "torch_cuda_runtime_probe_device": cuda_probe["device_name"],
        "torch_cuda_runtime_probe_error": cuda_probe["error"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ready_for_cuda_training": bool(
            requested_gpus == 0
            or (cuda_available and device_count >= requested_gpus and cuda_probe["ok"])
        ),
    }


def validate_training_profile(profile: HardwareProfile) -> Dict[str, object]:
    report = cuda_training_report(profile)
    if profile.num_gpus > 0 and not report["ready_for_cuda_training"]:
        raise RuntimeError(
            "Profile '{}' requests {} GPU(s), but PyTorch CUDA is not ready for runtime "
            "training on this machine. torch={}, cuda={}, device(s)={}, runtime_probe_error={!r}. "
            "Install a GPU build of PyTorch that supports this GPU architecture, or use "
            "`--profile cpu_debug` for CPU.".format(
                profile.name,
                profile.num_gpus,
                report["torch_version"],
                report["torch_cuda_version"],
                report["torch_cuda_device_names"],
                report["torch_cuda_runtime_probe_error"],
            )
        )
    return report


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
    if (
        torch_cuda_available()
        and torch_cuda_device_count() > 0
        and torch_cuda_runtime_probe()["ok"]
    ):
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

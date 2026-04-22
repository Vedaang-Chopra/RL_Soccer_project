import argparse
import os
from pathlib import Path

from soccer_twos_project.config import artifact_dirs, ensure_artifact_dirs


def find_progress_files(ray_results_root: Path):
    return sorted(ray_results_root.rglob("progress.csv"))


def label_for_progress(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        return "{} / {}".format(rel.parts[0], rel.parts[1])
    return path.parent.name


def load_progress(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    if "timesteps_total" not in df.columns or "episode_reward_mean" not in df.columns:
        return None
    return df[["timesteps_total", "episode_reward_mean"]].dropna()


def plot_single(label, df, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["timesteps_total"], df["episode_reward_mean"], linewidth=2)
    ax.set_title(label)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward Mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overlay(series, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in series:
        ax.plot(df["timesteps_total"], df["episode_reward_mean"], linewidth=2, label=label)
    ax.set_title("Soccer-Twos Agent Learning Curve Comparison")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward Mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def safe_filename(label: str) -> str:
    return (
        label.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("__", "_")
    )


def plot_results(args):
    dirs = ensure_artifact_dirs(args.artifact_root)
    ray_root = Path(args.ray_results or dirs["checkpoints"])
    out_dir = Path(args.output_dir or dirs["plots"])
    out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for progress_path in find_progress_files(ray_root):
        label = label_for_progress(progress_path, ray_root)
        if args.filter and args.filter not in label:
            continue
        df = load_progress(progress_path)
        if df is None or df.empty:
            continue
        series.append((label, df))
        single_out = out_dir / "{}.png".format(safe_filename(label))
        plot_single(label, df, single_out)
        print("Wrote", single_out)

    if not series:
        raise SystemExit("No usable progress.csv files found under {}".format(ray_root))
    overlay_out = out_dir / "learning_curve_comparison.png"
    plot_overlay(series, overlay_out)
    print("Wrote", overlay_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Soccer-Twos Ray training curves.")
    parser.add_argument("--artifact-root", default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"))
    parser.add_argument("--ray-results", help="Root containing Ray progress.csv files.")
    parser.add_argument("--output-dir", help="Directory for PNG plots.")
    parser.add_argument("--filter", help="Only plot labels containing this substring.")
    return parser.parse_args()


def main():
    plot_results(parse_args())


if __name__ == "__main__":
    main()

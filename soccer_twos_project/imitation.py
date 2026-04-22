import argparse
import importlib
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_twos_project.config import ensure_artifact_dirs
from soccer_twos_project.exporting import write_agent_package


BASELINE_FILE_ID = "1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE"


class BCPolicyNetwork(nn.Module):
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


def get_agent_class(module):
    from soccer_twos import AgentInterface
    import inspect

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls is not AgentInterface and issubclass(cls, AgentInterface):
            return cls
    raise ValueError("No AgentInterface subclass found in {}".format(module.__name__))


def done_all(done) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(done["__all__"])
        return bool(max(done.values())) if done else False
    return bool(done)


def build_action_reverse_lookup(action_space):
    if hasattr(action_space, "n"):
        return None, int(action_space.n)
    from gym_unity.envs import ActionFlattener

    flattener = ActionFlattener(action_space.nvec)
    reverse = {}
    for index in range(flattener.action_space.n):
        reverse[tuple(np.asarray(flattener.lookup_action(index), dtype=int).tolist())] = index
    return reverse, int(flattener.action_space.n)


def flatten_action(action, reverse_lookup):
    if reverse_lookup is None:
        return int(action)
    key = tuple(np.asarray(action, dtype=int).tolist())
    if key not in reverse_lookup:
        raise ValueError("Action {} is not in flattener lookup.".format(key))
    return reverse_lookup[key]


def download_baseline(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ceia_baseline_agent.zip"
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown first: pip install gdown") from exc
    gdown.download(id=BASELINE_FILE_ID, output=str(zip_path), quiet=False)
    try:
        shutil.unpack_archive(str(zip_path), str(output_dir))
    except shutil.ReadError:
        print("Downloaded baseline to {}, but it was not a recognized archive.".format(zip_path))
    print("Baseline output directory:", output_dir)


def collect_dataset(args):
    import soccer_twos

    dirs = ensure_artifact_dirs(args.artifact_root)
    dataset_path = Path(args.output or (dirs["datasets"] / "bc_expert_dataset.npz"))
    expert_module = importlib.import_module(args.expert_module)
    env = soccer_twos.make(render=False, base_port=args.base_port)
    expert = get_agent_class(expert_module)(env)
    reverse_lookup, action_size = build_action_reverse_lookup(env.action_space)

    observations = []
    actions = []
    episodes = 0
    while len(actions) < args.samples:
        obs = env.reset()
        episodes += 1
        while len(actions) < args.samples:
            blue_actions = expert.act({0: obs[0], 1: obs[1]})
            orange_actions = expert.act({0: obs[2], 1: obs[3]})
            for team_obs, team_actions in (
                ((obs[0], obs[1]), blue_actions),
                ((obs[2], obs[3]), orange_actions),
            ):
                for local_id, player_obs in enumerate(team_obs):
                    observations.append(np.asarray(player_obs, dtype=np.float32).reshape(-1))
                    actions.append(flatten_action(team_actions[local_id], reverse_lookup))
                    if len(actions) >= args.samples:
                        break
                if len(actions) >= args.samples:
                    break
            step_actions = {
                0: blue_actions[0],
                1: blue_actions[1],
                2: orange_actions[0],
                3: orange_actions[1],
            }
            obs, reward, done, info = env.step(step_actions)
            if done_all(done):
                break

    env.close()
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        observations=observations,
        actions=actions,
        obs_size=observations.shape[1],
        action_size=action_size,
        expert_module=args.expert_module,
        episodes=episodes,
    )
    print("Wrote dataset:", dataset_path)
    print("Samples:", len(actions), "Episodes:", episodes)


def train_bc(args):
    dirs = ensure_artifact_dirs(args.artifact_root)
    dataset_path = Path(args.dataset)
    data = np.load(dataset_path, allow_pickle=True)
    x = data["observations"].astype(np.float32)
    y = data["actions"].astype(np.int64)
    obs_size = int(data["obs_size"])
    action_size = int(data["action_size"])
    hidden_layers = [int(v) for v in args.hidden_layers.split(",") if v]

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(y))
    split = int(len(indices) * (1.0 - args.val_fraction))
    train_idx = indices[:split]
    val_idx = indices[split:]

    model = BCPolicyNetwork(obs_size, action_size, hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx]) if len(val_idx) else None
    y_val = torch.from_numpy(y[val_idx]) if len(val_idx) else None

    for epoch in range(args.epochs):
        order = torch.randperm(len(y_train))
        total_loss = 0.0
        model.train()
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            logits = model(x_train[batch_idx])
            loss = F.cross_entropy(logits, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)

        model.eval()
        train_loss = total_loss / len(y_train)
        if x_val is not None and len(y_val):
            with torch.no_grad():
                val_logits = model(x_val)
                val_loss = F.cross_entropy(val_logits, y_val).item()
                val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
            print(
                "epoch={} train_loss={:.4f} val_loss={:.4f} val_acc={:.3f}".format(
                    epoch + 1, train_loss, val_loss, val_acc
                )
            )
        else:
            print("epoch={} train_loss={:.4f}".format(epoch + 1, train_loss))

    output_dir = Path(args.output_dir or (dirs["submissions"] / args.agent_name))
    metadata = {
        "agent_name": args.agent_name,
        "author": args.author,
        "email": args.email,
        "description": args.description,
        "stage": "bc_imitation",
        "algo": "behavior_cloning",
        "criterion": "imitation learning",
        "source_checkpoint": str(dataset_path),
        "policy_id": "behavior_cloning",
        "obs_size": obs_size,
        "action_size": action_size,
        "hidden_layers": hidden_layers,
    }
    package_path = write_agent_package(output_dir, metadata, model.state_dict(), not args.no_zip)
    print("Wrote imitation package:", output_dir)
    if not args.no_zip:
        print("Wrote zip:", package_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior cloning pipeline for Soccer-Twos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download-baseline")
    dl.add_argument("--output-dir", default="ceia_baseline_download")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--expert-module", required=True)
    collect.add_argument("--samples", type=int, default=50000)
    collect.add_argument("--output")
    collect.add_argument("--base-port", type=int)
    collect.add_argument("--artifact-root", default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"))

    train = subparsers.add_parser("train")
    train.add_argument("--dataset", required=True)
    train.add_argument("--agent-name", default="soccer_bc_imitation")
    train.add_argument("--author", default="Your Name")
    train.add_argument("--email", default="your.email@gatech.edu")
    train.add_argument("--description", default="Behavior cloning Soccer-Twos agent.")
    train.add_argument("--hidden-layers", default="256,256")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output-dir")
    train.add_argument("--artifact-root", default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"))
    train.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "download-baseline":
        download_baseline(Path(args.output_dir))
    elif args.command == "collect":
        collect_dataset(args)
    elif args.command == "train":
        train_bc(args)


if __name__ == "__main__":
    main()

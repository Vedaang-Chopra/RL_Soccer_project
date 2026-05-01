import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

from soccer_twos_project.config import (
    ensure_artifact_dirs,
    json_safe,
    select_profile,
    write_json,
)
from soccer_twos_project.envs import create_rllib_env
from soccer_twos_project.training import STAGES, build_training_config


MODEL_TEMPLATE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
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
'''


AGENT_TEMPLATE = '''import json
import os

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import PolicyNetwork


class SoccerTorchAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        package_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(package_dir, "metadata.json")
        checkpoint_path = os.path.join(package_dir, "checkpoint.pth")
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.name = self.metadata.get("agent_name", "SoccerTorchAgent")
        self.action_mode = self.metadata.get("action_mode", "flat_discrete")
        self.action_nvec = self.metadata.get("action_nvec")
        self.flattener = None
        if self.action_mode == "flat_discrete" and hasattr(env.action_space, "nvec"):
            self.flattener = ActionFlattener(env.action_space.nvec)
            live_action_size = self.flattener.action_space.n
        elif hasattr(env.action_space, "n"):
            live_action_size = env.action_space.n
        else:
            live_action_size = self.metadata["action_size"]

        self.model = PolicyNetwork(
            self.metadata["obs_size"],
            self.metadata.get("action_size", live_action_size),
            self.metadata.get("hidden_layers", [512]),
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, observation):
        actions = {}
        with torch.no_grad():
            for player_id, obs in observation.items():
                state = np.asarray(obs, dtype=np.float32).reshape(-1)
                state = torch.from_numpy(state).float().unsqueeze(0)
                logits = self.model(state)
                if self.action_mode == "multidiscrete_branches":
                    actions[player_id] = self._branch_action(logits)
                else:
                    action_index = int(torch.argmax(logits, dim=-1).item())
                    if self.flattener is not None:
                        actions[player_id] = self.flattener.lookup_action(action_index)
                    else:
                        actions[player_id] = action_index
        return actions

    def _branch_action(self, logits):
        if not self.action_nvec:
            raise ValueError("metadata.json is missing action_nvec for branch action inference.")
        branches = []
        start = 0
        for branch_size in self.action_nvec:
            end = start + int(branch_size)
            branch_logits = logits[:, start:end]
            branches.append(int(torch.argmax(branch_logits, dim=-1).item()))
            start = end
        return np.asarray(branches, dtype=np.int64)
'''


INIT_TEMPLATE = "from .agent import SoccerTorchAgent\n"


REQUIREMENTS_TEMPLATE = '''gym==0.19.0
gym-unity==0.27.0
numpy
torch
'''


def trainer_class(algo: str):
    if algo.upper() == "PPO":
        from ray.rllib.agents.ppo import PPOTrainer

        return PPOTrainer
    if algo.upper() == "DQN":
        from ray.rllib.agents.dqn import DQNTrainer

        return DQNTrainer
    raise ValueError("Unsupported algo: {}".format(algo))


def observation_size(space) -> int:
    import numpy as np

    if hasattr(space, "shape") and space.shape:
        return int(np.prod(space.shape))
    raise ValueError("Cannot infer flat observation size from {}".format(space))


def action_space_details(space) -> Dict:
    import numpy as np

    if hasattr(space, "n"):
        return {
            "action_mode": "flat_discrete",
            "action_size": int(space.n),
            "action_output_size": int(space.n),
            "action_nvec": None,
        }
    if hasattr(space, "nvec"):
        nvec = [int(v) for v in np.asarray(space.nvec).tolist()]
        return {
            "action_mode": "multidiscrete_branches",
            "action_size": int(np.prod(nvec)),
            "action_output_size": int(np.sum(nvec)),
            "action_nvec": nvec,
        }
    raise ValueError("Cannot infer flat action size from {}".format(space))


def extract_mlp_state(policy, expected_action_size: int):
    # Prefer the RLlib state_dict layout when available. Module traversal order
    # is not reliable here: RLlib can register the logits head before the shared
    # hidden stack, which causes us to export only the output layer.
    try:
        return extract_mlp_state_from_weights(
            policy.model.state_dict(), expected_action_size
        )
    except Exception as e:
        print("extract_mlp_state_from_weights failed:", e)
        pass

    import torch

    linears = [
        module
        for module in policy.model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    selected = []
    for layer in linears:
        selected.append(layer)
        if layer.out_features == expected_action_size and len(selected) >= 1:
            break
    if not selected or selected[-1].out_features != expected_action_size:
        raise RuntimeError(
            "Could not locate action logits layer with {} outputs. Found linear "
            "layers: {}".format(
                expected_action_size,
                [(layer.in_features, layer.out_features) for layer in linears],
            )
        )

    hidden_layers = [layer.out_features for layer in selected[:-1]]
    state_dict = {}
    for idx, layer in enumerate(selected[:-1]):
        state_dict["layers.{}.weight".format(idx)] = layer.weight.detach().cpu()
        state_dict["layers.{}.bias".format(idx)] = layer.bias.detach().cpu()
    state_dict["output.weight"] = selected[-1].weight.detach().cpu()
    state_dict["output.bias"] = selected[-1].bias.detach().cpu()
    return state_dict, hidden_layers


def load_checkpoint_policy_weights(checkpoint: str, policy_id: str) -> Dict:
    import pickle

    with open(checkpoint, "rb") as f:
        checkpoint_state = pickle.load(f)
    worker_state = pickle.loads(checkpoint_state["worker"])
    policy_states = worker_state["state"]
    resolved_policy_id = resolve_policy_id(policy_id, policy_states)
    return policy_states[resolved_policy_id]["weights"]


def resolve_policy_id(policy_id: str, policy_container) -> str:
    if policy_id in policy_container:
        return policy_id
    aliases = {
        "default": "default_policy",
        "default_policy": "default",
    }
    alias = aliases.get(policy_id)
    if alias and alias in policy_container:
        print("Policy id '{}' not found; using '{}'.".format(policy_id, alias))
        return alias
    available = (
        list(policy_container.keys())
        if hasattr(policy_container, "keys")
        else list(policy_container)
    )
    raise ValueError("Policy id '{}' not found. Available policies: {}".format(policy_id, available))


def extract_mlp_state_from_weights(weights: Dict, expected_action_size: int):
    import torch

    hidden_pattern = re.compile(r"^_hidden_layers\.(\d+)\._model\.0\.(weight|bias)$")
    hidden_indices = sorted(
        {
            int(match.group(1))
            for key in weights
            for match in [hidden_pattern.match(key)]
            if match
        }
    )
    state_dict = {}
    hidden_layers = []
    for out_idx, hidden_idx in enumerate(hidden_indices):
        weight_key = "_hidden_layers.{}._model.0.weight".format(hidden_idx)
        bias_key = "_hidden_layers.{}._model.0.bias".format(hidden_idx)
        weight = torch.as_tensor(weights[weight_key]).detach().cpu()
        bias = torch.as_tensor(weights[bias_key]).detach().cpu()
        hidden_layers.append(int(weight.shape[0]))
        state_dict["layers.{}.weight".format(out_idx)] = weight
        state_dict["layers.{}.bias".format(out_idx)] = bias

    output_weight_key = "_logits._model.0.weight"
    output_bias_key = "_logits._model.0.bias"
    if output_weight_key not in weights or output_bias_key not in weights:
        raise RuntimeError(
            "Could not find RLlib logits weights. Available keys: {}".format(
                sorted(weights)[:20]
            )
        )
    output_weight = torch.as_tensor(weights[output_weight_key]).detach().cpu()
    output_bias = torch.as_tensor(weights[output_bias_key]).detach().cpu()
    if int(output_weight.shape[0]) != int(expected_action_size):
        raise RuntimeError(
            "Expected action output size {}, found logits size {}.".format(
                expected_action_size, output_weight.shape[0]
            )
        )
    state_dict["output.weight"] = output_weight
    state_dict["output.bias"] = output_bias
    return state_dict, hidden_layers


def restore_policy(checkpoint: str, stage: str, profile_name: str, policy_id: str):
    import ray
    from ray import tune

    spec = STAGES[stage]
    profile = select_profile(profile_name, smoke=True)
    config = build_training_config(stage, profile)
    config["num_workers"] = 0
    config["num_gpus"] = 0
    if "num_envs_per_worker" in config:
        config["num_envs_per_worker"] = 1

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)
    trainer = trainer_class(spec["algo"])(config=config, env="Soccer")
    resolved_policy_id = resolve_policy_id(policy_id, trainer.workers.local_worker().policy_map)
    policy = trainer.get_policy(resolved_policy_id)
    direct_weights = None
    try:
        trainer.restore(checkpoint)
        policy = trainer.get_policy(resolved_policy_id)
    except TypeError as exc:
        print("Trainer restore failed; loading policy weights directly from checkpoint:", exc)
        direct_weights = load_checkpoint_policy_weights(checkpoint, resolved_policy_id)
    return ray, trainer, policy, config, resolved_policy_id, direct_weights


def readme_text(metadata: Dict) -> str:
    return """# {agent_name}

**Agent name:** {agent_name}

**Author(s):** {author} ({email})

## Description

{description}

## Training Metadata

- Stage: `{stage}`
- Algorithm: `{algo}`
- Criterion: {criterion}
- Source checkpoint: `{source_checkpoint}`
- Observation size: `{obs_size}`
- Action size: `{action_size}`
- Action mode: `{action_mode}`
- Hidden layers: `{hidden_layers}`
""".format(**metadata)


def write_agent_package(
    output_dir: Path,
    metadata: Dict,
    state_dict,
    zip_output: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.py").write_text(MODEL_TEMPLATE)
    (output_dir / "agent.py").write_text(AGENT_TEMPLATE)
    (output_dir / "__init__.py").write_text(INIT_TEMPLATE)
    (output_dir / "requirements.txt").write_text(REQUIREMENTS_TEMPLATE)
    (output_dir / "README.md").write_text(readme_text(metadata))
    write_json(output_dir / "metadata.json", json_safe(metadata))

    import torch

    torch.save(
        {
            "state_dict": state_dict,
            "metadata": metadata,
        },
        output_dir / "checkpoint.pth",
    )
    if zip_output:
        return zip_agent_package(output_dir)
    return output_dir


def zip_agent_package(package_dir: Path) -> Path:
    zip_path = package_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_dir.rglob("*"):
            if path.is_file():
                zf.write(path, package_dir.name / path.relative_to(package_dir))
    return zip_path


def export_checkpoint(args):
    dirs = ensure_artifact_dirs(args.artifact_root)
    default_dir = dirs["submissions"] / args.agent_name
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)

    ray, trainer, policy, config, resolved_policy_id, direct_weights = restore_policy(
        args.checkpoint, args.stage, args.profile, args.policy_id
    )
    try:
        obs_size = observation_size(policy.observation_space)
        action_details = action_space_details(policy.action_space)
        if direct_weights is not None:
            state_dict, hidden_layers = extract_mlp_state_from_weights(
                direct_weights, action_details["action_output_size"]
            )
        else:
            state_dict, hidden_layers = extract_mlp_state(
                policy, action_details["action_output_size"]
            )
        spec = STAGES[args.stage]
        metadata = {
            "agent_name": args.agent_name,
            "author": args.author,
            "email": args.email,
            "description": args.description,
            "stage": args.stage,
            "algo": spec["algo"],
            "criterion": spec["criterion"],
            "source_checkpoint": args.checkpoint,
            "policy_id": resolved_policy_id,
            "obs_size": obs_size,
            "action_size": action_details["action_output_size"],
            "action_space_size": action_details["action_size"],
            "action_mode": action_details["action_mode"],
            "action_nvec": action_details["action_nvec"],
            "hidden_layers": hidden_layers,
            "training_config": json_safe(config),
        }
        package_path = write_agent_package(output_dir, metadata, state_dict, not args.no_zip)
    finally:
        trainer.stop()
        ray.shutdown()
    print("Wrote package:", output_dir)
    if not args.no_zip:
        print("Wrote zip:", package_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Export RLlib Soccer-Twos policy package.")
    parser.add_argument("--checkpoint", required=True, help="RLlib checkpoint path.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--policy-id", default="default_policy", help="RLlib policy id.")
    parser.add_argument("--profile", default="cpu_debug", help="Profile used to rebuild config.")
    parser.add_argument("--artifact-root", default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"))
    parser.add_argument("--output-dir", help="Output agent directory.")
    parser.add_argument("--agent-name", required=True, help="Final package/agent name.")
    parser.add_argument("--author", default="Your Name")
    parser.add_argument("--email", default="your.email@gatech.edu")
    parser.add_argument("--description", default="Trained Soccer-Twos policy.")
    parser.add_argument("--no-zip", action="store_true", help="Do not zip output package.")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before export.")
    return parser.parse_args()


def main():
    export_checkpoint(parse_args())


if __name__ == "__main__":
    main()

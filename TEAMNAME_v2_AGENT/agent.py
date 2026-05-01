import json
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

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from soccer_twos_project.config import ensure_artifact_dirs
from soccer_twos_project.mlagents_compat import patch_unity_environment_close


def get_agent_class(module):
    from soccer_twos import AgentInterface
    import inspect

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls is not AgentInterface and issubclass(cls, AgentInterface):
            return cls
    raise ValueError("No AgentInterface subclass found in {}".format(module.__name__))


def load_agent(module_name: str, base_port=None):
    patch_unity_environment_close()
    import soccer_twos

    env = soccer_twos.make(render=False, base_port=base_port)
    module = importlib.import_module(module_name)
    agent = get_agent_class(module)(env)
    env.close()
    return agent


def done_all(done) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(done["__all__"])
        return bool(max(done.values())) if done else False
    return bool(done)


def evaluate(agent1_name: str, agent2_name: str, episodes: int, base_port=None):
    patch_unity_environment_close()
    import soccer_twos

    agent1 = load_agent(agent1_name, base_port=base_port)
    agent2 = load_agent(agent2_name, base_port=base_port)
    env = soccer_twos.make(render=False, base_port=base_port)
    rows: List[Dict] = []

    for episode_idx in range(episodes):
        obs = env.reset()
        if episode_idx % 2 == 0:
            blue_agent, orange_agent = agent1, agent2
            blue_name, orange_name = agent1_name, agent2_name
        else:
            blue_agent, orange_agent = agent2, agent1
            blue_name, orange_name = agent2_name, agent1_name

        blue_return = 0.0
        orange_return = 0.0
        steps = 0
        while True:
            blue_actions = blue_agent.act({0: obs[0], 1: obs[1]})
            orange_actions = orange_agent.act({0: obs[2], 1: obs[3]})
            actions = {
                0: blue_actions[0],
                1: blue_actions[1],
                2: orange_actions[0],
                3: orange_actions[1],
            }
            obs, reward, done, info = env.step(actions)
            steps += 1
            blue_return += reward[0] + reward[1]
            orange_return += reward[2] + reward[3]
            if done_all(done):
                break

        rows.append(
            {
                "episode": episode_idx,
                "steps": steps,
                "blue_agent": blue_name,
                "orange_agent": orange_name,
                "blue_return": blue_return,
                "orange_return": orange_return,
                "agent1_return": blue_return if blue_name == agent1_name else orange_return,
                "agent2_return": blue_return if blue_name == agent2_name else orange_return,
            }
        )
        print("Episode {}/{} complete".format(episode_idx + 1, episodes))

    env.close()
    return rows, summarize(rows, agent1_name, agent2_name)


def summarize(rows: List[Dict], agent1_name: str, agent2_name: str) -> Dict:
    agent1_returns = [row["agent1_return"] for row in rows]
    agent2_returns = [row["agent2_return"] for row in rows]
    wins = sum(1 for row in rows if row["agent1_return"] > row["agent2_return"])
    losses = sum(1 for row in rows if row["agent1_return"] < row["agent2_return"])
    draws = len(rows) - wins - losses
    return {
        "agent1": agent1_name,
        "agent2": agent2_name,
        "episodes": len(rows),
        "agent1_reward_mean": float(np.mean(agent1_returns)) if rows else float("nan"),
        "agent2_reward_mean": float(np.mean(agent2_returns)) if rows else float("nan"),
        "agent1_reward_min": float(np.min(agent1_returns)) if rows else float("nan"),
        "agent1_reward_max": float(np.max(agent1_returns)) if rows else float("nan"),
        "agent2_reward_min": float(np.min(agent2_returns)) if rows else float("nan"),
        "agent2_reward_max": float(np.max(agent2_returns)) if rows else float("nan"),
        "agent1_wins": wins,
        "agent1_losses": losses,
        "draws": draws,
        "agent1_win_rate": wins / len(rows) if rows else float("nan"),
        "episode_len_mean": float(np.mean([row["steps"] for row in rows])) if rows else float("nan"),
    }


def write_outputs(rows: List[Dict], summary: Dict, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "{}_episodes.csv".format(label)
    json_path = out_dir / "{}_summary.json".format(label)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Wrote", csv_path)
    print("Wrote", json_path)


def write_summary(summary: Dict, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "{}_summary.json".format(label)
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Wrote", json_path)


def safe_label(agent1: str, agent2: str) -> str:
    return "{}_vs_{}".format(agent1, agent2).replace("/", "_").replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Headless Soccer-Twos evaluator.")
    parser.add_argument("-m", "--agent-module", help="Self-play agent module.")
    parser.add_argument("-m1", "--agent1-module", help="Team 1 agent module.")
    parser.add_argument("-m2", "--agent2-module", help="Team 2 agent module.")
    parser.add_argument("-e", "--episodes", type=int, default=100)
    parser.add_argument("-p", "--base-port", type=int)
    parser.add_argument("--artifact-root", default=os.environ.get("SOCCER_TWOS_DRIVE_ROOT"))
    parser.add_argument("--output-dir", help="Directory for JSON/CSV evaluation output.")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use soccer_twos.evaluate instead of the fallback evaluator.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.agent_module:
        agent1 = args.agent_module
        agent2 = args.agent_module
    elif args.agent1_module and args.agent2_module:
        agent1 = args.agent1_module
        agent2 = args.agent2_module
    else:
        raise SystemExit("Specify -m for self-play or -m1/-m2 for head-to-head.")

    dirs = ensure_artifact_dirs(args.artifact_root)
    out_dir = Path(args.output_dir or dirs["evals"])
    if args.official:
        from soccer_twos.evaluate import evaluate as official_evaluate

        summary = official_evaluate(agent1, agent2, args.episodes, args.base_port)
        write_summary(summary, out_dir, safe_label(agent1, agent2))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    rows, summary = evaluate(agent1, agent2, args.episodes, args.base_port)
    write_outputs(rows, summary, out_dir, safe_label(agent1, agent2))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

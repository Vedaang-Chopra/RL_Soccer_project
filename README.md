# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)
- See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the organized project layout.
- See [docs/FINAL_PROJECT_TRACKER.md](docs/FINAL_PROJECT_TRACKER.md) for the final-project checklist.

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run a small environment sanity example
python examples/legacy/example_random_players.py

### 6. Use the guided notebook sequence for the project workflow
jupyter notebook notebooks/00_environment_understanding.ipynb

Run the notebooks in order:

- `notebooks/00_environment_understanding.ipynb`
- `notebooks/01_training_smoke_and_tensorboard.ipynb`
- `notebooks/02_methods_baseline_shaping_curriculum_imitation.ipynb`
- `notebooks/03_full_training_pipeline.ipynb`
- `notebooks/04_submission_and_report.ipynb`

The original all-in-one runner remains available at `notebooks/CS8803_SoccerTwos_Project.ipynb`.

### 7. Or train from the command line
python -m soccer_twos_project.training train --stage ppo_baseline --profile auto

Available organized stages are `ppo_baseline`, `ppo_shaped`, `ppo_curriculum`,
`ppo_selfplay`, and `dqn_baseline`.

Legacy starter scripts live in [examples/legacy](examples/legacy).

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under [examples/agents](examples/agents).
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m examples.agents.example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 examples.agents.example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.


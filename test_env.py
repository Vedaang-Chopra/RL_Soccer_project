import soccer_twos
from soccer_twos import EnvType

env = soccer_twos.make(
    render=False,
    variation=EnvType.team_vs_policy,
    flatten_branched=True,
    single_player=True,
    opponent_policy=lambda *_: 0,
)
print("Action space:", env.action_space)
print("Action sample:", env.action_space.sample())
obs = env.reset()
obs, rew, done, info = env.step(env.action_space.sample())
print("Step returns obs:", type(obs))
print("Info:", info)

from pettingzoo.mpe import simple_adversary_v2

env = simple_adversary_v2.env(
    N=2,
    max_cycles=25,
    continuous_actions=False,
    render_mode="human",
)

env.reset()

data = []

while not (any(env.terminations.values()) or any(env.terminations.values())):
    env.render()
    action = env.action_spaces[env.agent_selection].sample()
    d = {
        "agent_selection": env.agent_selection,
        "observation": env.observe(env.agent_selection),
        "action": action,
        "observations": {a: env.observe(a) for a in env.possible_agents},
    }
    env.step(action)
    d["rewards"] = env.rewards
    d["infos"] = env.infos
    data.append(d)
print("done")


from envs.level0.tmps_env import env_level0
import envs.level0

env_config = {
    "num_agents": 4,
    "env_class": "ClutteredGoalCycleEnv",
    "grid_size": 13,
    "max_steps": 250,
    "clutter_density": 0.15,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
    "n_bonus_tiles": 3,
    "initial_reward": True,
    "penalty": -1.5,
}

if __name__ == '__main__':
    env = env_level0(env_config)

    # Start an episode!
    # Each observation from the environment contains a list of observaitons for each agent.
    # In this case there's only one agent so the list will be of length one.
    obs_list = env.reset()

    done = False
    while not done:
        env.render()  # OPTIONAL: render the whole scene + birds eye view

        # #player_action = human.action_step(obs_list[0]['pov'])
        # # The environment expects a list of actions, so put the player action into a list
        # agent_actions = [player_action]
        #
        # next_obs_list, rew_list, done, _ = env.step(agent_actions)
        #
        # obs_list = next_obs_list
#

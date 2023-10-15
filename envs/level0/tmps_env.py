from envs.bases.geo_base import GeoGridData
from envs.bases.env_base import tmps_env_base
import envs.bases.scenario_base
import os
import envs.level0

_path = os.path.dirname(os.path.abspath(__file__))
print(_path)


class env_level0(tmps_env_base):

    def __init__(self, configs):
        print('tmps_env __init__')

        # read geo metadata and make geo config. should be done first.
        geo_grid_data = GeoGridData(_path, '300x300.bmp', '300x300_4m.png')
        configs['map_width'] = geo_grid_data.width()
        configs['map_height'] = geo_grid_data.height()
        configs['geo_grid_data'] = geo_grid_data

        # make agent config
        agents = envs.level0.make_agents(configs['num_agents'], configs['map_width'], configs['obs_box_size'], configs['init_pos'])
        configs['agents'] = agents

        # make enemy config
        enemies = envs.bases.scenario_base.make_enemies()
        configs['enemies'] = enemies

        # number of enemies
        self.n = 4

        super().__init__(configs)
        print(configs)

    def reset(self, **kwargs):
        print(kwargs)
        obs = tmps_env_base.reset(self, **kwargs)
        return obs
        
    def step(self, actions):
        # print(actions)
        return tmps_env_base.step(self, actions)


        # for agent in self.agents:
        #     agent.action = actions[num]
        #     num += 1
        #     print(agent.action)




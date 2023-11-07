from envs.bases.agent_base import AgentInterface

health = 100,  # max 100 int
has_weapon = True,  # bool
num_ammunition = 70,  # int
view_range = 2000,  # meter float
weight = 23000,  # Kg
max_speed = 65,  # 60 km/h
max_force = 100,  # Torque N.m
max_turn_rate = 0.5,  # 0.5 RPM
rough_road_degrade_rate = 80,  # 야지 주행능력 80%
blue_red_team = 0  # 0 blue 1red


def make_agents(
        num_agents,  # max 4
        map_width,
        obs_box_size,
        init_pos,
        dyn_delta_t
):
    agents = [AgentInterface(
        agent_id=[f'agent_{i}'],  # i,
        health=health,
        has_weapon=has_weapon,
        num_ammunition=num_ammunition,
        view_range=view_range,
        weight=weight,
        max_speed=max_speed,
        max_force=max_force,
        max_turn_rate=max_turn_rate,
        rough_road_degrade_rate=rough_road_degrade_rate,
        blue_red_team=blue_red_team,
        obs_box_size=obs_box_size,
        map_width = map_width,
        init_position=init_pos[i],
        dyna_delta_t = dyn_delta_t
    ) for i in range(0, num_agents)]

    return agents


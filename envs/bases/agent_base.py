import gym
import numpy as np
from enum import IntEnum
from envs.bases.geo_base import geo_data_type
from envs.bases.DynK1A1acc import DynK1A1acc


class FightModeMoveActions(IntEnum):
    # NOTE: the first 4 actions also match the directional vector
    right = 0  # Move right
    down = 1  # Move down
    left = 2  # Move left
    up = 3  # Move up
    stop = 4  # stop
    no_op = 5  # dead


class FightModeFightActions(IntEnum):
    search = 0  # find enemy
    aiming = 1  # move turret
    fire = 2  # Fire
    no_op = 3  # dead


class AgentInterface:

    class FightModeMoveActions(IntEnum):
        # NOTE: the first 4 actions also match the directional vector
        right = 0  # Move right
        down = 1  # Move down
        left = 2  # Move left
        up = 3  # Move up
        fire = 4  # Fire
        aiming = 5  # move turret
        stop = 6  # stop
        search = 7  # find enemy
        no_op = 8  # dead

    def __init__(
            self,
            agent_id=0,
            health=100,  # max 100 int
            has_weapon=True,  # bool
            num_ammunition=70,  # int
            view_range=2000,  # meter float
            weight=23000,  # Kg
            max_speed=65,  # 60 km/h
            max_force=100,  # Torque N.m
            max_turn_rate=0.5,  # 0.5 RPM
            rough_road_degrade_rate=80,  # 야지 주행능력 80%
            blue_red_team=0,  # 0 blue 1red
            obs_box_size=50,
            map_width=500,
            init_position=(0., 0.)
    ):
        self.init_position = init_position
        self.init_num_ammunition = num_ammunition

        self.map_width = map_width
        self.obs_box_size = obs_box_size
        self.agent_id = agent_id
        self.health = health
        self.has_weapon = has_weapon
        self.num_ammunition = num_ammunition
        self.view_range = view_range
        self.weight = weight
        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn_rate = max_turn_rate
        self.rough_road_degrade_rate = rough_road_degrade_rate
        self.blue_red_team = blue_red_team
        self.position = init_position

        # self.map_width = map_width
        # self.map_height = map_height

        self.done = False
        self.behavior = 0
        self.closest_detected_enemy_dist = 0.0
        self.aiming_target_id = 0
        self.most_far_placed_enemy_dist = 0.0  # regardless of detected
        self.turret_abs_direction = 0.0  # 0.0 red means 3 o'clock, clock-wise
        self.current_speed = 0.0
        self.current_orientation = 0.0 # radian +X(3 o'clock) is 0.0, clock-wise
        self.list_detected_enemy = []

        self.reward = np.zeros(2)
        self.active = True

        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(-5.0, 10.0, dtype=float),  # 가속 m/s2
                gym.spaces.Box(-15.0, 15.0, dtype=float),  # 각가속 deg/s2
                gym.spaces.Discrete(len(FightModeMoveActions)),  # 교전 시 action
                gym.spaces.Discrete(len(FightModeFightActions)),  # 교전 시 action
                gym.spaces.Discrete(2),  # 탐색 0 or 교전 모드 1  flag
            )
        )

        # observation은 계속 추가해 갈 것임. 20230820 hoyabina
        # binary 여러개 말고 Box로.. 20230825 hoyabina
        self.observation_space_not_use = gym.spaces.Dict({
                'disable': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # R
                'obstacle': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # G
                'runway': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # B
                'object': gym.spaces.MultiBinary([obs_box_size,obs_box_size], seed=42),  # O
                'height': gym.spaces.Box(0.0, 256.0, shape=(obs_box_size,obs_box_size), dtype=np.float32)  # Height
            }
        )

        # 이 형태로 observation_space를 임시로 적용함. 20230825 hoyabina
        # R, G, B, Agent, Enemy. MultiBinary 여러개 이던 것을 하나로 합침. geo_grid와 type을 일치시킴.
        self.observation_space_rgbae = gym.spaces.Box(0, 1, shape=(obs_box_size,obs_box_size,5), dtype=np.uint16)  # Height
        # objects. rendering용.  object의 크기만큼 격자를 RGB로 채운다. agent(255,255,255) enemy(255,0,255). object들끼지 겹치지 않는 특성.
        self.observation_space_object = gym.spaces.Box(0, 255, shape=(obs_box_size,obs_box_size,3), dtype=np.uint8)  # Height
        # Height
        self.observation_space_height = gym.spaces.Box(0.0, 256.0, shape=(obs_box_size,obs_box_size), dtype=np.float32)  # Height
        # # objects position
        # self.observation_space_object_pos = gym.spaces.Box(0.0, map_width, shape=(8,2), dtype=np.float32)  # objects position
        # explore memory data 처음 탐색한 곳인지 기억용
        self.observation_space_explore = gym.spaces.Box(0, 1, shape=(map_width, map_width), dtype=np.uint8)  # global
        # Dict 타입을 써야 random access를 할 수 있다.
        self.observation_space = gym.spaces.Dict(
            {
                'rgbae': self.observation_space_rgbae,
                'objects': self.observation_space_object,
                # 'objects_pos': self.observation_space_object_pos,
                'height': self.observation_space_height,
                'explore': self.observation_space_explore
            }
        )


        self.action = self.action_space.sample()
        self.observation = self.observation_space.sample()
        self.explore_memory = np.zeros((map_width, map_width), dtype=np.uint8)
        self.class_dynamic = DynK1A1acc(1.0, 1, 1, self.init_position[0], self.init_position[1], self.current_orientation)

        print(init_position)

    # def set_obs_data(self, obs_):
    #     print(obs_)

    def reset(self):
        self.done = False
        self.active = True
        self.num_ammunition = self.init_num_ammunition
        self.behavior = 0
        self.closest_detected_enemy_dist = 0.0
        self.closest_detected_enemy_id = ''
        self.aiming_target_id = 0
        self.most_far_placed_enemy_dist = 0.0
        self.turret_abs_direction = 0.0  # 0.0 deg means 12 o'clock, counter-clock-wise
        self.position = self.init_position
        self.current_speed = 0.0
        self.current_orientation = 0.0
        self.list_detected_enemy = []
        self.reward[0] = -1
        self.reward[1] = 0
        self.observation['rgbae'].fill(0)
        self.observation['objects'].fill(0)
        self.observation['height'].fill(0)
        self.observation['explore'].fill(0)
        self.class_dynamic = DynK1A1acc(0.01, 1, 1, self.init_position[0], self.init_position[1], self.current_orientation)
        # 시작 지점을 이미 탐색한 것으로 설정.
        self.observation['explore'][int(self.position[0])][int(self.position[1])] = 1

    def do_dynamic(self, ax_, alpha_):
        return self.class_dynamic.sim_once(ax_, alpha_)

    def do_move_explore(self):
        #x, y, yaw, vx, yaw_rate, ax, alpha = self.do_dynamic(1.5, 12.1)
        x, y, yaw, vx, yaw_rate, ax, alpha = self.do_dynamic(self.action[0][0], self.action[1][0])
        # print(self.agent_id, "action[0]", self.action[0], "action[1]", self.action[1], x, y, yaw, yaw_rate, ax, alpha)
        if x < 0 or x > self.map_width or y < 0 or y > self.map_width:
            self.active = False
            #  todo reward <-- Not here. activation is not factor for reward
        else:
            self.position = (x, y)
            self.current_orientation = yaw

        # todo check position is where the obstacle is. <-- env_base step 에서 check함.

    def do_step(self):
        # 매 step 시작할 때 agent의 reward 중 explore 관련 reward마 초기화.
        self.reward[0] = -1
        # self.reward[1] = 0
        # self.list_detected_enemy = []  # 이것은 매 step 초기화하면 안됨. done이 성립할 수 없게 됨.

        if len(self.action) == 5:  # validate num of arguments
            if self.action[4] == 0:  # explore mode, 0:explore 1:fight
                self.do_move_explore()  # move first
                if self.observation['explore'][int(self.position[0])][int(self.position[1])] == 0:  # when move to un-explored area
                    self.set_reward_for_explore(0)  # then set explore-reward 0

                self.observation['explore'][int(self.position[0])][int(self.position[1])] = 1  # memorize explored area

            # else:  # fight mode
            #     print(self.agent_id, "Fight mode action", self.action[4])

        else:
            print(self.agent_id, "wrong action space count..")

    # def get_env_RBG_value(self):
    #     return

    def get_observation_space(self):
        return self.observation_space

    def add_reward_for_explore(self, value):
        self.reward[0] += value

    def set_reward_for_explore(self, value):
        self.reward[0] = value

    def add_reward_for_detect_enemy(self, value):
        self.reward[1] += value

    def do_reward(self):
        return

    @property
    def health(self):
        return self._health

    @health.setter
    def health(self, health_):
        self._health = health_

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior_):
        self._behavior = behavior_

    @property
    def closest_detected_enemy_dist(self):
        return self._closest_detected_enemy_dist

    @closest_detected_enemy_dist.setter
    def closest_detected_enemy_dist(self, dist_):
        self._closest_detected_enemy_dist = dist_

    @property
    def closest_detected_enemy_id(self):
        return self._closest_detected_enemy_id

    @closest_detected_enemy_id.setter
    def closest_detected_enemy_id(self, id_):
        self._closest_detected_enemy_id = id_

    @property
    def aiming_target_id(self):
        return self._aiming_target_id

    @aiming_target_id.setter
    def aiming_target_id(self, id_):
        self._aiming_target_id = id_

    @property
    def most_far_placed_enemy_dist(self):
        return self._most_far_placed_enemy_dist

    @most_far_placed_enemy_dist.setter
    def most_far_placed_enemy_dist(self, dist_):
        self._most_far_placed_enemy_dist = dist_

    @property
    def turret_abs_direction(self):
        return self._turret_abs_direction

    @turret_abs_direction.setter
    def turret_abs_direction(self, deg_):
        self._turret_abs_direction = deg_

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos_):
        self._position = pos_

    @property
    def current_speed(self):
        return self._current_speed

    @current_speed.setter
    def current_speed(self, speed_):
        self._current_speed = speed_

    @property
    def current_orientation(self):
        return self._current_orientation

    @current_orientation.setter
    def current_orientation(self, orientation_):
        self._current_orientation = orientation_

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, act_):
        self._active = act_

    def num_detected_enemy(self):
        return len(self.list_detected_enemy)

    def append_detected_enemy(self, enemy, reward_val):
        if enemy.enemy_id not in self.list_detected_enemy:
            self.list_detected_enemy.append(enemy.enemy_id)
            self.add_reward_for_detect_enemy(reward_val) # 기존에 찾아낸 적군이 아닐 경우, 보상 추가.. todo ??? 내가 아닌 다른 아군이 찾아낸 적군을 내가 다시 찾아 냈을 경우에는 보상이 없는가?

    # 시야에서 사리지면 detected 되지 않은 것으로 변경한다.
    # 이후, 다시 detected 되면 또 보상을 하게 되는데 맞는가?
    def remove_detected_enemy(self, enemy):
        if enemy.enemy_id in self.list_detected_enemy:
            self.list_detected_enemy.remove(enemy.enemy_id)





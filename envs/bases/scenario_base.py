
enemy_data = {
    'num_enemy': 4,
    'object_width': 3,
    'object_length': 7,
    'enemy_init_pos': ((88, 69), (190, 120), (67, 220), (195, 220))
}


class EnemyInterface:

    def __init__(
            self,
            enemy_id,
            health=100,
            init_position=(0, 0)
    ):
        self.init_position = init_position
        self.enemy_id = enemy_id
        self.health = health
        self.position = init_position
        self.active = True

    def add_damage(self, damage_):
        self.health -= damage_

    def reset(self):
        self.active = True
        self.position = self.init_position
        self.health = 100

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos_):
        self._position = pos_

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, act_):
        self._active = act_




def make_enemies():
    enemies = [EnemyInterface(
        enemy_id=[f'enemy_{i}'],  # i,
        health=100,
        init_position=enemy_data['enemy_init_pos'][i]
    ) for i in range(0, enemy_data['num_enemy'])]

    return enemies


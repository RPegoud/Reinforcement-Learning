import numpy as np
import pandas as pd

class Env():
    def __init__(self) -> None:
        self.coordinates = {
            'A': ((6,1),),
            'W': ((3, range(3)), (range(5,8), 3)),
            'T': ((range(3), 8), (3, range(8,12))),
            'P': ((6, 10), (0,11)),
            'G': ((1,10),)
        }
        self.generate_grid()
        self.generate_reward_map()

    def generate_grid(self):
        grid = np.zeros((8,12), dtype=np.object0)
        for key in list(self.coordinates.keys()):
            for values in self.coordinates[key]:
                grid[values] = key
        self.grid = pd.DataFrame(grid)        

    def generate_reward_map(self):
        reward_map = np.zeros((8,12), dtype=np.float32)
        reward_map[self.coordinates['G'][0]] = 1
        self.reward_map = pd.DataFrame(reward_map)

    def get_reward(self, coordinates:tuple):
        return self.reward_map.loc[coordinates]
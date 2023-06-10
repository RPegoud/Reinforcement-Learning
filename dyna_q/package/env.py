import pandas as pd
import numpy as np

class Env():
    def __init__(self) -> None:
        self.coordinates = {
            'A': ((6, 1),),
            'W': ((3, range(3)), (range(5, 8), 3)),
            'T': ((range(3), 8), (3, range(8, 12))),
            'P': ((6, 10), (0, 11)),
            'LP': ((1, 2),),
            'G': ((1, 10),)
        }
        self.generate_grid()
        self.generate_reward_map()

    def generate_grid(self):
        grid = np.zeros((8, 12), dtype=np.object0)
        for key in list(self.coordinates.keys()):
            for values in self.coordinates[key]:
                grid[values] = key
        self.grid = pd.DataFrame(grid)

    def generate_reward_map(self):
        reward_map = np.zeros((8, 12), dtype=np.float32)
        reward_map[self.coordinates['G'][0]] = 1
        self.reward_map = pd.DataFrame(reward_map)

    def get_reward(self, coordinates: tuple = None, reverse:bool=True):
        """
        Queries the reward map and returns the reward associated to the coordinates
        @reverse: - if the coordinates are derived from the agent state, set reverse to True
                    They have to be reversed before querying the dataframe as 
                    agent(state) = (x,y) = pd.Dataframe.loc(y,x) with (x,y) = (col, row)
                  - if the coordinates come from env.coordinates, then set reverse to False
                    as they are already in the (row, col) format
        """
        if reverse: return self.reward_map.loc[coordinates[::-1]]
        else: return self.reward_map.loc[coordinates]

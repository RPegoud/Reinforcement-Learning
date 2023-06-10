import numpy as np
from package import Env

class Agent():
    def __init__(self,
                 gamma: float = 1.0,  # undiscounted task
                 step_size: float = 0.1,
                 epsilon: float = 0.1,
                 ) -> None:
        self.env = Env()
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.n_actions = 4
        self.actions = list(range(self.n_actions))
        self.last_action = -1
        self.last_state = -1
        self.n_states = self.env.grid.size
        self.start_position = self.coord_to_state(
            list(reversed(self.env.coordinates.get('A')[0])))
        self.position = self.start_position
        self.q_values = self.init_q_values()
        self.random_generator = np.random.RandomState(seed=17)
        self.done = False

    def coord_to_state(self, coordinates: tuple) -> int:
        return coordinates[0]*10 + coordinates[1]

    def state_to_coord(self, state: int):
        return (int(state//10), state % 10)

    def init_q_values(self) -> None:
        q_values = {}
        rows, cols = self.env.grid.index, self.env.grid.columns
        for col in cols:
            for row in rows:
                q_values[self.coord_to_state((col, row))] = np.zeros(
                    4, dtype=np.float32)
        return q_values

    def update_coord(self, coord: tuple, action: int) -> tuple:
        """
        Given a state and an action, moves the agent on the grid
        If the agent encounters a wall or the edge of the grid, the initial position is returned
        If the agent falls into a whole ('T') or finds the goal ('G'), the episode ends
        """
        assert action in [0, 1, 2, 3], f"Invalid action {action}"
        x, y = coord
        if action == 0:
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            x -= 1

        # if the action moves the agent out of bounds
        if x not in range(0, self.env.grid.shape[1]):
            return coord
        if y not in range(0, self.env.grid.shape[0]):
            return coord

        # /!\ when parsing the dataframe x and y are reversed
        # if the agent bumps into a wall
        if self.env.grid.loc[y, x] == 'W':
            print('W')
            return coord
        # if the agent goes through the portal
        if self.env.grid.loc[y, x] == 'P':
            print('P')
            return (11, 0)
        # if the agent encounters a terminal state (whole or goal)
        if self.env.grid.loc[y, x] in ['T', 'G']:
            self.done = True

        self.position == (x, y)
        return (x, y)

    def update_state(self, state, action):
        assert action in [0, 1, 2, 3], f"Invalid action: {action}, should be in {[i for i in range(4)]}"
        coord = self.state_to_coord(state)
        updated_coord = self.update_coord(coord, action)
        updated_state = self.coord_to_state(updated_coord)
        self.position = updated_state
        return updated_state

    def argmax(self, q_values) -> int:
        """
        Selects the index of the highest action value
        Breaks ties randomly
        """
        return self.random_generator.choice(np.flatnonzero(q_values == np.max(q_values)))

    def epsilon_greedy(self, state) -> int:
        """
        Returns an action using an epsilon-greedy policy 
        w.r.t. the current action-value function
        """
        # convert the state to coordinates to query the q_values
        if self.random_generator.rand() < self.epsilon:
            action = self.random_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)
        return action

    def agent_start(self, state: int):
        """
        Called at the start of an episode, takes the first action 
        given the initial state
        """
        self.past_state = state
        self.past_action = self.epsilon_greedy(state)
        # take the action
        self.update_state(state, self.past_action)
        return self.past_action

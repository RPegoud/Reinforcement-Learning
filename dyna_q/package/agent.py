import numpy as np
from package import Env

class Agent():
    def __init__(self, 
                gamma:float=1.0, # undiscounted task 
                step_size:float=0.1, 
                epsilon:float=0.1, 
                planning_steps:int=100
                ) -> None:
        self.env = Env()
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.n_actions = 4
        self.actions = list(range(self.n_actions))
        self.last_action = -1
        self.last_state = -1
        self.n_states = self.env.grid.size
        self.start_position = self.coord_to_state(self.env.coordinates.get('A')[0])
        self.q_values = self.init_q_values()
        self.model = {} # model[state][action] = (new state, reward)
        self.random_generator = np.random.RandomState(seed=17)
        self.done = False

    def coord_to_state(self, coordinates:tuple) -> int:
        return coordinates[0]*10 + coordinates[1]
    
    def state_to_coord(self, state:int):
        return (int(state//10), state%10)
    
    def init_q_values(self) -> None:
        q_values = {}
        rows, cols = self.env.grid.index, self.env.grid.columns
        for col in cols:
            for row in rows:    
                q_values[self.coord_to_state((col,row))] = np.zeros(4, dtype=np.float32)
        return q_values

    def update_model(self, last_state:int, last_action:int, state:int, reward:int) -> None:
        """
        Adds a new transition to the model, if the state is encountered for 
        the first time, creates a new key
        """
        try: self.model[last_state][last_action] = (state, reward)
        except KeyError:
            self.model[last_state] = {}
            self.model[last_state][last_action] = (state, reward)

    def update_coord(self, coord, action):
        assert action in [0,1,2,3], f"Invalid action {action}"
        x,y = coord
        if action == 0:
            y-=1
        elif action == 1:
            x+=1
        elif action == 2:
            y+=1
        elif action == 3:
            x-=1

        # if the action moves the agent out of bounds
        if x not in range(0, self.env.grid.shape[1]):
            print(f'x out of bounds: {x} : {self.env.grid.shape[1]}')
            return coord
        if y not in range(0, self.env.grid.shape[0]):
            print(f'y out of bounds: {y} : {self.env.grid.shape[0]}')
            return coord
        return (x,y)
    
    def update_state(self, state, action):
        assert action in [0,1,2,3], f"Invalid action {action}"
        coord = self.state_to_coord(state)
        updated_coord = self.update_coord(coord, action)
        return self.coord_to_state(updated_coord)
    
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
    
    def planning_step(self):
        """
        Performs planning (indirect RL)
        """
        for _ in range(self.planning_steps):
            # select a visited state
            planning_state = self.random_generator.choice(list(self.model.keys()))
            # select a recorded action
            planning_action = self.random_generator.choice(list(self.model[planning_state].keys()))
            # get the predicted next state and reward
            next_state, reward = self.model[planning_state][planning_action]
            # update the values in case of non-terminal state

            # update the values in case of terminal state
            if next_state == -1:
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward - update)
                self.q_values[planning_state][planning_action] = update   
            # update the values in case of non-terminal state
            else:   
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward + self.gamma * np.max(self.q_values[next_state]) - update)
                self.q_values[planning_state][planning_action] = update

    def agent_start(self, state:int):
        """
        Called at the start of an episode, takes the first action 
        given the initial state
        """
        self.past_state = state
        self.past_action = self.epsilon_greedy(state)
        
        return self.past_action
    
    def step(self, state:int, reward:int, ):
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (reward + self.gamma * np.max(self.q_values[state]) - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update
        self.update_model(self.past_state, self.past_action, state, reward)
        # planning step
        self.planning_step()
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action
    
    def agent_end(self, reward:int):
        """
        Called once the agent reaches a terminal state 
        """
        # direct RL update for a terminal state
        update = self.q_values[self.past_state, self.past_action]
        update += self.step_size * (reward - update)
        self.q_values[self.past_state, self.past_action] = update
        # model update with next_action = -1
        self.update_model(self.past_state, self.past_action, -1, reward)
        # planning step
        self.planning_step()

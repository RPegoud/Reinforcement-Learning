import numpy as np
from package import Env
from package import Agent, plot_bar_chart, plot_heatmap
import pandas as pd
from tqdm.auto import tqdm
from plotly.subplots import make_subplots

class Agent():
    def __init__(self,
                 gamma: float = 0.1,  # undiscounted task
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
        self.start_position = self.coord_to_state(self.env.coordinates.get('A')[0][::-1])
        self.position = self.start_position
        self.q_values = self.init_state_action_dict()
        self.state_visits = self.init_state_dict(initial_value=0)
        self.random_generator = np.random.RandomState(seed=17)
        self.done = False
        self.n_steps = []

    def reset(self):
        self.done = False
        self.position = self.start_position
        self.last_action = -1
        self.last_state = -1

    def coord_to_state(self, coordinates: tuple) -> int:
        return coordinates[0]*10 + coordinates[1]

    def state_to_coord(self, state: int):
        return (int(state//10), state % 10)

    def init_state_action_dict(self) -> dict:
        output_dict = {}
        rows, cols = self.env.grid.index, self.env.grid.columns
        for col in cols:
            for row in rows:
                output_dict[self.coord_to_state((col, row))] = np.zeros(4, dtype=np.float32)
        return output_dict

    def init_state_dict(self, initial_value) -> dict:
        output_dict = {}
        rows, cols = self.env.grid.index, self.env.grid.columns
        for col in cols:
            for row in rows:
                output_dict[self.coord_to_state((col, row))] = initial_value
        return output_dict

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
            return coord
        # if the agent goes through the portal
        if self.env.grid.loc[y, x] == 'P':
            return (11, 0)
        # if the agent encounters falls into a trao
        if self.env.grid.loc[y, x] == 'T':
            self.done = True
        # if the agent finds the treasure
        if self.env.grid.loc[y,x] == 'G':
            self.done = True

        self.position == (x, y)
        return (x, y)

    def update_state(self, state, action) -> int:
        assert action in [0, 1, 2, 3], f"Invalid action: {action}, should be in {[i for i in range(4)]}"
        coord = self.state_to_coord(state)
        updated_coord = self.update_coord(coord, action)
        updated_state = self.coord_to_state(updated_coord)
        self.position = updated_state
        self.state_visits[self.position] += 1
        return updated_state

    def argmax(self, action_values) -> int:
        """
        Selects the index of the highest action value
        Breaks ties randomly
        """
        return self.random_generator.choice(np.flatnonzero(action_values == np.max(action_values)))

    def epsilon_greedy(self, state) -> int:
        """
        Returns an action using an epsilon-greedy policy 
        w.r.t. the current action-value function
        """
        # probability of epsilon of picking a random action
        if self.random_generator.rand() < self.epsilon:
            action = self.random_generator.choice(self.actions)
        # picking the action greedily w.r.t state action values
        else:
            action_values = self.q_values[state]
            action = self.argmax(action_values)
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

class Q_learning_Agent(Agent):
    def __init__(self, 
                 gamma: float = 1, 
                 step_size: float = 0.1, 
                 epsilon: float = 0.1
                 ) -> None:
        super().__init__(gamma, step_size, epsilon)
        self.name = "Q-learning"    

    def step(self, state: int, reward: int) -> None:
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * \
            (reward + self.gamma * np.max(self.q_values[state]) - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update
        self.update_model(self.past_state, self.past_action, state, reward)
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        self.update_state(state, action)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_end(self) -> None:
        """
        Called once the agent reaches a terminal state 
        """
        terminal_coordinates = self.state_to_coord(self.position)
        # the coordinates must be reversed when querying the dataframe
        reward = self.env.get_reward(terminal_coordinates)
        # direct RL update for a terminal state
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (reward - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update with next_action = -1
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.state_visits[self.past_state] += 1
    
    def play_episode(self) -> None:
        """
        Plays one episode (agent_start, agent_step, agent_end)
        Records the number of step during the episode
        """
        self.agent_start(self.start_position)
        episode_steps = 1
        while not self.done:
            self.step(self.position, 
                      self.env.get_reward(self.state_to_coord(self.position)))
            episode_steps+=1
        if self.position == self.coord_to_state(self.env.coordinates.get('G')[0]):
            self.cumulative_reward +=1
        self.n_steps.append(episode_steps)
        self.agent_end()
        self.reset()
    
    def fit(self, n_episode:int, plot_progress:list=None) -> None:
        """
        Plays n_episode episodes
        @plot_progress (list): calls the plot_agent_performances function for each episode in the list
        """
        self.episode_played = 0
        for idx in tqdm(range(n_episode), position=0, leave=True):
            if self.episode_played == 100:
                 self.env.activate_late_portal()
            self.play_episode()
            self.episode_played +=1
            self.rewards.append(self.cumulative_reward)
            if plot_progress != None:
                if idx in plot_progress:
                    self.plot_agent_performances()
        
    def state_to_matrix(self, dictionary:dict) -> pd.DataFrame:
        """
        Convert a dictionary of states (e.g. q_values and n_visits) to a 
        matrix representation matching the environment's grid representation
        """
        key_val = [(self.state_to_coord(key)[::-1], np.max(values)) \
            for (key, values) in list(dictionary.items())]
        matrix = pd.DataFrame(np.zeros((8,12)))
        for key, value in key_val:
            matrix.loc[key] = value
        matrix.index = [str(i) for i in matrix.index]
        matrix.columns = [str(c) for c in matrix.columns]
        return matrix
    
    def plot_agent_performances(self) -> None:
        q_values = self.state_to_matrix(self.q_values)
        state_visits = self.state_to_matrix(self.state_visits)

        n_steps = pd.DataFrame(self.n_steps, columns=['steps'])
        n_steps['is_optimal'] = np.where(n_steps.steps == 17,'#EF553B', '#636EFA')

        # create the plots
        heatmap1 = plot_heatmap(q_values, **{'colorbar':dict(x=0.45, y=0.78, len=0.473)})
        heatmap2 = plot_heatmap(state_visits, **{'colorbar':dict(x=1, y=0.78, len=0.473)})
        bar_chart = plot_bar_chart(n_steps, attribute='steps', color='is_optimal')

        # Create subplot figure
        fig = make_subplots(rows=2, cols=2, shared_xaxes=False, 
                            vertical_spacing=0.13, specs=[[{}, {}],[{"colspan": 2}, None]],
                            subplot_titles=("State value function","Number of total visits", "Number of steps per episode")
                            )

        # Add the heatmaps and bar chart to the subplot figure
        fig.add_trace(heatmap1, row=1, col=1)
        fig.add_trace(heatmap2, row=1, col=2)
        fig.add_trace(bar_chart, row=2, col=1)

        title = f"Agent: {self.name}, Number of episodes: {self.episode_played}<br>\
        <span style='font-size: 13px'>Model parameters: [learning rate: {self.step_size}, epsilon: {self.epsilon}, discount: {self.gamma}]</span>"
        fig.update_layout(height=900, width=1200, title=title)

        fig.show()

class Dyna_Q_Agent(Q_learning_Agent):
    def __init__(self, 
                 gamma: float = 1, 
                 step_size: float = 0.1, 
                 epsilon: float = 0.1, 
                 planning_steps: int = 100) -> None:
        super().__init__(gamma, step_size, epsilon)
        self.planning_steps = planning_steps
        self.model = {}  # model[state][action] = (new state, reward)
        self.name = "Dyna Q"    

    def update_model(self, last_state: int, last_action: int, state: int, reward: int) -> None:
        """
        Adds a new transition to the model, if the state is encountered for 
        the first time, creates a new key
        """
        try:
            self.model[last_state][last_action] = (state, reward)
            self.state_visits[last_state] += 1
        except KeyError:
            self.model[last_state] = {}
            self.model[last_state][last_action] = (state, reward)

    def planning_step(self) -> None:
            """
            Performs planning (indirect RL)
            """
            for _ in range(self.planning_steps):
                # select a visited state
                planning_state = self.random_generator.choice(
                    list(self.model.keys()))
                # select a recorded action
                planning_action = self.random_generator.choice(
                    list(self.model[planning_state].keys()))
                # get the predicted next state and reward
                next_state, reward = self.model[planning_state][planning_action]
                # update the values in case of terminal state
                if next_state == -1:
                    update = self.q_values[planning_state][planning_action]
                    update += self.step_size * (reward - update)
                    self.q_values[planning_state][planning_action] = update
                # update the values in case of non-terminal state
                else:
                    update = self.q_values[planning_state][planning_action]
                    update += self.step_size * (reward + self.gamma \
                                                * np.max(self.q_values[next_state]) - update)
                    self.q_values[planning_state][planning_action] = update

    def step(self, state: int, reward: int) -> None:
        """
        A step performed by the agent
        """
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * \
            (reward + self.gamma * np.max(self.q_values[state]) - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update
        self.update_model(self.past_state, self.past_action, state, reward)
        # planning step
        self.planning_step()
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        self.update_state(state, action)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action

    def agent_end(self) -> None:
        """
        Called once the agent reaches a terminal state 
        """
        terminal_coordinates = self.state_to_coord(self.position)
        # the coordinates must be reversed when querying the dataframe
        reward = self.env.get_reward(terminal_coordinates)
        # direct RL update for a terminal state
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * (reward - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update with next_action = -1
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.state_visits[self.past_state] += 1
        # planning step
        self.planning_step()

class Dyna_Q_plus_Agent(Dyna_Q_Agent):
    """
    In Dyna-Q+, a bonus reward is given for actions that haven't been tried for a long time
    as there are greater chances that the environment dynamics have changed
    The number of transitions since the last time (state, action) was tried is given by tau(state, action)
    The associated reward is given by: reward + kappa * sqrt(tau(state, action))
    """
    def __init__(self, 
                 gamma: float = 1, 
                 step_size: float = 0.1, 
                 epsilon: float = 0.1, 
                 planning_steps: int = 100,
                 kappa: float = 1e-3,
                 ) -> None:
        super().__init__(gamma, step_size, epsilon, planning_steps)
        self.name = "Dyna-Q+"
        self.kappa = kappa
        self.tau = self.init_state_action_dict()
        
    def update_model(self, last_state: int, last_action: int, state: int, reward: int) -> None:
        """
        Overwrite the Dyna-Q update_model function
        Now, when we visit a state for the first time, all the action that were not selected
        are initialized with 0, they will be updated at each time steps according to the Dyna-Q+ algorithm
        """
        if last_state not in self.model:
            self.model[last_state] = {last_action : (state, reward)}
        for action in self.actions:
            if action != last_action:
                self.model[last_state][action] = (last_state, 0)
        else:
            self.model[last_state][last_action] = (state, reward) 

    def update_tau(self, state:int, action:int) -> None:
        for key in list(self.tau.keys()):
            self.tau[key] +=1
        self.tau[state][action] = 0
            

    def planning_step(self) -> None:  
        """
        Overwrite the Dyna-Q planning_step function
        Performs planning (indirect RL) and adds a bonus to the transition reward
        The bonus is given by kappa * sqrt(tau(state, action))
        """
        for _ in range(self.planning_steps):
            # select a visited state
            planning_state = self.random_generator.choice(list(self.model.keys()))
            # select a recorded action
            planning_action = self.random_generator.choice(list(self.model[planning_state].keys()))
            # get the predicted next state and reward
            next_state, reward = self.model[planning_state][planning_action]
            # add the bonus reward
            reward += self.kappa * np.sqrt(self.tau[planning_state][planning_action])
            # update the values in case of terminal state
            if next_state == -1:
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward - update)
                self.q_values[planning_state][planning_action] = update
            # update the values in case of non-terminal state
            else:
                update = self.q_values[planning_state][planning_action]
                update += self.step_size * (reward + self.gamma \
                                            * np.max(self.q_values[next_state]) - update)
                self.q_values[planning_state][planning_action] = update
    
    def step(self, state: int, reward: int) -> None:
        """
        Overwrite the Dyna-Q step function
        At every step, we increment the last visit counter for every state action by 1
        The current state action pair is reset to 0
        """
        # direct RL update
        update = self.q_values[self.past_state][self.past_action]
        update += self.step_size * \
            (reward + self.gamma * np.max(self.q_values[state]) - update)
        self.q_values[self.past_state][self.past_action] = update
        # model update
        self.update_model(self.past_state, self.past_action, state, reward)
        # planning step
        self.planning_step()
        # action selection using the e-greedy policy
        action = self.epsilon_greedy(state)
        self.update_tau(state, action)
        self.update_state(state, action)
        # before performing the action, save the current state and action
        self.past_state = state
        self.past_action = action

        return self.past_action
from package import Agent, plot_bar_chart, plot_heatmap
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from plotly.subplots import make_subplots


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
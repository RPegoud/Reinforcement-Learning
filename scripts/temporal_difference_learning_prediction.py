import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm

N_NODES = 5
LOWER_BOUND = (-N_NODES//2)
UPPER_BOUND = (N_NODES//2)+1
print(f"Nodes: {N_NODES}\nLower bound: {LOWER_BOUND}\nUpper bound: {UPPER_BOUND}")
alphas = [.15, .1, .05]
GAMMA = 0.9

class Env():
    def __init__(self) -> None:
        # transition reward for each action pair
        # for each state, contains [reward_left, reward_right]
        rewards = np.zeros((N_NODES, 2), dtype=int)
        # only moving right from the last state gives a reward
        rewards[-1,1] = 1
        self.rewards = rewards
        self.states = [i for i in range(LOWER_BOUND, UPPER_BOUND+1)]
        # nodes = {state:transition_rewards[left, right]}
        self.nodes = {
            state:reward for state,reward in zip(self.states[1:],rewards)
        }

class Agent():
    def __init__(self) -> None:
        self.position = 0
        self.done = False
        self.action = None
        self.env = Env()
        self.values = {state:value for state, value in \
                                        zip(self.env.states, np.zeros(len(self.env.states)))}
        
    def step(self):
        """
        Performs a step according to the uniform random policy
        """
        s = self.position
        self.action = np.random.choice([0,1])
        if self.action == 1:
            self.position += 1
        else: self.position -= 1
        
        # check terminal state
        if self.position in [LOWER_BOUND, UPPER_BOUND]:
            self.done = True

        # s, s_prime, action
        return s, self.position, self.action
    
    def reset(self):
        self.position = 0
        self.done = False
        return self.position
    
    def rmse(self):
        """
        Returns the root mean squarred error 
        The error is considered as the difference between the prediction
        and the true state value
        """
        true_value_function = np.arange(1,N_NODES+1)/(N_NODES+1)
        delta = list(self.values.values())[1:-1] - true_value_function
        rmse = np.sqrt(np.mean(np.power(delta, 2)))
        return np.round(rmse,4)

class TD_Agent(Agent):
    def __init__(self, alpha:float, gamma:float) -> None:
        self.alpha = alpha
        self.gamma = gamma
        super().__init__()

    def update_value(self, s, s_prime, reward):
        """
        Update the state values according to the TD(0) algorithm
        The terminal state has no value, therefore the update is modified for s_prime = terminal
        """
        if not self.done:
            self.values[s] = self.values[s] + self.alpha*(reward + self.gamma * self.values[s_prime] - self.values[s])
        else: 
            self.values[s] = self.values[s] + self.alpha*(reward - self.values[s])

    def play_episode(self):
        s = self.reset() # initial state
        while True:
            s, s_prime, action = self.step()
            reward = self.env.nodes.get(s)[action]
            self.update_value(s, s_prime, reward)
            if self.done:
                break
            s = s_prime

class MC_Agent(Agent):
    def __init__(self, alpha:float, gamma:float) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.returns = []
        super().__init__()

    def update_value():
        """
        Update the values according to every-visit Monte Carlo
        """
        pass

    def play_episode(self):
        s = self.reset() # initial state
        G = 0 # initial estimated return
        episode = {
            'states' : [s],
            'actions' : [],
            'rewards' : [],
        }
        while True:
            s, s_prime, action = self.step()
            episode['states'].append(s_prime)
            episode['actions'].append(action)
            reward = self.env.nodes.get(s)[action]
            episode['rewards'].append(reward)
            if self.done:
                break
        

def get_runs(alphas, n_runs=10, n_episodes=100):
    """
    For each value of alpha, performs n_runs of n_episodes and records the error
    Runs for a specific alpha value are exported in a dataframe
    """
    errors = {}
    dataframes = {}
    for alpha in alphas:
        print(f'Computing errors for alpha: {alpha}')
        for run in tqdm(range(n_runs)):
            errors[f'{alpha}_{run}'] = []
            dataframes[f'{alpha}'] = pd.DataFrame()
            a = TD_Agent(alpha=alpha, gamma=GAMMA)
            for _ in range(n_episodes):
                a.play_episode()
                errors[f'{alpha}_{run}'].append(a.rmse())
            dataframes[f'{alpha}'] = pd.concat((dataframes[f'{alpha}'], pd.Series(errors)), axis=1)
    return dataframes


def average_run(dataframe:pd.DataFrame):
    """
    Return the average error per episode for each alpha
    """
    return pd.DataFrame(dataframe.to_list(), columns=range(100)).apply(np.mean)

def plot_runs(dataframes):
    """
    Plots the average error per episode for each alpha
    """
    averaged_runs = pd.DataFrame([average_run(dataframes[str(alpha)][0]) for alpha in alphas]).T
    averaged_runs.columns = alphas
    fig = px.line(averaged_runs,
            labels={'index':'Number of episodes', 'value':'RMSE', 'variable':'Alpha'},
            title="Root mean square errors of value estimations for 10 runs of TD(0)",
            height=600
            )
    fig.update_layout()
    fig.show()

if __name__ == "__main__":
    errors = get_runs(alphas, n_runs=1000, n_episodes=100)
    plot_runs(errors)

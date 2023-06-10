from package import Agent, Env
import numpy as np

class Dyna_Q_Agent(Agent):
    def __init__(self, 
                 gamma: float = 1, 
                 step_size: float = 0.1, 
                 epsilon: float = 0.1, 
                 planning_steps: int = 100) -> None:
        super().__init__(gamma, step_size, epsilon)
        self.planning_steps = planning_steps
        self.model = {}  # model[state][action] = (new state, reward)

    def update_model(self, last_state: int, last_action: int, state: int, reward: int) -> None:
        """
        Adds a new transition to the model, if the state is encountered for 
        the first time, creates a new key
        """
        try:
            self.model[last_state][last_action] = (state, reward)
        except KeyError:
            self.model[last_state] = {}
            self.model[last_state][last_action] = (state, reward)

    def planning_step(self):
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
                    update += self.step_size * \
                        (reward + self.gamma *
                        np.max(self.q_values[next_state]) - update)
                    self.q_values[planning_state][planning_action] = update

    def step(self, state: int, reward: int):
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

    def agent_end(self):
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
        # planning step
        self.planning_step()
    
    def play_episode(self):
        self.agent_start(self.start_position)
        while not self.done:
            print((self.position, self.state_to_coord(self.position)))
            action = self.step(self.position, 
                    self.env.get_reward(self.state_to_coord(self.position)))
            print(action)
        self.agent_end()
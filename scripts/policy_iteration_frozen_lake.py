import gym
import numpy as np
import pandas as pd
import seaborn as sns


def policy_iteration(p, n_states, n_actions, gamma, epsilon, max_iter=100):
    """
    @p: environment model: p[state][action] returns a tuple
        (p_transition, next_state, reward, done) for each new state s'
    @gamma: discount factor
    @epsilon: float indicating the tolerance level for policy convergence
    @max_iter: maximum number of iterations before interrupting training
    """
    # initialize the new policy
    value_function = np.zeros(n_states)
    policy = np.zeros(n_states)

    policy_stable = False
    n_iter = 0

    while not policy_stable and n_iter < max_iter:
        # perform policy evaluation and policy improvement
        value_function = policy_evaluation(p, n_states, policy, gamma, epsilon)
        new_policy = policy_improvement(p, n_states, n_actions, value_function, gamma)

        # compare the old and the updated policy
        delta = new_policy - policy

        # if the policy stops improving, i.e. old_policy = new_policy
        # the policy converged and we stop the iteration
        if np.linalg.norm(delta) == 0:
            print("Policy converged!")
            policy_stable = True

        policy = new_policy
        n_iter += 1

    if n_iter == max_iter:
        print("Policy iteration didn't converge, exiting")
        exit()

    return value_function, policy


def policy_evaluation(p, n_states, policy, gamma, epsilon, max_iter=100):
    """
    @p: environment model: p[state][action] returns a tuple
        (p_transition, next_state, reward, done) for each new state s'
    @policy: list containing one action to take in each state
    @gamma: discount factor
    @epsilon: float indicating the tolerance level for policy convergence
    @max_iter: maximum number of iterations before interrupting training
    """

    # initialization
    value_function = np.zeros(n_states)
    error = 1
    n_iter = 0

    # repeat the algorithm until convergence or maximum nummber
    # of iterations
    while error > epsilon and n_iter < max_iter:
        # initialize the new value function
        new_value_function = np.zeros(n_states)
        # sweep through all states
        # pick the policy's action for that state
        # get all the transition probabilities, rewards and subsequent states s'
        # for all transitions, compute the Bellman update for the value function
        for state in range(n_states):
            action = policy[state]
            transitions = p[state][action]
            for transition in transitions:
                prob, s_prime, reward, _ = transition
                new_value_function[state] += prob * (
                    reward + gamma * value_function[s_prime]
                )
        # check for convergence
        # update the value function with and increment the iteration counter
        error = np.max(np.abs(new_value_function - value_function))
        value_function = new_value_function
        n_iter += 1

    # if the maximum number of iterations is reached, exit the function
    if n_iter > max_iter:
        print("Policy evaluation didn't converge, exiting")
        exit()

    return value_function


def policy_improvement(p, n_states, n_actions, value_from_policy, gamma):
    """
    @p: environment model: p[state][action] returns a tuple
        (p_transition, next_state, reward, done) for each new state s'
    @value_from_policy: value estimation of each state before policy improvement
    @gamma: discount factor
    """

    # initialize the new policy
    new_policy = np.zeros(n_states, dtype=int)

    # iterate through all states and actions
    for state in range(n_states):
        # initialize the q values for each state
        q_values = np.zeros(n_actions)

        for action in range(n_actions):
            transitions = p[state][action]
            for transition in transitions:
                prob, s_prime, reward, done = transition
                # get Vpi for the next state s'
                old_estimate = value_from_policy[s_prime]
                # estimate the q value for the current action
                q_values[action] += prob * (reward + gamma * old_estimate)
        # make the policy greedy with regards to the q values
        best_action = np.argmax(q_values)
        new_policy[state] = best_action

    return new_policy


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    p = env.P
    value_function, policy = policy_iteration(
        p, n_states, n_actions, gamma=0.999, epsilon=1e-5
    )
    print(f"Value function: {value_function}")
    print(f"Policy: {policy}")
    sns.heatmap(pd.DataFrame(value_function.reshape(8, -1)))

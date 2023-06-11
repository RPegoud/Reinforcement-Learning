from package import Dyna_Q_Agent

if __name__ == "__main__":
    a = Dyna_Q_Agent(planning_steps=100, epsilon=0.1, gamma=0.9, step_size=0.25)
    a.fit(n_episode=400, plot_progress=[100, 250, 400])

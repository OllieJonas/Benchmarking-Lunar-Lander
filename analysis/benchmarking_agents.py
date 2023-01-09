import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.interpolate import interp1d

## Plotting graphs for RL

##     graph_num   = number of the graph being produced
##     data_points = array containing arrays of data points to be plotted
##     data_style  = array containing arrays of information and styles for data_points [colour, label, cobf colour]
##     filename    = name of the file containing the output graph
##     graph_title = title of the output graph
def plot_graph(graph_num, data_points, data_style, filename, graph_title):
    fig, ax = plt.subplots()
    ep_counter = np.arange(1,1001)
    for i in range(len(data_points)):
        # Plot original data points
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.set_title(graph_title)
        ax.plot(data_points[i], color=data_style[i][0], alpha=0.35)

        # Curve of Best Fit code
        gap = 50
        cobf = np.average(np.array(data_points[i]).reshape(-1, gap), axis=1)
        cobf_eps = ep_counter[0::gap]
        x_new = np.linspace(cobf_eps.min(), cobf_eps.max(), 500)
        f = interp1d(cobf_eps, cobf, kind="quadratic")
        y_smooth = f(x_new)
        ax.plot(x_new, y_smooth, color=data_style[i][2], label=data_style[i][1])
        ax.legend(loc="lower right")
        
    fn = str(graph_num) + filename
    plt.savefig(fn)
    plt.clf()
    return graph_num + 1

# [Cumulative, Average, No Time-Steps]
def read_file(filename, reward):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            data.append(row)
    for each in data:
        reward.append(each[1])
    return reward


# Data for Random agent over 1000 eps
random_reward = []
random_reward = read_file("random_results.csv", random_reward)
style_random = ["lightgray", "Random", "grey"]

# Data for Sarsa agent over 1000 eps
deep_sarsa_reward = []
deep_sarsa_reward = read_file("deep_sarsa_results.csv", deep_sarsa_reward)
style_deep_sarsa = ["lightcoral", "Deep Sarsa", "red"]

# Data for Deep Sarsa agent over 1000 eps
sarsa_reward = np.random.uniform(size = 1000, low = -200, high = 200)
style_sarsa = ["lightgreen", "Sarsa", "green"]

# Data for DQN agent over 1000 eps
dqn_reward = []
dqn_reward = read_file("dqn_results.csv", dqn_reward)
style_dqn = ["peachpuff", "DQN", "orange"]

# Data for DDPG agent over 1000 eps
ddpg_reward = []
ddpg_reward = read_file("ddpg_results.csv", ddpg_reward)
style_ddpg = ["lightskyblue", "DDPG", "dodgerblue"]

# Counter for graph number
counter = 1


## Plot individual graphs
counter = plot_graph(counter, [random_reward], [style_random], "-Random-1000.png", "Random Agent - 1000 Episodes")
#counter = plot_graph(counter, [sarsa_reward], [style_sarsa], "-Sarsa-1000.png", "Sarsa Agent - 1000 Episodes")
counter = plot_graph(counter, [deep_sarsa_reward], [style_deep_sarsa], "-Deep-Sarsa-1000.png", "Deep Sarsa Agent - 1000 Episodes")
counter = plot_graph(counter, [dqn_reward], [style_dqn], "-DQN-1000.png", "DQN Agent - 1000 Episodes")
counter = plot_graph(counter, [ddpg_reward], [style_ddpg], "-DDPG-1000.png", "DDPG Agent - 1000 Episodes")
        

## Plot individual graphs with Random as a benchmark plot
#counter = plot_graph(counter, [random_reward,sarsa_reward], [style_random,style_sarsa], "-Sarsa-Random-1000.png", "Sarsa vs Random - 1000 Episodes")
counter = plot_graph(counter, [random_reward,deep_sarsa_reward], [style_random,style_deep_sarsa], "-Deep-Sarsa-Random-1000.png", "Deep Sarsa vs Random - 1000 Episodes")
counter = plot_graph(counter, [random_reward,ddpg_reward], [style_random,style_ddpg], "-DDPG-Random-1000.png", "DDPG vs Random - 1000 Episodes")
counter = plot_graph(counter, [random_reward,dqn_reward], [style_random,style_dqn], "-DQN-Random-1000.png", "DQN vs Random - 1000 Episodes")

## Plot each graph against each other
counter = plot_graph(counter, [deep_sarsa_reward,dqn_reward], [style_deep_sarsa,style_dqn], "-Deep-Sarsa-DQN-1000.png", "Deep Sarsa vs DQN - 1000 Episodes")
counter = plot_graph(counter, [random_reward,dqn_reward,ddpg_reward], [style_random,style_dqn,style_ddpg], "-Random-DQN-DDPG-1000.png", "Random vs DQN vs DDPG - 1000 Episodes")


















import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.interpolate import interp1d
import math

## Plotting graphs for RL

##     graph_num   = number of the graph being produced
##     data_points = array containing arrays of data points to be plotted
##     data_style  = array containing arrays of information and styles for data_points [colour, label, cobf colour]
##     filename    = name of the file containing the output graph
##     graph_title = title of the output graph
##     smooth_only = bool - only plot smoothed version of graphs
def plot_graph(graph_num, data_points, data_style, filename, graph_title, smooth_only):
    fig, ax = plt.subplots()
    ep_counter = np.arange(1,1001)
    gap = 50
    x_final = np.linspace(25, 975, 50)
    for i in range(len(data_points)):

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(graph_title)

        # Plot original data points
        if not smooth_only:
            ax.plot(data_points[i], color=data_style[i][0], alpha=0.4)

        # Curve of Best Fit code
        cobf = np.average(np.array(data_points[i]).reshape(-1, gap), axis=1)
        cobf_eps = ep_counter[0::gap]
        x_new = np.linspace(cobf_eps.min(), cobf_eps.max(), 50)
        f = interp1d(cobf_eps, cobf, kind="quadratic")
        y_smooth = f(x_new)

        
        ax.plot(x_final, y_smooth, color=data_style[i][2], label=data_style[i][1])
        ax.legend(loc="lower right")

        # Line at 200 code
        ax.hlines(y=[200], xmin=[0], xmax=[1000], colors='grey', linestyles='--', lw=1)

        # min value lower mult of 200 for y axis
        min_value = 100000000
        for each in data_points:
            temp_min_value = math.floor(min(each)/200)*200
            if temp_min_value < min_value:
                min_value = temp_min_value

        # highest value mult 200 or 200 for success line if larger
        max_value = -100000000
        for each in data_points:
            temp_max_value = math.ceil(max(each)/200)*200
            if temp_max_value > max_value:
                max_value = temp_max_value
        if max_value < 400:
            max_value = 400
        
        ax.set_yticks(np.arange(min_value, max_value+1, 200))
        
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
        reward.append(each[0])
    return reward



### GRAPHS FOR REPORT ###

counter = 1  #counter for graph number

random_reward = []
random_reward = read_file("random.csv", random_reward)
style_random = ["lightgray", "Random", "grey"]

human_reward = []
human_reward = read_file("human.csv",human_reward)
style_human = ["plum", "Human", "violet"]

sarsa_reward = []
sarsa_reward = read_file("sarsa.csv", sarsa_reward)
style_sarsa = ["lightcoral", "SARSA", "red"]

dqn_reward = []
dqn_reward = read_file("dqn.csv", dqn_reward)
style_dqn = ["peachpuff", "DQN", "orange"]

ddpg_reward = []
ddpg_reward = read_file("ddpg.csv", ddpg_reward)
style_ddpg = ["darkgrey", "DDPG", "black"]

deep_sarsa_reward = []
deep_sarsa_reward = read_file("deep_sarsa.csv", deep_sarsa_reward)
style_deep_sarsa = ["lightgreen", "Deep Sarsa", "forestgreen"]

td3_reward = []
td3_reward = read_file("td3-1.csv", td3_reward)
style_td3 = ["lightskyblue", "TD3", "dodgerblue"]

sac_reward = []
sac_reward = read_file("sac.csv", sac_reward)
style_sac = ["mediumpurple", "SAC", "rebeccapurple"]

# Plot Individual Graphs
counter = plot_graph(counter, [random_reward], [style_random], "-Random-1000.png", "Random Agent - 1000 Episodes", False)
counter = plot_graph(counter, [human_reward], [style_human], "-Human-1000.png", "Human Agent - 1000 Episodes", False)
counter = plot_graph(counter, [sarsa_reward], [style_sarsa], "-Sarsa-1000.png", "Sarsa Agent - 1000 Episodes", False)
counter = plot_graph(counter, [dqn_reward], [style_dqn], "-DQN-1000.png", "DQN - 1000 Episodes", False)
counter = plot_graph(counter, [ddpg_reward], [style_ddpg], "-DDPG-1000.png", "DDPG Agent - 1000 Episodes", False)
counter = plot_graph(counter, [deep_sarsa_reward], [style_deep_sarsa], "-Deep-Sarsa-1000.png", "Deep Sarsa Agent - 1000 Episodes", False)
counter = plot_graph(counter, [td3_reward], [style_td3], "-TD3-1000.png", "TD3 Agent - 1000 Episodes", False)
counter = plot_graph(counter, [sac_reward], [style_sac], "-SAC-1000.png", "SAC Agent - 1000 Episodes", False)

# Plot Graphs with Random
counter = plot_graph(counter, [human_reward,random_reward], [style_human,style_random], "-Human-vs-Random-1000.png", "Human vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [sarsa_reward,random_reward], [style_sarsa,style_random], "-Sarsa-vs-Random-1000.png", "Sarsa  vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [dqn_reward,random_reward], [style_dqn,style_random], "-DQN-vs-Random-1000.png", "DQN vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [ddpg_reward,random_reward], [style_ddpg,style_random], "-DDPG-vs-Random-1000.png", "DDPG vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [deep_sarsa_reward,random_reward], [style_deep_sarsa,style_random], "-Deep-Sarsa-vs-Random-1000.png", "Deep Sarsa vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [td3_reward,random_reward], [style_td3,style_random], "-TD3-vs-Random-1000.png", "TD3 vs Random - 1000 Episodes", False)
counter = plot_graph(counter, [sac_reward, random_reward], [style_sac,style_random], "-SAC-vs-Random-1000.png", "SAC vs Random - 1000 Episodes", False)

# All on same axes
counter = plot_graph(counter, [random_reward,sarsa_reward,dqn_reward,ddpg_reward,deep_sarsa_reward,td3_reward,human_reward, sac_reward], [style_random,style_sarsa,style_dqn,style_ddpg,style_deep_sarsa,style_td3,style_human, style_sac], "-All-Agents-1000.png", "All Agents - 1000 Episodes", False)

# Just plotting smooth lines
counter = plot_graph(counter, [random_reward,sarsa_reward,dqn_reward,ddpg_reward,deep_sarsa_reward,td3_reward,human_reward, sac_reward], [style_random,style_sarsa,style_dqn,style_ddpg,style_deep_sarsa,style_td3,style_human, style_sac], "-All-Agents-1000.png", "All Agents - 1000 Episodes", True)

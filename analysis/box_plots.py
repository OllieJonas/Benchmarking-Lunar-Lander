### Box Plots ###

# This is the code used to generate the box-plots used in our report

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import csv


# [Cumulative, Average, No Time-Steps]
def read_file(filename, reward):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            data.append(row)
    for each in data:
        reward.append(each[0])
    reward = reward[-100:]
    return reward


# Creating dataset
ddpg = []
ddpg = read_file("ddpg.csv", ddpg)

human = []
human = read_file("human.csv", human)

sarsa = []
sarsa = read_file("sarsa.csv", sarsa)

deep_sarsa = []
deep_sarsa = read_file("deep_sarsa.csv", deep_sarsa)

dqn = []
dqn = read_file("dqn.csv", dqn)

td3 = []
td3 = read_file("td3-1.csv", td3)

random = []
random = read_file("random.csv", random)

sac = []
sac = read_file("sac.csv", sac)

def plot_boxes(data, labels, plot_title, filename):
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Cumulative Reward")

    # Creating axes instance
    # set showfliers to true to see outliers
    bp = ax.boxplot(data, patch_artist = True, notch =False, showfliers=False, vert = 0, boxprops=dict(facecolor='lightgrey', color='black'))


    ## Styling ################################################################
    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
            whisker.set(color ='black',linewidth = 1.5, linestyle =":")

    # changing color and linewidth of caps
    for cap in bp['caps']:
            cap.set(color ='black', linewidth = 2)

    # changing color and linewidth of medians
    for median in bp['medians']:
            median.set(color ='red', linewidth = 2)

    # changing style of fliers
    for flier in bp['fliers']:
            flier.set(marker ='D', color ='#e7298a', alpha = 0.5)
    ############################################################################
    
	
    # x-axis labels
    ax.set_yticklabels(labels)

    # Adding title
    plt.title(plot_title)

    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.savefig(filename)
    plt.clf()


## Data needs to be in the same order as strings for y axis labels
data        = [sac, dqn, random, human, sarsa, td3, deep_sarsa, ddpg]
labels      = ["SAC","DQN", "Random", "Human", "SARSA", "TD3", "Deep SARSA", "DDPG"]
plot_title  = "Box Plot: All Agents"
filename    = "Box-Plots-All-Agents.png"


### labels      - array of strings for y axis labels
### data        - array of arrays of data points
###                 - E.g. [ Data 1,   Data 2,  Data 3 ]
###                 -      [ [0,1,1], [1,1,1], [2,2,1] ]
### image_title - string fir title for the plot
### filename    - string ending with '.png' for the image to be saved as
plot_boxes(data, labels, plot_title, filename)

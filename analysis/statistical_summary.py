### Statistical Summary of Cumulative Rewards for Agents

import numpy as np
import statistics
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

data   = [sac, dqn, random, sarsa, human, td3, deep_sarsa, ddpg]
labels = ["SAC", "DQN", "Random", "SARSA", "Human", "TD3", "Deep SARSA", "DDPG"]

print("{:16s} {:7} {:12} {:7} {:10}".format("Algorithm", "IQR", "Median", "Mean", "Std Dev"))
print("-----------------------------------------------------")
for i in range(len(data)):
    q1, q2, q3 = np.percentile(data[i], [25, 50, 75])
    iqr = q3 - q1
    mean = sum(data[i])/len(data[i])
    std_dev = statistics.stdev(data[i])
    print("{:12s} {:7.2f} {:10.2f} {:10.2f} {:10.2f}".format(labels[i], iqr, q2, mean, std_dev))

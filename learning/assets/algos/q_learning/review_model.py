import matplotlib.pyplot as plt

import numpy as np
"""
This python file is used to review the model of the Q-learning algorithm. Only works for 1 joint robots.
"""

name = "Q.npy"
path = "/home/peter/OneDrive/Skrivebord/Uni stuff/Masters/Masters_project/mj_sim/data/Q_learning/Velocity_observation_Optimistic_Starting_Vel_punishment_20241101145207522/model/"



model = np.load(path + name)



for i in range(50):
    position = 20
    velocity = i
    value = position + velocity*25^1
    print("position: ", position)
    print("velocity: ", velocity)
    print("Value: ", value)
    print("Q values: ", model[value])
    print("\n")






#Print Q values for both actions
minus50velocity = np.array([model[i][0] for i in range(0, len(model))])
plus50velocity = np.array([model[i][1] for i in range(0, len(model))])

plt.clf()
plt.plot(range(len(minus50velocity)), minus50velocity, label="-50 velocity")
plt.plot(range(len(plus50velocity)), plus50velocity, label="50 velocity")
plt.legend()



plt.savefig(path + "Q_table.png")







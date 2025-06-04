import itertools
import numpy as np
import time
import torch

joint_logs = [[] for _ in range(7)]
# Generate all possible combinations for the specified number of joints



for i in range(2):
    print(joint_logs)
    joint_logs[i].append(i)

print(joint_logs)

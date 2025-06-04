


import os

import time

import matplotlib.pyplot as plt

import pandas as pd


import numpy as np



import numpy as np
import scipy.stats

import math
import torch
import torch.nn.functional as F
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h




def make_plot_from_tensor_board():



    output_file_address = "Data/graphs/"

    file_name = "C1_reward_injected_RL_OPT_sucks"

    full_directory = output_file_address + file_name



    data = []
    csv_data_path = "Data/CSV_data/C1/C1_reward_injected/RL_OPT_sucks"
    files = os.listdir(csv_data_path)
    print("files: ", files)

    for i in range(1):


        full_path = csv_data_path + "/"+ files[i]

        df = pd.read_csv(full_path)
        cropped_values = df["Value"][:130]


        data.append(cropped_values)

    data = np.array(data)

    # for i in range(10):

    #     for j in range(len(data[i])):
    #         data[i][j] = F.sigmoid(torch.tensor(data[i][j])).item()


    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    std_dev = np.std(data, axis=0) 

    confidence_val = 0.95
    interval_95 = mean_confidence_interval(data, confidence=confidence_val)


    os.makedirs("Data/graphs", exist_ok=True)

    expert_val = 0.9917
    plt.axhline(y=expert_val, color='r', linestyle='--', label=f"Expert demonstration reward ({expert_val})")


    plt.plot(mean, label='mean reward', color='blue')
    plt.fill_between(range(len(mean)), mean - interval_95, mean + interval_95, color='blue', alpha=0.3, label=f'± {confidence_val*100}% confidence interval')

    plt.xlabel("Training steps")
    plt.ylabel("True reward")
    plt.title("PPO training cloth manipulation")
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim(0,1.1)
    plt.tight_layout()
    plt.savefig( full_directory )
    plt.close()




make_plot_from_tensor_board()

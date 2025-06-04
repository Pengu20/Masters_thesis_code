


import os

import time

import matplotlib.pyplot as plt

import pandas as pd


import numpy as np



import numpy as np
import scipy.stats






def make_plot_from_tensor_board():

    #True reward perturbation
    # mean = [0.98990107, 0.97724697, 0.96179470, 0.94322909, 0.92441602, 0.89471307, 0.87835409, 0.83802821, 0.80840290, 0.7812695]
    # interval_95 = [0.00508726, 0.00688387, 0.01213520, 0.01862256, 0.02659422, 0.04321904, 0.04837965, 0.06709102, 0.08363471, 0.10214387]


    # State only expert pertubations
    # mean = [0.88611639, 0.88274276, 0.83416688, 0.76887459, 0.71148944, 0.63595277, 0.64228368, 0.64780450, 0.61798072, 0.5920972]
    # interval_95 = [0.19534418, 0.18952591, 0.23303071, 0.26511005, 0.28764309, 0.31987871, 0.32169232, 0.32912398, 0.33527728, 0.34524481]


    # state-action expert perturbations
    mean = [0.99518466, 0.99515814, 0.99504304, 0.99504924, 0.99498647, 0.99449265, 0.99442232, 0.99375975, 0.99351913, 0.99223650]
    interval_95 = [0.00222553, 0.00219248, 0.00247697, 0.00234208, 0.00265002, 0.00312168, 0.00347582, 0.00417914, 0.00462084, 0.00623303]
    
    # # Reward injected
    # mean = [0.00184062, 0.00184415, 0.00185480, 0.00188553, 0.00191611, 0.00194677, 0.00197599, 0.00193440, 0.00200319, 0.00206434]
    # interval_95 = [0.00222999, 0.00224052, 0.00226224, 0.00232251, 0.00233317, 0.00243468, 0.00240475, 0.00220633, 0.00228382, 0.00246887]


    mean = np.array(mean)

    interval_95 = np.array(interval_95)


    os.makedirs("Data/graphs", exist_ok=True)



    plt.plot(mean, label='mean expert reward', color='blue')
    
    plt.xticks(np.arange(0, len(mean), step=1), labels=np.arange(0,0.1, 0.01))
    plt.fill_between(range(len(mean)), mean - interval_95, mean + interval_95, color='blue', alpha=0.3, label=f'± {0.95*100}% confidence interval')
    plt.xlabel("state transition deviation")
    plt.ylabel("Discriminator reward")
    plt.ylim(0.95,1.00)
    plt.title("State-action AIRL perturbation test")
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()


    output_file_address = "Data/graphs/"

    file_name = "Perturbation_graph_state-action_AIRL.png"

    full_directory = output_file_address + file_name


    plt.savefig( full_directory )
    plt.close()




make_plot_from_tensor_board()

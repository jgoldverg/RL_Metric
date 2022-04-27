import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pydot
import matplotlib.pyplot as plt
from fileData import *
from netEnv import *
from dqnAgent import *
import time
import pickle
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import cmocean
from math import sqrt
import random
code_path = ".."

requiredFields=['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput','TotalEnergy','DataTransferEnergy']
LabelName='TotalAvgTput'
fileData_chameleon=ReadFile(code_path+'/Dataset/Chameleon_Combined_all.csv',requiredFields)
optimal_throughput_dictionary_chameleon=fileData_chameleon.return_map_for_tuple_to_throughput()
filedata_grouped_df=fileData_chameleon.get_grouped_df()
print("There are total %d number of groups based on FileCount, AvgFileSize,BufSize, Bandwidth, AvgRtt"%filedata_grouped_df.ngroups)

key=(32, 34.9238114, 40, 10, 30)
key_specific_test_logs=fileData_chameleon.return_group_specific_test_logs(key)
selectedgroup=environmentGroup(fileData_chameleon.get_grouped_df(),key)
a_group=selectedgroup.return_a_group()
print(f"selected group {(32, 34.9238114, 40, 10, 30)}")
print(f"selectedgroup.group_maximum_throughput {selectedgroup.group_maximum_throughput()}")
print(f"selectedgroup.return_group_max_throughput_parameters{selectedgroup.return_group_max_throughput_parameters()}")
env = NetEnvironment(selectedgroup)
group_from_grouped_df=selectedgroup.return_group_from_grouped_df()
total_key=0
for key in filedata_grouped_df.groups.keys():
    print(f"group key is {key}")
    total_key+=1
print(f"total number of keys are {total_key}")

env=NetEnvironment(selectedgroup)
agent=DQNAgent(env)
agent.warming_replay_buffer()
agent.training()
agent.save_model()
agent.load_model('agent(32, 34.9238114, 40, 10, 30)online_net','agent(32, 34.9238114, 40, 10, 30)target_net')
reward_per_episode_validation,action_list_per_episode=agent.validation()
agent.print_action_list_for_validation_episodes(action_list_per_episode)
agent.plot_validation_curve()

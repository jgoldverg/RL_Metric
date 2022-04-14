import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pydot
import matplotlib.pyplot as plt
from fileData import *
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

requiredFields=['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput','TotalEnergy','DataTransferEnergy']
LabelName='TotalAvgTput'
fileData_chameleon=ReadFile('/Users/jamil/Desktop/RL_Final/RL_Metric/Dataset/Chameleon_Combined_all.csv',requiredFields)
optimal_throughput_dictionary_chameleon=fileData_chameleon.return_map_for_tuple_to_throughput()
filedata_grouped_df=fileData_chameleon.get_grouped_df()
print("There are total %d number of groups based on FileCount, AvgFileSize,BufSize, Bandwidth, AvgRtt"%filedata_grouped_df.ngroups)
total_test_log=0
for key in filedata_grouped_df.groups.keys():
    print(f"group key is {key}")
    key_specific_test_logs=fileData_chameleon.return_group_specific_test_logs(key)
    total_test_log+=len(key_specific_test_logs)
    print(f"key_specific_test_logs {len(key_specific_test_logs)}")
print(f"Total Test logs {total_test_log}")
print(f"Total logs {len(fileData_chameleon.logs)}")



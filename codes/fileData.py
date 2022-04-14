# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-04T14:44:57-05:00


"""
History log are grouped based on 5 fields ('FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt')
and this tuple we will denote as key, group key, group tuple
"""


import math
import random
import numpy as np
import re
import sys
import pandas as pd
from collections import Counter
from collections import deque

"""
input: a pandas dataframe and a list of fields that needs to be extracted from the csv
output: pandas dataframe from the csv with onlt the input list of required fields
"""

def extractRequiredColumn(df,requiredFields):
    return df[df.columns[df.columns.isin(requiredFields)]]

"""
input: a directory where the csv files resides could handle multiple csv files location
output: concatenated pandas datframe (in case of multiple location) or pandas dataframe
"""
def load_dataset_from_file(dataset_file_location):
    result_df=pd.read_csv(dataset_file_location[0])
    if len(dataset_file_location)>1:
        for i in range(1,len(dataset_file_location)):
            temp_df=pd.read_csv(dataset_file_location[i])
            result_df=pd.concat([result_df, temp_df], axis=0, join='inner')
    return result_df

"""
log class to store each row of the history log (only required fields) as a log
"""

class Log:
    def __init__(self, serialNo, values):
        self.serialNo = serialNo
        self.values = values
        self.ranges=[]
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput','TotalEnergy','DataTransferEnergy']

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += "%d %s:%.2f" % (self.serialNo, self.names[i], self.values[i])
        return result

"""
Readfile Class to read a history log (.csv) and group logs based on ['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt']
and provide group and history log as a whole statistics
"""
class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):
        self.logs=[]
        self.dataset=pd.read_csv(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))#'P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput'
        self.grouped_df=self.requiredData.groupby(['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'])
        self.map_for_tuple_to_throughput=dict()
        self.map_for_group_key_to_all_logs=dict()
        self.map_for_group_key_to_test_logs=dict()
        for key,item in self.grouped_df:
            key_group_test_logs=[]
            a_group=self.grouped_df.get_group(key)
            group_max_throughput=a_group['TotalAvgTput'].max()
            self.map_for_tuple_to_throughput[key]=group_max_throughput
            number_of_rows=a_group.shape[0]
            selected_no_test_rows=math.ceil(number_of_rows*1)  #30% is test data
            a_group_test=a_group.sample(n=selected_no_test_rows)
            for index, row in a_group_test.iterrows():
                key_group_test_logs.append(Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))
            self.map_for_group_key_to_test_logs[key]=key_group_test_logs

    """
    return_map_for_tuple_to_throughput() returns a dictionary for
    each group (key) of logs and max throughput for that group
    """
    def return_map_for_tuple_to_throughput(self):
      return self.map_for_tuple_to_throughput

    """
    get_grouped_df() returns a generic.DataFrameGroupBy object which needs to be accessed
    by group keys
    """
    def get_grouped_df(self):
        return self.grouped_df
    """
    return_group_specific_test_logs() returns a list of log entries associated with
    that particular group keys
    """
    def return_group_specific_test_logs(self,group_key):
        return self.map_for_group_key_to_test_logs[group_key]



# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-04T14:44:57-05:00

import math
import random
import numpy as np
import re
import sys
import pandas as pd
from collections import Counter
from collections import deque

class Log:
    def __init__(self, serialNo, values):
        # each range is left inclusive and right exclusive, i.e., [left, right)
        self.serialNo = serialNo
        self.values = values
        # print(self.values)
        self.ranges=[]
        for value in values:
            self.ranges.append(value)
            self.ranges.append(value+1)
        # print(self.values, type(self.values))
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput','TotalEnergy','DataTransferEnergy']

    def is_intersect(self, dimension, left, right):
        return self.values[dimension]>=left and self.values[dimension]<=right

    def is_intersect_multi_dimension(self, ranges):
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def matches(self, packet):
        assert len(packet) == 5, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # src ip
            packet[0] + 1,
            packet[1] + 0,  # dst ip
            packet[1] + 1,
            packet[2] + 0,  # src port
            packet[2] + 1,
            packet[3] + 0,  # dst port
            packet[3] + 1,
            packet[4] + 0,  # protocol
            packet[4] + 1
        ])

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            # result += " %s:%.2f:[%d, %d) " % (self.names[i], self.values[i], self.ranges[i * 2],
            #                             self.ranges[i * 2 + 1])
            result += "%d %s:%.2f" % (self.serialNo, self.names[i], self.values[i])
        return result

class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):
        self.logs=[]
        self.test_logs=[]
        #self.train_logs=[]
        #dataset_file_location is a "list" of considered dataset files
        self.dataset=pd.read_csv(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        # print(self.dataset)
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))#'P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput'

        self.grouped_df=self.requiredData.groupby(['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'])
        #print(type(self.grouped_df))
        self.map_for_tuple_to_throughput=dict()
        for key,item in self.grouped_df:
          a_group=self.grouped_df.get_group(key)
          # print(a_group, type(a_group),'\n')
          group_max_throughput=a_group['TotalAvgTput'].max()
          self.map_for_tuple_to_throughput[key]=group_max_throughput
          number_of_rows=a_group.shape[0]
        #   print(a_group.shape)
        #   print("number_of_rows",number_of_rows)
          selected_no_test_rows=math.ceil(number_of_rows*0.3)  #30% is test data
        #   print("selected_no_test_rows", selected_no_test_rows)
          a_group_test=a_group.sample(n=selected_no_test_rows)
          # print(a_group_test, '\n')
          for index, row in a_group_test.iterrows():
              self.test_logs.append(Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))
        #self.test_logs=self.logs

    def return_map_for_tuple_to_throughput(self):
      return self.map_for_tuple_to_throughput

    def get_grouped_df(self):
        return self.grouped_df

    def return_test_logs(self):
      return self.test_logs

    def return_train_logs(self):
      return self.train_logs


def extractRequiredColumn(df,requiredFields):
    return df[df.columns[df.columns.isin(requiredFields)]]

def load_dataset_from_file(dataset_file_location):
    result_df=pd.read_csv(dataset_file_location[0])
    if len(dataset_file_location)>1:
        for i in range(1,len(dataset_file_location)):
            temp_df=pd.read_csv(dataset_file_location[i])
            result_df=pd.concat([result_df, temp_df], axis=0, join='inner')
    return result_df

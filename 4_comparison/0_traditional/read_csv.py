import numpy as np
import pandas as pd

name_file = r'E:\acocac\research\scripts\thesis_final\5_comparison\1b_traditional\test_dataset.csv'

data = pd.read_table(name_file, sep=',', header=0)  # -- one header
#
# y_data = data.iloc[:, 0]
# y = np.asarray(y_data.values, dtype='uint8')
#
# polygonID_data = data.iloc[:, 1]
# polygon_ids = polygonID_data.values
# polygon_ids = np.asarray(polygon_ids, dtype='uint16')
#
# X_data = data.iloc[:, 2:]
# X = X_data.values
# X = np.asarray(X, dtype='float32')
#
# print(X[0,:3])

# export method 1
# name_file = r'E:\acocac\research\scripts\thesis_final\5_comparison\1b_traditional\fcTS_train_10.csv'
#
# data = pd.read_table(name_file, sep=',', header=0) #-- one header
# data = pd.read_csv(name_file)
# bandsCollection = ['red', 'NIR','blue', 'green', 'SWIR1', 'SWIR2','SWIR3']
# data = data.melt(['ID','MCD12Q1v6','system:index'], value_vars=bandsCollection, var_name='cols',  value_name='vals')
# data = data.sort_values(['ID','system:index'])
# data = data.groupby(['ID','MCD12Q1v6'])['vals'].apply(lambda df: df.reset_index(drop=True)).unstack()
# data.reset_index(inplace=True)
#
# y_data = data.iloc[:,1]
# y = np.asarray(y_data.values, dtype='uint8')
# y = y -1
#
# polygonID_data = data.iloc[:,0]
# polygon_ids = polygonID_data.values
# polygon_ids = np.asarray(polygon_ids, dtype='uint16')
#
# X_data = data.iloc[:,2:]
# X = X_data.values
# X = np.asarray(X, dtype='float32')
# print(X.shape)
# print(X[0,0:7])

# export method 2
# name_file = r'E:\acocac\research\scripts\thesis_final\5_comparison\1b_traditional\nifcTS_test_samplesize16_tyear2015.csv'
#
# data = pd.read_table(name_file, sep=',', header=0) #-- one header
# # data = pd.read_csv(name_file)
# # print(data.head(3))
#
#
# # frames = [df1, df2, df3]
# #
# # result = pd.concat(frames)
#
# data.set_index('ID', inplace=True)
# a = pd.DataFrame(pd.concat([data.red, data.NIR]), columns=['xy']).reset_index()
#
# print(a.head())
# bandsCollection = ['red', 'NIR','blue', 'green', 'SWIR1', 'SWIR2','SWIR3']
# data = data.melt(['ID','MCD12Q1v6','system:index'], value_vars=bandsCollection, var_name='cols',  value_name='vals')
# data = data.sort_values(['ID','system:index'])
# data = data.groupby(['ID','MCD12Q1v6'])['vals'].apply(lambda df: df.reset_index(drop=True)).unstack()
# data.reset_index(inplace=True)
#
# y_data = data.iloc[:,1]
# y = np.asarray(y_data.values, dtype='uint8')
# y = y -1
#
# polygonID_data = data.iloc[:,0]
# polygon_ids = polygonID_data.values
# polygon_ids = np.asarray(polygon_ids, dtype='uint16')
#
# X_data = data.iloc[:,2:]
# X = X_data.values
# X = np.asarray(X, dtype='float32')
# print(X.shape)
# print(X[0,0:7])

# export method 3
from ast import literal_eval

name_file = r'E:\acocac\research\scripts\thesis_final\5_comparison\1b_traditional\train_dataset.csv'

data = pd.read_table(name_file, sep=',', header=0) #-- one header

y_data = data['class']
y = np.asarray(y_data.values, dtype='uint8')

data['array'] = data['array'].apply(lambda s: list(literal_eval(s)))
data['array'] = data['array'].apply(lambda s: np.concatenate(np.array(s)))

x = np.array(data['array']).tolist()
x = [j for j in x if len(j) == 368]

X = np.asarray(x, dtype='float32')

polygonID_data = data.index
polygon_ids = polygonID_data.values
polygon_ids = np.asarray(polygon_ids, dtype='uint16')
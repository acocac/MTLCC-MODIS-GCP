#!/usr/bin/python


"""
	Reading train and test files.
	Pre-processing steps.
"""


import os, sys
import argparse

import numpy as np
import pandas as pd
from ast import literal_eval

import math
import random
import itertools
import time

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------			PREPROCESSING			--------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
def readSITSData(name_file, year):
	"""
		Read the data contained in name_file
		INPUT:
			- name_file: file where to read the data
		OUTPUT:
			- X: variable vectors for each example
			- polygon_ids: id polygon (use e.g. for validation set)
			- Y: label for each example
	"""
	
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
	#
	# return  X, polygon_ids, y

	#input data sample
	# data = pd.read_table(name_file, sep=',', header=0)  # -- one header
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
	# return X, polygon_ids, y


	# #input data GEE
	data = pd.read_table(name_file, sep=',', header=0)  # -- one header

	y_data = data['class'] - 1

	y = np.asarray(y_data.values, dtype='uint8')

	data[ 'array' ] = data[ 'array' ].apply(lambda s: list(literal_eval(s)))
	data[ 'array' ] = data[ 'array' ].apply(lambda s: np.concatenate(np.array(s)))

	x = np.array(data[ 'array' ]).tolist()

	if year == 2001:
		nobs = 45
	else:
		nobs = 46
        
	ind2remove = [ ]
	for p, i in enumerate(x):
		if len(i) < nobs * 7:
			ind2remove.append(p)

	y = np.delete(y, ind2remove)

	x = [ j for j in x if len(j) == nobs * 7]

	if year == 2001:
		X = np.asarray(x, dtype='float32')
		X = [np.concatenate([X[t][0:7], X[t]]) for t in range(X.shape[0])]
		X = np.array(X)
	else:
		X = np.asarray(x, dtype='float32')

	doy = np.array(range(1,365,8))
	X = [np.concatenate([X[t],doy]) for t in range(X.shape[0])]
	X = np.array(X)

	polygonID_data = data.index
	polygon_ids = polygonID_data.values
	polygon_ids = np.asarray(polygon_ids, dtype='uint16')

	return X, polygon_ids, y
#-----------------------------------------------------------------------
def reshape_data(X, nchannels):
	"""
		Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
		INPUT:
			-X: original feature vector ()
			-feature_strategy: used features (options: SB, NDVI, SB3feat)
			-nchannels: number of channels
		OUTPUT:
			-new_X: data in the good format for Keras models
	"""
	
	return X.reshape(X.shape[0],int(X.shape[1]/nchannels),nchannels)

#-----------------------------------------------------------------------
def computingMinMax(X, per=2):
	min_per = np.percentile(X, per, axis=(0,1))
	max_per = np.percentile(X, 100-per, axis=(0,1))
	return min_per, max_per

#-----------------------------------------------------------------------
def normalizingData(X, min_per, max_per):
	return (X-min_per)/(max_per-min_per)

#EOF
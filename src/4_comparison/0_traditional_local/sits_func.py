#!/usr/bin/python

"""
	Reading train and test files.
"""


import numpy as np
import pandas as pd
from ast import literal_eval

def readSITSData(name_file, year, normalise=False):
	"""
		Read the data contained in name_file
		INPUT:
			- name_file: file where to read the data
			- year: target year
		OUTPUT:
			- X: variable vectors for each example
			- polygon_ids: id
			- Y: label for each example
	"""

	# #input data GEE
	data = pd.read_table(name_file, sep=',', header=0)  # -- one header

	y_data = data['class']

	y = np.asarray(y_data.values, dtype='uint8')

	data['array'] = data['array'].apply(lambda s: list(literal_eval(s)))
	data['array'] = data['array'].apply(lambda s: np.concatenate(np.array(s)))

	x = np.array(data[ 'array' ]).tolist()

	if year == '2001':
		nobs = 45
	else:
		nobs = 46
        
	ind2remove = [ ]
	for p, i in enumerate(x):
		if len(i) < nobs * 7:
			ind2remove.append(p)

	y = np.delete(y, ind2remove)

	x = [ j for j in x if len(j) == nobs * 7]

	if year == '2001':
		X = np.asarray(x, dtype='float32')
		X = [np.concatenate([X[t][0:7], X[t]]) for t in range(X.shape[0])]
		X = np.array(X)
	else:
		X = np.asarray(x, dtype='float32')

	doy = np.array(range(1,365,8))

	if normalise:
		X = normalize_fixed(X, -100, 16000)
		doy = doy / 365

	X = [np.concatenate([X[t],doy]) for t in range(X.shape[0])]
	X = np.array(X)

	polygonID_data = data.index
	polygon_ids = polygonID_data.values
	polygon_ids = np.asarray(polygon_ids, dtype='uint16')

	return X, polygon_ids, y

def computingMinMax(X, per=2):
	min_per = np.percentile(X, per, axis=(0,1))
	max_per = np.percentile(X, 100-per, axis=(0,1))
	return min_per, max_per

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

	return X.reshape(X.shape[0], int(X.shape[1]/nchannels), nchannels)

def normalize_fixed(X, min_per, max_per):
	x_normed = (X-min_per) / (max_per-min_per)
	return x_normed
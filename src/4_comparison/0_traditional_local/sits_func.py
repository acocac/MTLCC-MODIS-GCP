#!/usr/bin/python

"""
	Reading train and test files.
"""


import numpy as np
import pandas as pd
from ast import literal_eval

def readSITSData(name_file, year):
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
	X = [np.concatenate([X[t],doy]) for t in range(X.shape[0])]
	X = np.array(X)

	polygonID_data = data.index
	polygon_ids = polygonID_data.values
	polygon_ids = np.asarray(polygon_ids, dtype='uint16')

	return X, polygon_ids, y
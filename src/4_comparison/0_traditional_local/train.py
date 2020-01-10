#!/usr/bin/python

"""
	Main file
"""

import os, sys
import argparse

from sits_func import *

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn import over_sampling, under_sampling
from hyperopt import Trials
import pickle as pkl

import time

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#---------------------			MAIN			------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
def getBestModelfromTrials(trials, STATUS_OK):
    valid_trial_list = [trial for trial in trials
						if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['misc']['vals']

def prep_data(partition, period, datadir, ssize):
	X = [ ]
	Y = [ ]
	IDS = [ ]

	for y in period.split('|'):
		x, ids, y = readSITSData(
			os.path.join(datadir, 'fcTS_{}_ssize{}_tyear{}_LCfinalmapv6_LCProp2.csv'.format(partition,ssize, y)), y)

		X.append(x)
		Y.append(y)
		IDS.append(ids)

	X = np.vstack(X)

	Y = [ item for sublist in Y for item in sublist ]
	IDS = [ item for sublist in IDS for item in sublist ]

	target_others = [Y, IDS]

	Y, IDS = [np.array(t) for t in target_others]

	return X, Y, IDS

def main(args):
	classif_type = ["RF", "TempCNN", "GRU-RNNbi", "GRU-RNN"]
	if args.classifier not in classif_type:
		print("ERR: select an available classifier (RF, TempCNN, GRU-RNNbi or GRU-RNN)")
		sys.exit(1)

	model_dir = args.modeldir.replace('|', '')

	# Output filenames
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	X_train, y_train, ids_train = prep_data('train', args.train_on, args.datadir, args.ssize)

	STATUS_OK = 'ok'

	trials = pkl.load(
		open(args.trials,
			 'rb'))

	bestmodel = getBestModelfromTrials(trials.trials, STATUS_OK)

	if bestmodel['max_features'][0] == 0:
		max_features = 'auto'
	else:
		max_features = 'sqrt'

	if bestmodel['bootstrap'][0] == 0:
		bootstrap = 'True'
	else:
		bootstrap = 'False'

	resampling = over_sampling.RandomOverSampler(sampling_strategy='auto',
										  random_state=42)

	estimator = RandomForestClassifier(n_estimators=20, max_features=max_features,max_depth=bestmodel['max_depth'][0],
									   min_samples_leaf=bestmodel['min_samples_leaf'][0],min_samples_split=bestmodel['min_samples_split'][0],
									   bootstrap=bootstrap, n_jobs=-1, verbose=1)

	pl = make_pipeline_imb(resampling, estimator)

	#-- train a rf classifier
	print('Training with the best model with parameters: ', bestmodel)
	start_train_time = time.time()
	estimator_fit = pl.fit(X_train, y_train)
	res_time = round(time.time()-start_train_time, 2)
	print('Training time (s): ', res_time)

	model_file = os.path.join(model_dir, 'model-' + args.classifier + '.h5')
	#-- save the model
	joblib.dump(estimator_fit, model_file)
	print("Writing the model over")

#-----------------------------------------------------------------------
if __name__ == "__main__":
	try:
		if len(sys.argv) == 1:
			prog = os.path.basename(sys.argv[0])
			print('      '+sys.argv[0]+' [options]')
			print("     Help: ", prog, " --help")
			print("       or: ", prog, " -h")
			print("example 1 : python %s --classifier TempCNN " %sys.argv[0])
			sys.exit(-1)
		else:
			parser = argparse.ArgumentParser(description='Training RF, TempCNN or GRU-RNN models on SITS datasets')
			parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")

			parser.add_argument('--classifier', dest='classifier',
								help='classifier to train (RF/TempCNN/GRU-RNNbi/GRU-RNN)')
			parser.add_argument('--datadir', type=str, default=None,
								help='directory containing the data')
			parser.add_argument('-train_on', '--train_on', type=str, default="2001",
								help='train years')
			parser.add_argument('-ssize', '--ssize', type=int, default=10, help='Sample size')
			parser.add_argument('--channels', dest='channels',
								help='number of channels')
			parser.add_argument('-cpus', '--cpus', type=int, default=14, help='cpus')
			parser.add_argument('--trials', type=str, default=None,
								help='full path to file containing the trials')
			args = parser.parse_args()
			main(args)

	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)
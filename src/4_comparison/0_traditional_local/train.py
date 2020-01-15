#!/usr/bin/python

"""
	Main file
"""

import os, sys
import argparse

from sits_func import *

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn import over_sampling, under_sampling
from hyperopt import Trials
import pickle as pkl
from sklearn.metrics import accuracy_score

import time

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('modeldir', type=str,
					  help="directory containing model definition '.h5'")
  parser.add_argument('--classifier', dest='classifier',
					  help='classifier to train (RF/TempCNN/GRU-RNNbi/GRU-RNN)')
  parser.add_argument('--datadir', type=str, default=None,
					  help='directory containing the data')
  parser.add_argument('-train_on', '--train_on', type=str, default="2001",
					  help='train years')
  parser.add_argument('-ssize', '--ssize', type=int, default=10,
					  help='Sample size')
  parser.add_argument('-trials', '--trials', type=int, default=10,
					  help='Number of trials')
  parser.add_argument('--channels', dest='channels',
					  help='number of channels')
  parser.add_argument('-cpus', '--cpus', type=int, default=14,
					  help='cpus')
  parser.add_argument('--fold', type=str, default=None,
					  help='fold')

  args, _ = parser.parse_known_args(args=argv[ 1: ])
  return args


def getBestModelfromTrials(trials, STATUS_OK):
    valid_trial_list = [trial for trial in trials
						if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['misc']['vals']


def prep_data(partition,args):
	X = [ ]
	Y = [ ]
	IDS = [ ]

	for y in args.train_on.split('|'):
		if args.classifier == 'SVM':
			x, ids, y = readSITSData(
				os.path.join(args.datadir, 'fold{}'.format(args.fold),
							 'fcTS_{}_ssize{}_tyear{}_LCfinalmapv6_LCProp2.csv'.format(partition, args.ssize, y)),
				y, normalise=True)
		else:
			x, ids, y = readSITSData(
				os.path.join(args.datadir, 'fold{}'.format(args.fold),
							 'fcTS_{}_ssize{}_tyear{}_LCfinalmapv6_LCProp2.csv'.format(partition, args.ssize, y)),
				y)

		X.append(x)
		Y.append(y)
		IDS.append(ids)

	X = np.vstack(X)

	Y = [ item for sublist in Y for item in sublist ]
	IDS = [ item for sublist in IDS for item in sublist ]

	target_others = [Y, IDS]

	Y, IDS = [np.array(t) for t in target_others]

	return X, Y, IDS


def run_model(args):
	classif_type = ['RF','SVM']
	if args.classifier not in classif_type:
		print('ERR: select an available classifier (RF, SVM)')
		sys.exit(1)

	X_train, y_train, ids_train = prep_data('train', args)
	X_val, y_val, ids_val = prep_data('val', args)

	STATUS_OK = 'ok'

	trials = pkl.load(
		open(os.path.join(args.modeldir,'hpt','hyperopt_trials_niters{}_ssize{}.pkl'.format(args.trials,args.ssize)),
			 'rb'))

	bestmodel = getBestModelfromTrials(trials.trials, STATUS_OK)

	resampling = over_sampling.RandomOverSampler(sampling_strategy='auto',
												 random_state=42)
	if args.classifier == 'RF':
		if bestmodel['max_features'][0] == 0:
			max_features = 'auto'
		else:
			max_features = 'sqrt'

		if bestmodel['bootstrap'][0] == 0:
			bootstrap = 'True'
		else:
			bootstrap = 'False'

		estimator = RandomForestClassifier(n_estimators=int(bestmodel['n_estimators'][0]),
										   max_features=max_features,
										   max_depth=int(bestmodel['max_depth'][0]),
										   min_samples_leaf=int(bestmodel['min_samples_leaf'][0]),
										   min_samples_split=int(bestmodel['min_samples_split'][0]),
										   bootstrap=bootstrap,
										   n_jobs=-1,
										   verbose=1)
	else:
		c_lim = (-2, 7)
		g_lim = (-2, 4)

		C_space = [10 ** exp for exp in range(*c_lim)]
		gamma_space =  [10**exp for exp in range(*g_lim)]
		kernel_space = ['rbf']

		C = C_space[int(bestmodel['C'][0])]
		gamma = gamma_space[int(bestmodel['gamma'][0])]
		kernel = kernel_space[int(bestmodel['kernel'][0])]

		print('Best model using C = {} gamma = {} and kernel {}'.format(C, gamma, kernel))
		estimator = SVC(C=C,
						gamma=gamma,
						kernel=kernel,
						verbose=1)

	pl = make_pipeline_imb(resampling, estimator)

	#-- train a rf classifier
	print('Training with the best model with parameters: ', bestmodel)
	start_train_time = time.time()
	estimator_fit = pl.fit(X_train, y_train)
	train_time = round(time.time()-start_train_time, 2)
	print('Training time (s): ', train_time)

	#-- test a rf classifier
	start_train_time = time.time()
	test_score = estimator_fit.score(X_val, y_val)
	test_time = round(time.time()-start_train_time, 2)
	print("Test Score: ", test_score)
	print("Time elapsed: ", test_time)

	def makedir(outfolder):
		if not os.path.exists(outfolder):
			os.makedirs(outfolder)

	outdir = os.path.join(args.modeldir,'models')
	makedir(outdir)

	model_file = os.path.join(outdir, 'model-{}_fold{}.h5'.format(args.classifier,args.fold))
	#-- save the model
	joblib.dump(estimator_fit, model_file)
	print("Writing the model over {}".format(model_file))

	eval_label = ['OA', 'train_time', 'test_time']

	history = np.zeros((len(eval_label), 1))

	history_file = os.path.join(outdir,'trainingHistory-{}_fold{}.csv'.format(args.classifier,args.fold)) #-- only for deep learning models

	history[0]=test_score
	history[1]=train_time
	history[2]=test_time

	df = pd.DataFrame(np.transpose(history), columns=eval_label)
	df.to_csv(history_file)

def main():

	args = parse_arguments(sys.argv)

	run_model(args)

if __name__ == "__main__":
	try:
		main()

	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)
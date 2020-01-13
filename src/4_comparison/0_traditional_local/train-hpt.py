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
import pandas as pd

from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
from hyperopt.pyll.base import scope
from imblearn import over_sampling, under_sampling
from imblearn.pipeline import make_pipeline as make_pipeline_imb


def parse_arguments(argv):
	"""Parses execution arguments and replaces default values.

    Args:
      argv: Input arguments from sys.

    Returns:
      Dictionary of parsed arguments.
    """

	parser = argparse.ArgumentParser()

	parser.add_argument('modeldir', type=str,
						help="directory containing TF graph definition 'graph.meta'")
	parser.add_argument('--classifier', dest='classifier',
						help='classifier to train (RF/TempCNN/GRU-RNNbi/GRU-RNN)')
	parser.add_argument('--datadir', type=str, default=None,
						help='directory containing the data')
	parser.add_argument('-train_on', '--train_on', type=str, default="2001",
						help='train years')
	parser.add_argument('-ssize', '--ssize', type=int, default=10,
						help='Sample size')
	parser.add_argument('--channels', dest='channels',
						help='number of channels')
	parser.add_argument('--partition', type=str, default=None,
						help='all')
	parser.add_argument('-cpus', '--cpus', type=int, default=14,
						help='cpus')
	parser.add_argument('-num_eval', '--num_eval', type=int, default=5,
						help='cpus')
	parser.add_argument('-w', '--writemodel', action="store_true",
						help='write out pngs for each tile with prediction, label etc.')
	parser.add_argument('--fold', type=str, default=None,
						help='fold')

	args, _ = parser.parse_known_args(args=argv[1:])
	return args

class hseach:
	rf_param_hyperopt = {
		'type': 'rf',
		'n_estimators': scope.int(hp.quniform('n_estimators', 20, 2000, 10)),
		'max_features': hp.choice('max_features', [ 'auto', 'sqrt' ]),
		'max_depth': scope.int(hp.quniform('max_depth', 10, 110, 11)),
		'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)), #randint
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
		'bootstrap': hp.choice('bootstrap', [True,False]),
	}

	svm_param_hyperopt = {
		'type': 'svm',
		# 'C': scope.float(hp.lognormal('C', 0, 1)),
		'C': hp.choice('C', [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]),
		'kernel': hp.choice('kernel', ['rbf']),
		# 'gamma': scope.float(hp.uniform('gamma', 0.001, 10000))
		'gamma': hp.choice('gamma', [0.01, 0.1, 1, 10, 100, 1000]),
	}

def hyperopt(param_space, X_train, y_train, X_test, y_test, args):
	resampling = over_sampling.RandomOverSampler(sampling_strategy='auto',
												 random_state=42)

	start = time.time()

	def objective_function(params):
		classifier_type = params[ 'type' ]
		del params['type']
		if classifier_type == 'rf':
			clf = RandomForestClassifier(**params)
		elif classifier_type == 'svm':
			clf = SVC(**params)
		else:
			return 0


		pl = make_pipeline_imb(resampling, clf)

		score = cross_val_score(pl, X_train, y_train, n_jobs=args.cpus, cv=3).mean()
		return {'loss': -score, 'status': STATUS_OK}

	rstate = np.random.RandomState(1)  # <== Use any number here but fixed

	trials = Trials()
	best_param = fmin(objective_function,
					  param_space,
					  algo=tpe.suggest,
					  max_evals=args.num_eval,
					  trials=trials,
					  rstate=rstate)

	loss = [x['result']['loss'] for x in trials.trials]

	joblib.dump(trials, os.path.join(args.modeldir,'hyperopt_trials_niters{}_ssize{}.pkl'.format(args.num_eval, args.ssize)))

	# best_param_values = [ x for x in best_param.values() ]
	#
	# del best_param_values['classifier_type']
	#
	# if best_param_values[2] == 0:
	# 	max_features = 'auto'
	# else:
	# 	max_features = 'sqrt'
	#
	# if best_param_values[0] == 0:
	# 	bootstrap = 'True'
	# else:
	# 	bootstrap = 'False'
	#
	# print("Best parameters: ", best_param)
	#
	# clf_best = RandomForestClassifier(n_estimators=int(best_param_values[5]),
	# 								  max_features=max_features,
	# 							      max_depth=int(best_param_values[1]),
	# 								  min_samples_leaf=int(best_param_values[3]),
	# 								  min_samples_split=int(best_param_values[4]),
	# 								  bootstrap=bootstrap,
	# 								  n_jobs=args.cpus)
	#
	# pl = make_pipeline_imb(resampling, clf_best)
	#
	# # clf_best.fit(X_train, y_train)
	# estimator_fit = pl.fit(X_train, y_train)
	#
	print("")
	print("##### Results")
	print("Score best parameters: ", min(loss) * -1)
	print("Best parameters: ", best_param)
	# print("Test Score: ", estimator_fit.score(X_test, y_test))
	print("Time elapsed: ", round(time.time() - start, 2))
	print("Parameter combinations evaluated: ", args.num_eval)
	#
	# if args.writemodel:
	# 	model_file = os.path.join(args.modeldir, 'model-' + args.classifier + '.h5')
	# 	# -- save the model
	# 	joblib.dump(clf_best, model_file)
	# 	print("Writing the model over path {}".format(model_file))

	return trials


def prep_data(partition, args):
	X = [ ]
	Y = [ ]
	IDS = [ ]

	for y in args.train_on.split('|'):
		if args.classifier == 'SVM':
			x, ids, y = readSITSData(
				os.path.join(args.datadir, 'fold{}'.format(args.fold),
							 'fcTS_{}_ssize{}_tyear{}_LCfinalmapv6_LCProp2.csv'.format(partition, args.ssize, y)), y, normalise=True)
		else:
			x, ids, y = readSITSData(
				os.path.join(args.datadir, 'fold{}'.format(args.fold),
							 'fcTS_{}_ssize{}_tyear{}_LCfinalmapv6_LCProp2.csv'.format(partition, args.ssize, y)), y)

		X.append(x)
		Y.append(y)
		IDS.append(ids)

	X = np.vstack(X)

	Y = [ item for sublist in Y for item in sublist ]
	IDS = [ item for sublist in IDS for item in sublist ]

	target_others = [Y, IDS]

	Y, IDS = [np.array(t) for t in target_others]

	return X, Y, IDS


def run_hyperopt(args):
	classif_type = ['RF','SVM']
	if args.classifier not in classif_type:
		print('ERR: select an available classifier (RF, SVM)')
		sys.exit(1)

	if args.partition == 'all':
		X_train, y_train, ids_train = prep_data(args)
	elif args.partition == 'val':
		X_train, y_train, ids_train = prep_data('train', args)
		X_val, y_val, ids_val =  prep_data('val', args)

	# Output filenames
	target_period = args.train_on.replace('|', '')

	modeldir = os.path.join(args.modeldir,'{}_ssize{}_trials{}_trainon{}'.format(args.classifier,args.ssize,args.num_eval,target_period),'hpt')
	args.modeldir = modeldir
	if not os.path.exists(args.modeldir):
		os.makedirs(args.modeldir)

	classifiers_config = {
		'RF': {
			'hsearch': hseach.rf_param_hyperopt
		},
		'SVM': {
			'hsearch': hseach.svm_param_hyperopt
		}
	}

	param_hyperopt = classifiers_config[args.classifier]['hsearch']

	search_space = hp.choice('classifier_type', [param_hyperopt])

	results_hyperopt = hyperopt(search_space, X_train, y_train, X_val, y_val, args)

	if args.classifier == 'RF':
		tpe_results = pd.DataFrame({'loss': [ x[ 'loss' ] for x in results_hyperopt.results],
									'iteration': results_hyperopt.idxs_vals[0]['n_estimators'],
									'n_estimators': results_hyperopt.idxs_vals[ 1 ][ 'n_estimators' ],
									'max_features': results_hyperopt.idxs_vals[ 1 ][ 'max_features' ],
									'max_depth': results_hyperopt.idxs_vals[ 1 ][ 'max_depth' ],
									'min_samples_split': results_hyperopt.idxs_vals[ 1 ][ 'min_samples_split' ],
									'min_samples_leaf': results_hyperopt.idxs_vals[ 1 ][ 'min_samples_leaf' ],
									'bootstrap': results_hyperopt.idxs_vals[ 1 ][ 'bootstrap' ],
									},
								   )
	elif args.classifier == 'SVM':
		tpe_results = pd.DataFrame({'loss': [x['loss' ] for x in results_hyperopt.results],
									'iteration': results_hyperopt.idxs_vals[0]['C'],
									'C': results_hyperopt.idxs_vals[1]['C'],
									'kernel': results_hyperopt.idxs_vals[1]['kernel'],
									'gamma': results_hyperopt.idxs_vals[1]['gamma']
									},
								   )

	tpe_results.to_csv(os.path.join(args.modeldir,'trials_niters{}_ssize{}.csv'.format(args.num_eval, args.ssize)), index=False)


def main():

	args = parse_arguments(sys.argv)

	run_hyperopt(args)


if __name__ == "__main__":
	try:
		main()

	except(RuntimeError):
		print >> sys.stderr
		sys.exit(1)
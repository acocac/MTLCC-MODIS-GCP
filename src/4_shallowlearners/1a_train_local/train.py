"""
Train shallow learner models using a local machine

Example invocation::

    python 4_shallowlearners/train.py MODELDIR
        -c RF
        -d DATADIR
        -y 2001
        -s 3000
        -t 10
        -d 7
        -n 12
        -f 0
        -r Copernicusraw
        -b 1

acocac@gmail.com
"""

import argparse
import os
import pickle as pkl
import sys
import time

from imblearn import over_sampling
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC

from sits_func import *


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
  parser.add_argument('-m','--classifier', dest='classifier',
                      help='classifier to train (RF/SVM)')
  parser.add_argument('-d','--datadir', type=str, default=None,
                      help='directory containing the data')
  parser.add_argument('-y','-train_on', '--train_on', type=str, default="2001",
                      help='train years')
  parser.add_argument('-s','-ssize', '--ssize', type=int, default=10,
                      help='Sample size')
  parser.add_argument('-t','-trials', '--trials', type=int, default=10,
                      help='Number of trials')
  parser.add_argument('-d','--channels', dest='channels',
                      help='number of channels')
  parser.add_argument('-n','-cpus', '--cpus', type=int, default=14,
                      help='cpus')
  parser.add_argument('-f','--fold', type=str, default=None,
                      help='fold')
  parser.add_argument('-r', '--reference', type=str, required=True,
                      help='Dataset')
  parser.add_argument('-b', '--bestmodel', type=int, default=1,
                      help='int')

  args, _ = parser.parse_known_args(args=argv[ 1: ])
  return args


def getBestModelfromTrials(trials, order, STATUS_OK):
    valid_trial_list = [trial for trial in trials
                        if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]

    index_having_minumum_sort = np.argsort(losses)
    index_having_minumum_loss = index_having_minumum_sort[order-1]

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
                             'fcTS_{}_ssize{}_tyear{}_{}.csv'.format(partition, args.ssize, y, args.reference)),
                y, normalise=True)
        else:
            x, ids, y = readSITSData(
                os.path.join(args.datadir, 'fold{}'.format(args.fold),
                             'fcTS_{}_ssize{}_tyear{}_{}.csv'.format(partition, args.ssize, y, args.reference)),
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

    bestmodel = getBestModelfromTrials(trials.trials, args.bestmodel, STATUS_OK)

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

    model_file = os.path.join(outdir, 'model-{}_bm{}.h5'.format(args.classifier,args.bestmodel))
    #-- save the model
    joblib.dump(estimator_fit, model_file)
    print("Writing the model over {}".format(model_file))

    eval_label = ['OA', 'train_time', 'test_time']

    history = np.zeros((len(eval_label), 1))

    history_file = os.path.join(outdir,'trainingHistory-{}_bm{}.csv'.format(args.classifier,args.bestmodel))

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
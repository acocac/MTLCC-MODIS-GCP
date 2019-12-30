from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf

from constants.constants import *

from . import inputs
from . import model

import functools

import logging
from pathlib import Path
# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('--modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
  parser.add_argument('--datadir', type=str, default=None,
                      help='directory containing the data (defaults to environment variable $datadir)')
  parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU')
  # parser.add_argument('-d', '--train_on', type=str, default="2015", nargs='+',
  #                     help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('-d', '--dataset', type=str, default="2016",
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('-b', '--batchsize', type=int, default=32, help='batchsize')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  # parser.add_argument('-o', '--overwrite', action="store_true", help='overwrite graph. may lead to problems with checkpoints compatibility')
  parser.add_argument('-s', '--shuffle', type=bool, default=True, help="batch shuffling")
  parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                               "Will at each get_next randomly choose "
                                                                               "<temporal_samples> elements from "
                                                                               "timestack. Defaults to None -> no temporal sampling")
  parser.add_argument('--save_frequency', type=int, default=64, help="save frequency")
  parser.add_argument('--summary_frequency', type=int, default=64, help="summary frequency")
  parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
  parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
  parser.add_argument('--max_models_to_keep', type=int, default=5, help="maximum number of models to keep")

  parser.add_argument('--save_every_n_hours', type=int, default=1, help="save checkpoint every n hours")
  parser.add_argument('--queue_capacity', type=int, default=256, help="Capacity of queue")
  parser.add_argument('--allow_growth', type=bool, default=True, help="Allow dynamic VRAM growth of TF")
  parser.add_argument('--limit_batches', type=int, default=-1,
                      help="artificially reduce number of batches to encourage overfitting (for debugging)")
  parser.add_argument('-exp', '--experiment', type=str, default="bands", help='Experiment to train')
  parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6raw_LCType1",
                      help='Reference dataset to train')
  parser.add_argument('--learning_rate', type=float, default=None,
                      help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
  parser.add_argument('--convrnn_filters', type=int, default=8,
                      help="number of convolutional filters in ConvLSTM/ConvGRU layer")
  parser.add_argument('--convrnn_layers', type=int, default=1,
                      help="number of convolutional recurrent layers")
  parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
  # parser.add_argument('-t', '--writetiles', action="store_true",
  #                     help='write out pngs for each tile with prediction, label etc.')
  # parser.add_argument('-gt', '--writegt', action="store_true",
  #                     help='write out pngs for each tile label etc.')
  # parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')

  args, _ = parser.parse_known_args(args=argv[1:])
  return args

def evaluate(args):

    latest_ckpt = tf.train.latest_checkpoint(args.modeldir)

    data_image, data_label, data_iterator = inputs.input_fn_eval(args, mode=tf.estimator.ModeKeys.PREDICT)

    # Run tensor and generate 3 top data as array
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        sess.run([data_iterator.initializer])

        image_arr = [ ]
        label_arr = [ ]
        for i in range(3):
            (x250, x500, doy, year), label = sess.run([data_image, data_label])
            image_arr.append(x250)
            label_arr.append(label)

    # Predict 3 top data
    pred_fn = tf.contrib.predictor.from_saved_model(latest_ckpt)
    pred = pred_fn({'inputs': image_arr})

    # Output and Compare
    print('Predicted Values:', pred[ 'classes' ])
    print('Real Values:', label_arr)

def main():

  args = parse_arguments(sys.argv)

  tf.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Params
  #   params = {
  #       'dim_chars': 100,
  #       'dim': 300,
  #       'dropout': 0.5,
  #       'num_oov_buckets': 1,
  #       'epochs': 25,
  #       'batch_size': 20,
  #       'buffer': 15000,
  #       'filters': 50,
  #       'kernel_size': 3,
  #       'lstm_size': 100,
  #       'words': str(Path(DATADIR, 'vocab.words.txt')),
  #       'chars': str(Path(DATADIR, 'vocab.chars.txt')),
  #       'tags': str(Path(DATADIR, 'vocab.tags.txt')),
  #       'glove': str(Path(DATADIR, 'glove.npz'))
  #   }

  evaluate(args)

if __name__ == "__main__":
  main()
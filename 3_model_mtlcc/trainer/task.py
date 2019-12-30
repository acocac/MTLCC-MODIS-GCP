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

import json
import functools


def get_trial_id():
    """Returns the trial id if it exists, else "0"."""
    trial_id = json.loads(
        os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    return trial_id if trial_id else "1"


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
  parser.add_argument('-d', '--train_on', type=str, default="2015",
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('-dt', '--train_year', type=str, default="2015", help='Train year')
  parser.add_argument('-b', '--batchsize', type=int, default=32, help='batchsize')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  parser.add_argument('-e', '--epochs', type=int, default=5, help="epochs")
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
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
  parser.add_argument('--convrnn_filters', type=int, default=8,
                      help="number of convolutional filters in ConvLSTM/ConvGRU layer")
  parser.add_argument('--convrnn_layers', type=int, default=1,
                      help="number of convolutional recurrent layers")
  parser.add_argument('-step', '--step', type=str, default="training", help='step')

  args, _ = parser.parse_known_args(args=argv[1:])
  return args


def train_and_evaluate(args):
    """Runs model training and evaluation using TF Estimator API."""

    num_samples_train = 0
    num_samples_test = 0

    # if if num batches artificially reduced -> adapt sample size
    if args.limit_batches > 0:
        num_samples_train = args.limit_batches * args.batchsize * len(args.train_on.split(' '))
        num_samples_test = args.limit_batches * args.batchsize * len(args.train_on.split(' '))
    else:
        num_samples_train_out, _ = inputs.input_filenames(args, mode=tf.estimator.ModeKeys.TRAIN)
        num_samples_train += int(num_samples_train_out.get_shape()[0]) * len(args.train_on.split(' '))
        num_samples_test_out, _ = inputs.input_filenames(args, mode=tf.estimator.ModeKeys.EVAL)
        num_samples_test += len(num_samples_test_out) * len(args.train_on.split(' '))

    train_steps = num_samples_train / args.batchsize * args.epochs
    test_steps = num_samples_test / args.batchsize
    ckp_steps = num_samples_train / args.batchsize

    train_input_fn = functools.partial(
        inputs.input_fn_train_multiyear,
        args,
        mode=tf.estimator.ModeKeys.TRAIN
    )

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=train_steps)

    eval_input_fn = functools.partial(
        inputs.input_fn_train_multiyear,
        args,
        mode=tf.estimator.ModeKeys.EVAL
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        start_delay_secs=0,
        throttle_secs=1,  # eval no more than every x seconds
        steps=test_steps, # evals on x batches        steps=test_steps, # evals on x batches
        name='eval'
    )

    distribution = tf.contrib.distribute.MirroredStrategy()
    # multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    run_config = tf.estimator.RunConfig(
        # save_checkpoints_steps=args.save_frequency,
        # save_summary_steps=args.summary_frequency,
        train_distribute=distribution,
        # eval_distribute=distribution,
        save_checkpoints_steps=ckp_steps,
        save_summary_steps=ckp_steps,
        keep_checkpoint_max=args.max_models_to_keep,
        keep_checkpoint_every_n_hours=args.save_every_n_hours,
        model_dir=os.path.join(args.modeldir,get_trial_id()),
        log_step_count_steps=args.summary_frequency # set the frequency of logging steps for loss function
    )

    estimator = model.build_estimator(run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():

  args = parse_arguments(sys.argv)

  train_and_evaluate(args)


if __name__ == "__main__":
  main()
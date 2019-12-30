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

# import logging
# # from pathlib import Path
# # Logging
# Path('results').mkdir(exist_ok=True)
# tf.logging.set_verbosity(logging.INFO)
# handlers = [
#     logging.FileHandler('results/main.log'),
#     logging.StreamHandler(sys.stdout)
# ]
# logging.getLogger('tensorflow').handlers = handlers

tf.logging.set_verbosity(tf.logging.INFO)


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

    # def serving_input_receiver_fn():
    #     feature_placeholders = {
    #         'x250': tf.placeholder(dtype=tf.float32, shape=[ None, None, None, None, 2 ], name='x250'),
    #         'x500': tf.placeholder(dtype=tf.float32, shape=[ None, None, None, None, 5 ], name='x500'),
    #         'doy': tf.placeholder(dtype=tf.float32, shape=[ None, None ], name='doy'),
    #         'year': tf.placeholder(dtype=tf.float32, shape=[ None, None ], name='year')}
    #
    #     return tf.estimator.export.ServingInputReceiver(feature_placeholders, feature_placeholders)

    # latest_exporter = tf.estimator.LatestExporter(
    #     name="models",
    #     serving_input_receiver_fn=serving_input_receiver_fn,
    #     exports_to_keep=10)
    # best_exporter = tf.estimator.BestExporter(
    #     serving_input_receiver_fn=serving_input_receiver_fn,
    #     exports_to_keep=1)
    # exporters = [latest_exporter, best_exporter]

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
    #
    # exporter = tf.estimator.FinalExporter(
    #     'export', functools.partial(
    #         inputs.tfrecord_serving_input_fn,
    #         feature_spec=feature_spec,
    #         label_name='labels'))
    #

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        start_delay_secs=0,
        throttle_secs=1,  # eval no more than every x seconds
        steps=test_steps, # evals on x batches        steps=test_steps, # evals on x batches
        # # steps=None,  # evals on x batches
        # exporters=exporters,
        name='eval'
    )

    run_config = tf.estimator.RunConfig(
        # save_checkpoints_steps=args.save_frequency,
        # save_summary_steps=args.summary_frequency,
        save_checkpoints_steps=ckp_steps,
        save_summary_steps=ckp_steps,
        keep_checkpoint_max=args.max_models_to_keep,
        keep_checkpoint_every_n_hours=args.save_every_n_hours,
        model_dir=os.path.join(args.modeldir,get_trial_id()),
        log_step_count_steps=args.summary_frequency # set the frequency of logging steps for loss function
    )

    estimator = model.build_estimator(run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # utils.dump_object(estimator, model_output_path)
    #
    # if metadata.HYPERPARAMTER_TUNING:
    #     env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    #     # First find out if there's a task value on the environment variable.
    #     # If there is none or it is empty define a default one.
    #     task_data = env.get('task') or {'type': 'master', 'index': 0}
    #
    #     trial = task_data.get('trial')
    #     if trial is not None:
    #         output_dir = os.path.join(args.modeldir, trial)
    #     else:
    #         output_dir = args.output_path
    #
    #     # HyperparameterSpec object in the job request to match the chosen name
    #     hpt = hypertune.HyperTune()
    #     hpt.report_hyperparameter_tuning_metric(
    #         hyperparameter_metric_tag='Taxi Model Accuracy',
    #         metric_value=np.mean(scores),
    #         global_step=900)

    # ###method 1
    # your_feature_spec = {
    #     'x250': tf.placeholder(dtype=tf.float32, shape=[ None, None, None, None, 2 ], name='x250'),
    #     'x500': tf.placeholder(dtype=tf.float32, shape=[ None, None, None, None, 5 ], name='x500'),
    #     'doy': tf.placeholder(dtype=tf.float32, shape=[ None, None ], name='doy'),
    #     'year': tf.placeholder(dtype=tf.float32, shape=[ None, None ], name='year')
    # }
    #
    # def _serving_input_receiver_fn():
    #     serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
    #                                            name='x')
    #     # key (e.g. 'examples') should be same with the inputKey when you
    #     # buid the request for prediction
    #     receiver_tensors = {'examples': serialized_tf_example}
    #     features = tf.parse_example(serialized_tf_example, your_feature_spec)
    #     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    #
    # estimator.export_savedmodel(os.path.join(args.modeldir, "export"), serving_input_fn)

    # # ###method 2
    # def serving_input_fn():
    #     feature_spec = {
    #         'x250': tf.io.FixedLenFeature([ ], tf.string),
    #         'x500': tf.io.FixedLenFeature([ ], tf.string),
    #         'doy': tf.io.FixedLenFeature([ ], tf.string),
    #         'year': tf.io.FixedLenFeature([ ], tf.string),
    #     }
    #
    #     serialized_tf_example = tf.placeholder(dtype=tf.string,
    #                                            shape=[None],
    #                                            name='input_example_tensor')
    #
    #     receiver_tensors = {'example': serialized_tf_example}
    #
    #     features = tf.parse_example(serialized_tf_example, feature_spec)
    #
    #     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    #
    # exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)
    #
    # latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.modeldir,get_trial_id()))
    #
    # last_eval = estimator.evaluate(
    #     eval_input_fn,
    #     checkpoint_path=latest_ckpt,
    #     steps=1
    # )
    #
    # exporter.export(estimator, os.path.join(args.modeldir,get_trial_id()), latest_ckpt, last_eval, is_the_final_export=True)
    # print('exported')

    ##method 3
    # feature_format = {
    #     'x250/data': tf.io.FixedLenFeature([ ], tf.string),
    #     'x250/shape': tf.io.FixedLenFeature([ 4 ], tf.int64),
    #     'x250aux/data': tf.io.FixedLenFeature([ ], tf.string),
    #     'x250aux/shape': tf.io.FixedLenFeature([ 4 ], tf.int64),
    #     'x500/data': tf.io.FixedLenFeature([ ], tf.string),
    #     'x500/shape': tf.io.FixedLenFeature([ 4 ], tf.int64),
    #     'dates/doy': tf.io.FixedLenFeature([ ], tf.string),
    #     'dates/year': tf.io.FixedLenFeature([ ], tf.string),
    #     'dates/shape': tf.io.FixedLenFeature([ 1 ], tf.int64),
    #     'labels/data': tf.io.FixedLenFeature([ ], tf.string),
    #     'labels/shape': tf.io.FixedLenFeature([ 4 ], tf.int64)
    # }
    #
    # serialized_example = tf.placeholder(dtype=tf.string, shape=None,
    #                                        name='input_example_tensor')
    #
    # feature = tf.io.parse_single_sequence_example(serialized_example, feature_format)
    #
    # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #
    # export_model = classifier.export_savedmodel(os.path.join(args.modeldir,get_trial_id()), serving_input_receiver_fn)
    #
    # def serving_input_fn():
    #     feature_placeholders = {
    #             'x250': tf.placeholder(dtype=tf.float32, shape=[None,46, 24, 24, 2 ], name='x250'),
    #             'x500': tf.placeholder(dtype=tf.float32, shape=[46, 24, 24, 5 ], name='x500'),
    #             'doy': tf.placeholder(dtype=tf.float32, shape=[24*24], name='doy'),
    #             'year': tf.placeholder(dtype=tf.float32, shape=[24*24], name='year')
    #     }
    #
    #     return tf.estimator.export.ServingInputReceiver(feature_placeholders, feature_placeholders)
    #
    # estimator.export_savedmodel(export_dir_base=os.path.join(args.modeldir,get_trial_id()), serving_input_receiver_fn=serving_input_fn)

    #
    #
    # def tfrecords_serving_input_fn():
    #     """Build the serving inputs."""
    #
    #     # inputs = {'inputs':  tf.compat.v1.placeholder(tf.float32, [ None, 784 ])}
    #
    #     feature_placeholders = {
    #         'x250': tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 2], name='x250'),
    #         'x500': tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 5], name='x500'),
    #         'doy': tf.placeholder(dtype=tf.float32, shape=[None, None ], name='doy'),
    #         'year': tf.placeholder(dtype=tf.float32, shape=[None, None ], name='year')
    #     }
    #     # features = {
    #     #     key: tf.expand_dims(tensor, -1)
    #     #     for key, tensor in feature_placeholders.items()
    #     # }
    #     # x=tf.placeholder(dtype=tf.float32,shape=[None, None, None, None, None],name='x'),
    #     #
    #     # input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
    #     #     'x':x
    #     # })()
    #
    #     # x = tf.concat((feature_placeholders['x250'], feature_placeholders['x500'], feature_placeholders['doy'], feature_placeholders['year']), axis=-1, name="x")
    #
    #     #
    #     # inputs = {
    #     #     'x250': tf.placeholder(dtype=tf.float32,shape=[None, None, 24, 24, 2],name='x250'),
    #     #     'x500': tf.placeholder(dtype=tf.float32,shape=[None, None, 24, 24, 5],name='x500'),
    #     #     'doy': tf.placeholder(dtype=tf.float32,shape=[None, None, 24, 24, 1],name='doy'),
    #     #     'year': tf.placeholder(dtype=tf.float32, shape=[None, None, 24, 24, 1], name='year')
    #     # }  # the shape of this dict should match the shape of your JSON
    #     #
    #     # return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)
    #     return tf.estimator.export.ServingInputReceiver(features,feature_placeholders)
    #
    #     # exporter = tf.estimator.FinalExporter('MODIS-LC',
    #     #                                       tfrecords_serving_input_fn)
    #
    # ##TODO
    # exporter = estimator.export_saved_model(
    #     export_dir_base=os.path.join(args.modeldir, "export"),
    #     serving_input_receiver_fn=tfrecords_serving_input_fn)
    # print('model is saved ', exporter)


def main():

  args = parse_arguments(sys.argv)

  # tf.logging.set_verbosity(tf.compat.v1.logging.INFO)

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

  train_and_evaluate(args)


if __name__ == "__main__":
  main()
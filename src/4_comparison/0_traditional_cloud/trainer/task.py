# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Executes model training and evaluation."""

import argparse
import logging
import os

import hypertune
import numpy as np
from datetime import datetime
from sklearn import model_selection
from trainer import metadata
from trainer import model
from trainer import utils

import time

def _train_and_evaluate(estimator, output_dir, train_on, datadir, ssize):
    """Runs model training and evaluation.

    Args:
      estimator: (pipeline.Pipeline), Pipeline instance assemble pre-processing
        steps and model training
      dataset: (pandas.DataFrame), DataFrame containing training data
      output_dir: (string), directory that the trained model will be exported

    Returns:
      None
    """

    x_train, y_train, ids_train = utils.prep_data('train', train_on, datadir, ssize)
    x_val, y_val, ids_val = utils.prep_data('test', train_on, datadir, ssize)

    # x_train, y_train, x_val, y_val = utils.data_train_test_split(dataset)
    estimator.fit(x_train, y_train)

    # Write model and eval metrics to `output_dir`
    model_output_path = os.path.join(output_dir, 'model',
                                     metadata.MODEL_FILE_NAME)

    utils.dump_object(estimator, model_output_path)

    if metadata.HYPERPARAMTER_TUNING:
        # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)
        scores = model_selection.cross_val_score(estimator, x_val, y_val, cv=3)

        logging.info('Scores: %s', scores)

        # The default name of the metric is training/hptuning/metric.
        # We recommend that you assign a custom name
        # The only functional difference is that if you use a custom name,
        # you must set the hyperparameterMetricTag value in the
        # HyperparameterSpec object in the job request to match the chosen name
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=np.mean(scores),
            global_step=900)

def run_experiment(arguments):
    """Testbed for running model training and evaluation."""
    # Get data for training and evaluation

    logging.info('Arguments: %s', arguments)

    # Get estimator
    estimator = model.get_estimator(arguments)

    # Run training and evaluation
    _train_and_evaluate(estimator, arguments.job_dir, arguments.train_on, arguments.datadir, arguments.ssize)


def _parse_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log-level',
        help='Logging level.',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN',
        ],
        default='INFO',
    )

    parser.add_argument(
        '--job-dir',
        help='Output directory for exporting model and other metadata.',
        required=True,
    )

    parser.add_argument(
        '--n_estimators',
        help='Number of trees in random forest',
        default=50,
        type=int,
    )

    parser.add_argument(
        '--max_features',
        help='Number of features to consider at every split',
        default='auto',
        type=str,
    )

    parser.add_argument(
        '--max_depth',
        help='Maximum number of levels in tree',
        type=int,
        default=3,
    )

    parser.add_argument(
        '--min_samples_split',
        help='Minimum number of samples required to split a node',
        default=2,
        type=int,
    )

    parser.add_argument(
        '--min_samples_leaf',
        help='Minimum number of samples required at each leaf node',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--bootstrap',
        help='Method of selecting samples for training each tree',
        default='auto',
        type=str,
    )

    parser.add_argument(
        '--datadir',
        type=str,
        default=None,
        help='directory containing the data')

    parser.add_argument('--train_on',
                        type=str,
                        default="2001",
                        help='train years')

    parser.add_argument('--ssize',
                        type=int,
                        default=10,
                        help='Sample size')

    return parser.parse_args()


def main():
    """Entry point"""

    arguments = _parse_args()
    logging.basicConfig(level=arguments.log_level)
    # Run the train and evaluate experiment
    start_train_time = time.time()
    time_start = datetime.utcnow()
    run_experiment(arguments)
    train_time = round(time.time() - start_train_time, 2)
    print('Training time (s): ', train_time)

    time_end = datetime.utcnow()
    time_elapsed = time_end - time_start
    logging.info('Experiment elapsed time: {} seconds'.format(
        time_elapsed.total_seconds()))


if __name__ == '__main__':
    main()

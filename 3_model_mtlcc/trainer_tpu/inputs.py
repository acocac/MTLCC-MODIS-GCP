# Copyright 2019 Google Inc. All Rights Reserved.

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
"""Input and preprocessing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import os
import configparser
import csv
import numpy as np
from tensorflow.python.lib.io import file_io
import re

from trainer import utils
from constants.constants import *

from google.cloud import storage


def input_fn_train_singleyear(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.train_on[0],
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  # iterator = tfdataset.dataset.make_one_shot_iterator()
  iterator = tf.compat.v1.data.make_initializable_iterator(tfdataset)
  #
  tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

  features, labels = iterator.get_next()

  return features, labels


def input_fn_train_multiyear(args, params, mode):
  """Reads TFRecords and returns the features and labels."""

  def input_fn(params):

    datasets_dict = dict()

    for section in args.train_on.split(' '):
      datasets_dict[section] = dict()

      dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=section,
                        experiment=args.experiment, reference=args.reference)

      if mode == tf.estimator.ModeKeys.TRAIN:
        partition = TRAINING_IDS_IDENTIFIER
      elif mode == tf.estimator.ModeKeys.EVAL:
        partition = TESTING_IDS_IDENTIFIER
      elif mode == tf.estimator.ModeKeys.PREDICT:
        partition = EVAL_IDS_IDENTIFIER

      datasets_dict[ section ][ partition ] = dict()

      # datasets_dict[section][partition] = dict()
      tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                             args.fold,
                                                             args.batchsize,
                                                             prefetch_batches=args.prefetch,
                                                             num_batches=args.limit_batches)

      # iterator = tfdataset.dataset.make_one_shot_iterator()
      datasets_dict[section][partition]["tfdataset"] = tfdataset

    if len(args.train_on.split(' ')) > 1:

      ds = datasets_dict[args.train_on.split(' ')[0]][partition]["tfdataset"]

      for section in args.train_on.split(' ')[1:]:
        ds = ds.concatenate(datasets_dict[section][partition]["tfdataset"])

    else:
      ds = datasets_dict[args.train_on.split(' ')[0]][partition]["tfdataset"]

    # iterator = tfdataset.dataset.make_one_shot_iterator()
    # iterator = tf.compat.v1.data.make_initializable_iterator(ds)
    # # #
    # tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    #
    # features, labels = iterator.get_next()
    # # print(features, labels)
    # return features, labels

    ##MULTI-GPU
    return ds

  return input_fn

def input_fn_eval(args, mode):
  """Reads TFRecords and returns the features and labels."""

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=args.dataset,
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.PREDICT:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  # iterator = tfdataset.dataset.make_one_shot_iterator()
  iterator = tf.compat.v1.data.make_initializable_iterator(tfdataset)
  #
  tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

  features, labels = iterator.get_next()

  if mode == tf.estimator.ModeKeys.PREDICT:
    # return features
    return features, labels

  # if mode == tf.estimator.ModeKeys.PREDICT:
  #   return features, labels, iterator

  return features, labels


def input_filenames(args, mode):
  """Reads TFRecords and returns the features and labels."""

  if args.step == 'training':
    target = args.train_on.split(' ')[0]
  else:
    target = args.dataset

  dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=target,
                           experiment=args.experiment, reference=args.reference)

  if mode == tf.estimator.ModeKeys.TRAIN:
    partition = TRAINING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.EVAL:
    partition = TESTING_IDS_IDENTIFIER
  elif mode == tf.estimator.ModeKeys.PREDICT:
    partition = EVAL_IDS_IDENTIFIER

  # datasets_dict[section][partition] = dict()
  tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                         args.fold,
                                                         args.batchsize,
                                                         prefetch_batches=args.prefetch,
                                                         num_batches=args.limit_batches)

  return filenames, dataset


class Dataset():
  """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

  def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False, experiment="all",
               reference="MCD12Q1v6_cleaned", step="training"):
    self.verbose = verbose

    self.augment = augment

    self.experiment = experiment
    self.reference = reference

    # parser reads serialized tfrecords file and creates a feature object
    parser = utils.parser()
    if self.experiment == "bands" or self.experiment == "bandswodoy":
      self.parsing_function = parser.parse_example_bands
    elif self.experiment == "indices":
      self.parsing_function = parser.parse_example_bandsaux
    elif self.experiment == "bandsaux":
      self.parsing_function = parser.parse_example_bandsaux
    elif self.experiment == "all":
      self.parsing_function = parser.parse_example_bandsaux
    elif self.experiment == "bandswoblue":
      self.parsing_function = parser.parse_example_bandswoblue
    elif self.experiment == "bands250m" or self.experiment == "evi2":
      self.parsing_function = parser.parse_example_bands250m

    self.temp_samples = temporal_samples
    self.section = section
    self.step = step

    # if datadir is None:
    #    dataroot=os.environ["datadir"]
    # else:
    dataroot = datadir

    # csv list of geotransforms of each tile: tileid, xmin, xres, 0, ymax, 0, -yres, srid
    # use querygeotransform.py or querygeotransforms.sh to generate csv
    # fills dictionary:
    # geotransforms[<tileid>] = (xmin, xres, 0, ymax, 0, -yres)
    # srid[<tileid>] = srid
    self.geotransforms = dict()
    # https://en.wikipedia.org/wiki/Spatial_reference_system#Identifier
    self.srids = dict()
    with file_io.FileIO(os.path.join(dataroot, "geotransforms.csv"), 'r') as f:  # gcp
      # with open(os.path.join(dataroot, "geotransforms.csv"),'r') as f:
      reader = csv.reader(f, delimiter=',')
      for row in reader:
        # float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
        self.geotransforms[str(row[0])] = (
          float(row[1]), float(row[2]), int(row[3]), float(row[4]), int(row[5]), float(row[6]))
        self.srids[str(row[0])] = int(row[7])

    classes = os.path.join(dataroot, "classes_" + reference + ".txt")
    with file_io.FileIO(classes, 'r') as f:  # gcp
      # with open(classes, 'r') as f:
      classes = f.readlines()

    self.ids = list()
    self.classes = list()
    for row in classes:
      row = row.replace("\n", "")
      if '|' in row:
        id, cl = row.split('|')
        self.ids.append(int(id))
        self.classes.append(cl)

    ## create a lookup table to map labelids to dimension ids
    # # map data ids [0, 2, 4,..., nclasses_originalID]
    # labids = tf.constant(self.ids, dtype=tf.int64)
    #
    # # to dimensions [0, 1, 2, ... nclasses_orderID]
    # dimids = tf.constant(list(range(0, len(self.ids), 1)), dtype=tf.int64)

    # if self.step == "training":
    #   self.id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(labids, dimids),
    #                                                      default_value=-1)
    #
    #   self.inverse_id_lookup_table = tf.contrib.lookup.HashTable(
    #     tf.contrib.lookup.KeyValueTensorInitializer(dimids, labids),
    #     default_value=-1)

    # self.classes = [cl.replace("\n","") for cl in f.readlines()]

    cfgpath = os.path.join(dataroot, "dataset.ini")
    print(cfgpath)
    # load dataset configs
    datacfg = configparser.ConfigParser()
    with file_io.FileIO(cfgpath, 'r') as f:  # gcp
      datacfg.read_file(f)
    # datacfg.read(cfgpath)
    cfg = datacfg[section]

    self.tileidfolder = os.path.join(dataroot, "tileids")
    self.datadir = os.path.join(dataroot, cfg["datadir"])

    assert 'pix250' in cfg.keys()
    assert 'nobs' in cfg.keys()
    assert 'nbands250' in cfg.keys()
    assert 'nbands500' in cfg.keys()

    self.tiletable = cfg["tiletable"]

    self.nobs = int(cfg["nobs"])

    self.expected_shapes = self.calc_expected_shapes(int(cfg["pix250"]),
                                                     int(cfg["nobs"]),
                                                     int(cfg["nbands250"]),
                                                     int(cfg["nbands500"]),
                                                     )

    # expected datatypes as read from disk
    self.expected_datatypes = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)

  def calc_expected_shapes(self, pix250, nobs, bands250, bands500):
    pix250 = pix250;
    pix500 = pix250 / 2;
    x250shape = (nobs, pix250, pix250, bands250)
    x500shape = (nobs, pix500, pix500, bands500)
    doyshape = (nobs,)
    yearshape = (nobs,)
    labelshape = (nobs, pix250, pix250)

    return [x250shape, x500shape, doyshape, yearshape, labelshape]

  def transform_labels_training(self, feature):
    """
    1. take only first labelmap, as labels are not supposed to change
    2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
    """

    x250, x500, doy, year, labels = feature

    # take first label time [46,24,24] -> [24,24]
    # labels are not supposed to change over the time series
    # labels = self.id_lookup_table.lookup(labels)
    # labels = labels[0]

    return (x250, x500, doy, year), labels

  def transform_labels_evaluation(self,feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        """

        x250, x500, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        labels = labels[0]

        return x250, x500, doy, year, labels

  def normalize_old(self, feature):

    x250, x500, doy, year, labels = feature
    x250 = tf.scalar_mul(1e-4, tf.cast(x250, tf.float32))
    x500 = tf.scalar_mul(1e-4, tf.cast(x500, tf.float32))

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year / cancel year

    return x250, x500, doy, year, labels

  def normalize_bands250m(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x500, doy, year, labels = feature

    # normal minx/max domain
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # 250m
    # SR
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
    norm250m = [x_normed_red, x_normed_NIR]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    # cancel effect 500m
    x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32)

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, x500, doy, year, labels

  def normalize_evi2(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x500, doy, year, labels = feature

    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # indices
    fixed_range = [[-10000, 10000]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)

    norm250m = [x_normed_evi2]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    # cancel effect 500m
    x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32)

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)

    return norm250m, x500, doy, year, labels

  def normalize_bands(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x500, doy, year, labels = feature

    # normal minx/max domain
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # 250m
    # SR
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
    norm250m = [x_normed_red, x_normed_NIR]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    # 500m
    x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

    norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def normalize_bandswoblue(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x500, doy, year, labels = feature

    # normal minx/max domain
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # 250m
    # SR
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
    norm250m = [x_normed_red, x_normed_NIR]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    # 500m
    x_normed_green = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)

    norm500m = [x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def normalize_bandsaux(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x250aux, x500, doy, year, labels = feature

    x250aux = tf.tile(x250aux, [self.nobs, 1, 1, 1])

    # normal minx/max domain
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # SR
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    # 250m
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

    # 500m
    x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

    # bio 01
    fixed_range = [[-290, 320]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_bio01 = normalize_fixed(x250aux[:, :, :, 0], fixed_range, normed_range)

    # bio 12
    fixed_range = [[0, 11401]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_bio12 = normalize_fixed(x250aux[:, :, :, 1], fixed_range, normed_range)

    # elevation
    fixed_range = [[-444, 8806]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_ele = normalize_fixed(x250aux[:, :, :, 2], fixed_range, normed_range)

    norm250m = [x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def normalize_indices(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x250aux, x500, doy, year, labels = feature

    # normed values
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # SR
    # 250m
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

    # 500m
    x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

    # indices
    fixed_range = [[-10000, 10000]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)
    x_normed_ndwi = normalize_fixed(x250[:, :, :, 3], fixed_range, normed_range)
    x_normed_ndii1 = normalize_fixed(x250[:, :, :, 4], fixed_range, normed_range)
    x_normed_ndii2 = normalize_fixed(x250[:, :, :, 5], fixed_range, normed_range)
    x_normed_ndsi = normalize_fixed(x250[:, :, :, 6], fixed_range, normed_range)

    norm250m = [x_normed_red, x_normed_NIR, x_normed_evi2, x_normed_ndwi, x_normed_ndii1, x_normed_ndii2, x_normed_ndsi]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def normalize_all(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x250aux, x500, doy, year, labels = feature

    x250aux = tf.tile(x250aux, [self.nobs, 1, 1, 1])

    # normed values
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # SR
    # 250m
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

    # 500m
    x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

    # bio 01
    fixed_range = [[-290, 320]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_bio01 = normalize_fixed(x250aux[:, :, :, 0], fixed_range, normed_range)

    # bio 12
    fixed_range = [[0, 11401]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_bio12 = normalize_fixed(x250aux[:, :, :, 1], fixed_range, normed_range)

    # elevation
    fixed_range = [[-444, 8806]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_ele = normalize_fixed(x250aux[:, :, :, 2], fixed_range, normed_range)

    # indices
    fixed_range = [[-10000, 10000]]
    fixed_range = np.array(fixed_range)
    normed_range = np.array(normed_range)
    x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)
    x_normed_ndwi = normalize_fixed(x250[:, :, :, 3], fixed_range, normed_range)
    x_normed_ndii1 = normalize_fixed(x250[:, :, :, 4], fixed_range, normed_range)
    x_normed_ndii2 = normalize_fixed(x250[:, :, :, 5], fixed_range, normed_range)
    x_normed_ndsi = normalize_fixed(x250[:, :, :, 6], fixed_range, normed_range)

    norm250m = [x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele, x_normed_evi2, x_normed_ndwi,
                x_normed_ndii1, x_normed_ndii2, x_normed_ndsi]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) / 365

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def normalize_bandswodoy(self, feature):

    def normalize_fixed(x, current_range, normed_range):
      current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
      normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
      x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
              tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
      x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
        normed_min,
        tf.float32)
      return x_normed

    x250, x500, doy, year, labels = feature

    # normal minx/max domain
    fixed_range = [[-100, 16000]]
    fixed_range = np.array(fixed_range)
    normed_range = [[0, 1]]
    normed_range = np.array(normed_range)

    # 250m
    # SR
    x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
    x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
    norm250m = [x_normed_red, x_normed_NIR]
    norm250m = tf.stack(norm250m)
    norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

    # 500m
    x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
    x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
    x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
    x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
    x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

    norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
    norm500m = tf.stack(norm500m)
    norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

    doy = tf.cast(doy, tf.float32) - tf.cast(doy, tf.float32)

    year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

    return norm250m, norm500m, doy, year, labels

  def augment(self, feature):

    x250, x500, doy, year, labels = feature

    ## Flip UD
    # roll the dice
    condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

    # flip
    x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[1]), lambda: x250)
    x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[1]), lambda: x500)
    labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[1]), lambda: labels)

    ## Flip LR
    # roll the dice
    condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

    # flip
    x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[2]), lambda: x250)
    x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[2]), lambda: x500)
    labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[2]), lambda: labels)

    return x250, x500, doy, year, labels

  def temporal_sample(self, feature):
    """ randomy choose <self.temp_samples> elements from temporal sequence """

    n = self.temp_samples

    # skip if not specified
    if n is None:
      return feature

    x250, x500, doy, year, labels = feature

    max_obs = self.nobs

    shuffled_range = tf.random.shuffle(tf.range(max_obs))[0:n]

    idxs = -tf.nn.top_k(-shuffled_range, k=n).values

    x250 = tf.gather(x250, idxs)
    x500 = tf.gather(x500, idxs)
    doy = tf.gather(doy, idxs)
    year = tf.gather(year, idxs)

    return x250, x500, doy, year, labels

  def addIndices(self, features):

    def NDVI(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
      nd = 10000 * ((a - b) / (a + b))
      nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
      nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
      nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def NDWI(a, b):  # 10000*(double)(nir[ii]-swir1[ii]) / (double)(nir[ii]+swir1[ii]);
      nd = 10000 * ((a - b) / (a + b))
      nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def NDSI(a, b):  # 10000*(double)(green[ii]-swir2[ii]) / (double)(green[ii]+swir2[ii]);
      nd = 10000 * ((a - b) / (a + b))
      nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def NDII1(a, b):  # 10000*(double)(nir[ii]-swir2[ii]) / (double)(nir[ii]+swir2[ii])
      nd = 10000 * ((a - b) / (a + b))
      nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def NDII2(a, b):  # 10000*(double)(nir[ii]-swir3[ii]) / (double)(nir[ii]+swir3[ii]);
      nd = 10000 * ((a - b) / (a + b))
      nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    def resize(tensor, new_height, new_width):
      t = tf.shape(tensor)[0]
      h = tf.shape(tensor)[1]
      w = tf.shape(tensor)[2]
      d = tf.shape(tensor)[3]

      # stack batch on times to fit 4D requirement of resize_tensor
      stacked_tensor = tf.reshape(tensor, [t, h, w, d])
      reshaped_stacked_tensor = tf.compat.v1.image.resize_images(stacked_tensor, size=(new_height, new_width))
      return tf.reshape(reshaped_stacked_tensor, [t, new_height, new_width, d])

    x250, x250aux, x500, doy, year, labels = features

    px = tf.shape(x250)[2]

    x250 = tf.cast(x250, tf.float32)
    x500 = tf.cast(x500, tf.float32)

    x500_r = tf.identity(resize(x500, px, px))

    ndvi = NDVI(x250[:, :, :, 1], x250[:, :, :, 0])
    evi2 = EVI2(x250[:, :, :, 1], x250[:, :, :, 0])
    ndwi = NDWI(x250[:, :, :, 1], x500_r[:, :, :, 2])
    ndii1 = NDII1(x250[:, :, :, 1], x500_r[:, :, :, 3])
    ndii2 = NDII2(x250[:, :, :, 1], x500_r[:, :, :, 4])
    ndsi = NDSI(x500_r[:, :, :, 1], x500_r[:, :, :, 3])

    indices250m = [evi2, ndwi, ndii1, ndii2, ndsi]

    x250indices = tf.stack(indices250m)
    x250indices = tf.transpose(x250indices, [1, 2, 3, 0])

    x250m = tf.concat([x250, x250indices], axis=3)

    return x250m, x250aux, x500, doy, year, labels

  def addIndices250m(self, features):

    def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
      nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
      nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
      return tf.where(tf.is_finite(nd), nd, nd_inf)

    x250, x500, doy, year, labels = features

    x250 = tf.cast(x250, tf.float32)

    evi2 = EVI2(x250[:, :, :, 1], x250[:, :, :, 0])

    # indices250m = [evi2]
    indices250m = [evi2]

    x250indices = tf.stack(indices250m)
    x250indices = tf.transpose(x250indices, [1, 2, 3, 0])

    x250m = tf.concat([x250, x250indices], axis=3)

    return x250m, x500, doy, year, labels

  def MCD12Q1v6raw_LCType1(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 0 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6stable_LCType1(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 1 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6raw_LCProp1(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 2 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6stable_LCProp1(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 2 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6raw_LCProp2(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 4 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6stable01to15_LCProp2(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 5 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6stable01to03_LCProp2(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 6]

    return x250, x500, doy, year, labels

  def ESAraw(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 7 ]

    return x250, x500, doy, year, labels

  def ESAstable(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 8 ]

    return x250, x500, doy, year, labels

  def Copernicusraw(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 9 ]

    return x250, x500, doy, year, labels

  def Copernicusraw_fraction(self, feature):

    x250, x500, doy, year, labels = feature

    labels = tf.argmax(labels, axis=3)

    return x250, x500, doy, year, labels

  def watermask(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 10 ]

    return x250, x500, doy, year, labels

  def Copernicusnew_cf2others(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 11 ]

    return x250, x500, doy, year, labels

  def merge_datasets2own(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 12 ]

    return x250, x500, doy, year, labels

  def merge_datasets2HuHu(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 13 ]

    return x250, x500, doy, year, labels

  def merge_datasets2Tsendbazaretal2maps(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 14 ]

    return x250, x500, doy, year, labels

  def merge_datasets2Tsendbazaretal3maps(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 15 ]

    return x250, x500, doy, year, labels

  def mapbiomas_fraction(self, feature):

    x250, x500, doy, year, labels = feature

    labels = tf.argmax(labels, axis=3)

    return x250, x500, doy, year, labels

  def tilestatic(self, feature):

    x250, x500, doy, year, labels = feature

    labels = tf.tile(labels, [self.nobs, 1, 1, 1])

    return x250, x500, doy, year, labels

  def get_ids(self, partition, fold=0):

    def readids(path):
      with file_io.FileIO(path, 'r') as f:  # gcp
        # with open(path, 'r') as f:
        lines = f.readlines()
      ##ac            return [int(l.replace("\n", "")) for l in lines]
      return [str(l.replace("\n", "")) for l in lines]

    traintest = "{partition}_fold{fold}.tileids"
    eval = "{partition}.tileids"

    if partition == 'train':
      # e.g. train240_fold0.tileids
      path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
      return readids(path)
    elif partition == 'test':
      # e.g. test240_fold0.tileids
      path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
      return readids(path)
    elif partition == 'eval':
      # e.g. eval240.tileids
      path = os.path.join(self.tileidfolder, eval.format(partition=partition))
      return readids(path)
    else:
      raise ValueError("please provide valid partition (train|test|eval)")

  def create_tf_dataset(self, partition, fold, batchsize, prefetch_batches=None, num_batches=-1, threads=8,
                        drop_remainder=True, overwrite_ids=None):

    # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
    allids = self.get_ids(partition=partition, fold=fold)

    # set of ids present in local folder (e.g. 1.tfrecord)
    # tiles = os.listdir(self.datadir)
    blobs = list_blobs(self.datadir)  # gcp
    tiles = [b.name for b in blobs]  # gcp
    tiles = [os.path.basename(t) for t in tiles]  # gcp

    if tiles[0].endswith(".gz"):
      compression = "GZIP"
      #            ext = ".tfrecord.gz"
      ext = ".gz"
    else:
      compression = ""
      ext = ".tfrecord"

    downloaded_ids = [str(t.replace(".gz", "").replace(".tfrecord", "")) for t in tiles]

    allids = [i.strip() for i in allids]

    # intersection of available ids and partition ods
    if overwrite_ids is None:
      ids = list(set(downloaded_ids).intersection(allids))
    else:
      print("overwriting data ids! due to manual input")
      ids = overwrite_ids

    filenames = [os.path.join(self.datadir, str(id) + ext) for id in ids]

    if self.verbose:
      print(
        "dataset: {}, partition: {}, fold:{} {}/{} tiles downloaded ({:.2f} %)".format(self.section, partition, fold,
                                                                                       len(ids), len(allids),
                                                                                       len(ids) / float(
                                                                                         len(allids)) * 100))

    def mapping_function(serialized_feature):
      # read data from .tfrecords
      feature = self.parsing_function(serialized_example=serialized_feature)
      # sample n times out of the timeseries
      feature = self.temporal_sample(feature)
      # indices
      if self.experiment == "indices" or self.experiment == "all": feature = self.addIndices(feature)
      if self.experiment == "evi2": feature = self.addIndices250m(feature)
      # perform data normalization [0,1000] -> [0,1]
      if self.experiment == "bands250m": feature = self.normalize_bands250m(feature)
      if self.experiment == "bands": feature = self.normalize_bands(feature)
      if self.experiment == "bandswoblue": feature = self.normalize_bandswoblue(feature)
      if self.experiment == "bandsaux": feature = self.normalize_bandsaux(feature)
      if self.experiment == "indices": feature = self.normalize_indices(feature)
      if self.experiment == "all": feature = self.normalize_all(feature)
      if self.experiment == "evi2": feature = self.normalize_evi2(feature)
      if self.experiment == "bandswodoy": feature = self.normalize_bandswodoy(feature)

      feature = self.tilestatic(feature)

      if self.reference == "MCD12Q1v6raw_LCType1": feature = self.MCD12Q1v6raw_LCType1(feature)
      if self.reference == "MCD12Q1v6raw_LCProp1": feature = self.MCD12Q1v6raw_LCProp1(feature)
      if self.reference == "MCD12Q1v6raw_LCProp2": feature = self.MCD12Q1v6raw_LCProp2(feature)
      if self.reference == "MCD12Q1v6stable_LCType1": feature = self.MCD12Q1v6stable_LCType1(feature)
      if self.reference == "MCD12Q1v6stable_LCProp1": feature = self.MCD12Q1v6stable_LCProp1(feature)
      if self.reference == "MCD12Q1v6stable01to15_LCProp2": feature = self.MCD12Q1v6stable01to15_LCProp2(feature)
      if self.reference == "MCD12Q1v6stable01to03_LCProp2": feature = self.MCD12Q1v6stable01to03_LCProp2(feature)
      if self.reference == "ESAraw": feature = self.ESAraw(feature)
      if self.reference == "ESAstable": feature = self.ESAstable(feature)
      if self.reference == "Copernicusraw": feature = self.Copernicusraw(feature)
      if self.reference == "Copernicusraw_fraction": feature = self.Copernicusraw_fraction(feature)
      if self.reference == "Copernicusnew_cf2others": feature = self.Copernicusnew_cf2others(feature)
      if self.reference == "merge_datasets2own": feature = self.merge_datasets2own(feature)
      if self.reference == "merge_datasets2HuHu": feature = self.merge_datasets2HuHu(feature)
      if self.reference == "merge_datasets2Tsendbazaretal2maps": feature = self.merge_datasets2Tsendbazaretal2maps(
        feature)
      if self.reference == "merge_datasets2Tsendbazaretal3maps": feature = self.merge_datasets2Tsendbazaretal3maps(
        feature)
      if self.reference == "mapbiomas_fraction": feature = self.mapbiomas_fraction(feature)
      if self.reference == "watermask": feature = self.watermask(feature)

      # perform data augmentation
      if self.augment: feature = self.augment(feature)
      # flatten for tempCNN
      # if self.tempCNN: feature = self.
      # replace potentially non sequential labelids with sequential dimension ids
      # feature = self.transform_labels(feature)
      if self.step == "training": feature = self.transform_labels_training(feature)
      if not self.step == "training": feature = self.transform_labels_evaluation(feature)

      return feature

    if num_batches > 0:
      filenames = filenames[0:num_batches * batchsize]

    # shuffle sequence of filenames
    if partition == 'train':
      filenames = tf.random.shuffle(filenames)

    dataset = tf.data.TFRecordDataset(filenames, compression_type=compression, num_parallel_reads=threads)

    dataset = dataset.map(mapping_function, num_parallel_calls=threads)

    # repeat forever until externally stopped
    dataset = dataset.repeat()

    if drop_remainder:
      # dataset = dataset.apply(tf.data.batch_and_drop_remainder(int(batchsize))) #tpu
      dataset = dataset.batch(int(batchsize), drop_remainder=True)

    else:
      dataset = dataset.batch(int(batchsize))

    if prefetch_batches is not None:
      dataset = dataset.prefetch(prefetch_batches)

    # modelshapes are expected shapes of the data stacked as batch
    output_shape = []
    for shape in self.expected_shapes:
      output_shape.append(tf.TensorShape((batchsize,) + shape))

    return dataset, output_shape, self.expected_datatypes, filenames


def list_blobs(path):  # gcp
  """Lists all the blobs in the bucket."""
  storage_client = storage.Client()

  bucket_name, prefix = _split_gcs_path(path)

  bucket = storage_client.get_bucket(bucket_name)

  blobs = bucket.list_blobs(prefix=prefix)

  return (blobs)


def _split_gcs_path(path):  # gcp
  m = re.search('gs://([a-z0-9-_.]*)/(.*)', path, re.IGNORECASE)
  if not m:
    raise ValueError('\'{}\' is not a valid GCS path'.format(path))

  return m.groups()
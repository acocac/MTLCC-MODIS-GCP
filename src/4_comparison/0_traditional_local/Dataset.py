from MODparser import MODparser
import tensorflow as tf
import os
import configparser
import csv
import numpy as np
from tensorflow.python.lib.io import file_io

class Dataset():
  """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

  def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False, experiment="all",
               reference="MCD12Q1v6_cleaned", step="training"):
    self.verbose = verbose

    self.augment = augment

    self.experiment = experiment
    self.reference = reference

    # parser reads serialized tfrecords file and creates a feature object
    parser = MODparser()
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

    cfgpath = os.path.join(dataroot, "dataset.ini")
    print(cfgpath)
    # load dataset configs
    datacfg = configparser.ConfigParser()
    with file_io.FileIO(cfgpath, 'r') as f:  # gcp
      datacfg.read_file(f)

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

  def transform_evaluation(self, features):
        def resize(tensor, new_height, new_width):
            t = tf.shape(tensor)[0]
            h = tf.shape(tensor)[1]
            w = tf.shape(tensor)[2]
            d = tf.shape(tensor)[3]

            # stack batch on times to fit 4D requirement of resize_tensor
            stacked_tensor = tf.reshape(tensor, [t, h, w, d])
            reshaped_stacked_tensor = tf.image.resize_images(stacked_tensor, size=(new_height, new_width))
            return tf.reshape(reshaped_stacked_tensor, [t, new_height, new_width, d])

        x250, x500, doy, year, labels = features

        px = tf.shape(x250)[2]

        x250 = tf.cast(x250, tf.float32)
        x500 = tf.cast(x500, tf.float32)

        x500_r = tf.identity(resize(x500, px, px))

        x = tf.concat([x250, x500_r], axis=3)

        x = tf.transpose(x, [1, 2, 0, 3])

        x = tf.reshape(x, (tf.shape(x)[1] * tf.shape(x)[1], tf.shape(x)[2] * tf.shape(x)[3]))  # -1 means "all"

        # doy = doy[tf.newaxis, :]
        # doy = tf.tile(doy,[tf.shape(x)[0],1])
        #
        # doy = tf.cast(doy, tf.float32)
        #
        # x_final = tf.concat([x,doy], axis=1)

        return x

  def MCD12Q1v6stable01to15_LCProp2(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 5 ]

    return x250, x500, doy, year, labels

  def MCD12Q1v6stable01to03_LCProp2(self, feature):

    x250, x500, doy, year, labels = feature

    labels = labels[ :, :, :, 6]

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
    elif partition == 'eval' or partition == 'pred':
      # e.g. eval240.tileids
      path = os.path.join(self.tileidfolder, eval.format(partition=partition))
      return readids(path)
    else:
      raise ValueError("please provide valid partition (train|test|eval|pred)")

  def create_tf_dataset(self, partition, fold, batchsize, prefetch_batches=None, num_batches=-1, threads=8,
                        drop_remainder=False, overwrite_ids=None):

    # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
    allids = self.get_ids(partition=partition, fold=fold)

    # set of ids present in local folder (e.g. 1.tfrecord)
    # tiles = os.listdir(self.datadir)
    tiles = file_io.get_matching_files(os.path.join(self.datadir, '*.gz'))
    tiles = [os.path.basename(t) for t in tiles]

    if tiles[0].endswith(".gz"):
      compression = "GZIP"
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
      # indices
      feature = self.tilestatic(feature)

      if self.reference == "MCD12Q1v6stable01to15_LCProp2": feature = self.MCD12Q1v6stable01to15_LCProp2(feature)
      if self.reference == "MCD12Q1v6stable01to03_LCProp2": feature = self.MCD12Q1v6stable01to03_LCProp2(feature)

      # perform data augmentation
      if self.augment: feature = self.augment(feature)

      if not self.step == "training": feature = self.transform_evaluation(feature)

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
      dataset = dataset.apply(tf.data.batch_and_drop_remainder(int(batchsize)))
    else:
      dataset = dataset.batch(int(batchsize))

    if prefetch_batches is not None:
      dataset = dataset.prefetch(prefetch_batches)

    # model shapes are expected shapes of the data stacked as batch
    output_shape = []
    for shape in self.expected_shapes:
      output_shape.append(tf.TensorShape((batchsize,) + shape))

    return dataset, output_shape, self.expected_datatypes, filenames

def main():
    dataset = Dataset(datadir="/media/data/marc/tfrecords/fields/L1C/480", verbose=True, temporal_samples=30,section="2016")

    training_dataset, output_shapes, output_datatypes, fm_train = dataset.create_tf_dataset("train", 0, 1, 5, True, 32)

    iterator = training_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run([iterator.initializer, tf.tables_initializer()])
        x250, x500, doy, year, labels = sess.run(iterator.get_next())
        print(x250.shape)

if __name__ == "__main__":
    main()


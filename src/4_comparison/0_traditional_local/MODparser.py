import tensorflow as tf
import numpy as np
import os

class MODparser():
  """ defined the Sentinel 2 .tfrecord format """

  def __init__(self):

    self.feature_format = {
      'x250/data': tf.io.FixedLenFeature([], tf.string),
      'x250/shape': tf.io.FixedLenFeature([4], tf.int64),
      'x250aux/data': tf.io.FixedLenFeature([], tf.string),
      'x250aux/shape': tf.io.FixedLenFeature([4], tf.int64),
      'x500/data': tf.io.FixedLenFeature([], tf.string),
      'x500/shape': tf.io.FixedLenFeature([4], tf.int64),
      'dates/doy': tf.io.FixedLenFeature([], tf.string),
      'dates/year': tf.io.FixedLenFeature([], tf.string),
      'dates/shape': tf.io.FixedLenFeature([1], tf.int64),
      'labels/data': tf.io.FixedLenFeature([], tf.string),
      'labels/shape': tf.io.FixedLenFeature([4], tf.int64)
    }

    return None

  def write(self, filename, x250ds, x500ds, doy, year, labelsds):
    # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

    #         writer = tf.python_io.TFRecordWriter(filename)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(filename, options=options)

    x250 = x250ds.astype(np.int64)
    x500 = x500ds.astype(np.int64)
    doy = doy.astype(np.int64)
    year = year.astype(np.int64)
    labels = labelsds.astype(np.int64)

    # Create a write feature
    feature = {
      'x250/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x250.tobytes()])),
      'x250/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250.shape)),
      'x500/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x500.tobytes()])),
      'x500/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x500.shape)),
      'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
      'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
      'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
      'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
      'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

  def parse_example_bands(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """
    feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

    # decode and reshape
    x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

    x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                        tf.cast(feature[0]['labels/shape'], tf.int32))

    return x250, x500, doy, year, labels

  def parse_example_tempCNN(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """
    feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

    # decode and reshape
    x = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                        tf.cast(feature[0]['labels/shape'], tf.int32))

    return x, labels

  def parse_example_bandsaux(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """

    feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

    # decode and reshape
    x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))
    x250aux = tf.reshape(tf.decode_raw(feature[0]['x250aux/data'], tf.int64),
                         tf.cast(feature[0]['x250aux/shape'], tf.int32))

    x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                        tf.cast(feature[0]['labels/shape'], tf.int32))

    return x250, x250aux, x500, doy, year, labels

  def parse_example_bandswoblue(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """
    feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

    # decode and reshape
    x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

    x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                        tf.cast(feature[0]['labels/shape'], tf.int32))

    x500 = x500[:, :, :, 1:5]

    return x250, x500, doy, year, labels

  def parse_example_bands250m(self, serialized_example):
    """
    example proto can be obtained via
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    or by passing this function in dataset.map(.)
    """
    feature = tf.io.parse_single_sequence_example(serialized_example, self.feature_format)

    # decode and reshape
    x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))
    x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64),
                        tf.cast(feature[0]['labels/shape'], tf.int32))

    x500 = x250[:, :, :, 0:1]

    return x250, x500, doy, year, labels

  def read(self, filenames):
    """ depricated! """

    if isinstance(filenames, list):
      filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    elif isinstance(filenames, tf.FIFOQueue):
      filename_queue = filenames
    else:
      print ("please insert either list or tf.FIFOQueue")

    reader = tf.TFRecordReader()
    f, serialized_example = reader.read(filename_queue)

    feature = tf.parse_single_example(serialized_example, features=self.feature_format)

    # decode and reshape
    x250 = tf.reshape(tf.decode_raw(feature['x250/data'], tf.int64), tf.cast(feature['x250/shape'], tf.int32))
    x500 = tf.reshape(tf.decode_raw(feature['x500/data'], tf.int64), tf.cast(feature['x500/shape'], tf.int32))

    doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
    year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

    labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

    return x250, x500, doy, year, labels

  def read_and_return(self, filename):
    """ depricated! """

    # get feature operation containing
    feature_op = self.read([filename])

    with tf.Session() as sess:
      tf.global_variables_initializer()

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      return sess.run(feature_op)

  def get_shapes(self, sample):
    print("reading shape of data using the sample " + sample)
    data = self.read_and_return(sample)
    return [tensor.shape for tensor in data]

  def tfrecord_to_pickle(self, tfrecordname, picklename):
    import cPickle as pickle

    reader = tf.TFRecordReader()

    # read serialized representation of *.tfrecord
    filename_queue = tf.train.string_input_producer([tfrecordname], num_epochs=None)
    filename_op, serialized_example = reader.read(filename_queue)
    feature = self.parse_example(serialized_example)

    with tf.Session() as sess:
      sess.run([tf.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      feature = sess.run(feature)

      coord.request_stop()
      coord.join(threads)

    pickle.dump(feature, open(picklename, "wb"), protocol=2)


def test():
    print ("Running self test:")
    print ("temporary tfrecord file is written with random numbers")
    print ("tfrecord file is read back")
    print ("contents are compared")

    filename=r"E:\acocac\research\s2_cc\tmptile.tfrecord"

    # create dummy dataset
    x250 = (np.random.rand(6,48,48,6)*1e3).astype(np.int64)
    x500 = (np.random.rand(6,24,24,6)*1e3).astype(np.int64)
    labels = (np.random.rand(6,48,48)*1e3).astype(np.int64)
    doy = (np.random.rand(6)*1e3).astype(np.int64)
    year = (np.random.rand(6)*1e3).astype(np.int64)

    # init parser
    parser=MODparser()

    parser.write(filename, x250, x500, doy, year,labels)

    x250_, x500_, doy_, year_, labels_ = parser.read_and_return(filename)
    print(x250_)
    # test if wrote and read data is the same
    print ("TEST")
    if np.all(x250_==x250_) and np.all(x500_==x500_) and np.all(labels_==labels) and np.all(doy_==doy) and np.all(year_==year):
        print ("PASSED")
    else:
        print ("NOT PASSED")

    # remove file
    os.remove(filename)

    #return tf.reshape(x500, (1,48,48,6))
    #return feature['x500shape']

if __name__=='__main__':
    
    ##test()

    parser = MODparser()

    parser.tfrecord_to_pickle("1.tfrecord","1.pkl")
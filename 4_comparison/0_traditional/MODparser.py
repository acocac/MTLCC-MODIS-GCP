import tensorflow as tf
import numpy as np
import os

class MODparser():
    """ defined the  .tfrecord format """

    def __init__(self):
        self.feature_format_train = {
            'x/data': tf.FixedLenFeature([], tf.string),
            'x/shape': tf.FixedLenFeature([2], tf.int64),
            'labels/data': tf.FixedLenFeature([], tf.string),
            'labels/shape': tf.FixedLenFeature([1], tf.int64)
        }

        self.feature_format_eval = {
            'x250/data': tf.FixedLenFeature([], tf.string),
            'x250/shape': tf.FixedLenFeature([4], tf.int64),
            'x250aux/data': tf.FixedLenFeature([ ], tf.string),
            'x250aux/shape': tf.FixedLenFeature([ 4 ], tf.int64),
            'x500/data': tf.FixedLenFeature([], tf.string),
            'x500/shape': tf.FixedLenFeature([4], tf.int64),
            'dates/doy': tf.FixedLenFeature([], tf.string),
            'dates/year': tf.FixedLenFeature([], tf.string),
            'dates/shape': tf.FixedLenFeature([1], tf.int64),
            'labels/data': tf.FixedLenFeature([], tf.string),
            'labels/shape': tf.FixedLenFeature([4], tf.int64)
        }

        return None

    def parse_example_train(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """

        # feature = tf.parse_single_example(serialized_example, self.feature_format)
        feature = tf.parse_single_sequence_example(serialized_example, self.feature_format_train)

        # decode and reshape
        x = tf.reshape(tf.decode_raw(feature[0]['x/data'], tf.int64), tf.cast(feature[0]['x/shape'], tf.int32))
        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64), tf.cast(feature[0]['labels/shape'], tf.int32))

        return x, labels

    def parse_example_eval(self, serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """
        feature = tf.parse_single_sequence_example(serialized_example, self.feature_format_eval)

        # decode and reshape
        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))

        x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64), tf.cast(feature[0]['labels/shape'], tf.int32))

        return x250, x500, doy, year, labels

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
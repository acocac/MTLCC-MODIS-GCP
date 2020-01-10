"""
Chunk TFRecord data to ingest into the models

Example invocation::

    python 1_datapreparation/1_chunk_TFrecords.py
        -r /home/xx/
        -y 2009
        -b 3
        -p 24
        -d MCD12Q1v6
        -e train

acocac@gmail.com
"""

import tensorflow as tf
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

config.gpu_options.allow_growth = True

import numpy as np
import os
import sys
import geopandas as gpd
import argparse
import joblib

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-r','--rootdir', type=str, required=True, help='Root dir with raw folder')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set in GEE')
parser.add_argument('-y','--tyear', type=str, required=True, help='Target year')
parser.add_argument('-b','--maxblocks', type=int, required=True, help='Maximum number of blocks')
parser.add_argument('-d','--dataset', type=str, required=True, help='version of the MODIS land cover dataset v5 or v6')
parser.add_argument('-e','--exportblocks', type=str, required=True, help='Export blocks for train or evaluation')
parser.add_argument('-n','--nworkers', type=int, default=None, help='Number of workers (by default all)')
parser.add_argument('--noconfirm', action='store_true', help='Skip confirmation')

def confirm(prompt=None, resp=False):
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

class dataset_of2others():
    def __init__(self, arr):
        self.arr = arr

    def copernicus(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##MODIS to new classes
        nodata = [0, 6, 10]
        DF = [1]
        OF = [2]
        S = [3]
        NH = [4]
        Ba = [7]
        Bu = [8]
        HC = [9]
        W = [11,12]
        HW = [5]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, DF) ] = 1
        arr_reclass[ np.isin(arr, OF) ] = 2
        arr_reclass[ np.isin(arr, S) ] = 3
        arr_reclass[ np.isin(arr, NH) ] = 4
        arr_reclass[ np.isin(arr, Ba) ] = 5
        arr_reclass[ np.isin(arr, HC) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, W) ] = 8
        arr_reclass[ np.isin(arr, HW) ] = 9

        return arr_reclass

class aggregate_datasets2own():
    def __init__(self, arr):
        self.arr = arr

    def ESA(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        nodata = [0, 24, 37]
        Ba = [25, 26, 27, 28, 33, 34, 35]
        W = [36]
        Bu = [32]
        DF = [7, 8, 9, 11, 12, 14, 15, 17 ]
        OF = [10, 13, 16, 29, 30, 18]
        NH = [19, 23, 31]
        C = [1, 2, 3, 4, 5, 6]
        S = [20, 21, 22]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, Ba) ] = 1
        arr_reclass[ np.isin(arr, W) ] = 2
        arr_reclass[ np.isin(arr, Bu) ] = 3
        arr_reclass[ np.isin(arr, DF) ] = 4
        arr_reclass[ np.isin(arr, OF) ] = 5
        arr_reclass[ np.isin(arr, NH) ] = 6
        arr_reclass[ np.isin(arr, C) ] = 7
        arr_reclass[ np.isin(arr, S) ] = 8

        return arr_reclass

    def copernicus(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##ESA-CCI to new classes
        nodata = [ 0, 16, 20 ]
        Ba = [ 17 ]
        W = [ 21, 22 ]
        Bu = [ 19 ]
        DF = [ 1, 2, 3, 4, 5, 6 ]
        OF = [ 7, 8, 9, 10, 11, 12 ]
        NH = [ 14, 15 ]
        C = [ 18 ]
        S = [ 13 ]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, Ba) ] = 1
        arr_reclass[ np.isin(arr, W) ] = 2
        arr_reclass[ np.isin(arr, Bu) ] = 3
        arr_reclass[ np.isin(arr, DF) ] = 4
        arr_reclass[ np.isin(arr, OF) ] = 5
        arr_reclass[ np.isin(arr, NH) ] = 6
        arr_reclass[ np.isin(arr, C) ] = 7
        arr_reclass[ np.isin(arr, S) ] = 8

        return arr_reclass

    def MODIS(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##ESA-CCI to new classes
        nodata = [0, 2]
        Ba = [ 1 ]
        W = [ 3 ]
        Bu = [ 4 ]
        DF = [ 5 ]
        OF = [ 6 ]
        NH = [ 8 ]
        C = [7, 9, 10]
        S = [ 11 ]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, Ba) ] = 1
        arr_reclass[ np.isin(arr, W) ] = 2
        arr_reclass[ np.isin(arr, Bu) ] = 3
        arr_reclass[ np.isin(arr, DF) ] = 4
        arr_reclass[ np.isin(arr, OF) ] = 5
        arr_reclass[ np.isin(arr, NH) ] = 6
        arr_reclass[ np.isin(arr, C) ] = 7
        arr_reclass[ np.isin(arr, S) ] = 8

        return arr_reclass

class aggregate_datasets2HuHu():
    def __init__(self, arr):
        self.arr = arr

    def ESA(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        nodata = [0, 24, 37]
        C = [1, 2, 3, 4, 5, 6]
        F = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        G = [18, 19, 23, 25, 26, 27, 28]
        S = [20, 21, 22 ]
        We = [29, 30, 31]
        W = [36]
        Bu = [32]
        Ba = [33, 34, 35]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, C) ] = 1
        arr_reclass[ np.isin(arr, F) ] = 2
        arr_reclass[ np.isin(arr, G) ] = 3
        arr_reclass[ np.isin(arr, S) ] = 4
        arr_reclass[ np.isin(arr, We) ] = 5
        arr_reclass[ np.isin(arr, W) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, Ba) ] = 8

        return arr_reclass

    def MODIS(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##ESA-CCI to new classes
        nodata = [0, 15]
        C = [12, 14]
        F = [1, 2, 3, 4, 5]
        G = [8, 9, 10]
        S = [6, 7]
        We = [11]
        W = [17]
        Bu = [13]
        Ba = [16]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, C) ] = 1
        arr_reclass[ np.isin(arr, F) ] = 2
        arr_reclass[ np.isin(arr, G) ] = 3
        arr_reclass[ np.isin(arr, S) ] = 4
        arr_reclass[ np.isin(arr, We) ] = 5
        arr_reclass[ np.isin(arr, W) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, Ba) ] = 8

        return arr_reclass

class aggregate_datasets2Tsendbazaretal():
    def __init__(self, arr):
        self.arr = arr

    def ESA(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        nodata = [0, 24, 37]
        C = [1, 2, 3, 4, 5, 6]
        F = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 29, 30]
        G = [19, 23]
        S = [20, 21, 22]
        We = [31]
        W = [36]
        Bu = [32]
        Ba = [25, 26, 27, 28, 33, 34, 35]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, C) ] = 1
        arr_reclass[ np.isin(arr, F) ] = 2
        arr_reclass[ np.isin(arr, G) ] = 3
        arr_reclass[ np.isin(arr, S) ] = 4
        arr_reclass[ np.isin(arr, We) ] = 5
        arr_reclass[ np.isin(arr, W) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, Ba) ] = 8

        return arr_reclass

    def MODIS(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##ESA-CCI to new classes
        nodata = [0, 15]
        C = [12, 14]
        F = [1, 2, 3, 4, 5, 8, 9]
        G = [10]
        S = [6, 7]
        We = [11]
        W = [17]
        Bu = [13]
        Ba = [16]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, C) ] = 1
        arr_reclass[ np.isin(arr, F) ] = 2
        arr_reclass[ np.isin(arr, G) ] = 3
        arr_reclass[ np.isin(arr, S) ] = 4
        arr_reclass[ np.isin(arr, We) ] = 5
        arr_reclass[ np.isin(arr, W) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, Ba) ] = 8

        return arr_reclass

    def copernicus(self):
        """
        a function for aggregating from ESA-CCI classes to simple classes
        """
        ##ESA-CCI to new classes
        nodata = [0, 16, 20]
        C = [18]
        F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        G = [14]
        S = [13]
        We = [15]
        W = [21, 22]
        Bu = [19]
        Ba = [17]

        arr = self
        arr_reclass = arr.copy()
        arr_reclass[ np.isin(arr, nodata) ] = 0
        arr_reclass[ np.isin(arr, C) ] = 1
        arr_reclass[ np.isin(arr, F) ] = 2
        arr_reclass[ np.isin(arr, G) ] = 3
        arr_reclass[ np.isin(arr, S) ] = 4
        arr_reclass[ np.isin(arr, We) ] = 5
        arr_reclass[ np.isin(arr, W) ] = 6
        arr_reclass[ np.isin(arr, Bu) ] = 7
        arr_reclass[ np.isin(arr, Ba) ] = 8

        return arr_reclass

def frac_copernicus_new(thsld, frac_array):
    # aggregate forest types
    closedf = np.sum(frac_array[:, :, :, 1:7], axis=3)
    openf = np.sum(frac_array[:, :, :, 7:13], axis=3)

    # create dimensions to append
    nodata = np.expand_dims(frac_array[:, :, :, 0], axis=3)
    closedf = np.expand_dims(closedf, axis=3)
    openf = np.expand_dims(openf, axis=3)

    # append
    nodata_closedf = np.append(nodata, closedf, axis=3)
    forest = np.append(nodata_closedf, openf, axis=3)

    newcopernicus_frac = np.append(forest, frac_array[:, :, :, 13:], axis=3)

    alllabels = np.zeros((1, frac_array.shape[ 1 ], frac_array.shape[ 2 ]))

    # negative probability to find second
    arr_tmp = np.copy(newcopernicus_frac)
    arr_tmp[ 0, :, :, 1 ] = -0.1

    alllabels_tmp = np.argmax(arr_tmp, axis=3)
    alllabels_ori = np.argmax(newcopernicus_frac, axis=3)
    alllabels_mask = np.where(newcopernicus_frac[0, :, :, 1] < thsld, alllabels_tmp, alllabels)

    alllabels_final = np.where(newcopernicus_frac[0, :, :, 1] >= thsld, alllabels_ori, alllabels_mask)

    return (alllabels_final)

class MODparser():
    def __init__(self):
        self.feature_format = {
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
            'labels/shape': tf.FixedLenFeature([4], tf.int64),
            'labels_frac/data': tf.FixedLenFeature([], tf.string),
            'labels_frac/shape': tf.FixedLenFeature([4], tf.int64)
        }

        return None

    def write(self, filename, x250ds, x250auxds, x500ds, doy, year, labelsds,labelcont):
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(filename, options=options)

        x250 = x250ds.astype(np.int64)
        x250aux = x250auxds.astype(np.int64)
        x500 = x500ds.astype(np.int64)
        doy = doy.astype(np.int64)
        year = year.astype(np.int64)
        labels = labelsds.astype(np.int64)
        labels_frac = labelcont.astype(np.float32)

        feature = {
            'x250/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x250.tobytes()])),
            'x250/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250.shape)),
            'x250aux/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x250aux.tobytes()])),
            'x250aux/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250aux.shape)),
            'x500/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x500.tobytes()])),
            'x500/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x500.shape)),
            'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
            'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
            'labels_frac/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ labels_frac.tobytes() ])),
            'labels_frac/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_frac.shape)),
            'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
            'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
            'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def parse_example(self, serialized_example):
        feature = tf.parse_single_sequence_example(serialized_example, self.feature_format)

        x250 = tf.reshape(tf.decode_raw(feature[0]['x250/data'], tf.int64), tf.cast(feature[0]['x250/shape'], tf.int32))
        x250aux = tf.reshape(tf.decode_raw(feature[0]['x250aux/data'], tf.int64), tf.cast(feature[0]['x250aux/shape'], tf.int32))
        x500 = tf.reshape(tf.decode_raw(feature[0]['x500/data'], tf.int64), tf.cast(feature[0]['x500/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature[0]['dates/doy'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature[0]['dates/year'], tf.int64), tf.cast(feature[0]['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature[0]['labels/data'], tf.int64), tf.cast(feature[0]['labels/shape'], tf.int32))
        labels_frac = tf.reshape(tf.decode_raw(feature[0]['labels_frac/data'], tf.float32), tf.cast(feature[0]['labels_frac/shape'], tf.int32))

        return x250, x250aux, x500, doy, year, labels, labels_frac

def trans(array,ds):
    if ds == 'x':
      z = np.asarray(array).transpose(1,2,0,3)
    elif ds == 'label':
      z = np.asarray(array).transpose(1,2,0)
    return z

def split_2d(array, splits):
    x, y = splits
    return np.split(np.concatenate(np.split(array, y, axis=1)), x*y)

def retrans(array,ds):
  if ds == 'x':
    z = np.asarray(array).transpose(2,0,1,3)
  elif ds == 'label':
    z = np.asarray(array).transpose(2,0,1)
  return z

def parser_fn(predict_files, directory, class_path, n_patches, maxblocks, exportblocks, fn):

  parser = MODparser()

  ds = tf.data.TFRecordDataset(predict_files, compression_type='GZIP')

  parsedDataset = ds.map(parser.parse_example, num_parallel_calls=5)

  iterator = parsedDataset.make_one_shot_iterator()

  with tf.Session(config=config) as sess:

      for n in range(n_patches):
        x250, x250aux, x500, DOY, year, labels, labelsfrac = iterator.get_next()
        x250, x250aux, x500, DOY, year, labels, labelsfrac = sess.run([x250, x250aux, x500, DOY, year, labels, labelsfrac])

        #add copernicusraw_fraction
        labels_frac = np.argmax(labelsfrac, axis=3)
        newdataset = np.expand_dims(labels_frac, axis=3)

        labels = np.append(labels, newdataset, axis=3)

        ### add fraction closed forest to others dataset ###
        frac_copernicus = frac_copernicus_new(0.7, labelsfrac)
        frac_copernicus = dataset_of2others.copernicus(frac_copernicus)
        newdataset_frac = np.expand_dims(frac_copernicus, axis=3)

        labels = np.append(labels, newdataset_frac, axis=3)

        ### add own merge dataset ###
        cop_new = aggregate_datasets2own.copernicus(labels[ :, :, :, 8])
        ESA_new = aggregate_datasets2own.ESA(labels[ :, :, :, 6])
        MODIS_new = aggregate_datasets2own.MODIS(labels[ :, :, :, 4])

        alllabels = np.zeros((1, cop_new.shape[ 1 ], cop_new.shape[2], 9))

        for i in range(1, 9):
            class_cop = np.ma.masked_where(cop_new != i, cop_new)
            class_ESA = np.ma.masked_where(ESA_new != i, ESA_new)
            class_MODIS = np.ma.masked_where(MODIS_new != i, MODIS_new)

            l = class_cop * class_ESA * class_MODIS
            l = l.filled(0)
            l[ np.isin(l, pow(i, 3)) ] = i

            alllabels[ :, :, :, i ] = l

        newdataset = np.max(alllabels, axis=3)
        newdataset = np.expand_dims(newdataset, axis=3)

        labels = np.append(labels, newdataset, axis=3)

        ### add Hu and Hu merge dataset ###
        ESA_new = aggregate_datasets2HuHu.ESA(labels[ :, :, :, 6 ])
        MODIS_new = aggregate_datasets2HuHu.MODIS(labels[ :, :, :, 4 ])

        alllabels = np.zeros((1, ESA_new.shape[ 1 ], ESA_new.shape[ 2 ], 9))

        for i in range(1, 9):
            class_ESA = np.ma.masked_where(ESA_new != i, ESA_new)
            class_MODIS = np.ma.masked_where(MODIS_new != i, MODIS_new)

            l = class_ESA * class_MODIS
            l = l.filled(0)
            l[ np.isin(l, pow(i, 2)) ] = i

            alllabels[ :, :, :, i ] = l

        newdataset = np.max(alllabels, axis=3)
        newdataset = np.expand_dims(newdataset, axis=3)

        labels = np.append(labels, newdataset, axis=3)

        ### add Tsendbazaretal merge dataset original ###
        ESA_new = aggregate_datasets2Tsendbazaretal.ESA(labels[ :, :, :, 6 ])
        MODIS_new = aggregate_datasets2Tsendbazaretal.MODIS(labels[ :, :, :, 4 ])

        alllabels = np.zeros((1, ESA_new.shape[1], ESA_new.shape[2], 9))

        for i in range(1, 9):
            class_ESA = np.ma.masked_where(ESA_new != i, ESA_new)
            class_MODIS = np.ma.masked_where(MODIS_new != i, MODIS_new)

            l = class_ESA * class_MODIS
            l = l.filled(0)
            l[ np.isin(l, pow(i, 2)) ] = i

            alllabels[ :, :, :, i ] = l

        newdataset = np.max(alllabels, axis=3)
        newdataset = np.expand_dims(newdataset, axis=3)

        labels = np.append(labels, newdataset, axis=3)

        ### add Tsendbazaretal merge dataset new ###
        cop_new = aggregate_datasets2Tsendbazaretal.copernicus(labels[ :, :, :, 8])
        ESA_new = aggregate_datasets2Tsendbazaretal.ESA(labels[ :, :, :, 6])
        MODIS_new = aggregate_datasets2Tsendbazaretal.MODIS(labels[ :, :, :, 4])

        alllabels = np.zeros((1, ESA_new.shape[1], ESA_new.shape[2], 9))

        for i in range(1, 9):
            class_cop = np.ma.masked_where(cop_new != i, cop_new)
            class_ESA = np.ma.masked_where(ESA_new != i, ESA_new)
            class_MODIS = np.ma.masked_where(MODIS_new != i, MODIS_new)

            l = class_cop * class_ESA * class_MODIS
            l = l.filled(0)
            l[ np.isin(l, pow(i, 3)) ] = i

            alllabels[ :, :, :, i ] = l

        newdataset = np.max(alllabels, axis=3)
        newdataset = np.expand_dims(newdataset, axis=3)

        labels = np.append(labels, newdataset, axis=3)

        #### preprocessing big block to small blocks ###
        x250_r = trans(x250, 'x')
        x250aux_r = trans(x250aux, 'x')
        x500_r = trans(x500, 'x')
        # labels_r = trans(labels, 'label')
        labels_r = trans(labels, 'x')
        labelsfrac_r = trans(labelsfrac, 'x')

        x250_chunk_data = split_2d(x250_r, (maxblocks, maxblocks))
        x250aux_chunk_data = split_2d(x250aux_r, (maxblocks, maxblocks))
        x500_chunk_data = split_2d(x500_r, (maxblocks, maxblocks))
        labels_chunk_data = split_2d(labels_r, (maxblocks, maxblocks))
        labelsfrac_chunk_data = split_2d(labelsfrac_r, (maxblocks, maxblocks))

        all_blocks = range(0, maxblocks * maxblocks, 1)

        if exportblocks == "train":
            list1 = list(range(maxblocks-1,(maxblocks*(maxblocks-1))-1,maxblocks))
            list2 = list(range((maxblocks*(maxblocks-1))-1,(maxblocks*(maxblocks-1))+maxblocks,1))
            ignore = list1 + list2

            good_indices = list(set(all_blocks).difference(ignore))

            x250_ = [retrans(x250_chunk_data[i], 'x') for i in good_indices]
            x250aux_ = [retrans(x250aux_chunk_data[i], 'x') for i in good_indices]
            x500_ = [retrans(x500_chunk_data[i], 'x') for i in good_indices]
            labels_ = [retrans(labels_chunk_data[i], 'label') for i in good_indices]
            labelsfrac_ = [retrans(labelsfrac_chunk_data[i], 'label') for i in good_indices]

        elif exportblocks == "train_forest70" or exportblocks == "eval" or exportblocks == "crossyear":

            x250_ = [retrans(x250_chunk_data[i], 'x') for i in all_blocks]
            x250aux_ = [retrans(x250aux_chunk_data[i], 'x') for i in all_blocks]
            x500_ = [retrans(x500_chunk_data[i], 'x') for i in all_blocks]
            labels_ = [retrans(labels_chunk_data[i], 'x') for i in all_blocks]
            labelsfrac_ = [retrans(labelsfrac_chunk_data[i], 'x') for i in all_blocks]

        for t in range(0, len(x250_), 1):
            # write results with GZ format
            filename_out = str(t) + "_" + str(n) + "_" + str(int(fn))
            outdir_file = os.path.join(directory, filename_out + '.gz')

            parser.write(outdir_file, x250_[t], x250aux_[t], x500_[t], DOY, year, labels_[t], labelsfrac_[t])

            if exportblocks != "crossyear":
                # export class
                labels = labels_[t].astype(np.int64)

                reference = ['MCD12Q1v6raw_LCType1', 'MCD12Q1v6stable_LCType1', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6stable_LCProp1',
                             'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6stable01to15_LCProp2', 'MCD12Q1v6stable01to03_LCProp2',
                             'ESAraw', 'ESAstable',
                             'Copernicusraw','Copernicusraw_fraction','Copernicusnew_cf2others',
                             'mergedatasets2own','mergedatasets2HuHu','mergedatasets2Tsendbazaretal_ori','mergedatasets2Tsendbazaretal_new']

                for r in range(len(reference)):
                    labels_class = np.max(labels[:,:,:,r], axis=0).flatten()
                    np.save(os.path.join(class_path,reference[r], filename_out + '.npy'), labels_class)

                #fractional
                labels = labelsfrac_[t].astype(np.float32)
                labels_class = np.max(labels[ :, :, :, :], axis=0).flatten()

                np.save(os.path.join(class_path, 'Copernicusfrac', filename_out + '.npy'), labels_class)

if __name__ == '__main__':
    args = parser.parse_args()
    rootdir = args.rootdir
    tyear = args.tyear
    psize = args.psize
    dataset = args.dataset
    maxblocks = args.maxblocks
    exportblocks = args.exportblocks
    nworkers = args.nworkers

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    indir_data = os.path.join(rootdir,'combine',str(psize),'data'+ tyear[2:])

    tiles = [file for r, d, f in os.walk(os.path.join(indir_data)) for file in f]

    outdir = os.path.join(rootdir,'gz',str(int(psize / maxblocks)), dataset, 'data' + tyear[2:])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if exportblocks != "crossyear":
        class_path = os.path.join(rootdir,'gz',str(int(psize / maxblocks)), dataset,'classes','data' + tyear[2:])
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        reference = ['MCD12Q1v6raw_LCType1', 'MCD12Q1v6stable_LCType1', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6stable_LCProp1',
                     'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6stable01to15_LCProp2', 'MCD12Q1v6stable01to03_LCProp2',
                      'ESAraw', 'ESAstable',
                      'Copernicusraw', 'Copernicusfrac',
                      'Copernicusraw_fraction','Copernicusnew_cf2others',
                      'mergedatasets2own','mergedatasets2HuHu','mergedatasets2Tsendbazaretal_ori','mergedatasets2Tsendbazaretal_new']

        for r in reference:
            if not os.path.exists(os.path.join(class_path,r)):
                os.makedirs(os.path.join(class_path,r))
    else:
        class_path = None

    if len(tiles) == 0:
        print('No tiles to process... Terminating')
        sys.exit(0)

    print()
    print('Will process the following :')
    print('Number of tiles : %d' % len(tiles))
    print('nworkers : %s' % str(nworkers))
    print('Input data dir : %s' % str(indir_data))
    print('Output data dir : %s' % str(outdir))
    print()

    if not args.noconfirm:
        if not confirm(prompt='Proceed?', resp=True):
            sys.exit(0)

    # Launch the process
    if nworkers is not None and nworkers > 1:

        if exportblocks == 'train' or exportblocks == 'train_forest70':
            target_tiles = gpd.read_file(os.path.join(rootdir, 'geodata', 'blocks', str(psize), 'geojson', 'fileid.geojson'))

            train_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','train_fold0.tileids')),
                dtype='str').tolist()
            test_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','test_fold0.tileids')),
                dtype='str').tolist()

            target_tiles = train_tiles + test_tiles

            target_tiles = set([os.path.basename(tiles[0]).split('-')[0] + '-' + i.split('_')[-1] + '.gz' for i in target_tiles])

        elif exportblocks == 'eval' or exportblocks == "crossyear":
            target_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','eval.tileids')),
                dtype='str').tolist()

            target_tiles = set([os.path.basename(tiles[0]).split('-')[0] + '-' + i.split('_')[-1] + '.gz' for i in target_tiles])

        print('Using joblib.Parallel with nworkers=%d' % nworkers)
        print('Number of tiles subset for the AOI: %d' % len(target_tiles))

        joblib.Parallel(n_jobs=nworkers)(
            joblib.delayed(parser_fn)(os.path.join(indir_data,tile), outdir, class_path, sum(1 for _ in tf.python_io.tf_record_iterator(os.path.join(indir_data,tile), options=options)),
                                      maxblocks, exportblocks, os.path.splitext(os.path.basename(tile))[0].split('-')[-1].split('.')[0])
            for tile in target_tiles
        )

    else:
        if exportblocks == 'train' or exportblocks == 'train_forest70':
            target_tiles = gpd.read_file(os.path.join(rootdir, 'geodata', 'blocks', str(psize), 'geojson', 'fileid.geojson'))

            train_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','train_fold0.tileids')),
                dtype='str').tolist()
            test_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','test_fold0.tileids')),
                dtype='str').tolist()

            target_tiles = [train_tiles] + [test_tiles]

            target_tiles = set([os.path.basename(tiles[0]).split('-')[0] + '-' + i.split('_')[-1] + '.gz' for i in target_tiles])

        elif exportblocks == 'eval' or exportblocks == "crossyear":
            target_tiles = np.loadtxt(
                os.path.join(os.path.join(rootdir,'geodata','split',str(psize),'final','tileids','eval.tileids')),
                dtype='str').tolist()

            target_tiles = [target_tiles]

            target_tiles = set([os.path.basename(tiles[0]).split('-')[0] + '-' + i.split('_')[-1] + '.gz' for i in target_tiles])

        for tile in target_tiles:

            fn = os.path.splitext(os.path.basename(tile))[0].split('-')[-1].split('.')[0]

            n_patches = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.join(indir_data,tile), options=options))

            parser_fn(os.path.join(indir_data,tile), outdir, class_path, n_patches, maxblocks, exportblocks, fn)
"""
Merge 250-m and 500-m TFRecord.gz records in single TF records

Example invocation::

    python 1_datapreparation/0_merge_multisources.py
        -r /home/xx/
        -y 2009
        -p 24
        -t 46
        -d MCD12Q1v6

acocac@gmail.com
"""

import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

# config.gpu_options.allow_growth = True

import numpy as np
import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-r','--rootdir', type=str, required=True, help='Dir with input TFrecords.gz generated by GEE')
parser.add_argument('-y','--tyear', type=str, required=True, help='Target year')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set of the MODIS 250-m data')
parser.add_argument('-t','--timesteps', type=int, required=True, help='time steps')

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
            'labels_frac/data': tf.FixedLenFeature([ ], tf.string),
            'labels_frac/shape': tf.FixedLenFeature([4], tf.int64)
        }

        return None

    def write(self, filename, x250ds, x250auxds, x500ds, doy, year, labelsds, labelcont):
        # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(filename, options=options)

        for x in range(x250ds.shape[0]):
            x250 = x250ds[x].astype(np.int64)
            x250aux = x250auxds[x].astype(np.int64)
            x500 = x500ds[x].astype(np.int64)
            doy = doy.astype(np.int64)
            year = year.astype(np.int64)
            labels = labelsds[x].astype(np.int64)
            labels_frac = labelcont[x].astype(np.float32)

            # Create a write feature
            feature = {
                'x250/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ x250.tobytes() ])),
                'x250/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250.shape)),
                'x250aux/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x250aux.tobytes() ])),
                'x250aux/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x250aux.shape)),
                'x500/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ x500.tobytes() ])),
                'x500/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x500.shape)),
                'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ labels.tobytes() ])),
                'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
                'labels_frac/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels_frac.tobytes() ])),
                'labels_frac/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_frac.shape)),
                'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ doy.tobytes() ])),
                'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ year.tobytes() ])),
                'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

def merge_fn(ds_250m_spectral, ds_500m_spectral, ds_250m_aux, bands_250m_spectral, bands_500m_spectral, bands_250m_aux, version_discrete, version_continuous, tfiles, psize, timesteps, batchsize, project):

    parser = MODparser()
    filenm = os.path.basename(ds_250m_spectral[0]).split('-')[0]

    ds_250m_spectral = tf.data.TFRecordDataset(ds_250m_spectral, compression_type='GZIP')
    ds_500m_spectral = tf.data.TFRecordDataset(ds_500m_spectral, compression_type='GZIP')
    ds_250m_aux = tf.data.TFRecordDataset(ds_250m_aux, compression_type='GZIP')

    # bands250m_ = bands_250m_spectral + version

    # Dictionary with names as keys, features as values.
    featureNames = list(bands_250m_spectral)

    columns = [tf.FixedLenFeature([timesteps, psize, psize], tf.int64) for k in featureNames]

    featuresDict250m_spectral = dict(zip(featureNames, columns))

    # Dictionary with names as keys, features as values.
    bands250m_aux_ = bands_250m_aux + version_discrete

    featureNames = list(bands250m_aux_)

    columns = [tf.FixedLenFeature([1, psize, psize], tf.int64) for k in featureNames]

    featuresDict250m_aux = dict(zip(featureNames, columns))

    # Dictionary with names as keys, features as values.
    bands250m_aux_cont = version_continuous

    featureNames = list(bands250m_aux_cont)

    columns = [tf.FixedLenFeature([1, psize, psize], tf.float32) for k in featureNames]

    featuresDict250m_aux_cont = dict(zip(featureNames, columns))

    # Dictionary with names as keys, features as values.
    bands_500m_ = bands_500m_spectral + ['DOY', 'year']

    # Dictionary with names as keys, features as values.
    featureNames = list(bands_500m_)

    columns = [tf.FixedLenFeature([timesteps, psize / 2, psize / 2], tf.int64) for k in featureNames]

    featuresDict500m_spectral = dict(zip(featureNames, columns))

    def parse_tfrecord250m_spectral(example_proto):

        feat = tf.parse_single_sequence_example(example_proto, featuresDict250m_spectral)

        x250 = tf.stack([(feat[0][x]) for x in bands_250m_spectral], axis=0)
        x250 = tf.transpose(x250, [1, 2, 3, 0])

        return x250

    def parse_tfrecord250m_aux(example_proto):

        feat = tf.parse_single_sequence_example(example_proto, featuresDict250m_aux)

        x250_aux = tf.stack([(feat[0][x]) for x in bands_250m_aux], axis=0)
        x250_aux = tf.transpose(x250_aux, [1, 2, 3, 0])

        landcover = tf.stack([(feat[0][x]) for x in version_discrete], axis=0)
        landcover = tf.transpose(landcover, [1, 2, 3, 0])

        feat = tf.parse_single_sequence_example(example_proto, featuresDict250m_aux_cont)
        landcover_continuous = tf.stack([(feat[0][x]) for x in version_continuous], axis=0)
        landcover_continuous = tf.transpose(landcover_continuous, [1, 2, 3, 0])

        return x250_aux, landcover, landcover_continuous

    def parse_tfrecord500m_spectral(example_proto):

        feat = tf.parse_single_sequence_example(example_proto, featuresDict500m_spectral)

        x500 = tf.stack([(feat[0][x]) for x in bands_500m_spectral], axis=0)
        x500 = tf.transpose(x500, [1, 2, 3, 0])

        # for predictions over all area wo mask
        year = tf.to_float(tf.reduce_mean(feat[0].pop('year'), axis=[1, 2]))
        DOY = tf.to_float(tf.reduce_mean(feat[0].pop('DOY'), axis=[1, 2]))

        return x500, year, DOY

    # Map the function over the dataset
    parsedDataset250m_spectral = ds_250m_spectral.map(parse_tfrecord250m_spectral, num_parallel_calls=5)
    parsedDataset500m_spectral = ds_500m_spectral.map(parse_tfrecord500m_spectral, num_parallel_calls=5)
    parsedDataset250m_aux = ds_250m_aux.map(parse_tfrecord250m_aux, num_parallel_calls=5)

    parsedDataset250m_spectral = parsedDataset250m_spectral.batch(batchsize)
    parsedDataset500m_spectral = parsedDataset500m_spectral.batch(batchsize)
    parsedDataset250m_aux = parsedDataset250m_aux.batch(batchsize)

    iterator250m_spectral = parsedDataset250m_spectral.make_one_shot_iterator()
    iterator500m_spectral = parsedDataset500m_spectral.make_one_shot_iterator()
    iterator250m_aux = parsedDataset250m_aux.make_one_shot_iterator()

    nfiles = tfiles / batchsize

    if tfiles % batchsize != 0:
        nfiles = nfiles + 1

    filepaths = ['{}/{}.gz'.format(project, filenm + '-' + str(i)) for i in range(int(nfiles))]

    with tf.Session(config=config) as sess:

        for t in range(int(nfiles)):
            x250_spectral = iterator250m_spectral.get_next()
            x250_spectral = sess.run(x250_spectral)

            x250_aux, labels, labels_continuous = iterator250m_aux.get_next()
            x250_aux, labels, labels_continuous = sess.run([x250_aux, labels, labels_continuous])

            x500_spectral, year, DOY = iterator500m_spectral.get_next()
            x500_spectral, year, DOY = sess.run([x500_spectral, year, DOY])

            # x250 = np.concatenate((x250_spectral, x250_aux), axis=-1)

            parser.write(filepaths[t], x250_spectral, x250_aux, x500_spectral, DOY[0, :], year[0, :], labels, labels_continuous)

if __name__ == '__main__':
    args = parser.parse_args()
    rootdir = args.rootdir
    tyear = args.tyear
    psize = args.psize
    timesteps = args.timesteps
    project = os.path.basename(rootdir)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)


    fileNames_250m_spectral = sorted(glob.glob(os.path.join(rootdir,'raw','250m_spectral','p' + str(psize) + 'k0','data' + tyear[2:],'*.gz')),key=os.path.getctime)
    fileNames_500m_spectral = sorted(glob.glob(os.path.join(rootdir,'raw','500m_spectral','p' + str(int(psize/2)) + 'k0','data' + tyear[2:],'*.gz')),key=os.path.getctime)
    fileNames_250m_aux = sorted(glob.glob(os.path.join(rootdir,'raw','250m_aux','p' + str(psize) + 'k0','data' + tyear[2:],'*.gz')),key=os.path.getctime)

    n_patches_first = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_250m_spectral[0], options=options))
    n_patches_last = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_250m_spectral[-1], options=options))

    batchsize_merge = n_patches_first

    print(batchsize_merge)

    tfiles_250m_spectral = (n_patches_first * (len(fileNames_250m_spectral)-1)) + n_patches_last

    n_patches_first = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_500m_spectral[0], options=options))
    n_patches_last = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_500m_spectral[-1], options=options))

    tfiles_500m_spectral = (n_patches_first * (len(fileNames_500m_spectral)-1)) + n_patches_last

    n_patches_first = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_250m_aux[0], options=options))
    n_patches_last = sum(1 for _ in tf.python_io.tf_record_iterator(fileNames_250m_aux[-1], options=options))

    tfiles_250m_aux = (n_patches_first * (len(fileNames_250m_aux)-1)) + n_patches_last

    assert(tfiles_250m_spectral == tfiles_500m_spectral == tfiles_250m_aux)

    bands_250m_spectral = ['red', 'NIR']
    bands_500m_spectral = ['blue', 'green', 'SWIR1', 'SWIR2','SWIR3']
    bands_250m_aux = ['bio01', 'bio12', 'elevation','slope','aspect']
    version_discrete = ['MCD12Q1v6raw_LCType1', 'MCD12Q1v6stable_LCType1', 'MCD12Q1v6raw_LCProp1', 'MCD12Q1v6stable_LCProp1', 'MCD12Q1v6raw_LCProp2', 'MCD12Q1v6stable01to15_LCProp2', 'MCD12Q1v6stable01to03_LCProp2', 'ESAraw', 'ESAstable', 'Copernicusraw']

    if project.startswith("tile_"):
        v_start = ['mapbiomas']
        v_end = ['mapbiomas_' + str(i) for i in range(1,13,1)]
        version_continuous = v_start + v_end
        version_discrete = version_discrete + ['watermask']

    else:
        v_start = ['remapped']
        v_end = ['remapped_' + str(i) for i in range(1, 23, 1) ]
        version_continuous = v_start + v_end

    merge_project = os.path.join(rootdir,'combine',str(psize),"data" + tyear[2:])
    if not os.path.exists(merge_project):
        os.makedirs(merge_project)

    merge_fn(fileNames_250m_spectral, fileNames_500m_spectral, fileNames_250m_aux, bands_250m_spectral, bands_500m_spectral, bands_250m_aux, version_discrete, version_continuous, tfiles_250m_spectral, psize, timesteps, batchsize_merge, merge_project)
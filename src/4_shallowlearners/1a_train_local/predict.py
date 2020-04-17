"""
Model prediction from shallow learner models using a local machine

Example invocation::

    python 4_shallowlearners/predict.py MODELDIR
        -c RF
        -datadir DATADIR
        -v
        -t
        -b 1
        -d 2001
        -exp bands
        -step verification
        -ref Copernicusraw
        -f 0
        -bm 1

acocac@gmail.com
"""

import argparse
import math
import os
import sys
import threading
from os.path import join

import tensorflow as tf
from osgeo import gdal, osr
from sklearn.externals import joblib

import Dataset

PRED_IDS_IDENTIFIER = "pred"

PREDICTION_FOLDERNAME="prediction"
CONFIDENCES_FOLDERNAME="confidences"

import numpy as np

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Evalutation of models')
  parser.add_argument('modeldir', type=str,
                      help="directory containing H5 file 'model.h5'")
  parser.add_argument('--datadir', type=str, default=None,
                      help='directory containing the data (defaults to environment variable $datadir)')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  parser.add_argument('-t', '--writetiles', action="store_true",
                      help='write out pngs for each tile with prediction, label etc.')
  parser.add_argument('-b', '--batchsize', type=int, default=32,
                      help='batchsize')
  parser.add_argument('-d', '--dataset', type=str, default="2016",
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('--prefetch', type=int, default=6,
                      help="prefetch batches")
  parser.add_argument('--allow_growth', type=bool, default=True,
                      help="Allow VRAM growth")
  parser.add_argument('--storedir', type=str, default="tmp",
                      help="directory to store tiles")
  parser.add_argument('-exp', '--experiment', type=str, default="None")
  parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6raw_LCType1", help='Reference dataset to train')
  parser.add_argument('-step', '--step', type=str, default="evaluation",
                      help='step')
  parser.add_argument('-c', '--classifier', type=str, default="RF",
                      help='classifier')
  parser.add_argument('--fold', type=str, default=None,
                      help='fold')
  parser.add_argument('-bm', '--bestmodel', type=int, default=1,
                      help='int')

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def normalize_fixed(X, min_per, max_per):
    x_normed = (X-min_per) / (max_per-min_per)
    return x_normed


def eval(args):
    assert args.experiment

    print(args.dataset)

    joblib_file = os.path.join(args.modeldir,'model-{}_bm{}.h5'.format(args.classifier,args.bestmodel))

    # Load from file
    model = joblib.load(joblib_file)

    dataset = Dataset.Dataset(datadir=args.datadir, verbose=True, section=args.dataset, experiment=args.experiment,reference=args.reference, step=args.step)

    if args.verbose: print ("initializing training dataset")
    tfdataset, _, _, filenames = dataset.create_tf_dataset(PRED_IDS_IDENTIFIER, 0,
                                                           args.batchsize,
                                                           prefetch_batches=args.prefetch)
    iterator = tfdataset.make_one_shot_iterator()

    num_samples = len(filenames)

    batches = num_samples / args.batchsize


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth

    with tf.Session(config=config) as sess:

        def makedir(outfolder):
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

        # makedir(join(args.storedir))
        makedir(join(args.storedir, PREDICTION_FOLDERNAME))

        try:
            for i in range(0, int(batches)):

                x_test = iterator.get_next()
                x_test = sess.run(x_test)

                x_tile = x_test[0, :, : ]

                if args.dataset == '2001':
                    X = [np.concatenate([x_tile[t][0:7], x_tile[t]]) for t in range(x_tile.shape[0])]
                    X = np.array(X)
                else:
                    X = x_tile

                doy = np.array(range(1, 365, 8))

                if args.classifier == 'SVM':
                    X = normalize_fixed(X, -100, 16000)
                    doy = doy / 365

                X = [np.concatenate([X[t], doy]) for t in range(X.shape[0])]
                X = np.array(X)

                pred = model.predict(X)

                nx = int(math.sqrt(pred.shape[0]))
                pred = np.reshape(pred, (1, nx, nx))

                stepfrom = i * args.batchsize
                stepto = stepfrom + args.batchsize + 1

                files = filenames[stepfrom:stepto]

                if args.writetiles:

                    threadlist=list()

                    for tile in range(x_test.shape[0]):

                        tileid = str(os.path.basename(files[tile]).split(".")[0])

                        geotransform = dataset.geotransforms[tileid]
                        srid = dataset.srids[tileid]

                        threadlist.append(write_tile(pred[tile], files[tile],join(args.storedir, PREDICTION_FOLDERNAME), geotransform, srid))

                    # start all write threads!
                    for x in threadlist:
                        x.start()

                    # wait for all to finish
                    for x in threadlist:
                        x.join()

        except KeyboardInterrupt:
            print ("Evaluation aborted")
            pass

def write_tile(array, datafilename, outfolder, geotransform, srid):
    """gave up for now... wrapper around write_tile_ to implement multithreaded writing"""
    writeargs = (array, datafilename, outfolder, geotransform, srid)
    #write_tile_(array, datafilename, outfolder, geotransform, srid)
    thread = threading.Thread(target=write_tile_, args=writeargs)
    # thread.start()
    return thread


def write_tile_(array, datafilename, outfolder, geotransform, srid):

    tile = os.path.basename(datafilename).replace(".tfrecord", "").replace(".gz", "")

    outpath = os.path.join(outfolder, tile + ".tif")

    nx = array.shape[0]
    ny = array.shape[1]

    if array.dtype == int:
        gdaldtype = gdal.GDT_Int16
    else:
        gdaldtype = gdal.GDT_Float32

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(outpath, ny, nx, 1, gdaldtype)
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(srid)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file

    dst_ds.GetRasterBand(1).WriteArray(array)

    dst_ds.FlushCache()  # write to disk
    dst_ds = None  # save, close


def main(args):
    # if args.verbose: print "setting visible GPU {}".format(args.gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args = parse_arguments(sys.argv)

    eval(args)


if __name__ == "__main__":

    args = parse_arguments(sys.argv)

    main(args)

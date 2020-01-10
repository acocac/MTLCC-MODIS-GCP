import os
import argparse
import Dataset
import tensorflow as tf

import sys
from osgeo import gdal, osr
from os.path import join
from sklearn.externals import joblib
import math

import threading

EVAL_IDS_IDENTIFIER = "eval"

MASK_FOLDERNAME="mask"
GROUND_TRUTH_FOLDERNAME="ground_truth"
PREDICTION_FOLDERNAME="prediction"
LOSS_FOLDERNAME="loss"
CONFIDENCES_FOLDERNAME="confidences"

TRUE_PRED_FILENAME="truepred.npy"

import numpy as np

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Evalutation of models')
  parser.add_argument('modeldir', type=str, help="directory containing H5 file 'model.h5'")
  parser.add_argument('--datadir', type=str, default=None,
                      help='directory containing the data (defaults to environment variable $datadir)')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  parser.add_argument('-t', '--writetiles', action="store_true",
                      help='write out pngs for each tile with prediction, label etc.')
  parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
  parser.add_argument('-gt', '--writegt', action="store_true",
                      help='write out pngs for each tile label etc.')
  parser.add_argument('-b', '--batchsize', type=int, default=32,
                      help='batchsize')
  parser.add_argument('-d', '--dataset', type=str, default="2016",
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
  parser.add_argument('--allow_growth', type=bool, default=True, help="Allow VRAM growth")
  parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
  parser.add_argument('-exp', '--experiment', type=str, default="None")
  parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6raw_LCType1", help='Reference dataset to train')
  parser.add_argument('-step', '--step', type=str, default="evaluation", help='step')
  parser.add_argument('-classifier', '--classifier', type=str, default="RF", help='classifier')

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def eval(args):
    assert args.experiment

    print(args.dataset)

    joblib_file = os.path.join(args.modeldir,'model-{}.h5'.format(args.classifier))

    # Load from file
    model = joblib.load(joblib_file)

    dataset = Dataset.Dataset(datadir=args.datadir, verbose=True, section=args.dataset, experiment=args.experiment,reference=args.reference, step=args.step)

    if args.verbose: print ("initializing training dataset")
    tfdataset, _, _, filenames = dataset.create_tf_dataset(EVAL_IDS_IDENTIFIER, 0,
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

        makedir(join(args.storedir))

        try:
            for i in range(0, int(batches)):

                x_test = iterator.get_next()
                x_test = sess.run(x_test)

                x_tile = x_test[0, :, : ]

                pred = model.predict(x_tile)

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

                        threadlist.append(write_tile(pred[tile], files[tile],args.storedir, geotransform, srid))

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
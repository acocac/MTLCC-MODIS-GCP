from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from constants.constants import *

from . import inputs
from . import model

import functools

from os.path import join
import threading
import datetime
from osgeo import gdal, osr
import configparser

PREDICTION_FOLDERNAME="prediction"
CONFIDENCES_FOLDERNAME="confidences"

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
  parser.add_argument('-d', '--dataset', type=str, default="2016",
                      help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
  parser.add_argument('-b', '--batchsize', type=int, default=32, help='batchsize')
  parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
  parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
  parser.add_argument('--allow_growth', type=bool, default=True, help="Allow dynamic VRAM growth of TF")
  parser.add_argument('--limit_batches', type=int, default=-1,
                      help="artificially reduce number of batches to encourage overfitting (for debugging)")
  parser.add_argument('-exp', '--experiment', type=str, default="bands", help='Experiment to train')
  parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6raw_LCType1",
                      help='Reference dataset to train')
  parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
  parser.add_argument('-wt', '--writetiles', action="store_true",
                      help='write out pngs for each tile with prediction, label etc.')
  parser.add_argument('-gt', '--writegt', action="store_true",
                      help='write out pngs for each tile label etc.')
  parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
  parser.add_argument('-step', '--step', type=str, default="evaluation", help='step')
  parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                               "Will at each get_next randomly choose "
                                                                               "<temporal_samples> elements from "
                                                                               "timestack. Defaults to None -> no temporal sampling")
  parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
  parser.add_argument('--learning_rate', type=float, default=None,
                      help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
  parser.add_argument('--convrnn_filters', type=int, default=None,
                      help="number of convolutional filters in ConvLSTM/ConvGRU layer")
  parser.add_argument('--convrnn_layers', type=int, default=None,
                      help="number of convolutional recurrent layers")

  args, _ = parser.parse_known_args(args=argv[1:])
  return args


def train_and_evaluate(args):
    """Runs model training and evaluation using TF Estimator API."""

    num_samples_predict = 0

    # if if num batches artificially reduced -> adapt sample size
    if args.limit_batches > 0:
        num_samples_predict = args.limit_batches * args.batchsize
    else:
        filenames, dataset = inputs.input_filenames(args, mode=tf.estimator.ModeKeys.PREDICT)
        num_samples_predict += len(filenames)

    batches = num_samples_predict / args.batchsize

    cfgpath = os.path.join(args.modeldir, "params.ini")
    # load dataset configs
    datacfg = configparser.ConfigParser()
    with file_io.FileIO(cfgpath, 'r') as f:  # gcp
      datacfg.read_file(f)
    cfg = datacfg['flags']

    args.convrnn_filters = cfg['convrnn_filters']
    args.convrnn_layers = cfg['convrnn_layers']
    args.learning_rate = cfg['learning_rate']

    print(args.convrnn_filters, args.convrnn_layers, args.learning_rate)
    run_config = tf.estimator.RunConfig(
        model_dir=args.modeldir,
    )

    estimator = model.build_estimator(
        run_config)

    predict_input_fn = functools.partial(
        inputs.input_fn_eval,
        args,
        # mode=tf.estimator.ModeKeys.EVAL
        mode = tf.estimator.ModeKeys.PREDICT
    )

    # recognize digits from local fonts
    predictions = estimator.predict(input_fn=predict_input_fn,
                                    yield_single_examples=False)

    # create appropiate folders
    def makedir(outfolder):
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

    makedir(join(args.storedir, PREDICTION_FOLDERNAME))

    try:

        starttime = datetime.datetime.now()
        for i in range(0, int(batches)):
            # pbar.update(i)
            now = datetime.datetime.now()
            dt = now - starttime
            seconds_per_batch = dt.seconds / (i + 1e-10)
            seconds_to_go = seconds_per_batch * (batches - i)
            eta = now + datetime.timedelta(seconds=seconds_to_go)
            print("{} eval batch {}/{}, eta: {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), i, batches,
                                                        eta.strftime("%Y-%m-%d %H:%M:%S")))

            stepfrom = i * args.batchsize
            stepto = stepfrom + args.batchsize + 1

            files = filenames[stepfrom:stepto]

            pred = next(predictions)['pred']
            pred_sc = next(predictions)['pred_sc']

            pred = pred + 1

            if args.writetiles:

                threadlist = list()

                for tile in range(pred.shape[0]):
                    tileid = str(os.path.basename(files[tile]).split(".")[0])

                    geotransform = dataset.geotransforms[tileid]
                    srid = dataset.srids[tileid]

                    threadlist.append(
                        write_tile(pred[ tile ], files[ tile ], join(args.storedir, PREDICTION_FOLDERNAME),
                                   geotransform, srid))

                    if args.writeconfidences:

                        for cl in range(pred_sc.shape[ -1 ]):

                            classname = dataset.classes[ cl + 1 ].replace(" ", "_")

                            foldername = "{}_{}".format(cl + 1, classname)

                            outfolder = join(args.storedir, CONFIDENCES_FOLDERNAME, foldername)
                            makedir(outfolder)  # if not exists

                            threadlist.append(
                                write_tile(pred_sc[ tile, :, :, cl ], files[ tile ], outfolder, geotransform, srid))

                # start all write threads!
                for x in threadlist:
                    x.start()

                # wait for all to finish
                for x in threadlist:
                    x.join()

    except KeyboardInterrupt:
        print("Evaluation aborted")
        pass


def write_tile(array, datafilename, outfolder, geotransform, srid):
    writeargs = (array, datafilename, outfolder, geotransform, srid)
    thread = threading.Thread(target=write_tile_, args=writeargs)
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


def main():
  args = parse_arguments(sys.argv)

  tf.logging.set_verbosity(tf.compat.v1.logging.INFO)

  train_and_evaluate(args)


if __name__ == "__main__":
  main()
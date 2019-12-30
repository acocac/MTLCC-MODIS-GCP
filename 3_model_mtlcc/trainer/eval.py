from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf

from constants.constants import *

from . import inputs
from . import model

import functools

import logging
from pathlib import Path

from os.path import join
import threading
import datetime
from osgeo import gdal, osr

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

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
  parser.add_argument('-s', '--shuffle', type=bool, default=False, help="batch shuffling")
  parser.add_argument('-e', '--epochs', type=int, default=5, help="epochs")
  parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                               "Will at each get_next randomly choose "
                                                                               "<temporal_samples> elements from "
                                                                               "timestack. Defaults to None -> no temporal sampling")
  parser.add_argument('--save_frequency', type=int, default=64, help="save frequency")
  parser.add_argument('--summary_frequency', type=int, default=64, help="summary frequency")
  parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
  parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
  parser.add_argument('--max_models_to_keep', type=int, default=5, help="maximum number of models to keep")

  parser.add_argument('--save_every_n_hours', type=int, default=1, help="save checkpoint every n hours")
  parser.add_argument('--queue_capacity', type=int, default=256, help="Capacity of queue")
  parser.add_argument('--allow_growth', type=bool, default=True, help="Allow dynamic VRAM growth of TF")
  parser.add_argument('--limit_batches', type=int, default=-1,
                      help="artificially reduce number of batches to encourage overfitting (for debugging)")
  parser.add_argument('-exp', '--experiment', type=str, default="bands", help='Experiment to train')
  parser.add_argument('-ref', '--reference', type=str, default="MCD12Q1v6raw_LCType1",
                      help='Reference dataset to train')
  parser.add_argument('--learning_rate', type=float, default=None,
                      help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
  parser.add_argument('--convrnn_filters', type=int, default=8,
                      help="number of convolutional filters in ConvLSTM/ConvGRU layer")
  parser.add_argument('--convrnn_layers', type=int, default=1,
                      help="number of convolutional recurrent layers")
  parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
  # parser.add_argument('-t', '--writetiles', action="store_true",
  #                     help='write out pngs for each tile with prediction, label etc.')
  # parser.add_argument('-gt', '--writegt', action="store_true",
  #                     help='write out pngs for each tile label etc.')
  # parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
  parser.add_argument('-wt', '--writetiles', action="store_true",
                      help='write out pngs for each tile with prediction, label etc.')
  parser.add_argument('-gt', '--writegt', action="store_true",
                      help='write out pngs for each tile label etc.')
  parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
  parser.add_argument('-step', '--step', type=str, default="evaluation", help='step')

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

    run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=args.max_models_to_keep,
        keep_checkpoint_every_n_hours=args.save_every_n_hours,
        model_dir=args.modeldir,
        log_step_count_steps=args.summary_frequency # set the frequency of logging steps for loss function
    )

    estimator = model.build_estimator(
        run_config)

    # estimator_predictor = tf.contrib.predictor.fromestimator(estimator, tfrecords_serving_input_fn)

    predict_input_fn = functools.partial(
        inputs.input_fn_eval,
        args,
        # mode=tf.estimator.ModeKeys.EVAL
        mode = tf.estimator.ModeKeys.PREDICT
    )

    # # print(predict_input_fn)
    # # predictions = list(estimator.predict(input_fn=predict_input_fn))
    # #
    # # for p in predictions:
    # #     print(int(p['classes'][0]))
    # #     break
    #
    # # save_hook = tf.estimator.CheckpointSaverHook(args.modeldir, save_secs=1)
    # # estimator.predict(input_fn=train_input_fn, hooks=[save_hook])
    # # # now you will get graph.pbtxt which is used in SavedModel, and then
    # # estimator.export_savedmodel(
    # #     export_dir_base=os.path.join(args.modeldir, "export"),
    # #     serving_input_receiver_fn=tfrecords_serving_input_fn)
    #


    # recognize digits from local fonts
    predictions = estimator.predict(input_fn=predict_input_fn,
                                    yield_single_examples=False)

    # for i in range(2):
    #     # predicted_font_classes = next(predictions)
    #     predicted_font_classes = next(predictions)['pred']
    #     print(predicted_font_classes)

    # predictions = []
    # for p_dict in estimator.predict(input_fn=predict_input_fn, yield_single_examples=False):
    #     # p_dict is a dictionary with a 'predictions' key
    #     # predictions.extend(p_dict['predictions'].flatten())
    #     predictions.extend(p_dict['predictions'])
    #
    # print(p_dict)

    #method 3
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

                            # write_tile(conn, pred_sc[tile, :, :, cl], datafilename=files[tile], tiletable=args.tiletable, outfolder=outfolder)
                            threadlist.append(
                                write_tile(pred_sc[ tile, :, :, cl ], files[ tile ], outfolder, geotransform, srid))
                            # thread.start_new_thread(write_tile, writeargs)

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
    """gave up for now... wrapper around write_tile_ to implement multithreaded writing"""
    writeargs = (array, datafilename, outfolder, geotransform, srid)
    #write_tile_(array, datafilename, outfolder, geotransform, srid)
    thread = threading.Thread(target=write_tile_, args=writeargs)
    # thread.start()
    return thread


def write_tile_(array, datafilename, outfolder, geotransform, srid):

    tile = os.path.basename(datafilename).replace(".tfrecord", "").replace(".gz", "")

    outpath = os.path.join(outfolder, tile + ".tif")

    #curs = conn.cursor()

    #sql = "select ST_xmin(geom) as xmin, ST_ymax(geom) as ymax, ST_SRID(geom) as srid from {tiletable} where id = {tileid}".format(
    #    tileid=tile, tiletable=tiletable)
    #curs.execute(sql)
    #xmin, ymax, srid = curs.fetchone()

    nx = array.shape[0]
    ny = array.shape[1]
    #xres = 10
    #yres = 10
    #geotransform = (xmin, xres, 0, ymax, 0, -yres)

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

  # Params
  #   params = {
  #       'dim_chars': 100,
  #       'dim': 300,
  #       'dropout': 0.5,
  #       'num_oov_buckets': 1,
  #       'epochs': 25,
  #       'batch_size': 20,
  #       'buffer': 15000,
  #       'filters': 50,
  #       'kernel_size': 3,
  #       'lstm_size': 100,
  #       'words': str(Path(DATADIR, 'vocab.words.txt')),
  #       'chars': str(Path(DATADIR, 'vocab.chars.txt')),
  #       'tags': str(Path(DATADIR, 'vocab.tags.txt')),
  #       'glove': str(Path(DATADIR, 'glove.npz'))
  #   }

  print('ok')
  train_and_evaluate(args)


if __name__ == "__main__":
  main()
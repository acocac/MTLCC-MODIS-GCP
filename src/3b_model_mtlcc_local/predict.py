import tensorflow as tf
import os
from Dataset import Dataset
import argparse
import datetime
from osgeo import gdal, osr
#import psycopg2
from os.path import join
import sklearn.metrics as skmetrics

import threading

#os.environ["GDAL_DATA"] = os.environ["HOME"] + "/.conda/envs/MTLCC/share/gdal"

MODEL_GRAPH_NAME = "graph.meta"
PRED_IDS_IDENTIFIER = "pred"

MODEL_CFG_FILENAME = "params.ini"
MODEL_CFG_FLAGS_SECTION = "flags"
MODEL_CFG_MODEL_SECTION = "model"
MODEL_CFG_MODEL_KEY = "model"

MODEL_CHECKPOINT_NAME = "model.ckpt"

PREDICTION_FOLDERNAME="prediction"
CONFIDENCES_FOLDERNAME="confidences"
GROUND_TRUTH_FOLDERNAME="ground_truth"

TRUE_PRED_FILENAME="truepred.npy"

import numpy as np

# simple flag to track if graph is created in this session or has to be imported
graph_created_flag = False


def main(args):
    # if args.verbose: print "setting visible GPU {}".format(args.gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.datadir is None:
        args.datadir = os.environ["datadir"]

    #with open(os.path.join(os.environ["HOME"], ".pgpass"), 'r') as f:
    #    pgpass = f.readline().replace("\n", "")
    #host, port, db, user, password = pgpass.split(':')
    #conn = psycopg2.connect('postgres://{}:{}@{}/{}'.format(user, password, host, db))

    # train
    eval(args)


def eval(args):
    # assert args.max_models_to_keep
    # assert type(args.allow_growth)==bool
    # assert args.summary_frequency
    # assert args.save_frequency
    assert args.experiment

    dataset = Dataset(datadir=args.datadir, verbose=True, section=args.dataset, experiment=args.experiment, reference=args.reference, step=args.step)

    if args.verbose: print ("initializing training dataset")
    tfdataset, output_shapes, output_datatypes, filenames = dataset.create_tf_dataset(PRED_IDS_IDENTIFIER, 0,
                                                                                      args.batchsize,
                                                                                      False,
                                                                                      prefetch_batches=args.prefetch)
    iterator = tfdataset.make_initializable_iterator()

    num_samples = len(filenames)

    filenames = [f.split('-')[-1] for f in filenames]

    # load meta graph
    graph = os.path.join(args.modeldir, MODEL_GRAPH_NAME)
    _ = tf.train.import_meta_graph(graph)

    def get_operation(name):
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    iterator_handle_op = get_operation("data_iterator_handle")
    is_train_op = get_operation("is_train")
    global_step_op = get_operation("global_step")
    samples_seen_op = get_operation("samples_seen")
    predictions_scores_op = get_operation("prediction_scores")
    predictions_op = get_operation("predictions")
    labels_op = get_operation("targets")

    saver = tf.train.Saver(save_relative_paths=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth

    # os.environ[ "CUDA_DEVICE_ORDER" ] = "PCI_BUS_ID"
    # os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '-1'
    #
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

        sess.run([iterator.initializer])

        data_handle = sess.run(iterator.string_handle())

        latest_ckpt = tf.train.latest_checkpoint(args.modeldir)
        if latest_ckpt is not None:
            print ("restoring from " + latest_ckpt)
            saver.restore(sess, latest_ckpt)

        step, samples = sess.run([global_step_op, samples_seen_op])
        # current_epoch = samples / float(num_samples)
        batches = num_samples / args.batchsize

        # create appropiate folders
        def makedir(outfolder):
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

        print('wt')
        makedir(join(args.storedir, PREDICTION_FOLDERNAME))

        try:
            #pbar.start()
            starttime = datetime.datetime.now()
            for i in range(0, int(batches)):
                #pbar.update(i)
                now = datetime.datetime.now()
                dt = now - starttime
                seconds_per_batch = dt.seconds/(i+1e-10)
                seconds_to_go = seconds_per_batch*(batches - i)
                eta = now+datetime.timedelta(seconds=seconds_to_go)
                print ("{} eval batch {}/{}, eta: {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), i,batches, eta.strftime("%Y-%m-%d %H:%M:%S")))

                stepfrom = i * args.batchsize
                stepto = stepfrom + args.batchsize + 1

                # take with care...
                files = filenames[stepfrom:stepto]
                # normal training operation
                #print "{} evaluation step {}...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step)
                feed_dict = {iterator_handle_op: data_handle, is_train_op: False}

                for p in range(args.npatches):
                    pred, pred_sc, label = sess.run([predictions_op, predictions_scores_op, labels_op], feed_dict=feed_dict)

                    # add one since the 0 (unknown dimension is used for mask)
                    pred = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(pred + 1, dtype=tf.int64)))
                    # label = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(label, dtype=tf.int64)))

                    if args.writetiles:

                        threadlist=list()

                        for tile in range(pred.shape[0]):
                            tileid = str(os.path.basename(files[tile]).split(".")[0])
                            tileid = '{}_{}'.format(p, tileid)

                            if tileid in dataset.geotransforms:
                                geotransform = dataset.geotransforms[tileid]
                                srid = dataset.srids[tileid]

                                threadlist.append(write_tile(pred[tile], tileid, join(args.storedir,PREDICTION_FOLDERNAME), geotransform, srid, 'prediction'))

                                if args.writegt:
                                    makedir(join(args.storedir, GROUND_TRUTH_FOLDERNAME))
                                    threadlist.append(
                                        write_tile(label[tile], tileid, join(args.storedir, GROUND_TRUTH_FOLDERNAME),
                                                   geotransform, srid, 'gt'))

                                if args.writeconfidences:
                                    for cl in range(pred_sc.shape[-1]):
                                        classname = dataset.classes[cl+1].replace(" ","_")

                                        foldername="{}_{}".format(cl+1,classname)

                                        outfolder = join(args.storedir,CONFIDENCES_FOLDERNAME,foldername)
                                        makedir(outfolder) # if not exists

                                        # write_tile(conn, pred_sc[tile, :, :, cl], datafilename=files[tile], tiletable=args.tiletable, outfolder=outfolder)
                                        threadlist.append(write_tile(pred_sc[tile, :, :, cl], tileid, outfolder, geotransform, srid, 'confidences'))
                                        #thread.start_new_thread(write_tile, writeargs)

                            # start all write threads!
                            for x in threadlist:
                                x.start()

                            # wait for all to finish
                            for x in threadlist:
                                x.join()

        except KeyboardInterrupt:
            print ("Evaluation aborted at step {}".format(step))
            pass

        #pbar.finish()
def write_tile(array, datafilename, outfolder, geotransform, srid, target):
    """gave up for now... wrapper around write_tile_ to implement multithreaded writing"""
    writeargs = (array, datafilename, outfolder, geotransform, srid, target)
    #write_tile_(array, datafilename, outfolder, geotransform, srid)
    thread = threading.Thread(target=write_tile_, args=writeargs)
    # thread.start()
    return thread


def write_tile_(array, datafilename, outfolder, geotransform, srid, target):

    tile = os.path.basename(datafilename).replace(".tfrecord", "").replace(".gz", "")

    outpath = os.path.join(outfolder, tile + ".tif")

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
    if target == 'prediction' or target == 'gt':
        dst_ds = gdal.GetDriverByName('GTiff').Create(outpath, ny, nx, 1, gdaldtype, options = ['COMPRESS=LZW'])
    elif target == 'confidences':
        dst_ds = gdal.GetDriverByName('GTiff').Create(outpath, ny, nx, 1, gdaldtype)

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(srid)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file

    dst_ds.GetRasterBand(1).WriteArray(array)

    dst_ds.FlushCache()  # write to disk
    dst_ds = None  # save, close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evalutation of models')
    parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
    # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
    parser.add_argument('--datadir', type=str, default=None,
                        help='directory containing the data (defaults to environment variable $datadir)')
    parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
    parser.add_argument('-t', '--writetiles', action="store_true",
                        help='write out pngs for each tile with prediction, label etc.')
    parser.add_argument('-gt', '--writegt', action="store_true",
                        help='write out pngs for each tile label etc.')
    parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='batchsize')
    parser.add_argument('-d', '--dataset', type=str, default="2016",
                        help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
    parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
    # tiletable now read from dataset.ini
    #parser.add_argument('--tiletable', type=str, default="tiles240", help="tiletable (default tiles240)")
    parser.add_argument('--allow_growth', type=bool, default=True, help="Allow VRAM growth")
    parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
    parser.add_argument('-exp', '--experiment', type=str, default="None")
    parser.add_argument('-ref','--reference', type=str, default="MCD12Q1v6raw_LCType1", help='Reference dataset to train')
    parser.add_argument('-step','--step', type=str, default="evaluation", help='step')
    parser.add_argument('-npatches','--npatches', type=int, default=2, help='Number of patches per combined TFRecord file')

    args = parser.parse_args()

    main(args)

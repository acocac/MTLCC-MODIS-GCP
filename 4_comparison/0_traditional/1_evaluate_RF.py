import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Flatten, Lambda, \
    SpatialDropout1D, Concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, GRU, Bidirectional
import argparse
from Dataset import Dataset
import tensorflow as tf

import datetime
from osgeo import gdal, osr
from os.path import join
import sklearn.metrics as skmetrics

import threading

EVAL_IDS_IDENTIFIER = "eval"

MASK_FOLDERNAME="mask"
GROUND_TRUTH_FOLDERNAME="ground_truth"
PREDICTION_FOLDERNAME="prediction"
LOSS_FOLDERNAME="loss"
CONFIDENCES_FOLDERNAME="confidences"

TRUE_PRED_FILENAME="truepred.npy"

import numpy as np

def main(args):
    if args.datadir is None:
        args.datadir = os.environ["datadir"]

    # train
    eval(args)

def eval(args):
    assert args.experiment
    assert args.reference
    assert type(args.num_classes) == int

    dataset = Dataset(datadir=args.datadir, verbose=True, section=args.dataset, experiment=args.experiment, reference=args.reference, num_classes=args.num_classes)

    if args.verbose: print ("initializing training dataset")
    tfdataset, filenames = dataset.create_tf_dataset(EVAL_IDS_IDENTIFIER, 0,
                                                     args.batchsize,
                                                     False,
                                                     prefetch_batches=args.prefetch)

    num_samples = len(filenames)

    # ### verify datasets ###
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    print("make iterator")
    iterator = tfdataset.make_initializable_iterator()
    nextitr = iterator.get_next()

    with tf.Session(config=config) as sess:
        sess.run(iterator.initializer)
    #     print("retrieving one sample as numpy array (just for fun)")
        x, doy, labels = sess.run(iterator.get_next())
        print("x shape: " + str(x.shape))
        print("doy shape: " + str(doy.shape))
        print("labels shape: " + str(labels.shape))
    ###

#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = args.allow_growth
#     with tf.Session(config=config) as sess:
#         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
#
#         sess.run([iterator.initializer])
#
#         nclasses = args.num_classes
#         model = Archi_TempCNN(nclasses)
#
#         # model_weights = load_model(args.modeldir)
#         model.load_weights(args.modeldir)  # load model weights
#
#         batches = num_samples / args.batchsize
#
#         # create appropiate folders
#         def makedir(outfolder):
#             if not os.path.exists(outfolder):
#                 os.makedirs(outfolder)
#
#         makedir(join(args.storedir, PREDICTION_FOLDERNAME))
#         makedir(join(args.storedir, PREDICTION_FOLDERNAME))
#         makedir(join(args.storedir, GROUND_TRUTH_FOLDERNAME))
#
#         try:
#             truepred = np.empty((0, 2),dtype=int)
#
#             #pbar.start()
#             starttime = datetime.datetime.now()
#             for i in range(0, int(batches)):
#                 #pbar.update(i)
#                 now = datetime.datetime.now()
#                 dt = now - starttime
#                 seconds_per_batch = dt.seconds/(i+1e-10)
#                 seconds_to_go = seconds_per_batch*(batches - i)
#                 eta = now+datetime.timedelta(seconds=seconds_to_go)
#                 print ("{} eval batch {}/{}, eta: {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), i,batches, eta.strftime("%Y-%m-%d %H:%M:%S")))
#
#                 stepfrom = i * args.batchsize
#                 stepto = stepfrom + args.batchsize + 1
#
#                 # take with care...
#                 files = filenames[stepfrom:stepto]
#
#                 x, label = sess.run(nextitr)
#
#                 x = x[0,:,:,:]
#                 # label = label[0,:,:]
#
#                 pred_sc = model.predict(x)
#                 pred = pred_sc.argmax(axis=-1)
#
#                 nx = label.shape[-1]
#                 pred = np.reshape(pred, (1, nx, nx))
#                 pred_sc_reshape = np.reshape(pred_sc, (1, nx, nx, pred_sc.shape[1]))
#
#                 # add one since the 0 (unknown dimension is used for mask) - not for the tempCNN model
#                 # label = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(label + 1, dtype=tf.int64)))
#                 # pred = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(pred + 1, dtype=tf.int64)))
#
#                 y_true = label.flatten()
#                 y_pred = pred.flatten()
#
#                 truepred = np.row_stack((truepred, np.column_stack((y_true, y_pred))))
#
#                 if args.writetiles:
#
#                     threadlist=list()
#
#                     for tile in range(pred.shape[0]):
# #ac                        tileid = int(os.path.basename(files[tile]).split(".")[0])
#                         tileid = str(os.path.basename(files[tile]).split(".")[0])
#
#                         geotransform = dataset.geotransforms[tileid]
#                         srid = dataset.srids[tileid]
#
#                         threadlist.append(write_tile(pred[tile], files[tile], join(args.storedir,PREDICTION_FOLDERNAME), geotransform, srid))
#
#                         if args.writegt:
#                             threadlist.append(write_tile(label[tile], files[tile],join(args.storedir,GROUND_TRUTH_FOLDERNAME), geotransform, srid))
#
#                         if args.writeconfidences:
#                             print(pred_sc_reshape.shape[-1])
#                             for cl in range(pred_sc_reshape.shape[-1]):
#                                 print(cl)
#                                 classname = dataset.classes[cl+1].replace(" ","_")
#
#                                 foldername="{}_{}".format(cl+1,classname)
#
#                                 outfolder = join(args.storedir,CONFIDENCES_FOLDERNAME,foldername)
#                                 makedir(outfolder) # if not exists
#
#                                 # write_tile(conn, pred_sc[tile, :, :, cl], datafilename=files[tile], tiletable=args.tiletable, outfolder=outfolder)
#                                 threadlist.append(write_tile(pred_sc_reshape[tile, :, :, cl], files[tile], outfolder, geotransform, srid))
#                                 #thread.start_new_thread(write_tile, writeargs)
#
#                     # start all write threads!
#                     for x in threadlist:
#                         x.start()
#
#                     # wait for all to finish
#                     for x in threadlist:
#                         x.join()
#
#         except KeyboardInterrupt:
#             print ("Evaluation aborted at step {}".format(step))
#             pass
#
#     ## write metrics and confusion matrix
#     #csv_rows = []
#     #csv_rows.append(", ".join(dataset.classes[1:]))
#
#     np.save(join(args.storedir,TRUE_PRED_FILENAME), truepred)
#
#     truepred = np.load(join(args.storedir,TRUE_PRED_FILENAME))
#     print(truepred.shape)
#
#     y_true = truepred[:, 0]
#     y_pred = truepred[:, 1]
#
#     y_true2 = np.ma.MaskedArray(y_true, mask=y_true==0).compressed()
#     y_pred2 = np.ma.MaskedArray(y_pred, mask=y_true==0).compressed()
#
#     ids_without_unknown = np.array(dataset.ids[1:]) - 1
#     confusion_matrix = skmetrics.confusion_matrix(y_true2, y_pred2, labels=dataset.ids[1:])
#
#     with open(join(args.storedir,'confusion_matrix.csv'), 'w+') as f:
#         class_temp = dataset.classes[1:]
#         f.write(", ".join(dataset.classes[1:])+"\n")
#         np.savetxt(f, confusion_matrix, fmt='%d', delimiter=', ', newline='\n')
#
#     classreport = skmetrics.classification_report(y_true, y_pred, labels=dataset.ids[1:], target_names=dataset.classes[1:])
#
#     classes = np.column_stack((dataset.ids[1:], dataset.classes[1:]))
#     np.savetxt(join(args.storedir,'classes.csv'), classes, fmt='%s', delimiter=', ', newline='\n')
#
#     with open(join(args.storedir, 'classification_report.txt'), 'w') as f:
#         f.write(classreport)
#
#     dec = 2 # round scores to dec significant digits
#     prec, rec, fscore, sup = skmetrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=dataset.ids[1:])
#     avg_prec = skmetrics.precision_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')
#     avg_rec = skmetrics.recall_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')
#     avg_fscore = skmetrics.f1_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')
#
#     metrics = np.column_stack((classes, np.around(prec, dec), np.around(rec, dec), np.around(fscore, dec), sup))
#     with open(join(args.storedir, "metrics.csv"), 'w+') as f:
#         f.write('id,name,precision,recall,fscore,support\n')
#         np.savetxt(f, metrics, fmt='%s', delimiter=', ', newline='\n')
#         f.write(',,,,,\n'.format(prec=avg_prec, rec=avg_rec, fsc=avg_fscore))
#         f.write(',weight. avg,{prec},{rec},{fsc},\n'.format(prec=np.around(avg_prec, dec), rec=np.around(avg_rec, dec),
#                                                              fsc=np.around(avg_fscore, dec)))
#
#

        #pbar.finish()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evalutation of models')
    parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
    # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
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
    # tiletable now read from dataset.ini
    #parser.add_argument('--tiletable', type=str, default="tiles240", help="tiletable (default tiles240)")
    parser.add_argument('--allow_growth', type=bool, default=True, help="Allow VRAM growth")
    parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")
    parser.add_argument('-exp', '--experiment', type=str, default="None")
    parser.add_argument('-ref','--reference', type=str, default="MCD12Q1v6raw_LCType1", help='Reference dataset to train')
    parser.add_argument('-nc','--num_classes', type=int, default=17, help='number of classes')

    args = parser.parse_args()

    main(args)

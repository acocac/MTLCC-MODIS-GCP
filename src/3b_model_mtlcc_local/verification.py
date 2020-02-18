import tensorflow as tf
import os
from Dataset import Dataset
import argparse
import datetime
from osgeo import gdal, osr
from os.path import join

import threading

#os.environ["GDAL_DATA"] = os.environ["HOME"] + "/.conda/envs/MTLCC/share/gdal"

EVAL_IDS_IDENTIFIER = "viz"

MODEL_CFG_FILENAME = "params.ini"

import numpy as np

# simple flag to track if graph is created in this session or has to be imported
graph_created_flag = False


def main(args):
    # if args.verbose: print "setting visible GPU {}".format(args.gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.datadir is None:
        args.datadir = os.environ["datadir"]

    eval(args)

def eval(args):
    assert args.experiment
    GROUND_TRUTH_FOLDERNAME = args.reference

    print(args.dataset)

    dataset = Dataset(datadir=args.datadir, verbose=True, section=args.dataset, experiment=args.experiment, reference=args.reference, step=args.step)

    if args.verbose: print ("initializing training dataset")
    tfdataset, output_shapes, output_datatypes, filenames = dataset.create_tf_dataset(EVAL_IDS_IDENTIFIER, 0,
                                                                                      args.batchsize,
                                                                                      False,
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

        makedir(join(args.storedir, GROUND_TRUTH_FOLDERNAME))

        try:
            for i in range(0, int(batches)):

                x250, x500, DOY, year, label = iterator.get_next()
                x250, x500, DOY, year, label = sess.run([x250, x500, DOY, year, label])

                stepfrom = i * args.batchsize
                stepto = stepfrom + args.batchsize + 1

                files = filenames[stepfrom:stepto]

                if args.writetiles:

                    threadlist=list()

                    for tile in range(label.shape[0]):
#ac                        tileid = int(os.path.basename(files[tile]).split(".")[0])
                        tileid = str(os.path.basename(files[tile]).split(".")[0])

                        geotransform = dataset.geotransforms[tileid]
                        srid = dataset.srids[tileid]

                        threadlist.append(write_tile(label[tile], files[tile],join(args.storedir,GROUND_TRUTH_FOLDERNAME), geotransform, srid))

                    # start all write threads!
                    for x in threadlist:
                        x.start()

                    # wait for all to finish
                    for x in threadlist:
                        x.join()

        except KeyboardInterrupt:
            print ("Evaluation aborted")
            pass

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
    # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
    parser.add_argument('--datadir', type=str, default=None,
                        help='directory containing the data (defaults to environment variable $datadir)')
    parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
    parser.add_argument('-t', '--writetiles', action="store_true",
                        help='write out pngs for each tile with prediction, label etc.')
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

    args = parser.parse_args()

    main(args)

#!/usr/bin/env python3
# =============================================================================
# Date:     Feb, 2020
# Author:   Alejandro Coca
# Purpose:  Creates proximity rasters.
# =============================================================================

import re

import gdal

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-i','--indir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Out dir')
parser.add_argument('-t','--targetvalue', type=str, required=True,
                    help='target value')

def create_proximity_raster(src, dst, values, units='PIXEL'):
    """
    Creates a proximity raster using gdal.ComputeProximity. NoData pixels in
    the src raster will be considered NoData pixels in the dst raster.
    :param src: source raster filename
    :param dst: dest raster filename
    :return:    None
    """
    # open src raster
    ds = gdal.Open(src, 0)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # create dst raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(dst, cols, rows, 1, gdal.GDT_Int16,  options = ['COMPRESS=LZW'])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(sr)

    print(",".join(map(str, values)))
    # define options for gdal.ComputeProximity and execute it
    if values == "67":
        options = [
            f'VALUES={["6, 7"]}',
            f'DISTUNITS={units}',
            'USE_INPUT_NODATA=NO',
        ]
    else:
        options = [
            f'VALUES={",".join(map(str, values))}',
            f'DISTUNITS={units}',
            'USE_INPUT_NODATA=NO',
        ]

    gdal.ComputeProximity(ds.GetRasterBand(1), out_ds.GetRasterBand(1), options)

    del ds, out_ds


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    tar_val = args.targetvalue

    create_proximity_raster(indir, outdir, [tar_val])
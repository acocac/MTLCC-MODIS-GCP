from osgeo import gdal

import os
import glob
import re

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-i','--indir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')
parser.add_argument('-y', '--targetyear', type=str, required=True,
                    help='Target year')

def makedir(outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def export_raw(fileNames, outdir, fn, target):
    if target == 'prediction':
        cl_string = "-of GTiff -co COMPRESS=LZW -overwrite \
        -multi -wo NUM_THREADS=ALL_CPUS -s_srs EPSG:4326 -srcnodata 0 -dstnodata -9999"
    elif target == 'confidences':
        cl_string = "-of GTiff -co COMPRESS=LZW -overwrite \
        -multi -wo NUM_THREADS=ALL_CPUS -s_srs EPSG:4326 -srcnodata 0 -dstnodata -3.4028234663852886e+38"

    warp_options = gdal.WarpOptions(gdal.ParseCommandLine(cl_string))
    vrt_options = gdal.BuildVRTOptions(separate=False)

    vrt_output=os.path.join(outdir,'{}.vrt'.format(fn))
    gdal.BuildVRT(vrt_output, fileNames, options=vrt_options)

    output_tiff = vrt_output.replace(".vrt", ".tif")

    if not os.path.exists(output_tiff):
        gdal.Warp(output_tiff, vrt_output, options=warp_options)


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    tyear = args.targetyear

    ##predictions
    pred_indir = os.path.join(indir,'prediction')

    fileNames = glob.glob(os.path.join(pred_indir, '*.tif'))
    fileNames.sort(key=lambda f: int(re.sub('\D', '', f)))

    makedir(os.path.join(outdir,'prediction'))

    export_raw(fileNames, os.path.join(outdir,'prediction'), tyear, 'prediction')

    #confidences
    conf_indir = os.path.join(indir,'confidences')

    tclasses =  os.listdir(conf_indir)

    makedir(os.path.join(outdir,'confidences',tyear))

    for c in tclasses:
        fileNames = glob.glob(os.path.join(conf_indir,c,'*.tif'))
        # sort dataframe by numeric format
        fileNames.sort(key=lambda f: int(re.sub('\D', '', f)))

        export_raw(fileNames, os.path.join(outdir,'confidences',tyear), c, 'confidences')
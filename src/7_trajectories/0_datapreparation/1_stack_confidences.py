import os
import glob
import re
import numpy as np
import rasterio
import fiona
from rasterio.mask import mask

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

def read_file(file):
    with rasterio.open(file) as src:
        return (src.read(1))

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    tyear = args.targetyear

    ##predictions
    file_list = glob.glob(os.path.join(indir,'*.tif'))
    file_list = sorted(file_list, key=lambda x: int(os.path.basename(os.path.dirname(x)).partition('_')[ 0 ]))

    # array_list = [read_file_lc(x, aoi) for x in file_list ]
    array_list = [read_file(x) for x in file_list ]
    array_stack = np.stack(array_list)  # stack

    with rasterio.open(file_list[0]) as src:
        profile = src.profile

    out_profile = profile

    out_profile.update({
        'dtype': rasterio.float32,
        'count': array_stack.shape[0],
        'height': array_stack.shape[1],
        'width': array_stack.shape[2],
        'compress': 'lzw',
        'tiled': 'True', ##options added to export large raster
        'blockxsize': 384,
        'blockysize': 384,
        'BIGTIFF': 'YES'})

    makedir(outdir)

    with rasterio.open(os.path.join(outdir, '{}.tif'.format(tyear)), 'w', **out_profile) as dest:
        dest.write(array_stack)
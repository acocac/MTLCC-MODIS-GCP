import rasterio
import os

import numpy as np
import glob
import re
from rasterio.mask import mask
import fiona
import sys
import joblib

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-i','--indir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')
parser.add_argument('-n','--nworkers', type=int, default=None,
                    help='Number of workers (by default all)')
parser.add_argument('--noconfirm', action='store_true',
                    help='Skip confirmation')

def confirm(prompt=None, resp=False):
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False


def makedir(outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def read_file_target(raster, shape):
    with rasterio.open(raster) as src:
        out_image, out_transform = mask(src, shape, crop=True, invert=False, all_touched=False)
        profile = src.profile
        out_meta = src.meta
        return out_image, out_transform, out_meta, profile


def read_file_lc(raster, shape):
    with rasterio.open(raster) as src:
        out_image, out_transform = mask(src, shape, crop=True, invert=False, all_touched=False)
        out_image[out_image==0] = np.nan
        return out_image


def gen_cleanlc(outdir, year):
    # update profile
    lc_stack= lc_stack_list[year]

    out_profile = target_layers[0][3]
    out_profile.update({
        'dtype': rasterio.float32,
        'count': lc_stack.shape[0],
        'height': target_layers[0][0].shape[1],
        'width': target_layers[0][0].shape[2],
        'transform': target_layers[0][1],
        'compress': 'lzw'})

    with rasterio.open('{}\{}.tif'.format(outdir,year), 'w',
                       **out_profile) as dest:
        dest.write(lc_stack)

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    nworkers = args.nworkers

    makedir(outdir)

    # lc data
    lc_list = glob.glob(os.path.join(indir, '*.tif'))
    lc_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    # aux files
    auxdir = r'F:\acoca\research\gee\dataset\AMZ\implementation'
    aoi_file = os.path.join(auxdir,'aoi','aoi_bigger.shp')
    terrai_file = os.path.join(auxdir,'terrai','AMZ_decrease_2004_01_01_to_2019_06_10.tif')

    # read files
    with fiona.open(aoi_file, "r") as shapefile:
        aoi = [feature["geometry"] for feature in shapefile]

    target_list = [terrai_file]

    target_layers = [read_file_target(x, aoi) for x in target_list]

    lc_stack_list = [read_file_lc(x, aoi) for x in lc_list]

    if len(lc_stack_list) == 0:
        print('No tiles to process... Terminating')
        sys.exit(0)

    print()
    print('Will process the following :')
    print('nworkers : %s' % str(nworkers))
    print('Input data dir : %s' % str(indir))
    print('Output data dir : %s' % str(outdir))
    print()

    if not args.noconfirm:
        if not confirm(prompt='Proceed?', resp=True):
            sys.exit(0)

    # Launch the process
    if nworkers is not None and nworkers > 1:

        joblib.Parallel(n_jobs=nworkers)(
            joblib.delayed(gen_cleanlc)(outdir, tyear)
            for tyear in range(len(lc_stack_list))
        )
    else:
        for tyear in range(len(lc_stack_list)):
            gen_cleanlc(outdir, tyear)
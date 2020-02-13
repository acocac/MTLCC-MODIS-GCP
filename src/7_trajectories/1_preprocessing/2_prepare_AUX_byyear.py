import rasterio
import os

import numpy as np
import glob
import re
from rasterio.mask import mask
import fiona
from rasterio.features import sieve
import sys
import joblib

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-p','--preddir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-a','--auxdir', type=str, required=True,
                    help='Aux dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')
parser.add_argument('-y', '--targetyears', type=int, default="2016", nargs='+',
                    help='Target year(s)')
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
        # out_image = out_image.astype(np.float32)
        # out_image[out_image==0] = 255
        return out_image


def gen_cleanlc(outdir, year):
    # select target year and sieve isolated pixels
    target_terrai = target_layers[0][0] == year
    terra_target_sieve = sieve(target_terrai[0].astype('int32'), size=2)
    terra_target_sieve = np.expand_dims(terra_target_sieve, axis=0)

    # mask lc data
    new_msk = ((terra_target_sieve != 1) | (target_layers[1][0] != 0))  # working mask water and year
    lc_layers_target = [ np.ma.MaskedArray(x, mask=new_msk) for x in lc_layers ]

    # arrange stack
    lc_stack = np.ma.stack(lc_layers_target, axis=0)
    lc_stack = lc_stack.reshape((lc_stack.shape[0] * lc_stack.shape[1], lc_stack.shape[2], lc_stack.shape[3]))
    # lc_stack = lc_stack.filled(np.nan)
    lc_stack = lc_stack.filled(0)

    # update profile
    out_profile = target_layers[1][3]
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
    preddir = args.preddir
    auxdir = args.auxdir
    outdir = args.outdir
    tyears = args.targetyears
    nworkers = args.nworkers

    makedir(outdir)

    # lc data
    lcdir = os.path.join(preddir,'gee')
    lc_list = glob.glob(os.path.join(lcdir, '*.tif'))
    lc_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    # aux files
    aoi_file = os.path.join(auxdir,'aoi','amazon_raisg.shp')
    terrai_file = os.path.join(auxdir,'terrai','AMZ_decrease_2004_01_01_to_2019_06_10_new.tif')
    watermask_file = os.path.join(auxdir,'watermask','watermask_new.tif')

    # read files
    with fiona.open(aoi_file, "r") as shapefile:
        aoi = [feature["geometry"] for feature in shapefile]

    target_list = [terrai_file, watermask_file]

    target_layers = [read_file_target(x, aoi) for x in target_list]

    lc_layers = [read_file_lc(x, aoi) for x in lc_list]

    if len(tyears) == 0:
        print('No tiles to process... Terminating')
        sys.exit(0)

    print()
    print('Will process the following :')
    print('Number of tiles : %d' % len(tyears))
    print('nworkers : %s' % str(nworkers))
    print('Input data dir : %s' % str(preddir))
    print('Output data dir : %s' % str(outdir))
    print()

    if not args.noconfirm:
        if not confirm(prompt='Proceed?', resp=True):
            sys.exit(0)

    # Launch the process
    if nworkers is not None and nworkers > 1:

        joblib.Parallel(n_jobs=nworkers)(
            joblib.delayed(gen_cleanlc)(outdir, tyear)
            for tyear in tyears
        )
    else:
        for tyear in tyears:
            gen_cleanlc(outdir, tyear)
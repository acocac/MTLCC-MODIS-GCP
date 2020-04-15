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

parser.add_argument('-i','--indir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')
parser.add_argument('-y', '--targetyears', type=int, default="2016", nargs='+',
                    help='Target year(s)')
parser.add_argument('-s','--size', type=int, required=True,
                    help='sieve size')

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


def sieve_year(y, size):
    # select target year and sieve isolated pixels
    target_terrai = target_layers[ 0 ][ 0 ] == y
    terra_target_sieve = sieve(target_terrai[ 0 ].astype('int32'), size=4)
    terra_target_sieve = np.expand_dims(terra_target_sieve, axis=0)

    new_msk = ((terra_target_sieve != 1) | (target_layers[ 1 ][ 0 ] != 0))  # working mask water and year

    lc_layers_target = np.ma.MaskedArray(target_terrai, mask=new_msk)

    return lc_layers_target

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    tyears = args.targetyears
    size = args.size

    makedir(outdir)

    # lc data
    lcdir = os.path.join(indir,'mapbiomas','tif')
    lc_list = glob.glob(os.path.join(lcdir, '*.tif'))
    lc_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    # aux files
    aoi_file = os.path.join(indir,'aoi','amazon_raisg.shp')
    terrai_file = os.path.join(indir,'terrai','AMZ_decrease_2004_01_01_to_2019_06_10.tif')
    watermask_file = os.path.join(indir,'watermask','watermask.tif')

    # read files
    with fiona.open(aoi_file, "r") as shapefile:
        aoi = [feature["geometry"] for feature in shapefile]

    target_list = [terrai_file, watermask_file]

    target_layers = [read_file_target(x, aoi) for x in target_list]

    sYear = tyears[0]
    eYear = tyears[1]

    terrai_layers = [sieve_year(x, size) for x in range(sYear,eYear)]

    a = np.ma.stack(terrai_layers, axis=0)
    z = a.reshape((a.shape[ 0 ] * a.shape[ 1 ], a.shape[ 2 ], a.shape[ 3 ]))
    u = z.filled(0)
    y = u * 1
    final = u.sum(axis=0)

    out_profile = target_layers[ 0 ][ 3 ]

    out_profile.update({
        'dtype': rasterio.int32,
        'count': target_layers[ 0 ][ 0 ].shape[ 0 ],
        'height': target_layers[ 0 ][ 0 ].shape[ 1 ],
        'width': target_layers[ 0 ][ 0 ].shape[ 2 ],
        'transform': target_layers[ 0 ][ 1 ],
        'compress': 'lzw'})

    with rasterio.open('{}\AMZ_sieve_terrai_{}_{}_sieve{}.tif'.format(outdir,sYear, eYear, size), 'w',
                       **out_profile) as dest:
        dest.write(final, 1)


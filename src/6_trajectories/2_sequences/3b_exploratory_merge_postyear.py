import os
import glob
import re
import numpy as np
import rasterio
import fiona
from rasterio.mask import mask

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-g','--geodir', type=str, required=True,
                    help='Geo dir')
parser.add_argument('-a','--auxdir', type=str, required=True,
                    help='Aux dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')

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
        return out_image

if __name__ == '__main__':
    args = parser.parse_args()
    geodir = args.geodir
    auxdir = args.auxdir
    outdir = args.outdir

    makedir(outdir)

    # targets = ['entropyP','turbulence']
    targets = ['entropy','turbulence']

    for t in targets:
        ##predictions
        file_list = glob.glob(os.path.join(geodir,'{}*.tif'.format(t)))

        # aux files
        aoi_file = os.path.join(auxdir, 'aoi', 'amazon_raisg.shp')

        # read files
        with fiona.open(aoi_file, "r") as shapefile:
            aoi = [ feature[ "geometry" ] for feature in shapefile ]

        target_layers = [read_file_target(x, aoi) for x in file_list[0:1]]

        array_list = [read_file_lc(x, aoi) for x in file_list ]

        array_stack = np.stack(array_list)  # stack
        array_stack = np.max(array_stack, axis=0)

        # update profile
        out_profile = target_layers[0][3]
        out_profile.update({
            'dtype': rasterio.float32,
            'count': 1,
            'height': target_layers[0][0].shape[1],
            'width': target_layers[0][0].shape[2],
            'transform': target_layers[0][1],
            'compress': 'lzw',
            'blockxsize': 384,
            'blockysize': 384,
            'BIGTIFF': 'YES'})


        with rasterio.open(os.path.join(outdir, '{}.tif'.format(t)), 'w', **out_profile) as dest:
            dest.write(array_stack)
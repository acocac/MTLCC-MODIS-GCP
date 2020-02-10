#!/bin/env python

"""
Create a land cover map from a probability raster file by applying a
maximum likelihood estimate to each pixel. This script reads a 17 band
raster and outputs a single band raster containing the index of the
input band with the maximum value.

Author: Parker Abercrombie <parker@pabercrombie.com>
"""

import os
import sys
import numpy as np

import gdal
from gdalconst import *

import configfile

WATER_MASK = 1
REPLACEMENT = 17

NO_DATA = 255

TILE_WIDTH = 2400
TILE_HEIGHT = 2400

def usage():
    print """Usage: mle tiles.conf stabilize.conf

Arguments:
   tiles.conf:      File that lists MODIS tiles to process, one per line.
   stabilize.conf:  Configuration file. Default: stabilize.conf
"""

def process_tile(in_file_name, out_file_name, lw_mask):
    dataset = gdal.Open(in_file_name, GA_ReadOnly)
    if dataset is None:
        print("Can't read dataset {}".format(in_file_name))

    probs = dataset.ReadAsArray(0, 0, 2400, 2400)
    probs = np.rollaxis(probs, 1)
    probs = np.rollaxis(probs, 2, 1)

    nodata = np.sum(probs, 2) < 0.0005

    mle = np.argmax(probs, 2) + 1
    mle = mle.astype('uint8')

    mle[nodata] = NO_DATA

    # Apply land/water mask
    mle[lw_mask == WATER_MASK] = REPLACEMENT

    if not os.path.exists(os.path.dirname(out_file_name)):
        os.makedirs(os.path.dirname(out_file_name))

    with file(out_file_name, 'wb') as out:
        mle.tofile(out)

    if len(sys.argv) > 1:
        tiles_file = sys.argv[1]
    else:
        usage()
        return

def main():
    if len(sys.argv) > 1:
        tiles_file = sys.argv[1]
    else:
        usage()
        return

    if len(sys.argv) > 2:
        config_file = sys.argv[2]
    else:
        config_file = 'stabilize.conf'

    config = configfile.parse(config_file)

    start_year = int(config['start_year'])
    end_year = int(config['end_year'])

    input_pattern = config['in_pattern']
    output_pattern = config['out_pattern_map']
    lw_mask_pattern = config['lw_mask_pattern']

    with open(tiles_file) as f:
        for tile in f:
            tile = tile.strip()
            if tile[0] == '#':
                continue # Skip comment lines

            lw_mask = np.fromfile(lw_mask_pattern.format(TILE=tile), dtype='uint8')
            lw_mask = lw_mask.reshape((TILE_WIDTH, TILE_HEIGHT))

            for year in xrange(start_year, end_year + 1):
                in_file = input_pattern.format(YEAR=year, TILE=tile)
                out_file = output_pattern.format(YEAR=year, TILE=tile)

                print in_file
                process_tile(in_file, out_file, lw_mask)

if __name__ == "__main__":
    main()

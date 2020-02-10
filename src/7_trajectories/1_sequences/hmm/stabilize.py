#!/bin/env python
#$ -pe omp 12
#$ -l h_rt=48:00:00

"""
Apply Markov chain algorithms to stabilize a series of land cover
maps. This program takes as input a time series of probability
raster files for a MODIS tile, and outputs either a series of
adjusted probability files, or classified raster files (depending on
the stabilization algorithm).

Parameters are read from a configuration file.

Usage: stabilize (tile | tiles.conf) stabilize.conf

Author: Parker Abercrombie <parker@pabercrombie.com>
"""

import sys
import os
import re
import numpy as np
import multiprocessing
import logging

import gdal
from gdalconst import *

try:
    import hmm
    import configfile
except:
    sys.stderr.write("Error importing hmm.py. Are you running this program as "
                     "a batch job on the computing cluster? Try submitting "
                     "stabilize.sh instead.\n")
    exit(1)

# Global configuration parameters. These values are read from the
# configutation file.
start_year = None
end_year = None
n_labels = None

change_prob = None
transition_matrix = None

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.WARNING)

def usage():
    print """Usage: stabilize (tile | tiles.conf) stabilize.conf

Arguments:
   tile:            MODIS tile ID (e.g. h08v05).
   tiles.conf:      File that lists MODIS tiles to process, one per line.
   stabilize.conf:  Configuration file. Default: stabilize.conf
"""

def ensure_dir_exists(path):
    """
    Test if a directory exists, and create the directory (including
    intermediate directories) if it does not. Raises an exception if
    the directory cannot be created, or if the path exists, but is not
    a directory.
    """
    try:
        os.makedirs(path)
    except:
        if not os.path.isdir(path):
            raise

class DataReader:
    """Iterator to read raster files in chunks of rows."""

    def __init__(self, pattern, lw_mask_pattern, tile, years, chunk_size=100):
        """
        Create a new data reader to read input files for a tile over a
        series of years.

        Parameters:
           tile:       MODIS tile identifier string (e.g. 'h08v05')
           years:      List or tuple of years to process.
           chunk_size: Number of rows to read as a chunk.
        """

        self.prob_rasters = {}

        self.years = years
        self.chunk_size = chunk_size

        for year in years:
            filename = pattern.format(TILE=tile, YEAR=year)
            dataset = gdal.Open(filename, GA_ReadOnly)
            if dataset is None:
                raise IOError("Can't read dataset {}".format(filename))
            self.prob_rasters[year] = dataset

            # Make sure that all of the rasters are the same dimensions
            if year == self.years[0]:
                self.x_size, self.y_size = (dataset.RasterXSize, dataset.RasterYSize)
            elif not ((self.x_size, self.y_size) == (dataset.RasterXSize, dataset.RasterYSize)):
                raise Exception('Incorrect dimension for data set {}. Expected {} x {} but got {} x {}'.format(
                        filename, self.x_size, self.y_size, dataset.RasterXSize, dataset.RasterYSize))

        self.lw_mask = np.fromfile(lw_mask_pattern.format(TILE=tile), dtype='uint8')
        self.lw_mask = self.lw_mask.reshape((2400, 2400))

    def __iter__(self):
        self.row = 0
        return self

    def read_chunk(self, year, row):
        raster = self.prob_rasters[year]
        return raster.ReadAsArray(0, row, raster.RasterXSize, self.chunk_size)

    def next(self):
        if self.row >= self.y_size:
            raise StopIteration

        # Read rows from each of the yearly raster files, and stack them into one matrix
        print "Reading rows {} to {}".format(self.row, self.row + self.chunk_size)
        pixels = np.vstack([self.read_chunk(year, self.row) for year in self.years])

        mask = self.lw_mask[self.row:(self.row + self.chunk_size), :].reshape((1, self.chunk_size, self.x_size))
        pixels = np.vstack((mask,  pixels))

        self.row += self.chunk_size

        return pixels

def process_tile(pool, tile, in_file_pattern, out_file_pattern, lw_mask_pattern,
                 function, out_data_type):
    """
    Process one tile. Read input files, apply stabilization algorithm
    over each pixel's time series, and write output.
    
    Arguments:
      pool - Multi-processing pool that will handle work.
      tile - Tile ID of tile to process (e.g. 'h08v05').
      in_file_pattern - File pattern of input files. TILE and YEAR
                        patterns will be replaced with tile ID and
                        four digit year to form the input file paths.
      out_file_pattern - File pattern of output files.
      lw_mask_pattern - File pattern for land/water mask. Water pixels
                        will not processed for stabilization.
      function - Function to apply to each pixel.
      out_data_type - GDAL data type of the output file (e.g. GDT_Float32).
    """
    print "==== Processing tile {} ====".format(tile)

    years = range(start_year, end_year + 1)
    reader = DataReader(in_file_pattern, lw_mask_pattern, tile, years)

    # Apply the stabilization algorithm using the process pool to
    # distribute the work. The main process will handle all file IO
    # through the data reader, and pass chunks of the input
    # rasters to each of the worker processes.
    result_it = pool.imap(function, reader)
    #import itertools; result_it = itertools.imap(function, reader)

    # Store output raster objects in a dictionary. They will be opened
    # as needed in the loop below.
    out_rasters = {}

    # Iterate over results as they are handed back from the
    # processing pool. Write each result chunk into the
    # appropriate output files.
    for i, labels in enumerate(result_it):
        for year in years:
            n_bands = labels.shape[0] / len(years)

            # Find the output file for this year of data. Open the
            # file if it is not already open.
            try:
                dataset = out_rasters[year]
            except KeyError:
                filename = out_file_pattern.format(TILE=tile, YEAR=year)

                ensure_dir_exists(os.path.dirname(filename))
                dataset = gdal.GetDriverByName('ENVI').Create(filename, reader.x_size, reader.y_size,
                                                              n_bands, out_data_type)
                if dataset is None:
                    raise IOError("Can't create dataset {}".format(filename))
                out_rasters[year] = dataset

            for band in xrange(1, n_bands + 1):
                indx = (year - start_year) * n_bands + band - 1
                dataset.GetRasterBand(band).WriteArray(labels[indx, :, :], 0, i * reader.chunk_size)

# Wrappers to invoke hmm functions using numpy.apply_along_axis. Note
# that it is not possible to create these functions dynamically
# because they must be serialized and passed to the child processes
# (this is a limitation of the multiprocessing library).
def hmm_forward_wrapper(pixels):
    """Wrapper function to invoke hmm.forward on the first axis of an array."""
    try:
        logger.debug("Processing {} pixels".format(pixels.shape[1] * pixels.shape[2]))
        return np.apply_along_axis(hmm.forward, 0, pixels, n_labels, transition_matrix)
    except KeyboardInterrupt:
        # Ignore keyboard interrupt in child process. This will be
        # handled by parent process.
        pass

def hmm_forward_backward_wrapper(pixels):
    """Wrapper function to invoke hmm.forward_backward on the first axis of an array."""
    try:
        logger.debug("Processing {} pixels".format(pixels.shape[1] * pixels.shape[2]))
        return np.apply_along_axis(hmm.forward_backward, 0, pixels, n_labels, transition_matrix)
    except KeyboardInterrupt:
        # Ignore keyboard interrupt in child process. This will be
        # handled by parent process.
        pass

def hmm_viterbi_wrapper(pixels):
    """Wrapper function to invoke hmm.viterbi on the first axis of an array."""
    try:
        logger.debug("Processing {} pixels".format(pixels.shape[1] * pixels.shape[2]))
        return np.apply_along_axis(hmm.viterbi, 0, pixels, n_labels, transition_matrix)
    except KeyboardInterrupt:
        # Ignore keyboard interrupt in child process. This will be
        # handled by parent process.
        pass
    
algorithm_library = {
    'forward' :          (hmm_forward_wrapper,          GDT_Float32),
    'forward_backward' : (hmm_forward_backward_wrapper, GDT_Float32),
    'viterbi' :          (hmm_viterbi_wrapper,          GDT_Byte)
}

def main():
    global transition_matrix
    global start_year
    global end_year
    global n_labels

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
    n_labels = int(config['n_labels'])

    # Construct the transition matrix
    change_prob = float(config['transition_prior'])
    transition_matrix = np.empty((n_labels, n_labels))
    transition_matrix.fill(change_prob)
    np.fill_diagonal(transition_matrix, 1.0 - change_prob)

    input_pattern = config['in_pattern']
    lw_mask_pattern = config['lw_mask_pattern']

    try:
        algorithm, out_data_type = algorithm_library[config['algorithm']]
    except KeyError:
        sys.stderr.write('Unknown algorithm: {}\n'.format(config['algorithm']))
        sys.stderr.write('Options are:\n')
        for a in algorithm_library.keys():
            sys.stderr.write('\t{}\n'.format(a))
        return

    # Use either the map or probs output pattern depending on the
    # algorithm. Viterbi generates a classification map (one band),
    # and the other algorithms generate marginal probability files
    # (one band for each label).
    if algorithm == 'viterbi':
        output_pattern = config['out_pattern_map']
    else:
        output_pattern = config['out_pattern_probs']

    try:
        print "Starting pool of {} processes...".format(multiprocessing.cpu_count())
    except NotImplementedError:
        print "Starting process pool..."

    pool = multiprocessing.Pool()
    try:
        if re.match('h\d?\dv\d?\d', tiles_file):
            tiles = [tiles_file]
        else:
            with open(tiles_file, 'r') as tile_file:
                tiles = tile_file.readlines()

        for tile in tiles:
            tile = tile.strip()
            if tile[0] == '#':
                continue # Skip comments

            process_tile(pool, tile, input_pattern, output_pattern, lw_mask_pattern,
                         algorithm, out_data_type)
    except KeyboardInterrupt:
        pool.terminate()
    except IOError, e:
        sys.stderr.write('Error: {}\n'.format(str(e)))

    pool.close()

if __name__ == "__main__":    
    np.set_printoptions(precision=3, linewidth=1000)
    main()

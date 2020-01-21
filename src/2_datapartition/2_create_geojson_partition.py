"""
Generate partitions fromn GRID of TFrecords

Example invocation::

    python 2_datapartition/2_create_geojson_partition.py
        -r /home/xx/
        -p 24
        -f 0

acocac@gmail.com
"""

import geopandas as gpd
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-r','--rootdir', type=str, required=True, help='Dir')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set of the MODIS 250-m data')
parser.add_argument('-f','--fold', type=int, required=True, help='fold')

def parfiles(indir, f):
    traintiles = np.loadtxt(os.path.join(indir, 'train_fold{}.tileids'.format(f)), dtype='str')
    testtiles = np.loadtxt(os.path.join(indir, 'test_fold{}.tileids'.format(f)), dtype='str')
    evaltiles = np.loadtxt(os.path.join(indir, 'eval.tileids'), dtype='str')
    return(traintiles, testtiles, evaltiles)

if __name__ == '__main__':
    args = parser.parse_args()
    rootdir = args.rootdir
    psize = args.psize
    fold = args.fold

    tileiddir = os.path.join(rootdir, 'geodata','split', str(psize), 'final', 'tileids')

    train, test, eval = parfiles(tileiddir, fold)

    #load merge patch and fileid geojson
    df_all = gpd.read_file(os.path.join(rootdir, 'geodata','split', str(psize), 'raw', 'tileid','tileid.geojson'))

    if not 'id_fn' in df_all.columns:
        df_all['id_fn'] = df_all.patch_id.astype(str) + '_' + df_all.file_id.astype(str)

    df_all.loc[df_all['id_fn'].isin(train), 'split'] = 0
    df_all.loc[df_all['id_fn'].isin(test), 'split'] = 1
    df_all.loc[df_all['id_fn'].isin(eval), 'split'] = 2

    outdir = os.path.join(rootdir, 'geodata','split', str(psize), 'final', 'geojson')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.isfile(os.path.join(outdir,'fold{}'.format(fold),'.geojson')):
        df_all.to_file(driver = 'GeoJSON', filename= os.path.join(outdir,'fold{}'.format(fold) + '.geojson'))
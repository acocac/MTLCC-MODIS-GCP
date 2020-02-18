"""
Generate partitions fromn GRID of TFrecords

Example invocation::

    python 2_datapartition/3_assign_partition_GZfiles_train.py
        -r /home/xx/
        -y 2009
        -p 24
        -b 3
        -f 0

acocac@gmail.com
"""

import geopandas as gpd
import os
import numpy as np
import argparse
import glob
import pandas as pd
import configparser

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-r','--rootdir', type=str, required=True, help='Dir with input TFrecords.gz generated by GEE')
parser.add_argument('-y','--tyear', type=str, required=True, help='Target year')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set of the MODIS 250-m data')
parser.add_argument('-b','--blocks', type=int, required=True, help='blocks per patch')
parser.add_argument('-f','--fold', type=int, required=True, help='fold')
parser.add_argument('-ref','--reference', type=str, required=True, help='reference map')

def parfiles(indir, f):
    traintiles = np.loadtxt(os.path.join(indir, 'train_fold{}.tileids'.format(f)), dtype='str')
    testtiles = np.loadtxt(os.path.join(indir, 'test_fold{}.tileids'.format(f)), dtype='str')
    evaltiles = np.loadtxt(os.path.join(indir, 'eval.tileids'), dtype='str')

    if not isinstance(testtiles, list):
        testtiles = [str(testtiles)]

    if not isinstance(evaltiles, list):
        evaltiles = [str(evaltiles)]

    return(traintiles, testtiles, evaltiles)

if __name__ == '__main__':
    args = parser.parse_args()
    rootdir = args.rootdir
    tyear = args.tyear
    psize = args.psize
    fold = args.fold
    blocks = args.blocks
    reference = args.reference

    tileiddir = os.path.join(rootdir, 'geodata','split', str(psize), 'final', 'tileids')

    train, test, eval = parfiles(tileiddir, fold)

    #load merge patch and fileid geojson
    # df_all = gpd.read_file(os.path.join(rootdir, 'geodata','split', str(psize), 'raw', 'tileid','tileid.geojson'))

    #project dir
    projectdir = os.path.join(rootdir, 'gz', str(int(psize / blocks)), 'multiple')

    #GZ dir
    GZdir = os.path.join(projectdir, 'data' + tyear[2:])

    #list gz files
    # filesnm = [os.path.splitext(os.path.basename(x))[0] for x in sorted(glob.glob(os.path.join(GZdir,'*.gz')),key=os.path.getctime)]
    filesnm = [os.path.splitext(os.path.basename(f))[ 0 ] for f in os.listdir(GZdir) if f.endswith(".gz")]

    # np.savetxt(os.path.join(rootdir, 'gz', str(int(psize / blocks)), 'multiple','filesnm.txt'), filesnm, fmt="%s")
    #
    #create tileids
    tilesid_df = pd.DataFrame(filesnm)
    names = {0: 'file', 1: 'id_fn'}

    tilesid_df = tilesid_df.iloc[:, 0].str.split('_', 1, expand=True).rename(columns=names)
    tilesid_df['file_nm'] = tilesid_df['file'].astype(str) + '_' + tilesid_df['id_fn'].astype(str)

    outdir = os.path.join(rootdir, 'gz', str(int(psize / blocks)), 'multiple','tileids')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savetxt(os.path.join(outdir,'train_fold{}'.format(fold) + '.tileids'), np.array(tilesid_df.loc[tilesid_df['id_fn'].isin(train),'file_nm']).astype(str),fmt='%s')
    np.savetxt(os.path.join(outdir,'test_fold{}'.format(fold) + '.tileids'), np.array(tilesid_df.loc[tilesid_df['id_fn'].isin(test),'file_nm']).astype(str),fmt='%s')
    np.savetxt(os.path.join(outdir,'eval.tileids'), np.array(tilesid_df.loc[tilesid_df['id_fn'].isin(eval),'file_nm']).astype(str),fmt='%s')

    # create geotransform
    if not os.path.isfile(os.path.join(projectdir, 'geotransforms.csv')):
        geo_df = pd.DataFrame(filesnm)

        geo_df['1'] = 0
        geo_df['2'] = 250
        geo_df['3'] = 0
        geo_df['4'] = 0
        geo_df['5'] = 0
        geo_df['6'] = -250
        geo_df['7'] = 32632

        geo_df.to_csv(os.path.join(projectdir, 'geotransforms.csv'), index=None, header=None)

    # create config
    if not os.path.isfile(os.path.join(projectdir, 'dataset.ini')):
        config = configparser.ConfigParser()
        config[tyear] = {'pix250': int(psize / blocks),
                           'nbands250': '11',
                           'nbands500': '5',
                           'nobs': '46',
                           'datadir': 'data' + tyear[2:],
                           'sqlwhere': '"where date is not null and year=2001"',
                           'tiletable': 'tiles23',
                           'fieldtable': 'fields2009',
                           'level': 'L1C'}


        with open(os.path.join(projectdir, 'dataset.ini'), 'w') as configfile:
            config.write(configfile)

    if not os.path.isfile(os.path.join(projectdir, 'classes.txt')):

        if reference == 'MCD12Q1v6raw_LCType1':
            classes_pd = pd.DataFrame(np.arange(0, 18))

            classes = ['unknown',
                       'Evergreen needleleaf forest',
                       'Evergreen broadleaf forest',
                       'Deciduous needleleaf forest',
                       'Deciduous broadleaf forest',
                       'Mixed forest','Closed shrublands',
                       'Open shrublands',
                       'Woody savannas',
                       'Savannas','Grasslands',
                       'Permanent wetlands',
                       'Croplands',
                       'Urban and built-up',
                       'Cropland natural vegetation mosaic',
                       'Snow and ice',
                       'Barren or sparsely vegetated','Water']

        elif reference == 'MCD12Q1v6raw_LCProp1':
            classes_pd = pd.DataFrame(np.arange(0, 17))

            classes = [ 'unknown',
                        'Barren',
                        'Permanent Snow and Ice',
                        'Water Bodies',
                        'Evergreen Needleleaf Forests',
                        'Evergreen Broadleaf Forests',
                        'Deciduous Needleleaf Forests',
                        'Deciduous Broadleaf Forests',
                        'Mixed Broadleaf Needleleaf Forests',
                        'Mixed Broadleaf Evergreen Deciduous Forests',
                        'Open Forests',
                        'Sparse Forests',
                        'Dense Herbaceous',
                        'Sparse Herbaceous',
                        'Dense Shrublands',
                        'Shrubland Grassland Mosaics',
                        'Sparse Shrublands']

        elif reference == 'MCD12Q1v6raw_LCProp2':
            classes_pd = pd.DataFrame(np.arange(0, 12))

            classes = [ 'unknown',
                        'Barren',
                        'Permanent Snow and Ice',
                        'Water Bodies',
                        'Urban and Builtup Lands',
                        'Dense Forests',
                        'Open Forests',
                        'Forest Cropland Mosaics',
                        'Natural Herbaceous',
                        'Natural Herbaceous Croplands Mosaics',
                        'Herbaceous Croplands',
                        'Shrublands']

        elif reference == 'ESAraw':
            classes_pd = pd.DataFrame(np.arange(0, 38))

            classes = [ 'unknown',
                        'Cropland rainfed',
                        'Cropland rainfed Herbaceous cover',
                        'Cropland rainfed Tree or shrub cover',
                        'Cropland irrigated or post-flooding',
                        'Mosaic cropland gt 50 natural vegetation tree-shrub-herbaceous cover lt 50',
                        'Mosaic natural vegetation gt 50 cropland lt 50',
                        'Tree cover broadleaved evergreen closed to open gt 15',
                        'Tree cover  broadleaved  deciduous  closed to open gt 15',
                        'Tree cover  broadleaved  deciduous  closed gt 40',
                        'Tree cover  broadleaved  deciduous  open 15 to 40',
                        'Tree cover  needleleaved  evergreen  closed to open gt 15',
                        'Tree cover  needleleaved  evergreen  closed gt 40',
                        'Tree cover  needleleaved  evergreen  open 15 to 40',
                        'Tree cover  needleleaved  deciduous  closed to open gt 15',
                        'Tree cover  needleleaved  deciduous  closed gt 40',
                        'Tree cover  needleleaved  deciduous  open 15 to 40',
                        'Tree cover  mixed leaf type',
                        'Mosaic tree and shrub gt 50 herbaceous cover lt 50',
                        'Mosaic herbaceous cover gt 50 - tree and shrub lt 50',
                        'Shrubland',
                        'Shrubland evergreen',
                        'Shrubland deciduous',
                        'Grassland',
                        'Lichens and mosses',
                        'Sparse vegetation tree-shrub-herbaceous cover lt 15',
                        'Sparse tree lt 15',
                        'Sparse shrub lt 15',
                        'Sparse herbaceous cover lt 15',
                        'Tree cover flooded fresh or brakish water',
                        'Tree cover flooded saline water',
                        'Shrub or herbaceous cover flooded water',
                        'Urban areas',
                        'Bare areas',
                        'Consolidated bare areas',
                        'Unconsolidated bare areas',
                        'Water bodies',
                        'Permanent snow and ice']

        elif reference == 'Copernicusraw':
            classes_pd = pd.DataFrame(np.arange(0, 23))

            classes = [ 'unknown',
                        'Closed forest evergreen needleleaf',
                        'Closed forest deciduous needleleaf',
                        'Closed forest evergreen broadleaf',
                        'Closed forest deciduous broadleaf',
                        'Closed forest mixed',
                        'Closed forest unknown',
                        'Open forest evergreen needleleaf',
                        'Open forest deciduous needleleaf',
                        'Open forest evergreen broadleaf',
                        'Open forest deciduous broadleaf',
                        'Open forest mixed',
                        'Open forest unknown',
                        'Shrubs',
                        'Herbaceous vegetation',
                        'Herbaceous wetland',
                        'Moss and lichen',
                        'Bare - sparse vegetation',
                        'Cultivated and managed vegetation-agriculture cropland',
                        'Urban - built up',
                        'Snow and Ice',
                        'Permanent water bodies',
                        'Open sea']

        classes_pd['1'] = classes

        classes_pd.to_csv(os.path.join(projectdir, 'classes_' + reference + '.txt'), index= None, header = None, sep= "|")
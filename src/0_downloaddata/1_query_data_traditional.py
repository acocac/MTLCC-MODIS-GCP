"""
Query and export MODIS data into CSV to train shallow learners

Example invocation::

    python 0_downloaddata/1_query_data.py
        -o DIRNAME
        -r MCD12Q1v6stable01to03_LCProp2_major
        -y 2009
        -i 3000
        -f 0
        -s Gcloud
        -b gs://test

acocac@gmail.com
"""

import ee
import argparse
import os
import time

ee.Initialize()

parser = argparse.ArgumentParser(description='Query and download MODIS data from GEE')

parser.add_argument('-o','--outdir', type=str, required=True, help='Output directory')
parser.add_argument('-r','--reference', type=str, required=True, help='Dataset')
parser.add_argument('-y','--tyear', type=int, required=True, help='Target year')
parser.add_argument('-p','--partition', type=str, required=True, help='Partition', default=0)
parser.add_argument('-i','--instances', type=int, required=True, help='Number of samples', default=0)
parser.add_argument('-f','--fold', type=str, required=True, help='Fold', default=0)
parser.add_argument('-s','--storage', type=str, required=True, help='Gdrive or Gcloud', default=0)
parser.add_argument('-b','--bucket', type=str, required=False, help='bucket name', default=None)

#Filter pixels by QA
def maskQAv6(img):
    msk = img.select(['QC']).neq(0).neq(2).neq(4)
    return img.updateMask(msk).unmask(9999)

def filtermap_MODIS(collection, start, end, scheme):
    ## load lc collection
    lc_data = ee.ImageCollection(collection).filterDate(start, end)

    if collection == 'MODIS/006/MCD12Q1':

        lc_data_clean = lc_data.map(maskQAv6)

        std = lc_data_clean.select(scheme).reduce(ee.Reducer.stdDev())

        mask3 = std.eq(0)

        reference_target = ee.Image(lc_data_clean.first()).select(scheme)

        target_final = ee.Image(reference_target).updateMask(mask3).unmask(9999)

        ##reclassify no target to mask values
        target_final = target_final.where(target_final.eq(9999), 0)

    if scheme == 'LC_Prop1':
        target_final = target_final.remap([0, 1, 2, 3, 11, 12, 13, 14, 15, 16, 21, 22, 31, 32, 41, 42, 43],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    elif scheme == 'LC_Prop2':
        target_final = target_final.remap([0, 1, 2, 3, 9, 10, 20, 25, 30, 35, 36, 40],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    return target_final.unmask(0)

def unmaskvalues(img):
    return img.unmask(0)

def createsample(samples):
    long = samples.get("longitude")
    lat = samples.get("latitude");
    geom = ee.Algorithms.GeometryConstructors.Point([long, lat]);
    return samples.setGeometry(geom);

def samplingcollection(img):
    out = img.select(['array']).sampleRegions(
        collection=samples_ft,
        properties=['class'],
        scale=scale,
        projection=proj,
        tileScale=16
    )
    return out

if __name__ == '__main__':
    args = parser.parse_args()
    outdir = args.outdir
    reference = args.reference
    tyear = args.tyear
    partition = args.partition
    instances = args.instances
    storage = args.storage
    bucket = args.bucket
    fold = args.fold

    ##input data
    #land cover maps
    MCD12Q1v6 = 'MODIS/006/MCD12Q1'

    #scale
    terra = ee.Image('users/acocac/latin_decrease_2004_01_01_to_2018_05_09')

    ##Get information about projection.
    proj = terra.select(0).projection().crs().getInfo()
    scale = terra.select(0).projection().nominalScale().getInfo()

    ##AOI
    aoi = ee.FeatureCollection('users/acocacbasic/thesis/c5/fold' + fold + '_all_hpt')

    ##time period
    tS = str(tyear) + '-01-01'
    tE = str(tyear) + '-12-31'

    #### LC map ####
    if reference == 'MCD12Q1v6stable01to03_LCProp2_major':
        finalmapv6_LCProp2_ = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2003) + '-12-31', 'LC_Prop2')
        print(str(2003) + '-12-31', 'LC_Prop2')
    elif reference == 'MCD12Q1v6stable01to15_LCProp2_major':
        finalmapv6_LCProp2_ = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2015) + '-12-31', 'LC_Prop2')
        print(str(2015) + '-12-31', 'LC_Prop2')

    finalmapv6_LCProp2_reclass = finalmapv6_LCProp2_.remap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                      [0, 1, 0, 2, 3, 4, 5, 0, 6, 0, 7, 8])

    targetmap = finalmapv6_LCProp2_reclass

    #### MODIS ####
    MODIS_BANDS = ['sur_refl_b01', 'sur_refl_b02']
    DEF_BANDS = ['red', 'NIR']

    MODIS250m_coll = ee.ImageCollection('MODIS/006/MOD09Q1').filterDate(tS, tE).select(MODIS_BANDS, DEF_BANDS)

    MODIS_BANDS =['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07']
    DEF_BANDS =['blue', 'green', 'SWIR1', 'SWIR2', 'SWIR3']

    MODIS500m_coll = ee.ImageCollection('MODIS/006/MOD09A1').filterDate(tS, tE).select(MODIS_BANDS, DEF_BANDS)

    MODIS_coll = ee.ImageCollection(MODIS250m_coll.combine(MODIS500m_coll))

    MODIS_coll_filled = MODIS_coll.map(unmaskvalues)

    #### SAMPLING ####
    if partition == 'train':
        target = aoi.filter(ee.Filter.eq("split", 0))
    elif partition == 'val':
        target = aoi.filter(ee.Filter.eq("split", 1))

    #preprocess
    targetmap = targetmap.addBands(ee.Image.pixelLonLat())
    targetmap = targetmap.rename(["class", "longitude", "latitude" ]).cast({"class": "int8"})

    #process
    samples_ft = targetmap.stratifiedSample(numPoints=instances,
                                         classBand= "class",
                                         region=target,
                                         seed=4,
                                         dropNulls=True,
                                         scale=scale,
                                         projection=proj,
                                         tileScale=16)

    # Convert
    samples = samples_ft.map(createsample)

    samples = samples.filter(ee.Filter.neq("class", 0))

    #extract
    ni = MODIS_coll_filled.toArray()

    trainCollection = ni.select(['array']).sampleRegions(
        collection=samples,
        properties=['class'],
        scale=scale,
        projection=proj,
        tileScale=16
    )


    if storage == 'Gdrive':
        task = ee.batch.Export.table.toDrive(collection=trainCollection,
                                             description='fcTS_{}_ssize{}_tyear{}_{}'.format(partition,instances,tyear,reference),
                                             folder=outdir,
                                             fileNamePrefix= 'fcTS_{}_ssize{}_tyear{}_{}'.format(partition,instances,tyear,reference),
                                             fileFormat='CSV')

    else:
        outfile = outdir + '/comparison/input/fold{}'.format(fold) + '/' + 'fcTS_{}_ssize{}_tyear{}_{}'.format(partition,instances,tyear,reference)
        task = ee.batch.Export.table.toCloudStorage(
                        collection=trainCollection,
                        description='fcTS_{}_ssize{}_tyear{}_{}'.format(partition, instances, tyear, reference),
                        fileNamePrefix=outfile,
                        bucket=bucket,
                        fileFormat='CSV')

    try:
        task.start()
        if storage == 'Gdrive':
            print('Exporting in CSV format {} samples (instances) per class available for period {} to {} and storing into GDrive {} folder'.format(instances, tS, tE,outdir))
        else:
            print('Exporting in CSV format {} samples (instances) per class available for period {} to {} and storing into GCloud Bucket gs://{}/{} folder'.format(instances, tS, tE, bucket, outfile))
        print(task.status())
        time.sleep(30)

    except Exception as str_error:
        print("Error ", str_error)
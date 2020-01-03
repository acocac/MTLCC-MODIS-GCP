"""
Query and export MODIS data into TFRecord format

Example invocation::

    python 0_downloaddata/1_query_data.py
        -o DIRNAME
        -r 250m
        -y 2009
        -p 384

acocac@gmail.com
"""

import ee
import argparse

ee.Initialize()

parser = argparse.ArgumentParser(description='Query and downloand MODIS data from GEE')

parser.add_argument('-o','--outdir', type=str, required=True, help='Output directory')
parser.add_argument('-r','--resolution', type=str, required=True, help='MODIS resolution')
parser.add_argument('-y','--tyear', type=int, required=True, help='Target year')
parser.add_argument('-p','--psize', type=int, required=True, help='Patch size')
parser.add_argument('-k','--ksize', type=int, required=False, help='Kernel size', default=0)
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

def filtermap_ESA(collection, scheme):

    std = collection.select(scheme).reduce(ee.Reducer.stdDev())

    mask = std.eq(0)

    reference_target = ee.Image(collection.first()).select(scheme)

    target_final = ee.Image(reference_target).updateMask(mask).unmask(9999)

    ##reclassify no target to mask values
    target_final = target_final.where(target_final.eq(9999), 0)

    return target_final.unmask(0)

def adddata250m(img):
    img = img.addBands(srtm.rename('elevation').int()).addBands(ee.Terrain.slope(srtm).rename('slope').int()).addBands(ee.Terrain.aspect(srtm).rename('aspect').int()).addBands(bio01.rename('bio01').int()).addBands(bio12.rename('bio12').int())\
        .addBands(rawmapv6_LCType1.rename('MCD12Q1v6raw_LCType1').int()).addBands(finalmapv6_LCType1.rename('MCD12Q1v6stable_LCType1').int())\
        .addBands(rawmapv6_LCProp1.rename('MCD12Q1v6raw_LCProp1').int()).addBands(finalmapv6_LCProp1.rename('MCD12Q1v6stable_LCProp1').int())\
        .addBands(rawmapv6_LCProp2.rename('MCD12Q1v6raw_LCProp2').int()).addBands(finalmapv6_LCProp2_01to15.rename('MCD12Q1v6stable01to15_LCProp2').int()).addBands(finalmapv6_LCProp2_01to03.rename('MCD12Q1v6stable01to03_LCProp2').int()) \
        .addBands(rawmap_ESA.rename('ESAraw').int()).addBands(finalmap_ESA.rename('ESAstable').int())\
        .addBands(raw_copernicus.rename('Copernicusraw').int()) \
        .addBands(copernicus_fraction)

    return img

def adddataverification(img):
    img = img.addBands(water_mask_final.rename('watermask').int()) \
    .addBands(mapbiomas_fraction)
    return img

def adddata500m(img):
    year = ee.Image.constant(img.date().get('year').add(0))
    day = ee.Image.constant(img.date().getRelative('day', 'year').add(1))
    img = img.addBands(day.rename('DOY').int()).addBands(year.rename('year').int())
    return img

def unmaskvalues(img):
    return img.unmask(0)

def multiband2imgColl_esa(x_band):
    xband = esa_v207.select([x_band])
    reclass = xband.remap([0, 10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140, 150, 151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
    return reclass

def multiband2imgColl_mapbiomas(x_band):
    xband = mapbiomas.select([x_band])
    reclass = xband.remap([27,3,4,5,6,11,12,13,14,22,32,33,34],[0,1,2,3,4,5,6,7,8,9,10,11,12])
    return reclass

def imgColl2multiband(image, previous):
    return ee.Image(previous).addBands(ee.Image(image))

def downscale(img):
    ##Load a MOD image.
    terra = ee.Image('users/acocac/latin_decrease_2004_01_01_to_2018_05_09')

    ##Get information about the LAI projection.
    proj = terra.select(0).projection().crs().getInfo()
    scale = terra.select(0).projection().nominalScale().getInfo()

    img_downscale = img.reduceResolution(
        reducer=ee.Reducer.mode(),
        maxPixels=65535
    ).reproject(
        crs=proj,
        scale=scale,
    )

    return (img_downscale)

def reduceToFraction(img):
    terra = ee.Image('users/acocac/latin_decrease_2004_01_01_to_2018_05_09')

    ##Get information about projection.
    proj = terra.select(0).projection().crs().getInfo()
    scale = terra.select(0).projection().nominalScale().getInfo()

    res = img.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=65535
    ).reproject(
        crs=proj,
        scale=scale,
    )

    return res

def fraction_copernicus(classid):

    target_class = raw_copernicus.eq(ee.Number(classid))

    target = reduceToFraction(target_class);

    target_fc = ee.Image(target)

    return target_fc.set('class', classid)

def fraction_mapbiomas(classid):

    target_class = raw_mapbiomas.eq(ee.Number(classid))

    target = reduceToFraction(target_class);

    target_fc = ee.Image(target)

    return target_fc.set('class', classid)

def fraction_rule(classid):

    target_class = raw_copernicus.eq(ee.Number(classid))
    forest_resample = resampled_copernicus.eq(ee.Number(classid))

    target = reduceToFraction(target_class);

    target_fc = ee.Image(target.gt(0.7))
    targ_ofu = target_fc.eq(forest_resample)

    return targ_ofu.set('class', classid)

def buffer1km(feature):
    feature_out = feature.setGeometry(feature.geometry().buffer(1000))
    return feature_out

if __name__ == '__main__':
    args = parser.parse_args()
    res = args.resolution
    outdir = args.outdir
    tyear = args.tyear
    psize = args.psize
    ksize = args.ksize
    storage = args.storage
    bucket = args.bucket

    ##input data
    #land cover maps
    MCD12Q1v6 = 'MODIS/006/MCD12Q1'
    esa_v207 = ee.Image('users/acocac/esacci')
    # copernicus_v201 = ee.Image('users/acocacbasic/copernicus_AMZ')
    copernicus_dir = 'users/acocacbasic/copernicus'
    copernicus_tiles = ['W060N00','W060N20','W080N00','W080N20']
    copernicus_path = [copernicus_dir + '/' + i for i in copernicus_tiles]
    mapbiomas = ee.Image('projects/mapbiomas-raisg/public/collection1/mapbiomas_raisg_panamazonia_collection1_integration_v1')
    gsw = ee.Image('JRC/GSW1_1/GlobalSurfaceWater')

    #auxiliary
    srtm = ee.Image('CGIAR/SRTM90_V4')
    bio = ee.Image('WORLDCLIM/V1/BIO')
    bio01 = bio.select('bio01')
    bio12 = bio.select('bio12')

    #scale
    terra = ee.Image('users/acocac/latin_decrease_2004_01_01_to_2018_05_09')

    ##AOI
    if outdir == 'AMZ':
        aoi = ee.FeatureCollection('ft:1Tv0-78XXpd0qCWbrylwFWl4Pc3W1vdmnZqiEMmVs', 'geometry'); #RAISG
    elif outdir.startswith("tile_"):
        tiles = ee.FeatureCollection('users/acocacbasic/thesis/model/patchid_AMZ384_250mSpectral')
        _, patch_id, file_id = outdir.split("_")
        aoi = tiles.filter(ee.Filter.And(ee.Filter.eq("file_id", int(file_id)), ee.Filter.eq("patch_id", int(patch_id))))
    else:
        aoi = ee.FeatureCollection('users/acocacbasic/test_aois/tile_1_612')
    # aoi = ee.FeatureCollection('users/acocac/tile4')
    # aoi = ee.FeatureCollection('users/acocac/thesis/fieldcampaigns/fc1_2004_01_01_2013_04_07')

    ##time period
    tS = str(tyear) + '-01-01'
    tE = str(tyear) + '-12-31'

    #### MODIS ####
    rawmapv6_LCType1 = filtermap_MODIS(MCD12Q1v6, tS, tE, 'LC_Type1');
    #finalmapv6_LCType1 = filtermap(MCD12Q1v6, str(tyear - 1) + '-01-01', str(tyear + 1) + '-12-31', 'LC_Type1');
    finalmapv6_LCType1 = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2015) + '-12-31', 'LC_Type1');
    rawmapv6_LCProp1 = filtermap_MODIS(MCD12Q1v6, tS, tE, 'LC_Prop1');
    #finalmapv6_LCProp1 = filtermap(MCD12Q1v6, str(tyear - 1) + '-01-01', str(tyear + 1) + '-12-31', 'LC_Prop1');
    finalmapv6_LCProp1 = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2015) + '-12-31', 'LC_Prop1');
    rawmapv6_LCProp2 = filtermap_MODIS(MCD12Q1v6, tS, tE, 'LC_Prop2');
    #finalmapv6_LCProp2 = filtermap(MCD12Q1v6, str(tyear - 1) + '-01-01', str(tyear + 1) + '-12-31', 'LC_Prop2');
    finalmapv6_LCProp2_01to15 = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2015) + '-12-31', 'LC_Prop2');
    finalmapv6_LCProp2_01to03 = filtermap_MODIS(MCD12Q1v6, str(2001) + '-01-01', str(2003) + '-12-31', 'LC_Prop2');

    # rawmapv6_LCType1 = downscale(rawmapv6_LCType1)
    # finalmapv6_LCType1 = downscale(finalmapv6_LCType1)
    # rawmapv6_LCProp1 = downscale(rawmapv6_LCProp1)
    # finalmapv6_LCProp1 = downscale(finalmapv6_LCProp1)
    # rawmapv6_LCProp2 = downscale(rawmapv6_LCProp2)
    # finalmapv6_LCProp2 = downscale(finalmapv6_LCProp2)

    #### ESA ####
    esa_v207 = esa_v207.select(
        ['b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24'])

    esa_bands = esa_v207.bandNames()

    #single year
    ESA_300m = ee.ImageCollection(esa_bands.map(multiband2imgColl_esa))  # convert multiband to imgcollection

    ESA_image = ESA_300m.iterate(imgColl2multiband, ee.Image())

    ESA_image = ee.Image(ESA_image).select(
        [ 'remapped', 'remapped_1', 'remapped_2', 'remapped_3', 'remapped_4', 'remapped_5',
          'remapped_6', 'remapped_7', 'remapped_8', 'remapped_9', 'remapped_10', 'remapped_11', 'remapped_12',
          'remapped_13','remapped_14'],  ## old names
        [ '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
          '2014', '2015' ]  ## new names
    )

    if tyear <= 2015:
        rawmap_ESA = ESA_image.select(str(tyear))
    else:
        rawmap_ESA = ESA_image.select(str(2015))

    rawmap_ESA = rawmap_ESA.unmask(0)
    rawmap_ESA = downscale(rawmap_ESA)

    #multiple
    finalmap_ESA = filtermap_ESA(ESA_300m, 'remapped');
    finalmap_ESA = downscale(finalmap_ESA)

    #### Copernicus ####
    sample_copernicus = ee.Image(copernicus_path[0])
    proj = sample_copernicus.select(0).projection().crs().getInfo()
    scale = sample_copernicus.select(0).projection().nominalScale().getInfo()

    copernicus_v201 = ee.ImageCollection(copernicus_path).mosaic().reproject(crs=proj,scale=scale)

    raw_copernicus = copernicus_v201.remap([0, 111, 113, 112, 114, 115, 116, 121, 123, 122, 124, 125, 126, 20, 30, 90, 100, 60, 40, 50, 70, 80, 200],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

    raw_copernicus = raw_copernicus.unmask(0)
    resampled_copernicus = downscale(raw_copernicus)

    #Copernicus continuous
    classes = ee.List.sequence(0, 22);
    target_fc = ee.ImageCollection.fromImages(classes.map(fraction_copernicus).flatten())
    copernicus_fraction = target_fc.iterate(imgColl2multiband, ee.Image())

    #### Mapbiomas ####
    if outdir.startswith("tile_"):

        mapbiomas_bands = mapbiomas.bandNames()

        mapbiomas_30m = ee.ImageCollection(mapbiomas_bands.map(multiband2imgColl_mapbiomas))  # convert multiband to imgcollection

        mapbiomas_image = mapbiomas_30m.iterate(imgColl2multiband, ee.Image())

        mapbiomas_30m = ee.Image(mapbiomas_image).select(
            ['remapped_1', 'remapped_2', 'remapped_3', 'remapped_4', 'remapped_5',
          'remapped_6', 'remapped_7', 'remapped_8', 'remapped_9', 'remapped_10',
        'remapped_11', 'remapped_12', 'remapped_13', 'remapped_14', 'remapped_15',
          'remapped_16', 'remapped_17'],  ## old names
            ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
            '2014', '2015', '2016', '2017']  ## new names
        )

        if tyear <= 2017:
            yeart = tyear
            raw_mapbiomas = mapbiomas_30m.select(str(yeart))
        else:
            yeart = 2017
            raw_mapbiomas = mapbiomas_30m.select(str(yeart))

        # raw_mapbiomas = mapbiomas.remap([27,3,5,6,11,4,12,13,14,32,22,33,34],[0,1,1,1,1,2,2,2,2,2,3,4,4])
        # raw_mapbiomas = raw_mapbiomas.remap([27,3,4,5,6,11,12,13,14,22,32,33,34],[0,1,2,3,4,5,6,7,8,9,10,11,12])

        classes = ee.List.sequence(0, 12);
        target_fc = ee.ImageCollection.fromImages(classes.map(fraction_mapbiomas).flatten())
        mapbiomas_fraction = target_fc.iterate(imgColl2multiband, ee.Image())

        v_start = [str(yeart)]
        v_end = [str(yeart) + '_' + str(i) for i in range(1, 13, 1)]
        v_original = v_start + v_end

        v_start = ['mapbiomas']
        v_end = ['mapbiomas_' + str(i) for i in range(1, 13, 1)]
        v_rename = v_start + v_end

        mapbiomas_fraction = ee.Image(mapbiomas_fraction).select(
            v_original,  ## old names
            v_rename  ## new names
        )

        ##water mask
        scale = terra.projection().nominalScale().getInfo()

        occurrence = gsw.select('occurrence')
        water_mask = occurrence.gt(90).unmask(0)
        water_mask_resampled = downscale(water_mask)

        water_vct = water_mask_resampled.reduceToVectors(
            geometry=aoi,
            scale=scale,
            geometryType='polygon',
            eightConnected=True,
            bestEffort=True,
            maxPixels=1e9
        ).filterMetadata('label', 'equals', 1)

        water_1km_vct = water_vct.map(buffer1km)

        water_1km_raster = water_1km_vct.reduceToImage(['label'], ee.Reducer.mode())

        terrai_mask = terra.gte(0)

        water_mask_final = ee.Image(terrai_mask.updateMask(water_1km_raster).unmask(0))
        water_mask_final = water_mask_final.remap([0, 1],[1, 0])

    if res == "500m_spectral":

        # MODIS_BANDS =['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
        # DEF_BANDS =['red', 'NIR', 'blue', 'green', 'SWIR1', 'SWIR2', 'SWIR3']
        MODIS_BANDS =['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'];
        DEF_BANDS =['blue', 'green', 'SWIR1', 'SWIR2', 'SWIR3']

        MODIS_coll = ee.ImageCollection('MODIS/006/MOD09A1').filterDate(tS, tE).select(MODIS_BANDS, DEF_BANDS)

        MODIS_coll = MODIS_coll.map(adddata500m)

        scale = terra.select(0).projection().nominalScale().getInfo() * 2
        proj = terra.select(0).projection().crs().getInfo()
        print('Res: ' + str(scale))
        print('CRS: ' + str(proj))

    elif res == "250m_spectral":

        MODIS_BANDS = ['sur_refl_b01', 'sur_refl_b02'];
        DEF_BANDS = ['red', 'NIR']

        MODIS_coll = ee.ImageCollection('MODIS/006/MOD09Q1').filterDate(tS, tE).select(MODIS_BANDS, DEF_BANDS)

        # MODIS_coll = MODIS_coll.map(adddata250m)

        scale = terra.select(0).projection().nominalScale().getInfo()
        proj = terra.select(0).projection().crs().getInfo()
        print('Res: ' + str(scale))
        print('CRS: ' + str(proj))

    elif res == "250m_aux":

        MODIS_BANDS = ['sur_refl_b01'];
        DEF_BANDS = ['red']

        MODIS_coll = ee.ImageCollection('MODIS/006/MOD09Q1').first().select(MODIS_BANDS, DEF_BANDS)

        MODIS_coll = adddata250m(MODIS_coll)
        if outdir.startswith("tile_"):
            MODIS_coll = adddataverification(MODIS_coll)

        scale = terra.select(0).projection().nominalScale().getInfo()
        proj = terra.select(0).projection().crs().getInfo()
        print('Res: ' + str(scale))
        print('CRS: ' + str(proj))

    if res != "250m_aux":
        MODIS_coll = MODIS_coll.map(unmaskvalues)

        ##select target bands and convert to array
        ni = MODIS_coll.toArrayPerBand()
        depth = ee.Array(MODIS_coll.size()).repeat(0,ni.bandNames().length()).getInfo()

    else:
        # MODIS_coll = unmaskvalues(MODIS_coll)
        #
        # ni = ee.ImageCollection(MODIS_coll).toArrayPerBand()
        # depth = ee.Array(ee.ImageCollection(MODIS_coll).size()).repeat(0,ni.bandNames().length()).getInfo()

        ni = ee.ImageCollection(MODIS_coll).toArrayPerBand()
        depth = ee.Array(1).repeat(0,ni.bandNames().length()).getInfo()
        # depth = ee.Array(ee.ImageCollection(MODIS_coll).size()).repeat(0,ni.bandNames().length()).getInfo()

    if storage == 'Gdrive':
        task = ee.batch.Export.image.toDrive(image=ni,
                                             description= '{}_p{}k{}_{}'.format(outdir, psize, ksize, res),
                                             folder=outdir,
                                             region=aoi.geometry().bounds().getInfo()["coordinates"],
                                             scale=scale,
                                             crs = proj,
                                             fileFormat='TFRecord',
                                             maxPixels=1e9,
                                             formatOptions = {
                                                 'patchDimensions': [psize, psize],
                                                 'tensorDepths': depth.getInfo(),
                                                 'collapseBands': False,
                                                 'compressed': True,
                                                 'maxFileSize': 1000000000
                                             }
                                             )

    else:

        outfile = outdir + '/raw/' + res + '/p{}k{}'.format(psize, ksize) + '/' + 'data' + str(str(tyear)[2:]) + '/' + '{}_p{}k{}-'.format(outdir, psize, ksize)
        task = ee.batch.Export.image.toCloudStorage(
            image=ni,
            description='{}_p{}k{}_{}'.format(outdir, psize, ksize, res),
            fileNamePrefix=outfile,
            bucket=bucket,
            region=aoi.geometry().bounds().getInfo()["coordinates"],
            scale=scale,
            crs=proj,
            fileFormat='TFRecord',
            maxPixels=1e9,
            formatOptions={
                'patchDimensions': [psize, psize],
                'tensorDepths': depth,
                'collapseBands': False,
                'compressed': True,
                'maxFileSize': 1000000000 # 400000000 1000000000 to control number of patches per tfrecord
            }
        )

    try:
            task.start()
            if storage == 'Gdrive':
                if res != "250m_aux":
                    print('Exporting in TF format {} images available for period {} to {} and storing into GDrive {} folder'.format(MODIS_coll.size().getInfo(), tS, tE,outdir))
                else:
                    print('Exporting in TF format aux data available and storing into GDrive {} folder'.format(outdir))
            else:
                if res != "250m_aux":
                    print('Exporting in TF format {} images available for period {} to {} and storing into GCloud Bucket gs://{}/{} folder'.format(MODIS_coll.size().getInfo(), tS, tE, bucket, outfile))
                else:
                    print('Exporting in TF format aux data available and storing into GCloud Bucket gs://{}/{} folder'.format(bucket, outfile))
            print(task.status())
    except Exception as str_error:
            print("Error ", str_error)
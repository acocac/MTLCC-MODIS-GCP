from MODparser import MODparser
import tensorflow as tf
import os
import configparser
import csv
import numpy as np

class Dataset():
    """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

    def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False, experiment="bands", reference="MCD12Q1v6stable01to15_LCProp2_major", step='evaluation'):
        self.verbose = verbose

        self.augment = augment

        self.experiment = experiment
        self.reference = reference
        self.step = step

        # parser reads serialized tfrecords file and creates a feature object
        parser = MODparser()
        if self.experiment == "bands" or self.experiment == "bandswodoy": self.parsing_function = parser.parse_example_bands
        elif self.experiment == "indices": self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "bandsaux": self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "all": self.parsing_function = parser.parse_example_bandsaux
        elif self.experiment == "bandswoblue": self.parsing_function = parser.parse_example_bandswoblue
        elif self.experiment == "bands250m" or self.experiment == "evi2": self.parsing_function = parser.parse_example_bands250m

        if self.experiment == "bands" and self.reference == "Copernicusraw_fraction": self.parsing_function = parser.parse_example_bands_fraction
        if self.experiment == "bands" and self.reference != "Copernicusraw_fraction" or self.reference != "mapbiomas_fraction": self.parsing_function = parser.parse_example_bands
        if self.experiment == "bands" and self.reference == "mapbiomas_fraction": self.parsing_function = parser.parse_example_bands_fraction

        self.temp_samples = temporal_samples
        self.section = section

        # if datadir is None:
        #    dataroot=os.environ["datadir"]
        # else:
        dataroot = datadir

        # csv list of geotransforms of each tile: tileid, xmin, xres, 0, ymax, 0, -yres, srid
        # use querygeotransform.py or querygeotransforms.sh to generate csv
        # fills dictionary:
        # geotransforms[<tileid>] = (xmin, xres, 0, ymax, 0, -yres)
        # srid[<tileid>] = srid
        self.geotransforms = dict()
        # https://en.wikipedia.org/wiki/Spatial_reference_system#Identifier
        self.srids = dict()

        with open(os.path.join(dataroot, "geotransforms.csv"),'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
#ac                self.geotransforms[int(row[0])] = (
#ac                float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
#ac                self.srids[int(row[0])] = int(row[7])
                # self.geotransforms[str(row[0])] = (
                # float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
                self.geotransforms[str(row[0])] = (
                    float(row[1]), float(row[2]), int(row[3]), float(row[4]), int(row[5]), float(row[6]))
                self.srids[str(row[0])] = int(row[7])

        classes = os.path.join(dataroot,"classes_" + reference + ".txt")
        with open(classes, 'r') as f:
            classes = f.readlines()

        self.ids=list()
        self.classes=list()
        for row in classes:
            row=row.replace("\n","")
            if '|' in row:
                id,cl = row.split('|')
                self.ids.append(int(id))
                self.classes.append(cl)
        
        ## create a lookup table to map labelids to dimension ids

        # map data ids [0, 1, 2, 3, 5, 6, 8, 9, 12, 13, 15, 16, 17, 19, 22, 23, 24, 25, 26]
        labids = tf.constant(self.ids, dtype=tf.int64)

        # to dimensions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        #dimids = tf.constant(range(len(self.ids)), dtype=tf.int64)
        dimids = tf.constant(list(range(0,len(self.ids),1)),dtype=tf.int64)

        if self.step != "verification":
            self.id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(labids, dimids),
                                                default_value=-1)

            self.inverse_id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(dimids,labids),
                                                default_value=-1)

        #self.classes = [cl.replace("\n","") for cl in f.readlines()]

        cfgpath = os.path.join(dataroot, "dataset.ini")
        # load dataset configs
        datacfg = configparser.ConfigParser()
        datacfg.read(cfgpath)
        cfg = datacfg[section]

        self.tileidfolder = os.path.join(dataroot, "tileids")
        self.datadir = os.path.join(dataroot, cfg["datadir"])

        assert 'pix250' in cfg.keys()
        assert 'nobs' in cfg.keys()
        assert 'nbands250' in cfg.keys()
        assert 'nbands500' in cfg.keys()

        self.tiletable=cfg["tiletable"]

        self.nobs = int(cfg["nobs"])

        self.expected_shapes = self.calc_expected_shapes(int(cfg["pix250"]),
                                                         int(cfg["nobs"]),
                                                         int(cfg["nbands250"]),
                                                         int(cfg["nbands500"]),
                                                         )


        # expected datatypes as read from disk
        self.expected_datatypes = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)

    def calc_expected_shapes(self, pix250, nobs, bands250, bands500):
        pix250 = pix250;
        pix500 = pix250 / 2;
        x250shape = (nobs, pix250, pix250, bands250)
        x500shape = (nobs, pix500, pix500, bands500)
        doyshape = (nobs,)
        yearshape = (nobs,)
        labelshape = (nobs, pix250, pix250)

        return [x250shape, x500shape, doyshape, yearshape, labelshape]

    def transform_labels(self,feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
        """

        x250, x500, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        # labels = labels[0]
        labels = self.id_lookup_table.lookup(labels)

        return x250, x500, doy, year, labels

    def transform_labels_verification(self,feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
        """

        x250, x500, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        labels = labels[0]
        # labels = self.id_lookup_table.lookup(labels)

        return x250, x500, doy, year, labels

    def transform_labels_prediction(self,feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
        """

        x250, x500, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        labels = labels[0]
        # labels = self.id_lookup_table.lookup(labels)

        return x250, x500, doy, year, labels

    def normalize_bands250m(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        #normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #250m
        #SR
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
        norm250m = [x_normed_red, x_normed_NIR]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        #cancel effect 500m
        x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32) #wo year

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        #year = tf.cast(year, tf.float32) - 2016
        #year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32) #wo year

        return norm250m, x500, doy, year, labels

    def normalize_evi2(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #indices
        fixed_range = [[-10000, 10000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)

        norm250m = [x_normed_evi2]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        #cancel effect 500m
        x500 = tf.cast(x500, tf.float32) - tf.cast(x500, tf.float32) #wo year

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        # year = tf.cast(year, tf.float32) - 2016
        # year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, x500, doy, year, labels

    def normalize_bands(self, feature):

        x250, x500, doy, year, labels = feature
        x250 = tf.scalar_mul(1e-4, tf.cast(x250, tf.float32))
        x500 = tf.scalar_mul(1e-4, tf.cast(x500, tf.float32))

        doy = tf.cast(doy, tf.float32) / 365

        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year / cancel year

        return x250, x500, doy, year, labels

    def normalize_bands2(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        #normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #250m
        #SR
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
        norm250m = [x_normed_red, x_normed_NIR]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        #500m
        x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

        norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        #year = tf.cast(year, tf.float32) - 2016
        #year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32) #wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandswoblue(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        #normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #250m
        #SR
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
        norm250m = [x_normed_red, x_normed_NIR]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        #500m
        x_normed_green = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)

        norm500m = [x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        #year = tf.cast(year, tf.float32) - 2016
        #year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32) #wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandsaux(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        x250aux = tf.tile(x250aux, [self.nobs,1,1,1])

        #normed values
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        # SR
        # normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)

        # 250m
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

        #bio 01
        fixed_range = [[-290, 320]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio01 = normalize_fixed(x250aux[:, :, :, 0], fixed_range, normed_range)

        #bio 12
        fixed_range = [[0, 11401]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio12 = normalize_fixed(x250aux[:, :, :, 1], fixed_range, normed_range)

        #elevation
        fixed_range = [[-444, 8806]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_ele = normalize_fixed(x250aux[:, :, :, 2], fixed_range, normed_range)

        norm250m = [x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        # year = tf.cast(year, tf.float32) - 2016
        # year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_indices(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        #normed values
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        # SR
        # normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)

        # 250m
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

        #indices
        fixed_range = [[-10000, 10000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)
        x_normed_ndwi = normalize_fixed(x250[:, :, :, 3], fixed_range, normed_range)
        x_normed_ndii1 = normalize_fixed(x250[:, :, :, 4], fixed_range, normed_range)
        x_normed_ndii2 = normalize_fixed(x250[:, :, :, 5], fixed_range, normed_range)
        x_normed_ndsi = normalize_fixed(x250[:, :, :, 6], fixed_range, normed_range)

        norm250m = [x_normed_red, x_normed_NIR, x_normed_evi2, x_normed_ndwi, x_normed_ndii1, x_normed_ndii2, x_normed_ndsi]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        # year = tf.cast(year, tf.float32) - 2016
        # year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_all(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x250aux, x500, doy, year, labels = feature

        x250aux = tf.tile(x250aux, [self.nobs,1,1,1])

        # normed values
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        # SR
        # normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)

        # 250m
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)

        # 500m
        x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

        # bio 01
        fixed_range = [[-290, 320]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio01 = normalize_fixed(x250aux[:, :, :, 0], fixed_range, normed_range)

        # bio 12
        fixed_range = [[0, 11401]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_bio12 = normalize_fixed(x250aux[:, :, :, 1], fixed_range, normed_range)

        # elevation
        fixed_range = [[-444, 8806]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_ele = normalize_fixed(x250aux[:, :, :, 2], fixed_range, normed_range)

        #indices
        fixed_range = [[-10000, 10000]]
        fixed_range = np.array(fixed_range)
        normed_range = np.array(normed_range)
        x_normed_evi2 = normalize_fixed(x250[:, :, :, 2], fixed_range, normed_range)
        x_normed_ndwi = normalize_fixed(x250[:, :, :, 3], fixed_range, normed_range)
        x_normed_ndii1 = normalize_fixed(x250[:, :, :, 4], fixed_range, normed_range)
        x_normed_ndii2 = normalize_fixed(x250[:, :, :, 5], fixed_range, normed_range)
        x_normed_ndsi = normalize_fixed(x250[:, :, :, 6], fixed_range, normed_range)

        norm250m = [x_normed_red, x_normed_NIR, x_normed_bio01, x_normed_bio12, x_normed_ele, x_normed_evi2, x_normed_ndwi, x_normed_ndii1, x_normed_ndii2, x_normed_ndsi]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) / 365

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        # year = tf.cast(year, tf.float32) - 2016
        # year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32)  # wo year

        return norm250m, norm500m, doy, year, labels

    def normalize_bandswodoy(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x250, x500, doy, year, labels = feature

        #normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #250m
        #SR
        x_normed_red = normalize_fixed(x250[:, :, :, 0], fixed_range, normed_range)
        x_normed_NIR = normalize_fixed(x250[:, :, :, 1], fixed_range, normed_range)
        norm250m = [x_normed_red, x_normed_NIR]
        norm250m = tf.stack(norm250m)
        norm250m = tf.transpose(norm250m, [1, 2, 3, 0])

        #500m
        x_normed_blue = normalize_fixed(x500[:, :, :, 0], fixed_range, normed_range)
        x_normed_green = normalize_fixed(x500[:, :, :, 1], fixed_range, normed_range)
        x_normed_SWIR1 = normalize_fixed(x500[:, :, :, 2], fixed_range, normed_range)
        x_normed_SWIR2 = normalize_fixed(x500[:, :, :, 3], fixed_range, normed_range)
        x_normed_SWIR3 = normalize_fixed(x500[:, :, :, 4], fixed_range, normed_range)

        norm500m = [x_normed_blue, x_normed_green, x_normed_SWIR1, x_normed_SWIR2, x_normed_SWIR3]
        norm500m = tf.stack(norm500m)
        norm500m = tf.transpose(norm500m, [1, 2, 3, 0])

        doy = tf.cast(doy, tf.float32) - tf.cast(doy, tf.float32)

        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        #year = tf.cast(year, tf.float32) - 2016
        #year = tf.cast(year, tf.float32) - 2002 #including year
        year = tf.cast(year, tf.float32) - tf.cast(year, tf.float32) #wo year

        return norm250m, norm500m, doy, year, labels

    def augment(self, feature):

        x250, x500, doy, year, labels = feature

        ## Flip UD

        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[1]), lambda: x250)
        x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[1]), lambda: x500)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[1]), lambda: labels)


        ## Flip LR

        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x250 = tf.cond(condition, lambda: tf.reverse(x250, axis=[2]), lambda: x250)
        x500 = tf.cond(condition, lambda: tf.reverse(x500, axis=[2]), lambda: x500)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[2]), lambda: labels)

        return x250, x500, doy, year, labels

    def temporal_sample(self, feature):
        """ randomy choose <self.temp_samples> elements from temporal sequence """

        n = self.temp_samples

        # skip if not specified
        if n is None:
            return feature

        x250, x500, doy, year, labels = feature

        # data format 1, 2, 1, 2, -1,-1,-1
        # sequence lengths indexes are negative values.
        # sequence_lengths = tf.reduce_sum(tf.cast(x250[:, :, 0, 0, 0] > 0, tf.int32), axis=1)

        # tf.sequence_mask(sequence_lengths, n_obs)

        # max_obs = tf.shape(x250)[1]
        max_obs = self.nobs

        shuffled_range = tf.random_shuffle(tf.range(max_obs))[0:n]

        idxs = -tf.nn.top_k(-shuffled_range, k=n).values

        x250 = tf.gather(x250, idxs)
        x500 = tf.gather(x500, idxs)
        doy = tf.gather(doy, idxs)
        year = tf.gather(year, idxs)

        return x250, x500, doy, year, labels

    # MODIS bands stored in x250 and x500
    # x250[:,:,:,0] = 'red'
    # x250[:,:,:,1] = 'NIR'
    # x500[:,:,:,0] = 'blue'
    # x500[:,:,:,1] = 'green'
    # x500[:,:,:,2] = 'SWIR1'
    # x500[:,:,:,3] = 'SWIR2'
    # x500[:,:,:,4] = 'SWIR3'

    # auxiliary data stored in x250
    # x250[:,:,:,2] = 'bio1'
    # x250[:,:,:,3] = 'bio2'
    # x500[:,:,:,4] = 'elevation'
    # x500[:,:,:,5] = 'slope'
    # x500[:,:,:,6] = 'aspect'

    def addIndices(self, features):

        def NDVI(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
            nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDWI(a, b):  # 10000*(double)(nir[ii]-swir1[ii]) / (double)(nir[ii]+swir1[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDSI(a, b):  #10000*(double)(green[ii]-swir2[ii]) / (double)(green[ii]+swir2[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDII1(a, b):  # 10000*(double)(nir[ii]-swir2[ii]) / (double)(nir[ii]+swir2[ii])
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def NDII2(a, b):  # 10000*(double)(nir[ii]-swir3[ii]) / (double)(nir[ii]+swir3[ii]);
            nd = 10000 * ((a - b) / (a + b))
            nd_inf = 10000 * ((a - b) / (a + b + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        def resize(tensor, new_height, new_width):
            t = tf.shape(tensor)[0]
            h = tf.shape(tensor)[1]
            w = tf.shape(tensor)[2]
            d = tf.shape(tensor)[3]

            # stack batch on times to fit 4D requirement of resize_tensor
            stacked_tensor = tf.reshape(tensor, [t, h, w, d])
            reshaped_stacked_tensor = tf.image.resize_images(stacked_tensor, size=(new_height, new_width))
            return tf.reshape(reshaped_stacked_tensor, [t, new_height, new_width, d])

        x250, x250aux, x500, doy, year, labels = features

        px = tf.shape(x250)[2]

        x250 = tf.cast(x250, tf.float32)
        x500 = tf.cast(x500, tf.float32)

        x500_r = tf.identity(resize(x500, px, px))

        # ndvi = NDVI(x250[:, :, :, 1], x250[:, :, :, 0])
        evi2 = EVI2(x250[:, :, :, 1], x250[:, :, :, 0])
        ndwi = NDWI(x250[:, :, :, 1], x500_r[:, :, :, 2])
        ndii1 = NDII1(x250[:, :, :, 1], x500_r[:, :, :, 3])
        ndii2 = NDII2(x250[:, :, :, 1], x500_r[:, :, :, 4])
        ndsi = NDII2(x500_r[:, :, :, 1], x500_r[:, :, :, 3])

        # indices250m = [evi2]
        indices250m = [evi2, ndwi, ndii1, ndii2, ndsi]

        x250indices = tf.stack(indices250m)
        x250indices = tf.transpose(x250indices, [1, 2, 3, 0])

        x250m = tf.concat([x250, x250indices], axis=3)

        return x250m, x250aux, x500, doy, year, labels

    def addIndices250m(self, features):

        def EVI2(a, b):  # 10000*2.5*(nir[ii] - red[ii])/(nir[ii] + (2.4*red[ii]) + 10000);
            nd = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000))
            nd_inf = 10000 * 2.5 * ((a - b) / (a + (2.4 * b) + 10000 + 0.000001))
            return tf.where(tf.is_finite(nd), nd, nd_inf)

        x250, x500, doy, year, labels = features

        x250 = tf.cast(x250, tf.float32)

        evi2 = EVI2(x250[:, :, :, 1], x250[:, :, :, 0])

        # indices250m = [evi2]
        indices250m = [evi2]

        x250indices = tf.stack(indices250m)
        x250indices = tf.transpose(x250indices, [1, 2, 3, 0])

        x250m = tf.concat([x250, x250indices], axis=3)

        return x250m, x500, doy, year, labels

    def MCD12Q1v6raw_LCType1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 0 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable_LCType1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 1 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 2 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable_LCProp1(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 2 ]

        return x250, x500, doy, year, labels

    # def MCD12Q1v6raw_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 4 ]
    #
    #   return x250, x500, doy, year, labels
    #
    # def MCD12Q1v6stable01to15_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 5 ]
    #
    #   return x250, x500, doy, year, labels
    #
    # def MCD12Q1v6stable01to03_LCProp2(self, feature):
    #
    #   x250, x500, doy, year, labels = feature
    #
    #   labels = labels[ :, :, :, 6]
    #
    #   return x250, x500, doy, year, labels

    def ESAraw(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 7 ]

        return x250, x500, doy, year, labels

    def ESAstable(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 8 ]

        return x250, x500, doy, year, labels

    def Copernicusraw(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 9 ]

        return x250, x500, doy, year, labels

    def Copernicusraw_fraction(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.argmax(labels, axis=3)

        return x250, x500, doy, year, labels

    def watermask(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 10 ]

        return x250, x500, doy, year, labels

    def Copernicusnew_cf2others(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 11 ]

        return x250, x500, doy, year, labels

    def merge_datasets2own(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 12 ]

        return x250, x500, doy, year, labels

    def merge_datasets2HuHu(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 13 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal2maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 14 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal3maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 15 ]

        return x250, x500, doy, year, labels

    def merge_datasets2Tsendbazaretal3maps(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 15 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 16 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable01to15_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 17 ]

        return x250, x500, doy, year, labels

    def MCD12Q1v6stable01to03_LCProp2_major(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 18 ]

        return x250, x500, doy, year, labels

    def mapbiomas_fraction(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.argmax(labels, axis=3)

        return x250, x500, doy, year, labels


    def tilestatic(self, feature):

        x250, x500, doy, year, labels = feature

        labels = tf.tile(labels, [self.nobs,1,1,1])

        return x250, x500, doy, year, labels

    def get_ids(self, partition, fold=0):

        def readids(path):
            with open(path, 'r') as f:
                lines = f.readlines()
##ac            return [int(l.replace("\n", "")) for l in lines]
            return [str(l.replace("\n", "")) for l in lines]

        traintest = "{partition}_fold{fold}.tileids"
        eval = "{partition}.tileids"

        if partition == 'train':
            # e.g. train240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'test':
            # e.g. test240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'eval' or partition == 'pred' or partition == 'viz':
            # e.g. eval240.tileids
            path = os.path.join(self.tileidfolder, eval.format(partition=partition))
            return readids(path)
        else:
            raise ValueError("please provide valid partition (train|test|eval)")

    def create_tf_dataset(self, partition, fold, batchsize, shuffle, prefetch_batches=None, num_batches=-1, threads=8,
                          drop_remainder=False, overwrite_ids=None):

        # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
        allids = self.get_ids(partition=partition, fold=fold)

        # set of ids present in local folder (e.g. 1.tfrecord)
        tiles = os.listdir(self.datadir)

        if tiles[0].endswith(".gz"):
            compression = "GZIP"
#            ext = ".tfrecord.gz"
            ext = ".gz"
        else:
            compression = ""
            ext = ".tfrecord"

##ac        downloaded_ids = [int(t.replace(".gz", "").replace(".tfrecord", "")) for t in tiles]
        downloaded_ids = [str(t.replace(".gz", "").replace(".tfrecord", "")) for t in tiles]

        # intersection of available ids and partition ods
        if overwrite_ids is None:
            ids = list(set(downloaded_ids).intersection(allids))
        else:
            print ("overwriting data ids! due to manual input")
            ids = overwrite_ids

        filenames = [os.path.join(self.datadir, str(id) + ext) for id in ids]

        if self.verbose:
            print ("dataset: {}, partition: {}, fold:{} {}/{} tiles downloaded ({:.2f} %)".format(self.section, partition, fold, len(ids), len(allids),
                                                                               len(ids) / float(len(allids)) * 100))

        def mapping_function(serialized_feature):
            # read data from .tfrecords
            feature = self.parsing_function(serialized_example=serialized_feature)
            # sample n times out of the timeseries
            feature = self.temporal_sample(feature)
            #indices
            if self.experiment == "indices" or self.experiment == "all": feature = self.addIndices(feature)
            if self.experiment == "evi2": feature = self.addIndices250m(feature)
            # perform data normalization [0,1000] -> [0,1]
            if self.experiment == "bands250m": feature = self.normalize_bands250m(feature)
            if self.experiment == "bands": feature = self.normalize_bands(feature)
            if self.experiment == "bandswoblue": feature = self.normalize_bandswoblue(feature)
            if self.experiment == "bandsaux": feature = self.normalize_bandsaux(feature)
            if self.experiment == "indices": feature = self.normalize_indices(feature)
            if self.experiment == "all": feature = self.normalize_all(feature)
            if self.experiment == "evi2": feature = self.normalize_evi2(feature)
            if self.experiment == "bandswodoy": feature = self.normalize_bandswodoy(feature)

            feature = self.tilestatic(feature)

            if self.step == "training" or self.step == "evaluation" or self.step == "verification":
                if self.reference == "MCD12Q1v6raw_LCType1": feature = self.MCD12Q1v6raw_LCType1(feature)
                if self.reference == "MCD12Q1v6raw_LCProp1": feature = self.MCD12Q1v6raw_LCProp1(feature)
                if self.reference == "MCD12Q1v6raw_LCProp2": feature = self.MCD12Q1v6raw_LCProp2(feature)
                if self.reference == "MCD12Q1v6raw_LCProp2_major": feature = self.MCD12Q1v6raw_LCProp2_major(feature)
                if self.reference == "MCD12Q1v6stable_LCType1": feature = self.MCD12Q1v6stable_LCType1(feature)
                if self.reference == "MCD12Q1v6stable_LCProp1": feature = self.MCD12Q1v6stable_LCProp1(feature)
                if self.reference == "MCD12Q1v6stable01to15_LCProp2": feature = self.MCD12Q1v6stable01to15_LCProp2(feature)
                if self.reference == "MCD12Q1v6stable01to03_LCProp2": feature = self.MCD12Q1v6stable01to03_LCProp2(feature)
                if self.reference == "MCD12Q1v6stable01to15_LCProp2_major": feature = self.MCD12Q1v6stable01to15_LCProp2_major(
                    feature)
                if self.reference == "MCD12Q1v6stable01to03_LCProp2_major": feature = self.MCD12Q1v6stable01to03_LCProp2_major(
                    feature)
                if self.reference == "ESAraw": feature = self.ESAraw(feature)
                if self.reference == "ESAstable": feature = self.ESAstable(feature)
                if self.reference == "Copernicusraw": feature = self.Copernicusraw(feature)
                if self.reference == "Copernicusraw_fraction": feature = self.Copernicusraw_fraction(feature)
                if self.reference == "Copernicusnew_cf2others": feature = self.Copernicusnew_cf2others(feature)
                if self.reference == "merge_datasets2own": feature = self.merge_datasets2own(feature)
                if self.reference == "merge_datasets2HuHu": feature = self.merge_datasets2HuHu(feature)
                if self.reference == "merge_datasets2Tsendbazaretal2maps": feature = self.merge_datasets2Tsendbazaretal2maps(
                    feature)
                if self.reference == "merge_datasets2Tsendbazaretal3maps": feature = self.merge_datasets2Tsendbazaretal3maps(
                    feature)
                if self.reference == "mapbiomas_fraction": feature = self.mapbiomas_fraction(feature)
                if self.reference == "watermask": feature = self.watermask(feature)

                print('Step {} using reference {}'.format(self.step, self.reference))

            elif self.step == "prediction":
                feature = self.MCD12Q1v6raw_LCType1(feature)
                print('Step {} using reference {}'.format(self.step, 'dummy dataset'))

            # perform data augmentation
            if self.augment: feature = self.augment(feature)
            # flatten for tempCNN
            # if self.tempCNN: feature = self.
            # replace potentially non sequential labelids with sequential dimension ids
            if self.step == "training": feature = self.transform_labels(feature)
            if self.step == "verification": feature = self.transform_labels_verification(feature)
            # if self.step == "prediction": feature = self.transform_labels_prediction(feature)

            return feature

        if num_batches > 0:
            filenames = filenames[0:num_batches * batchsize]

        # shuffle sequence of filenames
        if partition == 'train':
            filenames = tf.random_shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, compression_type=compression)

        dataset = dataset.map(mapping_function, num_parallel_calls=threads)

        # repeat forever until externally stopped
        dataset = dataset.repeat()

        # Don't trust buffer size -> manual shuffle beforehand
        # if shuffle:
        #    dataset = dataset.shuffle(buffer_size=int(min_after_dequeue))

        if drop_remainder:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(int(batchsize)))
        else:
            dataset = dataset.batch(int(batchsize))

        if prefetch_batches is not None:
            dataset = dataset.prefetch(prefetch_batches)

        # assign output_shape to dataset
        # modelshapes are expected shapes of the data stacked as batch
        output_shape = []
        for shape in self.expected_shapes:
            output_shape.append(tf.TensorShape((batchsize,) + shape))

        return dataset, output_shape, self.expected_datatypes, filenames


def main():
    dataset = Dataset(datadir="/media/data/marc/tfrecords/fields/L1C/480", verbose=True, temporal_samples=30,section="2016")

    training_dataset, output_shapes, output_datatypes, fm_train = dataset.create_tf_dataset("train", 0, 1, 5, True, 32)

    iterator = training_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run([iterator.initializer, tf.tables_initializer()])
        x250, x500, doy, year, labels = sess.run(iterator.get_next())
        print(x250.shape)

if __name__ == "__main__":
    main()

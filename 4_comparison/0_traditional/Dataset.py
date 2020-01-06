from MODparser import MODparser
import tensorflow as tf
import os
import configparser
import csv
import numpy as np

class Dataset():
    """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

    def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False, experiment="all", reference="MCD12Q1v6_cleaned", num_classes=17):
        self.verbose = verbose

        self.augment = augment

        self.experiment = experiment
        self.reference = reference
        self.num_classes = num_classes

        self.temp_samples = temporal_samples
        self.section = section

        parser = MODparser()
        if experiment == "eval": self.parsing_function = parser.parse_example_eval
        if experiment == "train": self.parsing_function = parser.parse_example_train

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

    def transform_labels_eval(self,feature):
        x, doy, labels = feature

        return x, doy, labels

    def transform_labels_training(self,feature):
        x, labels = feature

        labels = tf.one_hot(labels, self.num_classes)

        return x, labels

    def normalize_bands(self, feature):

        def normalize_fixed(x, current_range, normed_range):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
            x_normed = (tf.cast(x, tf.float32) - tf.cast(current_min, tf.float32)) / (
                    tf.cast(current_max, tf.float32) - tf.cast(current_min, tf.float32))
            x_normed = x_normed * (tf.cast(normed_max, tf.float32) - tf.cast(normed_min, tf.float32)) + tf.cast(
                normed_min,
                tf.float32)
            return x_normed

        x, labels = feature

        #normal minx/max domain
        fixed_range = [[-100, 16000]]
        fixed_range = np.array(fixed_range)
        normed_range = [[0, 1]]
        normed_range = np.array(normed_range)

        #SR
        x_normed = normalize_fixed(x, fixed_range, normed_range)

        return x_normed, labels

    def tempCNN_eval(self, features):
        def resize(tensor, new_height, new_width):
            t = tf.shape(tensor)[0]
            h = tf.shape(tensor)[1]
            w = tf.shape(tensor)[2]
            d = tf.shape(tensor)[3]

            # stack batch on times to fit 4D requirement of resize_tensor
            stacked_tensor = tf.reshape(tensor, [t, h, w, d])
            reshaped_stacked_tensor = tf.image.resize_images(stacked_tensor, size=(new_height, new_width))
            return tf.reshape(reshaped_stacked_tensor, [t, new_height, new_width, d])

        x250, x500, doy, year, labels = features

        px = tf.shape(x250)[2]

        x250 = tf.cast(x250, tf.float32)
        x500 = tf.cast(x500, tf.float32)

        x500_r = tf.identity(resize(x500, px, px))

        x = tf.concat([x250, x500_r], axis=3)

        x = tf.transpose(x, [1, 2, 0, 3])

        x = tf.reshape(x, (tf.shape(x)[1] * tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]))  # -1 means "all"

        labels = labels[0, :, :]

        return x, doy, labels

    def MCD12Q1v6raw_LCType1_train(self, feature):
            x, labels = feature

            labels = labels[0]

            return x, labels

    def MCD12Q1v6raw_LCProp1_train(self, feature):
            x, labels = feature

            labels = labels[2]

            return x, labels

    def MCD12Q1v6raw_LCProp2_train(self, feature):
            x, labels = feature

            labels = labels[4]

            return x, labels

    def ESAraw_train(self, feature):
            x, labels = feature

            labels = labels[6]

            return x, labels

    def Copernicusraw_train(self, feature):
            x, labels = feature

            labels = labels[7]

            return x, labels

    def Copernicusnew_all_train(self, feature):
            x, labels = feature

            labels = labels[8]

            return x, labels

    def Copernicusnew_cebf_train(self, feature):
            x, labels = feature

            labels = labels[9]

            return x, labels

    def MCD12Q1v6raw_LCType1_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[:,:,:,0]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp1_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 2]

        return x250, x500, doy, year, labels

    def MCD12Q1v6raw_LCProp2_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 4]

        return x250, x500, doy, year, labels

    def ESAraw_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 6]

        return x250, x500, doy, year, labels

    def Copernicusraw_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 7]

        return x250, x500, doy, year, labels

    def Copernicusnew_all_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 8]

        return x250, x500, doy, year, labels

    def Copernicusnew_cebf_eval(self, feature):

        x250, x500, doy, year, labels = feature

        labels = labels[ :, :, :, 9]

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
        elif partition == 'eval':
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

            if self.experiment == "eval" and self.reference == "MCD12Q1v6raw_LCType1": feature = self.MCD12Q1v6raw_LCType1_eval(feature)
            if self.experiment == "eval" and self.reference == "MCD12Q1v6raw_LCProp1": feature = self.MCD12Q1v6raw_LCProp1_eval(feature)
            if self.experiment == "eval" and self.reference == "MCD12Q1v6raw_LCProp2": feature = self.MCD12Q1v6raw_LCProp2_eval(feature)
            if self.experiment == "eval" and self.reference == "ESAraw": feature = self.ESAraw_eval(feature)
            if self.experiment == "eval" and self.reference == "Copernicusraw": feature = self.Copernicusraw_eval(feature)
            if self.experiment == "eval" and self.reference == "Copernicusnew_all": feature = self.Copernicusnew_all_eval(feature)
            if self.experiment == "eval" and self.reference == "Copernicusnew_cebf": feature = self.Copernicusnew_cebf_eval(feature)
            if partition == "eval": feature = self.tempCNN_eval(feature)

            if partition == "eval": feature = self.transform_labels_eval(feature)
            return feature

        if num_batches > 0:
            filenames = filenames[0:num_batches * batchsize]

        # shuffle sequence of filenames
        if shuffle:
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

        return dataset, filenames

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


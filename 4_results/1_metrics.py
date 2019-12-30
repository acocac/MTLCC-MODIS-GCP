import numpy as np
import os
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score

import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-i','--indir', type=str, required=True, help='Indir dir with conf folder')
parser.add_argument('-o','--outdir', type=str, required=True, help='Outdir dir')
parser.add_argument('-d','--dataset', type=str, required=True, help='Dataset')
parser.add_argument('-e','--experiment', type=str, required=True, help='Experiment')
parser.add_argument('-y','--targetyear', type=int, required=True, help='Target year')
parser.add_argument('-m','--model', type=str, required=True, help='Model')
parser.add_argument('-ep','--epochs', type=int, required=True, help='Epochs')
parser.add_argument('-f','--folds', type=list, required=True, help='Folds')
parser.add_argument('-c','--ckpt', type=int, required=True, help='Ckpt')
parser.add_argument('-l','--level', type=str, required=True, help='Level')

class classes:
    classes_MCD12Q1v6LCType1 = [ 'NoData','Evergreen needleleaf forest', 'Evergreen broadleaf forest',
                                 'Deciduous needleleaf forest', 'Deciduous broadleaf forest',
                                 'Mixed forest', 'Closed shrublands', 'Open shrublands',
                                 'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                                 'Croplands', 'Urban and built-up', 'Cropland natural vegetation mosaic',
                                 'Snow and ice', 'Barren or sparsely vegetated', 'Water' ]

    shortname_MCD12Q1v6LCType1 = ['NoData', 'ENF', 'EBF',
                                 'DNF', 'DBF',
                                 'MF', 'CS', 'OS',
                                 'WS', 'S', 'G', 'PW',
                                 'C', 'Bu', 'CN',
                                 'SI', 'Ba', 'W']

    colors_MCD12Q1v6LCType1 = [ '#ababab', '#05450a', '#086a10', '#54a708',
                                '#78d203', '#009900', '#c6b044',
                                '#dcd159', '#dade48', '#fbff13',
                                '#b6ff05', '#27ff87', '#c24f44',
                                '#fa0000', '#ff6d4c', '#69fff8',
                                '#f9ffa4', '#1c0dff' ]

    classes_MCD12Q1v6LCProp1 = [ 'NoData','Barren',
                                 'Permanent Snow and Ice',
                                 'Water Bodies',
                                 'Evergreen Needleleaf Forests',
                                 'Evergreen Broadleaf Forests',
                                 'Deciduous Needleleaf Forests',
                                 'Deciduous Broadleaf Forests',
                                 'Mixed Broadleaf-Needleleaf Forests',
                                 'Mixed Broadleaf Evergreen-Deciduous Forests',
                                 'Open Forests',
                                 'Sparse Forests',
                                 'Dense Herbaceous',
                                 'Shrubs',
                                 'Sparse Herbaceous',
                                 'Dense Shrublands',
                                 'Shrubland-Grassland Mosaics',
                                 'Sparse Shrublands' ]

    colors_MCD12Q1v6LCProp1 = [ '#ababab', '#f9ffa4', '#69fff8', '#1c0dff',
                                '#05450a', '#086a10', '#54a708',
                                '#78d203', '#005a00', '#009900',
                                # '#006c00','#00d000','#b6ff05', #old
                                '#52b352', '#00d000', '#b6ff05',
                                '#98d604', '#dcd159', '#f1fb58',
                                '#fbee65' ]

    classes_MCD12Q1v6LCProp2 = ['NoData',
        'Barren',
        'Permanent Snow and Ice',
        'Water Bodies',
        'Urban and Built-up Lands',
        'Dense Forests',
        'Open Forests',
        'Forest/Cropland Mosaics',
        'Natural Herbaceous',
        'Natural Herbaceous-Croplands Mosaics',
        'Herbaceous Croplands',
        'Shrublands' ]

    shortname_MCD12Q1v6LCProp2 = ['NoData', 'Ba', 'SI',
                                'W', 'Bu',
                                'DF', 'OF', 'FCM',
                                'NH', 'NHCM', 'C',
                                'S']

    colors_MCD12Q1v6LCProp2 = [ '#ababab', '#f9ffa4', '#69fff8', '#1c0dff',
                                '#fa0000', '#003f00', '#006c00',
                                '#e3ff77', '#b6ff05', '#93ce04',
                                '#77a703', '#dcd159' ]

    classes_esa = [ 'NoData','Cropland rainfed',
                    'Cropland rainfed Herbaceous cover',
                    'Cropland rainfed Tree or shrub cover',
                    'Cropland irrigated or post-flooding',
                    'Mosaic cropland gt 50 natural vegetation (tree/shrub/herbaceous cover) lt 50',
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
                    'Mosaic herbaceous cover gt 50 / tree and shrub lt 50',
                    'Shrubland',
                    'Shrubland evergreen',
                    'Shrubland deciduous',
                    'Grassland',
                    'Lichens and mosses',
                    'Sparse vegetation (tree/shrub/herbaceous cover) lt 15',
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
                    'Permanent snow and ice' ]

    colors_esa = [ '#ababab', '#ffff64', '#ffff64', '#ffff00',
                   '#aaf0f0', '#dcf064', '#c8c864',
                   '#006400', '#00a000', '#00a000',
                   '#aac800', '#003c00', '#003c00',
                   '#005000', '#285000', '#285000',
                   '#286400', '#788200', '#8ca000',
                   '#be9600', '#966400', '#966400',
                   '#be9600', '#ffb432', '#ffdcd2',
                   '#ffebaf', '#ffc864', '#ffd278',
                   '#ffebaf', '#00785a', '#009678',
                   '#00dc82', '#c31400', '#fff5d7',
                   '#dcdcdc', '#fff5d7', '#0046c8',
                   '#ffffff' ]

    classes_copernicus = [ 'NoData','Closed forest evergreen needleleaf',
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
                           'Open sea' ]

    colors_copernicus = [ '#ababab', '#58481f',
                          '#70663e',
                          '#009900',
                          '#00cc00',
                          '#4e751f',
                          '#007800',
                          '#666000',
                          '#8d7400',
                          '#8db400',
                          '#a0dc00',
                          '#929900',
                          '#648c00',
                          '#ffbb22',
                          '#ffff4c',
                          '#0096a0',
                          '#fae6a0',
                          '#b4b4b4',
                          '#f096ff',
                          '#fa0000',
                          '#f0f0f0',
                          '#0032c8',
                          '#000080' ]

    classes_copernicus_cf2others = [ 'NoData','Closed forest',
                                     'Open forest',
                                     'Shrubs',
                                     'Herbaceous vegetation',
                                     'Bare / sparse vegetation',
                                     'Urban / built up',
                                     'Cropland)',
                                     'Water bodies',
                                     'Herbaceous wetland' ]

    shortname_copernicus_cf2others = ['NoData', 'DF', 'OF',
                                 'S', 'HV',
                                 'Ba', 'Bu', 'C',
                                 'W', 'HW']

    colors_copernicus_cf2others = ['#ababab', '#086a10',
                                    '#54a708',
                                    '#ffbb22',
                                    '#ffff4c',
                                    '#b4b4b4',
                                    '#fa0000',
                                    '#f096ff',
                                    '#0032c8',
                                    '#0096a0']

    classes_merge_datasets2own = ['NoData',
                        'Barren',
                        'Water Bodies',
                        'Urban and Built-up Lands',
                        'Dense Forests',
                        'Open Forests',
                        'Natural Herbaceous',
                        'Croplands',
                        'Shrublands']

    shortname_merge_datasets2own = ['NoData', 'Ba',
                                 'W', 'Bu',
                                 'DF', 'OF',
                                 'NH', 'C',
                                 'S']

    colors_merge_datasets2own = ['#ababab','#f9ffa4','#1c0dff',
                               '#fa0000','#003f00',
                               '#006c00','#b6ff05',
                               '#77a703','#dcd159']

    classes_mapbiomas = ['NoData','Forest Formation', 'Savanna Formation',
                                 'Mangrove', 'Flooded forest',
                                 'Wetland', 'Grassland', 'Other non forest natural formation',
                                 'Farming', 'Non vegetated area', 'Salt flat', 'River, Lake and Ocean',
                                 'Glacier']

    colors_mapbiomas = [ '#ababab','#009820','#00FE2D','#68743A','#74A5AF','#3CC2A6','#B9AE53','#F3C13C','#FFFEB5','#EC9999','#FD7127','#001DFC','#FFFFFF']

def helpCalcKappa(ctable):
    total = np.sum(ctable)
    ctable = ctable / total
    categories = len(ctable)

    # fraction of agreement
    pagrm = 0
    for i in range(0, categories):
        pagrm = pagrm + ctable[ i, i ]

    # expected fraction of agreement subject to the observed distribution
    pexpct = 0
    for i in range(0, categories):
        pexpct = pexpct + np.sum(ctable[ i, : ]) * np.sum(ctable[ :, i ])

    # maximum  fraction  of  agreement  subject  to  the  observed  distribution
    pmax = 0
    for i in range(0, categories):
        pmax = pmax + np.min([ np.sum(ctable[ i, : ]), np.sum(ctable[ :, i ]) ])

        # kappa Index:
    kappa = (pagrm - pexpct) / (1 - pexpct)

    # kappa of location:
    kappa_loc = (pagrm - pexpct) / (pmax - pexpct)

    # kappa of histogram:
    kappa_hist = (pmax - pexpct) / (1 - pexpct)

    # chance agreement:
    chance_agrm = 100 * min((1 / categories), pagrm, pexpct)

    # quantity agreement:
    if min((1 / categories), pexpct, pagrm) == (1 / categories):
        quant_agrm = 100 * min((pexpct - 1 / categories), pagrm - 1 / categories)
    else:
        quant_agrm = 0

        # quantity disagreement:
    quant_dagrm = 100 * (1 - pmax)

    # allocation agreement:
    all_agrm = 100 * max(pagrm - pexpct, 0)

    # allocation disagreement:
    all_dagrm = 100 * (pmax - pagrm)

    kappa_comp = kappa, kappa_loc, kappa_hist, chance_agrm, quant_agrm, quant_dagrm, all_agrm, all_dagrm

    return kappa_comp

def mr_metrics(confusion_matrix, level):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    if level == 'global':
        return overall_accuracy, kappa, np.mean(precision), np.mean(recall), np.mean(f1)
    elif level == 'perclass':
        # Per class accuracy
        cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)

        return overall_accuracy, kappa, precision, recall, f1, cl_acc

class ClassMetric(object):
    def __init__(self, num_classes=2, ignore_index=0):
        self.num_classes = num_classes
        _range = -0.5, num_classes - 0.5
        self.range = np.array((_range, _range), dtype=np.int64)
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

        self.store = dict()

        self.earliness_record = list()

    def _update(self, o, t):
        t = t.flatten()
        o = o.flatten()
        # confusion matrix
        n, _, _ = np.histogram2d(t, o, bins=self.num_classes, range=self.range)
        self.hist += n

    def add(self, stats):
        for key, value in stats.items():

            value = value.data.cpu().numpy()

            if key in self.store.keys():
                self.store[key].append(value)
            else:
                self.store[key] = list([value])

        return dict((k, np.stack(v).mean()) for k, v in self.store.items())

    def update_confmat(self, target, output):
        self._update(output, target)
        return self.accuracy()

    def update_earliness(self,earliness):
        self.earliness_record.append(earliness)
        return np.hstack(self.earliness_record).mean()

    def accuracy(self):
        """
        https: // en.wikipedia.org / wiki / Confusion_matrix
        Calculates over all accuracy and per class classification metrics from confusion matrix
        :param confusion_matrix numpy array [n_classes, n_classes] rows True Classes, columns predicted classes:
        :return overall accuracy
                and per class metrics as list[n_classes]:
        """
        confusion_matrix = self.hist

        if type(confusion_matrix) == list:
            confusion_matrix = np.array(confusion_matrix)

        if level == 'global':
            overall_accuracy, kappa, precision, recall, f1 = mr_metrics(confusion_matrix, level)
            return (overall_accuracy, kappa, precision, recall, f1)

        elif level == 'perclass':
            overall_accuracy, kappa, precision, recall, f1, cl_acc = mr_metrics(confusion_matrix, level)

            return dict(
                overall_accuracy=overall_accuracy,
                kappa=kappa,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=cl_acc
            )

def confusionmatrix2table(cm, ids=None, classnames=None, outfile=None):
    overall_accuracy, kappa, precision, recall, f1, cl_acc = mr_metrics(cm, 'perclass')
    kn, k_loc, k_hist, c_agrm, q_agrm, q_dagrm, a_agrm, a_dagrm = helpCalcKappa(confusion_matrix(y_true2, y_pred2))

    support = cm.sum(1) # 0 -> prediction 1 -> ground truth

    if classnames is None:
        classnames = np.array(["{}".format(i) for i in range(confusion_matrix.shape[0])])

    df = pd.DataFrame([ids, classnames, list(np.round(cl_acc*100,2)), list(support.astype(int))]).T

    cols = ["ID", "class","accuracy","#pixels"]
    df.columns = cols

    # add empty row
    df = df.append(pd.Series([np.nan,np.nan,np.nan,np.nan], index=cols), ignore_index=True)
    sum = int(df["#pixels"].sum())
    df = df.append(pd.Series(["per class", round(overall_accuracy*100, 2), sum], index=["class","accuracy","#pixels"]), ignore_index=True)
    df = df.append(pd.Series([q_dagrm], index=["accuracy"]),ignore_index=True)
    df = df.append(pd.Series([a_dagrm], index=["accuracy"]),ignore_index=True)

    df = df.set_index(["ID", "class"])

    return(df)

def confusionmatrix2table_perclass(dataset, oa, acc_perclass, q_dagrm, a_dagrm, ids=None, classnames=None, shortnames=None, outfile=None):

    oa_mean = np.mean(oa)
    oa_std = np.std(oa)
    q_dagrm_mean = np.mean(q_dagrm)
    q_dagrm_std = np.std(q_dagrm)
    a_dagrm_mean = np.mean(a_dagrm)
    a_dagrm_std = np.std(a_dagrm)

    acc_mean = np.mean(acc_perclass,axis=0)
    acc_std = np.std(acc_perclass,axis=0)

    support = cm.sum(1) # 0 -> prediction 1 -> ground truth

    df = pd.DataFrame([ids, classnames, shortnames, list(np.round(acc_mean*100,2)), list(np.round(acc_std*100,2)), list(support.astype(int)), list(support.astype(int)/sum(support.astype(int)))]).T

    cols = ["ID", "class","short","accuracy_mean","accuracy_std","support","persupport"]
    df.columns = cols
    df['dataset'] = dataset

    df = df.set_index(["dataset", "ID", "class","short"])

    print("writing tabular to "+outfile)
    df.to_csv(outfile, index=True)

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    dataset = args.dataset
    exp = args.experiment
    tyear = args.targetyear
    model = args.model
    epochs = args.epochs
    folds = args.folds
    ckpt = args.ckpt
    level = args.level

    if not os.path.exists(os.path.join(outdir, 'raw', level)):
        os.makedirs(os.path.join(outdir, 'raw', level))

    datafolder = os.path.join(indir, exp, "ep" + str(epochs), model)

    #create dict dataset
    dataset_config = {
        'MCD12Q1v6raw_LCType1': {
            'labels': classes.classes_MCD12Q1v6LCType1,
            'shortname': classes.shortname_MCD12Q1v6LCType1,
            'classes': 22,
            'short': 'M17'
        },
        'MCD12Q1v6raw_LCProp1': {
            'labels': classes.classes_MCD12Q1v6LCProp1,
            'classes': 16,
            'short': 'M16'
        },
        'MCD12Q1v6raw_LCProp2': {
            'labels': classes.classes_MCD12Q1v6LCProp2,
            'classes': 11,
            'short': 'M11'
        },
        'ESAraw': {
            'labels': classes.classes_esa,
            'classes': 37,
            'short': 'E37'
        },
        'Copernicusraw': {
            'labels': classes.classes_copernicus,
            'classes': 22,
            'short': 'C22'
        },
        'Copernicusnew_cf2others': {
            'labels': classes.classes_copernicus_cf2others,
            'shortname': classes.shortname_copernicus_cf2others,
            'classes': 9,
            'short': 'C9h'
        },
        'merge_datasets2own': {
            'labels': classes.classes_merge_datasets2own,
            'shortname': classes.shortname_merge_datasets2own,
            'classes': 8,
            'short': 'O8h'
        },
        'MCD12Q1v6stable_LCType1': {
            'labels': classes.classes_MCD12Q1v6LCType1,
            'shortname': classes.shortname_MCD12Q1v6LCType1,
            'classes': 22,
            'short': 'M17h'
        },
        'MCD12Q1v6stable_LCProp2': {
            'labels': classes.classes_MCD12Q1v6LCProp2,
            'shortname': classes.shortname_MCD12Q1v6LCProp2,
            'classes': 11,
            'short': 'M11h'
        },
        'ESAstable': {
            'labels': 37,
            'classes': classes.classes_esa,
            'short': 'E37h'
        },
    }

    n_classes = dataset_config[dataset]['classes']
    shortdataset = dataset_config[dataset]['short']

    oa = [ ]
    prec = [ ]
    rec = [ ]
    fscore = [ ]
    kappa = [ ]

    if level == 'global':
        kappa_loc = []
        kappa_hist = []
        chance_agrm= []
        quant_agrm = []
        quant_dagrm= []
        all_agrm = []
        all_dagrm = []

    if level == 'perclass':
        ac_mean = []
        ac_stf = []
        quant_dagrm= []
        all_dagrm = []
        ids_all = []

    for i in range(0, len(folds)):
        fold_path = os.path.join(model + "64_15_fold" + str(folds[ i ]) + "_" + dataset + "_" + str(ckpt), str(tyear))
        quantevalpath = os.path.join(datafolder, fold_path)  # 'full' or 'demo'
        data = np.load(os.path.join(quantevalpath, "truepred.npy"))

        y_true = data[:,0]
        y_pred = data[:,1]

        y_true2 = np.ma.MaskedArray(y_true, mask=y_true == 0).compressed()
        y_pred2 = np.ma.MaskedArray(y_pred, mask=y_true == 0).compressed()

        if level == 'global':
            a, k, p, r, f = mr_metrics(confusion_matrix(y_true2,y_pred2), level)
            kn, k_loc, k_hist, c_agrm, q_agrm, q_dagrm, a_agrm, a_dagrm = helpCalcKappa(confusion_matrix(y_true2,y_pred2))

            oa.append(round(a * 100, 2))
            prec.append(round(p * 100, 2))
            rec.append(round(r * 100, 2))
            fscore.append(round(f * 100, 2))
            kappa.append(kn)
            kappa_loc.append(k_loc)
            kappa_hist.append(k_hist)
            chance_agrm.append(c_agrm)
            quant_agrm.append(q_agrm)
            quant_dagrm.append(q_dagrm)
            all_agrm.append(a_agrm)
            all_dagrm.append(a_dagrm)

            results = {'oa': oa, 'prec': prec, 'rec': rec, 'fscore': fscore, 'kappa': kappa, 'kappa_loc': kappa_loc,
                       'kappa_hist': kappa_hist, 'chance_agrm': chance_agrm, 'quant_agrm':quant_agrm, 'quant_dagrm':quant_dagrm,
                       'all_agrm':all_agrm, 'all_dagrm':all_dagrm}

            results_df = pd.DataFrame(results, columns=['oa', 'prec', 'rec', 'fscore',
                                                         'kappa','kappa_hist', 'chance_agrm', 'quant_agrm',
                                                         'quant_dagrm','all_agrm', 'all_dagrm'])
            results_df = results_df.T

            results_df['dataset'] = dataset
            results_df['epoch'] = epochs
            results_df['shortname'] = shortdataset
            results_df['metric'] = results_df.index

            results_df.to_csv(os.path.join(outdir,'raw',level,'ep'+ str(epochs) + '_' + shortdataset + '.csv'), index = False)

        elif level == 'perclass':
            # metrics = mr_metrics(confusion_matrix(y_true2, y_pred2), level)
            cm = confusion_matrix(y_true2, y_pred2)
            ids = np.unique(y_true2)
            ids_all.append(ids)

            overall_accuracy, kappa, precision, recall, f1, cl_acc = mr_metrics(cm, 'perclass')
            kn, k_loc, k_hist, c_agrm, q_agrm, q_dagrm, a_agrm, a_dagrm = helpCalcKappa(confusion_matrix(y_true2,y_pred2))
            support = cm.sum(1)  # 0 -> prediction 1 -> ground truth

            f = f1_score(y_true, y_pred, average=None)
            r = recall_score(y_true, y_pred, average=None)
            p = precision_score(y_true, y_pred, average=None)

            ac_mean.append(cl_acc)
            oa.append(overall_accuracy)
            quant_dagrm.append(q_dagrm)
            all_dagrm.append(a_dagrm)

            prec.append(p)
            rec.append(r)
            fscore.append(f)

    if level == 'perclass':
        labels = dataset_config[dataset]['labels']
        shortname = dataset_config[dataset]['shortname']

        ids_all = np.unique(ids_all)

        newlabels= [labels[i] for i in ids_all]
        newshort= [shortname[i] for i in ids_all]

        outfile = os.path.join(outdir, 'raw', level, dataset + '.csv')
        confusionmatrix2table_perclass(shortdataset, oa, fscore, quant_dagrm, all_dagrm, ids_all, newlabels, newshort, outfile)

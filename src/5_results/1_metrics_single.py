import numpy as np
import os
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score

import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-i','--indir', type=str, required=True, help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True, help='Outdir dir')
parser.add_argument('--dataset', type=str, default=None, help='Dataset')
parser.add_argument('-y','--targetyear', type=str, required=True, help='Target year')
parser.add_argument('-l','--level', type=str, required=True, help='Level')
parser.add_argument('-f','--folds', type=list, required=True, help='Folds')

class classes:
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
    level = args.level

    def makedir(outfolder):
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

    #create storedir
    makedir(outdir)

    #create dict dataset
    dataset_config = {
        'MCD12Q1v6stable01to03_LCProp2': {
            'labels': classes.classes_MCD12Q1v6LCProp2,
            'shortname': classes.shortname_MCD12Q1v6LCProp2,
            'classes': 11,
            'short': 'M11h01to03'
        }
    }

    n_classes = dataset_config[dataset]['classes']
    shortdataset = dataset_config[dataset]['short']

    data = np.load(os.path.join(indir, 'truepred_' + args.targetyear + '.npy'))

    y_true2 = data[:,0]
    y_pred2 = data[:,1]

    y_true2 = np.asarray(y_true2, dtype='int32')
    y_pred2 = np.asarray(y_pred2, dtype='int32')

    if level == 'global':
        a, k, p, r, f = mr_metrics(confusion_matrix(y_true2,y_pred2), level)
        kn, k_loc, k_hist, c_agrm, q_agrm, q_dagrm, a_agrm, a_dagrm = helpCalcKappa(confusion_matrix(y_true2,y_pred2))

        oa = round(a * 100, 2)
        prec = round(p * 100, 2)
        rec = round(r * 100, 2)
        fscore = round(f * 100, 2)
        kappa = kn
        kappa_loc = k_loc
        kappa_hist = k_hist
        chance_agrm = c_agrm
        quant_agrm = q_agrm
        quant_dagrm = q_dagrm
        all_agrm = a_agrm
        all_dagrm = a_dagrm

        results = {'oa': oa, 'prec': prec, 'rec': rec, 'fscore': fscore, 'kappa': kappa, 'kappa_loc': kappa_loc,
                   'kappa_hist': kappa_hist, 'chance_agrm': chance_agrm, 'quant_agrm':quant_agrm, 'quant_dagrm':quant_dagrm,
                   'all_agrm':all_agrm, 'all_dagrm':all_dagrm}

        results_df = pd.DataFrame(results, columns=['oa', 'prec', 'rec', 'fscore',
                                                    'kappa','kappa_hist', 'chance_agrm', 'quant_agrm',
                                                    'quant_dagrm','all_agrm', 'all_dagrm'], index=[0])

        results_df = results_df.T

        results_df['dataset'] = dataset
        results_df['shortname'] = shortdataset
        results_df['metric'] = results_df.index

        results_df.to_csv(os.path.join(outdir,shortdataset + '_'+ args.targetyear + '_' + args.level + '.csv'), index = False)

    elif level == 'perclass':
        # metrics = mr_metrics(confusion_matrix(y_true2, y_pred2), level)
        cm = confusion_matrix(y_true2, y_pred2)
        ids = np.unique(y_true2)

        overall_accuracy, kappa, precision, recall, f1, cl_acc = mr_metrics(cm, 'perclass')
        kn, k_loc, k_hist, c_agrm, q_agrm, q_dagrm, a_agrm, a_dagrm = helpCalcKappa(confusion_matrix(y_true2,y_pred2))
        support = cm.sum(1)  # 0 -> prediction 1 -> ground truth

        f = f1_score(y_true2, y_pred2, average=None)
        r = recall_score(y_true2, y_pred2, average=None)
        p = precision_score(y_true2, y_pred2, average=None)

        ac_mean = cl_acc
        oa = overall_accuracy
        quant_dagrm = q_dagrm
        all_dagrm = a_dagrm

        prec = p
        rec = r
        fscore = f

        labels = dataset_config[dataset]['labels']
        shortname = dataset_config[dataset]['shortname']

        ids_all = ids

        newlabels= [labels[i] for i in ids_all]
        newshort= [shortname[i] for i in ids_all]
        print(newlabels)
        outfile = os.path.join(outdir,shortdataset + '_'+ args.targetyear + '_' + args.level + '.csv')
        confusionmatrix2table_perclass(shortdataset, oa, fscore, quant_dagrm, all_dagrm, ids_all, newlabels, newshort, outfile)

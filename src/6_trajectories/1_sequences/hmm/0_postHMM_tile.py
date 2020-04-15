import numpy as np
import rasterio
import os
import glob

import hmm
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-i','--indir', type=str, required=True, help='Indir dir with conf folder')
parser.add_argument('-o','--outdir', type=str, required=True, help='Outdir dir')
parser.add_argument('-s','--suffix', type=str, required=True, help='filename suffix')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set in GEE')
parser.add_argument('-t','--tile', type=str, required=True, help='target tile')
parser.add_argument('-d','--dataset', type=str, required=True, help='dataset')
parser.add_argument('-e','--experiment', type=str, required=True, help='experiment')
parser.add_argument('-sy','--startyear', type=int, required=True, help='experiment')
parser.add_argument('-ti','--terrai', type=bool, required=False, default=False, help='terrai')

class classes:
    classes_MCD12Q1v6LCType1 = [ 'NoData','Evergreen needleleaf forest', 'Evergreen broadleaf forest',
                                 'Deciduous needleleaf forest', 'Deciduous broadleaf forest',
                                 'Mixed forest', 'Closed shrublands', 'Open shrublands',
                                 'Woody savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                                 'Croplands', 'Urban and built-up', 'Cropland natural vegetation mosaic',
                                 'Snow and ice', 'Barren or sparsely vegetated', 'Water' ]

    # classes_MCD12Q1v6LCType1 = [ 'NoData','Forest', 'Forest',
    #                              'Forest', 'Forest',
    #                              'Forest', 'Shrublands', 'Shrublands',
    #                              'Woody savannas', 'Grasslands', 'Grasslands', 'Permanent wetlands',
    #                              'Croplands', 'Urban and built-up', 'Croplands',
    #                              'Snow and ice', 'Barren or sparsely vegetated', 'Water']
    #
    colors_MCD12Q1v6LCType1 = [ '#ababab', '#05450a', '#086a10', '#54a708',
                                '#78d203', '#009900', '#c6b044',
                                '#dcd159', '#dade48', '#fbff13',
                                '#b6ff05', '#27ff87', '#c24f44',
                                '#fa0000', '#ff6d4c', '#69fff8',
                                '#f9ffa4', '#1c0dff' ]


    shortname_MCD12Q1v6LCType1 = ['NoData', 'ENF', 'EBF',
                                 'DNF', 'DBF',
                                 'MF', 'CS', 'OS',
                                 'WS', 'S', 'G', 'PW',
                                 'C', 'Bu', 'CN',
                                 'SI', 'Ba', 'W']

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

    # colors_MCD12Q1v6LCProp2 = [ '#ababab', '#f9ffa4', '#69fff8', '#1c0dff',
    #                             '#fa0000', '#003f00', '#006c00',
    #                             '#e3ff77', '#b6ff05', '#93ce04',
    #                             '#77a703', '#dcd159' ]
    colors_MCD12Q1v6LCProp2 = [ '#d4d2d2', '#f9ffa4', '#69fff8', '#1c0dff',
                                '#fa0000', '#003f00', '#006c00',
                                '#e3ff77', '#b6ff05', '#93ce04',
                                '#f096ff', '#dcd159' ]

    short_MCD12Q1v6LCProp2 = ['NoData','Ba', 'SI',
                                 'W', 'Bu',
                                 'DF', 'OF', 'FCM',
                                 'NH', 'NHCM', 'HC',
                                 'S']

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
                                     'Cultivated and managed vegetation/agriculture (cropland)',
                                     'Water bodies',
                                     'Herbaceous wetland' ]

    short_copernicus_cf2others = ['NoData','DF', 'OF',
                                 'S', 'HV',
                                 'Ba', 'Bu', 'C',
                                 'W', 'HW']

    colors_copernicus_cf2others = ['#d4d2d2', '#003f00',
                                    '#006c00',
                                    '#ffbb22',
                                    '#b6ff05',
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

    colors_merge_datasets2own = ['#ababab','#f9ffa4','#1c0dff',
                               '#fa0000','#003f00',
                               '#006c00','#b6ff05',
                               '#77a703','#dcd159']

    classes_mapbiomas = ['NoData','Forest Formation', 'Savanna Formation',
                                 'Mangrove', 'Flooded forest',
                                 'Wetland', 'Grassland', 'Other non forest natural formation',
                                 'Farming', 'Non vegetated area', 'Salt flat', 'River, Lake and Ocean',
                                 'Glacier']

    # colors_mapbiomas = ['#ababab','#009820','#00FE2D','#68743A','#74A5AF','#3CC2A6','#B9AE53','#F3C13C','#FFFEB5','#EC9999','#FD7127','#001DFC','#FFFFFF']
    colors_mapbiomas = ['#d4d2d2','#003f00','#006c00','#68743A','#74A5AF','#3CC2A6','#b6ff05','#F3C13C','#f096ff','#f9ffa4','#FD7127','#0032c8','#FFFFFF']

    short_mapbiomas = ['NoData','F','S', 'M',
                                 'Ff', 'We', 'G',
                                 'NFN', 'Fm', 'NVA',
                                 'Sf','W','SI']

def read_file(file):
    with rasterio.open(file) as src:
        return (src.read(1))


def post(probs):
    probs = np.insert(probs, 0, 0)
    labels = hmm.forward_backward_mle(probs, n_labels, transition_matrix)
    return (labels)


def change(post_array):
    change_array = np.zeros([13, post_array[0].shape[0], post_array[0].shape[1]])

    for i in range(len(post_array) - 1):
        listarra = [ post_array[ i ], post_array[ i + 1 ] ]
        a = np.std(listarra, axis=0)
        change_array[i][a > 0] = 1

    change_period = change_array.sum(axis=0)

    return (change_period)


def plot_year_LC(post_files, syear, colormap, vmin, vmax, title):
    gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.05, hspace=0.2)

    fig, axes = plt.subplots(nrows=2, ncols=9, sharex=False, sharey=False, figsize=(16, 4), gridspec_kw=gridspec_kwargs)
    axes[-1, -1].axis('off')
    axes[-1, -2].axis('off')
    axes[-1, -2 ].set_axis_off()

    axs = axes.flatten()

    for i, data in enumerate(post_files):
        axs[i].set_title(str(i + syear), fontsize='xx-large')
        axs[i].imshow(data, cmap=colormap, interpolation='none', vmin=vmin, vmax=vmax)
        axs[i].axis('off')

    fig.text(0.08, 0.5, title, fontsize='xx-large', weight='bold', ha='center', va='center', rotation='vertical')

    fig.tight_layout()

    return(fig)


def plot_year_terrai(post_files, target, syear, colormap, vmin, vmax, title):
    gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.05, hspace=0.2)

    fig, axes = plt.subplots(nrows=1, ncols=6, sharex=False, sharey=False, figsize=(16, 8), gridspec_kw=gridspec_kwargs)

    axs = axes.flatten()

    for i, data in enumerate(post_files):
        # axs[i].set_title(str(i * 4 + syear), fontsize='xx-large') ##if all

        axs[i].set_title(str(target[i] + syear), fontsize='xx-large')

        # if i != len(post_files)-1:
        #     axs[i].set_title(str(i * 2 + syear), fontsize='xx-large')
        # else:
        #     axs[i].set_title(str(i * 2 + syear - 1), fontsize='xx-large')
        axs[i].imshow(data, cmap=colormap, interpolation='none', vmin=vmin, vmax=vmax)
        axs[i].axis('off')

    # fig.text(0.08, 0.5, title, fontsize='xx-large', weight='bold', ha='center', va='center', rotation='vertical')
    fig.text(0.08, 0.5, title, fontsize='xx-large', ha='center', va='center', rotation='vertical')

    plt.tight_layout()

    return(fig)

def plot_year_terrai_chapter(post_files, target, syear, colormap, vmin, vmax, title):
    gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.05, hspace=0.2)

    fig, axes = plt.subplots(nrows=1, ncols=6, sharex=False, sharey=False, figsize=(16, 8), gridspec_kw=gridspec_kwargs)

    axs = axes.flatten()

    for i, data in enumerate(post_files):
        # axs[i].set_title(str(i * 4 + syear), fontsize='xx-large') ##if all
        if title == 'Mapbiomas':
            if i == 1:
                axs[i].set_title(str(target[i] + syear), fontsize='xx-large', weight='bold')
            else:
                axs[i].set_title(str(target[i] + syear), fontsize='xx-large')

        # if i != len(post_files)-1:
        #     axs[i].set_title(str(i * 2 + syear), fontsize='xx-large')
        # else:
        #     axs[i].set_title(str(i * 2 + syear - 1), fontsize='xx-large')
        axs[i].imshow(data, cmap=colormap, interpolation='none', vmin=vmin, vmax=vmax)
        axs[i].axis('off')

    # fig.text(0.08, 0.5, title, fontsize='xx-large', weight='bold', ha='center', va='center', rotation='vertical')
    fig.text(0.08, 0.5, title, fontsize='xx-large', ha='center', va='center', rotation='vertical')

    plt.tight_layout()

    return(fig)

if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    suffix = args.suffix
    tile = args.tile
    psize = args.psize
    dataset = args.dataset
    experiment = args.experiment
    syear = args.startyear
    terrai = args.terrai

    #create dict colors
    dataset_config = {
        'MCD12Q1v6raw_LCType1': {
            'classes': classes.classes_MCD12Q1v6LCType1,
            'colors': classes.colors_MCD12Q1v6LCType1},
        'MCD12Q1v6raw_LCProp1': {
            'classes': classes.classes_MCD12Q1v6LCProp1,
            'colors': classes.colors_MCD12Q1v6LCProp1},
        'MCD12Q1v6raw_LCProp2': {
            'classes': classes.classes_MCD12Q1v6LCProp2,
            'colors': classes.colors_MCD12Q1v6LCProp2},
        'ESAraw': {
            'classes': classes.classes_esa,
            'colors': classes.colors_esa},
        'Copernicusraw': {
            'classes': classes.classes_copernicus,
            'colors': classes.colors_copernicus},
        'Copernicusnew_cf2others': {
            'classes': classes.classes_copernicus_cf2others,
            'colors': classes.colors_copernicus_cf2others,
            'shortname': classes.short_copernicus_cf2others,
            'short': 'C9*'},
        'merge_datasets2own': {
            'classes': classes.classes_merge_datasets2own,
            'colors': classes.colors_merge_datasets2own},
        'MCD12Q1v6stable_LCType1': {
            'classes': classes.classes_MCD12Q1v6LCType1,
            'colors': classes.colors_MCD12Q1v6LCType1},
        'MCD12Q1v6stable_LCProp2': {
            'classes': classes.classes_MCD12Q1v6LCProp2,
            'colors': classes.colors_MCD12Q1v6LCProp2,
            'shortname': classes.short_MCD12Q1v6LCProp2,
            'short': 'M11*'},
        'mapbiomas': {
            'classes': classes.classes_mapbiomas,
            'colors': classes.colors_mapbiomas,
            'shortname': classes.short_mapbiomas,
            'short': 'Mapbiomas'}
    }

    colors = dataset_config[dataset]['colors']
    labels = dataset_config[dataset]['classes']
    n_labels = len(colors)-1

    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(range(n_labels), colors)

    if terrai == False:

        n_years = 2018 - (syear-1)
        change_prob = 0.1
        transition_matrix = np.empty((n_labels, n_labels))
        transition_matrix.fill(change_prob)

        if dataset == 'MCD12Q1v6raw_LCType1':
            transition_matrix[:,:5] = 0.1
        elif dataset == 'MCD12Q1v6raw_LCProp1':
            transition_matrix[:,3:9] = 0.1
        elif dataset == 'MCD12Q1v6raw_LCProp2':
            transition_matrix[:,4] = 0.1
        elif dataset == 'ESAraw':
            transition_matrix[:,6:17] = 0.1
        elif dataset == 'Copernicusraw':
            transition_matrix[:,:6] = 0.1
        elif dataset == 'Copernicusnew_cf2others':
            transition_matrix[ :,0] = 0.1
        elif dataset == 'merge_datasets2own':
            transition_matrix[:,3] = 0.1

        np.fill_diagonal(transition_matrix, 1.0 - change_prob)

        file_list = glob.glob(os.path.join(indir,'*/confidences/**/*.tif'))
        file_list = sorted(file_list, key=lambda x: int(os.path.basename(os.path.dirname(x)).partition('_')[ 0 ]))

        array_list = [read_file(x) for x in file_list ]
        array_stack = np.stack(array_list)  # stack

        # reshape
        array_stack = np.reshape(array_stack, (n_labels, n_years, psize * psize))
        array_stack = np.transpose(array_stack, [1, 0, 2])
        array_stack = np.reshape(array_stack, (n_years * n_labels, psize * psize))

        post_stack = np.apply_along_axis(post, 0, array_stack)
        post_stack = np.reshape(post_stack, (post_stack.shape[0], psize, psize))

        if not os.path.exists(os.path.join(outdir,'pngs')):
            os.makedirs(os.path.join(outdir,'pngs'))

        with rasterio.open(file_list[0]) as src:
            meta = src.meta

        meta.update(dtype=rasterio.int32)

        if not os.path.exists(os.path.join(outdir,'geoTIFF')):
            os.makedirs(os.path.join(outdir,'geoTIFF'))

        # Write output file
        for id, layer in enumerate(post_stack, start=1):
            with rasterio.open(os.path.join(outdir,'geoTIFF', str(
                    (syear-1) + id) + '_postHMM_' + suffix + '.tif'), 'w',
                               **meta) as dst:
                dst.write(layer.astype(rasterio.int32), 1)

        #RAW
        pred_list = glob.glob(os.path.join(indir, '*/prediction/*.tif'))
        pred_list = pred_list[:-1]
        pred_files = [read_file(x) for x in pred_list]
        fig_map_pre = plot_year_LC(pred_files, syear, colormap, vmin=0, vmax=len(colors)-1, title='Raw LC maps')
        fig_map_pre.savefig(os.path.join(outdir,'pngs','map_preHMM_' + suffix + '.png'), bbox_inches='tight',pad_inches = 0)

        #POST-PROCESSING
        post_stack = post_stack[:-1,:,:]
        fig_map_post = plot_year_LC(post_stack, syear, colormap, vmin=0, vmax=len(colors)-1, title='Smoothed LC maps')
        fig_map_post.savefig(os.path.join(outdir,'pngs','map_postHMM_' + suffix + '.png'), bbox_inches='tight',pad_inches = 0)

        ##legend##
        pre_post = np.concatenate((post_stack,np.array(pred_files)),axis=0)
        uniqueValues = uniqueValues = np.unique(pre_post)
        uniqueValues = uniqueValues.astype(int).tolist()
        uniqueValues = [0] + uniqueValues

        # Extracting handles and labels
        # newcolors = [colors[i-1] for i in uniqueValues.tolist()]
        # newlabels = [labels[i-1] for i in uniqueValues.tolist()]
        newcolors = [colors[i] for i in uniqueValues]
        newlabels = [labels[i] for i in uniqueValues]

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [ f("s", newcolors[i]) for i in range(len(newcolors))]
        # legend = plt.legend(handles, labels, loc=5, framealpha=1, frameon=False)
        legend = plt.legend(bbox_to_anchor=(0.5, -1.5), handletextpad=0.001, frameon=False, loc='lower center',
                            borderaxespad=0., labels=newlabels, ncol=3, fontsize='xx-large', markerscale=2)

        os.path.join(outdir, 'pngs', 'map_postHMM_' + suffix + '.png')

        def export_legend(legend, filename=os.path.join(outdir, 'pngs', 'legend_' + suffix + '.png'),
                          expand=[ -5, -5, 5, 5 ]):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)

        export_legend(legend)

        print('Pre- and Post-HMM PNG files created for ' + dataset + '_' + suffix)

    else:
        shortname = dataset_config[ dataset ]['shortname']
        short = dataset_config[ dataset ]['short']

        #TERRA-I
        file_list = glob.glob(os.path.join(outdir, 'terrai/*' + dataset + '.asc'))
        dyear = os.path.basename(file_list[0])[15:19]
        target_idx = int(dyear) - int(syear)


        # target = list(range(0, 16, 2))
        # target1 = list(range(0, 14, 2))
        #target2 = [ 13 ]
        ## for general
        # target = list(range(0, 18, 4))

        ##for chapter 4
        target1 = [0]
        target2 = [target_idx]
        target3 = list(range(10, 18, 2))

        target = target1 + target2 + target3

        file_list = [file_list[i] for i in target]
        #
        array_list = [read_file(x) for x in file_list ]
        array_stack = np.stack(array_list)  # stack

        uniqueValues = np.unique(array_stack)
        uniqueValues = uniqueValues.astype(int).tolist()
        uniqueValues.remove(-9999)
        # uniqueValues = [0] + uniqueValues

        # Extracting handles and labels
        newcolors = [colors[i] for i in uniqueValues]
        newlabels = [shortname[i] for i in uniqueValues]

        f = lambda m, c: plt.plot([ ], [ ], marker=m, color=c, ls="none")[ 0 ]
        handles = [ f("s", newcolors[ i ]) for i in range(len(newcolors)) ]
        # legend = plt.legend(handles, labels, loc=5, framealpha=1, frameon=False)
        # legend = plt.legend(bbox_to_anchor=(0.5, -1.5), handletextpad=0.001, frameon=False, loc='lower center',
        #                     borderaxespad=0., labels=newlabels, ncol=2, fontsize='xx-large', markerscale=2)

        legend = plt.legend(handles, newlabels, fontsize='small', bbox_to_anchor=(2, 3), ncol=2, markerscale=1,
                          borderpad=0.6, handletextpad=0.05, frameon=True)
        legend.set_title(short, prop={'size': 'medium', 'weight': 'heavy'})

        def export_legend(legend, filename=os.path.join(outdir, 'pngs', 'legend_def' + dyear + '_' + suffix + '.png'),
                          expand=[ -5, -5, 5, 5 ]):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent()
            bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi=150, bbox_inches=bbox)
            # im = Image.open(filename)
            # im.save(filename, dpi=[ 300, 300 ])

        export_legend(legend)

        #fig_map_terrai = plot_year_terrai(array_stack, 2004, colormap, vmin=0, vmax=len(colors) - 1, title='Post-LC')
        fig_map_terrai = plot_year_terrai_chapter(array_stack, target, 2001, colormap, vmin=0, vmax=len(colors) - 1, title=short)
        fig_map_terrai.savefig(os.path.join(outdir,'pngs','map_terrai_' + suffix + '.png'), bbox_inches='tight',pad_inches = 0)

        print('Post-LC Terra-i PNG files created for ' + dataset + '_' + suffix)

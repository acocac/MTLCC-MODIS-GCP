import numpy as np
import rasterio
import os
import glob

import hmm
import matplotlib
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Export gee data to visualise in the GEE code editor')

parser.add_argument('-i','--indir', type=str, required=True, help='Indir dir with conf folder')
parser.add_argument('-o','--outdir', type=str, required=True, help='Outdir dir')
parser.add_argument('-p','--psize', type=int, required=True, help='patch size value set in GEE')
parser.add_argument('-t','--tile', type=str, required=True, help='target tile')
parser.add_argument('-d','--dataset', type=str, required=True, help='dataset')
parser.add_argument('-e','--experiment', type=str, required=True, help='experiment')
parser.add_argument('-sy','--startyear', type=int, required=True, help='experiment')
parser.add_argument('-ti','--terrai', type=bool, required=False, default=False, help='terrai')

class classes:
    classes_mapbiomas = ['NoData','Forest Formation', 'Savanna Formation',
                                 'Mangrove', 'Flooded forest',
                                 'Wetland', 'Grassland', 'Other non forest natural formation',
                                 'Farming', 'Non vegetated area', 'Salt flat', 'River, Lake and Ocean',
                                 'Glacier']

    colors_mapbiomas = [ '#ababab','#009820','#00FE2D','#68743A','#74A5AF','#3CC2A6','#B9AE53','#F3C13C','#FFFEB5','#EC9999','#FD7127','#001DFC','#FFFFFF']

def read_file(file):
    with rasterio.open(file) as src:
        return (src.read(1))

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


def plot_year_terrai(post_files, syear, colormap, vmin, vmax, title):
    gridspec_kwargs = dict(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.05, hspace=0.2)

    fig, axes = plt.subplots(nrows=1, ncols=8, sharex=False, sharey=False, figsize=(16, 8), gridspec_kw=gridspec_kwargs)

    axs = axes.flatten()

    for i, data in enumerate(post_files):
        if i != len(post_files)-1:
            axs[i].set_title(str(i * 2 + syear), fontsize='xx-large')
        else:
            axs[i].set_title(str(i * 2 + syear - 1), fontsize='xx-large')
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
    tile = args.tile
    psize = args.psize
    dataset = args.dataset
    experiment = args.experiment
    syear = args.startyear
    terrai = args.terrai

    #create dict colors
    dataset_config = {
        'mapbiomas': {
            'classes': classes.classes_mapbiomas,
            'colors': classes.colors_mapbiomas}
    }

    colors = dataset_config[dataset]['colors']
    labels = dataset_config[dataset]['classes']
    n_labels = len(colors)-1

    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(range(n_labels), colors)

    #RAW
    gt_list = glob.glob(os.path.join(indir, '*/mapbiomas/*.tif'))
    gt_list = gt_list[:-1]
    gt_files = [read_file(x) for x in gt_list]
    fig_map = plot_year_LC(gt_files, syear, colormap, vmin=0, vmax=len(colors)-1, title='Mapbiomas LC maps')
    fig_map.savefig(os.path.join(outdir,'pngs','map_mapbiomas.png'), bbox_inches='tight',pad_inches = 0)

    ##legend##
    uniqueValues = uniqueValues = np.unique(np.array(gt_files))
    uniqueValues = uniqueValues.astype(int).tolist()
    if not 0 in uniqueValues:
        uniqueValues = [0] + uniqueValues

    # Extracting handles and labels
    newcolors = [colors[i] for i in uniqueValues]
    newlabels = [labels[i] for i in uniqueValues]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [ f("s", newcolors[i]) for i in range(len(newcolors))]
    # legend = plt.legend(handles, labels, loc=5, framealpha=1, frameon=False)
    legend = plt.legend(bbox_to_anchor=(0.5, -1.5), handletextpad=0.001, frameon=False, loc='lower center',
                        borderaxespad=0., labels=newlabels, ncol=3, fontsize='xx-large', markerscale=2)

    def export_legend(legend, filename=os.path.join(outdir, 'pngs', 'legend_' + dataset + '.png'),
                      expand=[ -5, -5, 5, 5 ]):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    export_legend(legend)

    print('LC map PNG files created for ' + dataset)

    #TERRA-I
    file_list = glob.glob(os.path.join(outdir, 'terrai/*' + dataset + '.asc'))
    target1 = list(range(0, 14, 2))
    target2 = [13]
    target = target1 + target2
    file_list = [file_list[i] for i in target]
    print(file_list)

    array_list = [ read_file(x) for x in file_list ]
    array_stack = np.stack(array_list)  # stack

    fig_map_terrai = plot_year_terrai(array_stack, 2004, colormap, vmin=0, vmax=len(colors) - 1, title='Post-LC')
    fig_map_terrai.savefig(os.path.join(outdir,'pngs','map_terrai_' + dataset + '.png'), bbox_inches='tight',pad_inches = 0)

    print('Post-LC Terra-i PNG files created for ' + dataset)

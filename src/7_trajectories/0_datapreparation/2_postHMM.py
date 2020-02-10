import mtlchmm
import glob
import os
import rasterio

import argparse

parser = argparse.ArgumentParser(description='Create clean LC data for seq analysis')

parser.add_argument('-i','--indir', type=str, required=True,
                    help='Indir dir')
parser.add_argument('-o','--outdir', type=str, required=True,
                    help='Outdir dir')

def makedir(outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def run(args):

    makedir(args.outdir)

    #settings
    hmm_model = mtlchmm.MTLCHMM(method='forward-backward',
                                transition_prior=0.1,
                                block_size=2000,
                                n_jobs=12,
                                tiled=True,
                                compress='lzw',
                                assign_class=True,
                                class_list=list(range(1,9)),
                                track_blocks=True,
                                out_dir=args.outdir)

    lc_probabilities = glob.glob(os.path.join(args.indir, '*.tif'))

    hmm_model.fit_predict(lc_probabilities)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
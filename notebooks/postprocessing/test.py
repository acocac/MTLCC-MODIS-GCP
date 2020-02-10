import mtlchmm
import rasterio
import glob
import os
import numpy as np


def makedir(outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def run():

    out_dir = r'E:\acocac\research\c4\tile_0_201\eval\test\hmm'
    makedir(out_dir)

    hmm_model = mtlchmm.MTLCHMM(method='forward-backward',
                                transition_prior=0.1,
                                block_size=384/4,
                                n_jobs=2,
                                tiled=True,
                                compress='lzw',
                                assign_class=True,
                                class_list=list(range(1,12)),
                                track_blocks=True,
                                out_dir=out_dir)

    indir_lcc = r'E:\acocac\research\c4\tile_0_201\eval\test\raw'
    lc_probabilities = glob.glob(os.path.join(indir_lcc, '*.tif'))

    hmm_model.fit_predict(lc_probabilities)

if __name__ == '__main__':
    run()


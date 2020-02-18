import numpy as np
import argparse
import sys
import rasterio
import glob
import os

TRUE_PRED_FILENAME="truepred.npy"

def parse_arguments(argv):
  """Parses execution arguments and replaces default values.

  Args:
    argv: Input arguments from sys.

  Returns:
    Dictionary of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Evalutation of models')
  parser.add_argument('--preddir', type=str,
                      help="directory containing predictions")
  parser.add_argument('--verdir', type=str, default=None,
                      help='directory containing verification "ground truth"')
  parser.add_argument('--dataset', type=str, default='2001',
                      help='dataset (year)')
  parser.add_argument('--bestmodel', type=str, default=None,
                      help='bestmodel')
  parser.add_argument('--experiment', type=str, default=None,
                      help='experiment')

  args, _ = parser.parse_known_args(args=argv[1:])

  return args


def read_file(file):
    with rasterio.open(file) as src:
        return (src.read(1))

def eval(args):

    truepred = np.empty((0, 2), dtype=int)

    if args.experiment == '0_tl':
        verdir_list = glob.glob(os.path.join(args.verdir, '*.tif'))

        target_id = os.path.basename(verdir_list[0]).split('_')[1]

        pred_out = read_file(os.path.join(args.preddir, '{}_0.tif'.format(target_id)))
        ver_out = read_file(os.path.join(args.verdir, '0_{}_0.tif'.format(target_id)))

    else:
        pred_list = glob.glob(os.path.join(args.preddir, '*.tif'))
        verdir_list = glob.glob(os.path.join(args.verdir, '*.tif'))

        assert(len(pred_list) == len(verdir_list))

        # Read all data as a list of numpy arrays
        pred = [read_file(x) for x in pred_list]
        ver = [read_file(x) for x in verdir_list]

        pred_out = np.stack(pred, axis=0)
        ver_out = np.stack(ver, axis=0)

    pred_out = pred_out.flatten()
    ver_out = ver_out.flatten()

    def makedir(outfolder):
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

    #create storedir
    storedir = args.preddir.replace('pred','metrics')
    storedir = os.path.dirname(storedir)
    makedir(storedir)

    y_true = np.ma.MaskedArray(ver_out, mask=ver_out==0).compressed()
    y_pred = np.ma.MaskedArray(pred_out, mask=ver_out==0).compressed()

    truepred = np.row_stack((truepred, np.column_stack((y_true, y_pred))))

    np.save(os.path.join(storedir,'truepred_{}.npy'.format(args.dataset)), truepred)

    print('NPY files predictions and ground truth successfully created')

def main(args):
    args = parse_arguments(sys.argv)

    eval(args)

if __name__ == "__main__":

    args = parse_arguments(sys.argv)

    main(args)





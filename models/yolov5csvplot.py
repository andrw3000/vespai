import os
import sys
import glob
import argparse
import pandas as pd

# Root in `vespalert`
if not os.path.basename(os.getcwd()) == 'vespalert':
    assert "Please run `csvplot.py` from root directory `vespalert`."

sys.path.insert(0, os.path.join(os.getcwd(), '../models/yolov5'))
from utils.plots import plot_results

parser = argparse.ArgumentParser()
parser.add_argument('--dir',
                    type=str,
                    default=os.path.join(''),
                    help="Path to directory containing `results*.csv` files.",
                    )
parser.add_argument('--nrows',
                    type=int,
                    default=50,
                    help="Number of row from CSV file to plot.")
args = parser.parse_args()

if __name__ == "__main__":

    # Plot entire results file
    csv_dir = os.path.join('../models/yolov5/runs/train', args.dir)
    plot_results(file='', dir=csv_dir)

    # Plot fist nrows rows
    csv_files = glob.glob(os.path.join(csv_dir, 'results*.csv'))
    new_dir = os.path.join(csv_dir, 'first{}eps'.format(args.nrows))
    os.makedirs(new_dir, exist_ok=True)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        new_file = os.path.join(new_dir, os.path.basename(csv_file))
        df[:args.nrows].to_csv(new_file, index=False)

    plot_results(file='', dir=new_dir)

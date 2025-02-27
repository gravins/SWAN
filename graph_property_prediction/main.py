import os
import torch

import ray
import time
import argparse
import datetime
from utils import DATA
from conf import CONFIGS
from utils.gpp_dataset import TASKS
from model_selection import model_selection
from utils.io import create_if_not_exist, join

# Ignore warnings
from sklearn.exceptions import UndefinedMetricWarning


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

ray.init(address='local', num_cpus=40)  # local ray initialization

print('Settings:')
print('\tKMP_SETTING:', os.environ.get('KMP_SETTING'))
print('\tOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print('\tKMP_BLOCKTIME:', os.environ.get('KMP_BLOCKTIME'))
print('\tMALLOC_CONF:', os.environ.get('MALLOC_CONF'))
print('\tLD_PRELOAD:', os.environ.get('LD_PRELOAD'))
print()

if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task',
                        help='The name of the GraphPropPred task.',
                        default=TASKS[0],
                        choices=TASKS)
    parser.add_argument('--model_name',
                        help='The model name.',
                        default=list(CONFIGS.keys())[0],
                        choices=CONFIGS.keys())
    parser.add_argument('--epochs', help='The number of epochs.', default=1500, type=int)
    parser.add_argument('--early_stopping',
                        help='Training stops if the selected metric does not improve for X epochs',
                        type=int,
                        default=100)
    parser.add_argument('--save_dir', help='The saving directory.', default='.')
    parser.add_argument('--cpus', help='The number of CPUs per configuration.', default=5, type=int)
    parser.add_argument('--gpus', help='The percentage of GPU per configuration.', default=0., type=float)
    args = parser.parse_args()

    print(args)
    assert args.epochs >= 1, 'The number of epochs should be greather than 0'
    args.save_dir = os.path.abspath(args.save_dir)

    p = join(args.save_dir, 'GraphPropPred')
    create_if_not_exist(p)
    p = join(p, args.task)
    create_if_not_exist(p)
    exp_dir = join(p, args.model_name)
    create_if_not_exist(exp_dir)

    # Run model selection
    best_conf_res = model_selection(model_name=args.model_name,
                                    early_stopping_patience=args.early_stopping,
                                    epochs=args.epochs,
                                    task=args.task,
                                    data_dir=os.path.join(args.save_dir, 'data'),
                                    exp_dir=exp_dir,
                                    num_cpus=args.cpus,
                                    num_gpus=args.gpus)

    print(best_conf_res)
    ray.shutdown()
    elapsed = time.time() - t0
    print(datetime.timedelta(seconds=elapsed))

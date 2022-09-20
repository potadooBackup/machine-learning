from enum import IntEnum
import os

import torch

# Tuning
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

class ProblemType(IntEnum):
    CLASSIFICATION = 0
    REGRESSION = 1

PROBLEM_TYPE = ProblemType.REGRESSION

# Const config
const_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 300,
    'random_seed': 5203000,
}

# Data root
DATA_ROOT_DIR = os.path.abspath('./dataset')
# Number of parallel processes for data fetching
NUM_WORKERS = 0

# For search run
NUM_SAMPLES = 5 # the number of runs

# Experiment config
config = {
    'batch_size': tune.choice([8, 16, 32]),
    'lr': tune.loguniform(1e-5, 1e-1),
}

search_algo = OptunaSearch()

# Schduler to stop bad performing trails.
scheduler = ASHAScheduler(
    reduction_factor = 2,
    max_t = const_config['n_epochs'],
    grace_period = 1,
)

# if __name__ == '__main__':
#     ray.init(num_cpus=3, num_gpus=0)
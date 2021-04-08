# This file is to develop our clustering baselines

# Archived.

########### IMPORT SECTION ###############
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets.random_task_dataset import RandomTaskDataset
from datasets.random_ova_dataset import RandomOvaDataset
from models.predictor import meanPred
from models.clustering_selector import ClusteringSelector
from utils import test_baseline
import constants

########### PARAMETER SETTING ############
test = 1
use_mode = "cosine"

if test == 1:

    # input files
    csv_path = ['/deep/group/activelearn/data/ova/meta_tst/non_holdout.csv',
                '/deep/group/activelearn/data/ova/meta_tst/holdout.csv',
                '/deep/group/activelearn/data/ova/meta_train/non_holdout.csv',
                '/deep/group/activelearn/data/ova/meta_val/non_holdout.csv']

    # output results for plotting
    output_path = ['/deep/group/activelearn/experiments/ova/CheXpert/meta_tst/clustering/non_holdout/',
                   '/deep/group/activelearn/experiments/ova/CheXpert/meta_tst/clustering/holdout/',
                   '/deep/group/activelearn/experiments/ova/CheXpert/meta_train/clustering/',
                   '/deep/group/activelearn/experiments/ova/CheXpert/meta_val/clustering/']

else:
    # input files
    csv_path = ['/deep/group/activelearn/data/level1/meta_tst/non_holdout_positives.csv',
                '/deep/group/activelearn/data/level1/meta_tst/holdout_positives.csv',
                '/deep/group/activelearn/data/level1/meta_train/positives.csv',
                '/deep/group/activelearn/data/level1/meta_val/positives.csv']

    normal_path = ['/deep/group/activelearn/data/level1/meta_tst/no_findings.csv',
                   '/deep/group/activelearn/data/level1/meta_tst/no_findings.csv',
                   '/deep/group/activelearn/data/level1/meta_train/no_findings.csv',
                   '/deep/group/activelearn/data/level1/meta_val/no_findings.csv']

    # output results for plotting
    output_path = ['/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/non_holdout/',
                   '/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/holdout/',
                   '/deep/group/activelearn/experiments/level1/meta_train/cluster_baseline/cosine_auc/',
                   '/deep/group/activelearn/experiments/level1/meta_val/cluster_baseline/cosine_auc/']


if __name__ == "__main__":

    for x in range(len(csv_path)):

        #skip training and test for computing sampling size of 1000
        if x == 2 or x == 3:
            continue

        cond = constants.NON_HOLD_OUT

        if x == 1:
            cond = constants.HOLD_OUT

        num_tasks = constants.NUM_META_TEST_TASKS

        if x == 2:
            num_tasks = constants.NUM_META_TRAIN_TASKS

        if test == 1:
            dset_all = RandomOvaDataset(csv_path=csv_path[x],
                                        unlabeled_pool_size=constants.UNLABELED_POOL_SIZE,
                                        unlabeled_pos_frac=constants.UNLABELED_POS_FRAC,
                                        query_set_size=constants.QUERY_SET_SIZE,
                                        query_pos_frac=constants.QUERY_POS_FRAC,
                                        conditions_used=cond,
                                        num_tasks=num_tasks,
                                        deterministic=True)
        else:
            dset_all = RandomTaskDataset(positive_csv_path=csv_path[x],
                                         normal_csv_path=normal_path[x],
                                         unlabeled_pool_size=constants.UNLABELED_POOL_SIZE,
                                         unlabeled_pos_frac=constants.UNLABELED_POS_FRAC,
                                         query_set_size=constants.QUERY_SET_SIZE,
                                         query_pos_frac=constants.QUERY_POS_FRAC,
                                         conditions_used=cond,
                                         num_tasks=num_tasks,
                                         deterministic=True)

        test_pred_all = meanPred(mode = use_mode)

        selector = ClusteringSelector()

        test_baseline(csv_path[x], output_path = output_path[x],selector = selector, dset_all = dset_all, predictor = test_pred_all)


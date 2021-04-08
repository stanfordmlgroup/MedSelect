# This file is to develop our clustering baselines for multilabel setting

# Archived.

########### IMPORT SECTION ###############
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets.multi_task_dataset import MultiTaskDataset
from models.predictor import meanPred
from models.clustering_selector import ClusteringSelector
from utils import test_baseline
import constants

########### PARAMETER SETTING ############
sampling_size = [40,80,100,200,400]
use_mode = "cosine"

# input files
positive_path = ['/deep/group/activelearn/data/level1/meta_tst/non_holdout_positives.csv']

normal_path = ['/deep/group/activelearn/data/level1/meta_tst/no_findings.csv']

if __name__ == "__main__":

    for x in range(len(positive_path)):

        cond = constants.NON_HOLD_OUT

        num_tasks = constants.NUM_META_TEST_TASKS

        dset_all = MultiTaskDataset(positive_csv_path=positive_path[x],
                                    normal_csv_path=normal_path[x],
                                    unlabeled_pool_size=constants.UNLABELED_POOL_SIZE,
                                    query_set_size=constants.QUERY_SET_SIZE,
                                    conditions_used=cond,
                                    num_tasks=num_tasks,
                                    deterministic=True)

        test_pred_all = meanPred(mode = use_mode, labels = torch.tensor([[1,1],[1,0],[0,1],[0,0]]))

        selector = ClusteringSelector()

        test_baseline(positive_path[x], sampling_size = sampling_size, num_cond = 2, dset_all = dset_all, 
                      selector = selector, predictor = test_pred_all)


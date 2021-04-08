# This file is modified to test ALL baselines.

########### IMPORT SECTION ###############
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist

from datasets.random_task_dataset import RandomTaskDataset, MEAN_AGE, STD_AGE
from models.clustering_selector import ClusteringSelector
from models.lstm_selector import LSTMSelector
from models.predictor import meanPred
from models.svm_predictor import svmPred
from utils import test_baseline, collate_fn, test_selector
import constants

########### PARAMETER SETTING ############
best_models = {10:'model_epoch5_iter30',
               20:'model_epoch5_iter150',
               40:'model_epoch5_iter30',
               80:'model_epoch4_iter150',
               100:'model_epoch5_iter90',
               200:'model_epoch1_iter90',
               400:'model_epoch1_iter90'}

nh = [97,95]
h = [7,5]

# input files
csv_path = ['/deep/group/activelearn/data/level1/meta_tst/non_holdout_positives.csv',
            '/deep/group/activelearn/data/level1/meta_tst/holdout_positives.csv']

normal_path = ['/deep/group/activelearn/data/level1/meta_tst/no_findings.csv',
               '/deep/group/activelearn/data/level1/meta_tst/no_findings.csv']

# output results for plotting: base random baseline
output_path0 = ['/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/non_holdout/base_',
                '/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/holdout/base_']

# output results for plotting: random baseline
output_path1 = ['/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/non_holdout/',
                '/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/holdout/']

# output results for plotting: clustering baseline
output_path2 = ['/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/non_holdout/',
                '/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/holdout/']

if __name__ == "__main__":

    for k in constants.K:

        for x in range(len(csv_path)):


            if x == 1:
                print("%%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%")
                cond = constants.HOLD_OUT
            else:
                print("%%%%%%%%%%%%%%%%%%%% NON HOLD OUT %%%%%%%%%%%%%%%%%%%")
                cond = constants.NON_HOLD_OUT

            dset_all = RandomTaskDataset(positive_csv_path=csv_path[x],
                                         normal_csv_path=normal_path[x],
                                         unlabeled_pool_size=constants.UNLABELED_POOL_SIZE,
                                         unlabeled_pos_frac=constants.UNLABELED_POS_FRAC,
                                         query_set_size=constants.QUERY_SET_SIZE,
                                         query_pos_frac=constants.QUERY_POS_FRAC,
                                         conditions_used=cond,
                                         num_tasks=constants.NUM_META_TEST_TASKS,
                                         deterministic=True,
                                         im_emb=True,
                                         age_sex_lat=True)

            FL = {}
            FL["max_overall"] = []
            FL["mean_overall"] = []
            FL["range_overall"] = []
            FL["max_lat"] = []
            FL["mean_lat"] = []
            FL["range_lat"] = []
            FL["max_fro"] = []
            FL["mean_fro"] = []
            FL["range_fro"] = []
            FL["max_overall_cos"] = []
            FL["mean_overall_cos"] = []
            FL["range_overall_cos"] = []
            FL["max_lat_cos"] = []
            FL["mean_lat_cos"] = []
            FL["range_lat_cos"] = []
            FL["max_fro_cos"] = []
            FL["mean_fro_cos"] = []
            FL["range_fro_cos"] = []
            loader_FL = torch.utils.data.DataLoader(dataset=dset_all, batch_size=1, collate_fn=collate_fn)
            for i, data in enumerate(loader_FL):
                emb = data["pool"][0,:,:512].cpu().numpy()
                is_fro = data["pool"][0,:,514].cpu().numpy()
                idx_fro = np.argwhere(is_fro == 1).flatten()
                idx_lat = np.argwhere(is_fro == 0).flatten()
                overall_met = pdist(emb,'euclidean')
                fro_met = pdist(emb[idx_fro,:],'euclidean')
                lat_met = pdist(emb[idx_lat,:],'euclidean')
                FL["max_overall"].append(np.max(overall_met))
                FL["mean_overall"].append(np.mean(overall_met))
                FL["range_overall"].append(np.max(overall_met) - np.min(overall_met))
                FL["max_lat"].append(np.max(lat_met))
                FL["mean_lat"].append(np.mean(lat_met))
                FL["range_lat"].append(np.max(lat_met)-np.min(lat_met))
                FL["max_fro"].append(np.max(fro_met))
                FL["mean_fro"].append(np.mean(fro_met))
                FL["range_fro"].append(np.max(fro_met)-np.min(fro_met))
                overall_met = pdist(emb,'cosine')
                fro_met = pdist(emb[idx_fro,:],'cosine')
                lat_met = pdist(emb[idx_lat,:],'cosine')
                FL["max_overall_cos"].append(np.max(overall_met))
                FL["mean_overall_cos"].append(np.mean(overall_met))
                FL["range_overall_cos"].append(np.max(overall_met) - np.min(overall_met))
                FL["max_lat_cos"].append(np.max(lat_met))
                FL["mean_lat_cos"].append(np.mean(lat_met))
                FL["range_lat_cos"].append(np.max(lat_met)-np.min(lat_met))
                FL["max_fro_cos"].append(np.max(fro_met))
                FL["mean_fro_cos"].append(np.mean(fro_met))
                FL["range_fro_cos"].append(np.max(fro_met)-np.min(fro_met))
                if i%100 == 0:
                    print("complete %d tasks."%(i))

            for key in FL.keys():
                print("Mean for "+key+" is: %.4f"%(np.mean(np.array(FL[key]))))
            stat_output = pd.DataFrame.from_dict(FL)
            filename = output_path0[x]+str(k)+"_FLstat.csv"
            stat_output.to_csv(filename, index = False)
            print("Output written to file: "+filename+"\n")


            test_pred_rand0 = meanPred(save_frontal = False)
            test_pred_rand1 = meanPred()
            test_pred_clust = meanPred()
            test_pred_lstm = meanPred()


            if x == 0:
                test_baseline(csv_path[x], output_path = output_path0[x], sampling_size = [k], dset_all = dset_all, seed = nh[0],
                              predictor = test_pred_rand0)

                test_baseline(csv_path[x], output_path = output_path1[x], sampling_size = [k], dset_all = dset_all, seed = nh[1],
                              predictor = test_pred_rand1)

            else:
                test_baseline(csv_path[x], output_path = output_path0[x], sampling_size = [k], dset_all = dset_all, seed = h[0],
                              predictor = test_pred_rand0)

                test_baseline(csv_path[x], output_path = output_path1[x], sampling_size = [k], dset_all = dset_all, seed = h[1],
                              predictor = test_pred_rand1)


            clustering = ClusteringSelector()
            test_baseline(csv_path[x], output_path = output_path2[x], sampling_size = [k], selector = clustering, dset_all = dset_all,
                          predictor = test_pred_clust)

            test_ld = torch.utils.data.DataLoader(dataset=dset_all, batch_size=constants.BATCH_SIZE,
                                                  collate_fn=collate_fn, num_workers=4)
            selector = LSTMSelector(input_size=(constants.USE_IMG*512 + constants.USE_ASL*3))

            test_selector(selector,test_pred_lstm,
                          f'/deep/group/activelearn/lstm_selector/k={k}/ckpt/only_img/{best_models[k]}',
                          test_ld,
                          k=k)

            comp_auc = pd.DataFrame.from_dict(test_pred_lstm.auc)
            print("AUC Score overall: %.4f, std: %.4f"%(comp_auc["overall"].mean(),comp_auc["overall"].std()))
            print("AUC Score frontal: %.4f, std: %.4f"%(comp_auc["frontal"].mean(),comp_auc["frontal"].std()))
#            print("AUC Score lateral: %.4f, std: %.4f"%(comp_auc["lateral"].mean(),comp_auc["lateral"].std()))
            print("\n")

#            if x == 0:
#                filename = f'/deep/group/activelearn/lstm_selector/k={k}/ckpt/only_img/{best_models[k]}_frocomp.csv'
#            else:
#                filename = f'/deep/group/activelearn/lstm_selector/k={k}/ckpt/only_img/{best_models[k]}_holdout_frocomp.csv'
#            comp_auc.to_csv(filename,index = False)
#            print("Output written to file: "+filename+"\n")


            res = test_pred_rand1.get_wasserstein(test_pred_rand0, MEAN_AGE, STD_AGE)
            wa_df = pd.DataFrame.from_dict(res)
            filename = output_path1[x]+str(k)+"_wacomp.csv"
            wa_df.to_csv(filename,index = False)
            print("Output written to file: "+filename+"\n")

            res = test_pred_clust.get_wasserstein(test_pred_rand0, MEAN_AGE, STD_AGE)
            wa_df = pd.DataFrame.from_dict(res)
            filename = output_path2[x]+str(k)+"_wacomp.csv"
            wa_df.to_csv(filename,index = False)
            print("Output written to file: "+filename+"\n")

            res = test_pred_lstm.get_wasserstein(test_pred_rand0, MEAN_AGE, STD_AGE)
            wa_df = pd.DataFrame.from_dict(res)
            if x == 0:
                filename = f'/deep/group/activelearn/lstm_selector/k={k}/ckpt/only_img/{best_models[k]}_wacomp.csv'
            else:
                filename = f'/deep/group/activelearn/lstm_selector/k={k}/ckpt/only_img/{best_models[k]}_holdout_wacomp.csv'
            wa_df.to_csv(filename,index = False)
            print("Output written to file: "+filename+"\n")


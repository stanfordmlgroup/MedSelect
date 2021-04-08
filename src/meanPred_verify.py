# Archived.
# Test script.

# This file verify calculation of meanCosSimPred and meanEuclidPred
# Using: 1) a baby toy example with small data, all printed out
#        2) a toy example with 10 actual tasks
#        3) a simple baseline testing with 50 actual tasks

# Update on Dec. 01, 2020:
# 1. modify baby toy testing for multi-label method 1 and 2

# TODO from Dec. 01, 2020:
# 1. test multi-label method 1 and 2 on real data

########### IMPORT SECTION ###############
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.multi_task_dataset import MultiTaskDataset
from models.predictor import meanPred
from constants import UNLABELED_POOL_SIZE, UNLABELED_POS_FRAC, QUERY_SET_SIZE, QUERY_POS_FRAC
from utils import collate_fn

########### PARAMETER TESTING ############
parser = argparse.ArgumentParser()
parser.add_argument('--testmode',dest='mode',default='cosine',help='choose from: cosine, euclidean')
args = parser.parse_args()

if __name__ == '__main__':

    print("************* START OF BABY TOY TESTING *********")

    #### Start of step-by-step checking section:
    labels = torch.tensor([(1,1),(1,0),(0,1),(0,0)])

    if args.mode == 'cosine':
        check_correctness = meanPred()
        check_correctness_mult2 = meanPred(labels = labels, verbose = True)
    else:
        check_correctness = meanPred(mode = "euclidean")
        check_correctness_mult2 = meanPred(mode = "euclidean",labels = labels, verbose = True)

    simple_data = {}
    torch.manual_seed(58)
    simple_data['pool'] = torch.randn((1,8,5))
    torch.manual_seed(13)
    simple_data['query'] = torch.randn((1,4,5))
    simple_data['pool_labels'] = torch.Tensor([1,1,1,1,0,0,0,0])
    simple_data['mask'] = torch.Tensor([1,0,0,1,0,1,0,1])
    simple_data['query_labels'] = torch.Tensor([1,0,0,1])
    simple_data['pool_labels'] = torch.reshape(simple_data['pool_labels'],(1,8))
    simple_data['mask'] = torch.reshape(simple_data['mask'],(1,8))
    simple_data['query_labels'] = torch.reshape(simple_data['query_labels'],(1,4))

    print("------------Score: binary structure------------------------------")
    print("simplified data used: ")
    print(simple_data)
    y_score = check_correctness(simple_data)
    print(y_score)

    print("------------Score: multiclass method 2 --------------------------")
    simple_data['pool_labels'] = torch.Tensor([[1,1],[1,0],[0,1],[0,1],[0,0],[0,0],[1,0],[1,0]])
    simple_data['query_labels'] = torch.Tensor([[1,1],[1,0],[0,0],[0,1]])
    simple_data['pool_labels'] = torch.reshape(simple_data['pool_labels'],(1,8,2))
    simple_data['query_labels'] = torch.reshape(simple_data['query_labels'],(1,4,2))
    y_score = check_correctness_mult2(simple_data)
    print(y_score)
    print(y_score.shape)

    print("Test functionality of auroc:")
    for i in range(y_score.shape[0]):
#         auc = roc_auc_score(simple_data["query_labels"][i,:],y_score[i,:])
#         print("%.4f"%(auc))
        for j in range(y_score.shape[2]):
            auc = roc_auc_score(simple_data["query_labels"][i,:,j],y_score[i,:,j])
            print("%.4f"%(auc))

    #### End of step-by-step checking section

    print("************* START OF TOY TESTING **************:")

    #### Start of original testing section:

    dset = MultiTaskDataset(positive_csv_path='/deep/group/activelearn/data/level1/meta_train/positives.csv',
                             normal_csv_path='/deep/group/activelearn/data/level1/meta_train/no_findings.csv',
                             unlabeled_pool_size=UNLABELED_POOL_SIZE,
                             query_set_size=QUERY_SET_SIZE,
                             conditions_used=["Cardiomegaly", "Consolidation"],
                             num_tasks=10,
                             deterministic=True)

    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=10, collate_fn=collate_fn)

    if args.mode == 'cosine':
        test_score_mult2 = meanPred(labels = torch.tensor([[1,1],[1,0],[0,1],[0,0]]))
        test_score = meanPred()
    else:
        test_score_mult2 = meanPred(mode = "euclidean",labels = torch.tensor([[1,1],[1,0],[0,1],[0,0]]))
        test_score = meanPred(mode = "euclidean")

    for i, data in enumerate(loader):

        data['pool_labels'] = torch.transpose(data['pool_labels'],1,2)
        data['query_labels'] = torch.transpose(data['query_labels'],1,2)
        # create random mask
        selected = np.zeros((data['pool_labels'].shape[0],data['pool_labels'].shape[1]))

        for k in range(data['pool'].shape[0]):
            np.random.seed(k+7)
            selected_index = np.random.choice(range(UNLABELED_POOL_SIZE),100,replace = False)
            selected[k,selected_index] = 1
        data['mask'] = torch.Tensor(selected)


        print("------------Score: multi method 2------------------------------")
        y_score = test_score_mult2(data)
        print("According score is:")
        for i in range(y_score.shape[0]):
#             auc = roc_auc_score(data["query_labels"][i,:],y_score[i,:])
#             print("%.4f"%(auc))
            for j in range(y_score.shape[2]):
                auc = roc_auc_score(data["query_labels"][i,:,j],y_score[i,:,j])
                print("cond %d: %.4f"%(j,auc))

        if i >= 0:
            break

    #### End of original testing section

    print("************* START OF SIMPLE BASELINE TESTING **************")

    #### Start of baseline accuracy calculation:

    loader = torch.utils.data.DataLoader(dataset=dset, batch_size=50, collate_fn=collate_fn)

    sample_list = [10,20,40,80,100,200,1000]

    for i, data in enumerate(loader):

        data['pool_labels'] = torch.transpose(data['pool_labels'],1,2)
        data['query_labels'] = torch.transpose(data['query_labels'],1,2)

        for j in range(len(sample_list)):

            print("Now sampling: %d X-rays.." %(sample_list[j]))
            # create random mask
            selected = np.zeros((data['pool_labels'].shape[0],data['pool_labels'].shape[1]))

            for k in range(data['pool'].shape[0]):
                np.random.seed(k+j+5)
                selected_index = np.random.choice(range(UNLABELED_POOL_SIZE),sample_list[j],replace = False)
                selected[k,selected_index] = 1

            data['mask'] = torch.Tensor(selected)

            print("------------Score: multi method 2------------------------------")
            y_score = test_score_mult2(data)
            auc = np.zeros((y_score.shape[0],y_score.shape[2]))
            for i in range(y_score.shape[0]):
                for j in range(y_score.shape[2]):
                    auc_i = roc_auc_score(data['query_labels'][i,:,j],y_score[i,:,j])
                    auc[i][j] = auc_i
            ave_auc = np.mean(auc,axis = 0)
            std_auc = np.std(auc,axis = 0)
            print("AUC Score mean: ")
            print(ave_auc)
            print("AUC Score std: ")
            print(std_auc)
            print("\n")

    #### End of baseline accuracy calculation

    print("************* START OF WEIGHT TESTING **************: multiclass not available")
    exit(1)
    for i, data in enumerate(loader):

        weight_pos = data["pool_labels"]
        weight_neg = torch.zeros((data["pool_labels"].shape[0],data["pool_labels"].shape[1]))
        weight_neg = torch.masked_fill(weight_neg,(weight_pos == 0), 1.0)

        num_pos = torch.sum(data["pool_labels"],dim = 1,keepdim = True)
        num_neg = torch.sub(data["pool_labels"].shape[1],num_pos)

        weight_pos = torch.div(weight_pos,num_pos)
        weight_neg = torch.div(weight_neg,num_neg)

        weight = torch.stack([weight_pos,weight_neg],axis = 2)


        print("weight of shape:")
        print(weight.shape)

        data["mask"] = torch.ones(data["pool_labels"].shape)

        print("selecting all samples with mask == 1 everywhere and average equally (old structure)")
        y_score = test_score(data)
        print("According score is:")
        for i in range(y_score.shape[0]):
            auc = roc_auc_score(data['query_labels'][i,:],y_score[i,:])
            print("%.4f"%(auc))

        print("selecting all samples with a weight parameter")
        y_score = test_score(data,weight)
        print("According score is:")
        for i in range(y_score.shape[0]):
            auc = roc_auc_score(data['query_labels'][i,:],y_score[i,:])
            print("%.4f"%(auc))

        if i >= 0:
            break


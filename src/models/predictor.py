# This file contains predictors of our model

# Currently contains: average-based predictor

# Update on Dec. 01, 2020:
# 1. add multi-label method 1 and 2 to predictor (excluding weight section)
# 2. multi-label method 1 and 2 pass toy testing
# 3. remove prediction mode as can be derived from score
# 4. remove verbose mode as results verified for binary case
# 5. add verbose mode to test multi-label cases

# ToDo from Dec. 01, 2020:
# 1. better annotations - hard to understand, some are incorrect
# 2. multi-label method 1 and 2 not yet tested on real data
# 3. implement multi-label when using weight
# 4. optimize code and annotation, remove redundancy

############# IMPORT SECTION ##################
import numpy as np
import torch
from scipy.spatial.distance import pdist,squareform
from scipy.stats import kurtosis, skew, wasserstein_distance
from sklearn.metrics import roc_auc_score

############## Average-based Cosine Similarity ################

class meanPred(torch.nn.Module):
    """This is the predictor using average-based distance metrics."""

    def __init__(self,mode = "cosine", save_metrics = True, verbose = False, save_frontal = True, save_lateral = False):
        """
        Initialize meanPred class

        Args:
            mode (string): choose from: {"cosine", "euclidean"}
            save_metrics (bool): whether to save distance metrics for interpretation
            verbose (bool): for testing purpose, support only multi-label method
            save_frontal (bool): whether to save frontal and lateral auc differently
            save_lateral (bool): whether to save lateral with frontal or just the frontal
        """
        super(meanPred,self).__init__()

        self.save_metrics = save_metrics
        self.mode = mode
        self.verbose = verbose
        self.save_frontal = save_frontal
        self.save_lateral = save_lateral

        if self.save_metrics:
            self.dist = {}
            self.FLdist = {}
            self.dist["max_l2"] = []
            self.dist["mean_l2"] = []
            self.dist["range_l2"] = []
            self.dist["max_fro"] = []
            self.dist["mean_fro"] = []
            self.dist["range_fro"] = []
            self.dist["max_fro_cos"] = []
            self.dist["mean_fro_cos"] = []
            self.dist["range_fro_cos"] = []
            self.dist["max_l1"] = []
            self.dist["mean_l1"] = []
            self.dist["max_inf"] = []
            self.dist["mean_inf"] = []
            self.dist["max_cos"] = []
            self.dist["mean_cos"] = []
            self.dist["range_cos"] = []
            self.dist["age"] = []
            self.dist["sex"] = []
            self.dist["laterality"] = []
            self.raw_dist = {}
            self.raw_dist["l2"] = []
            self.raw_dist["l1"] = []
            self.raw_dist["inf"] = []
            self.raw_dist["cos"] = []
            self.raw_dist["age"] = []

        if self.save_frontal:
            self.auc = {}
            self.auc["overall"] = []
            self.auc["frontal"] = []
            if self.save_lateral:
                self.auc["lateral"] = []
            self.auc["index"]=[]

    def clear(self):
        if self.save_metrics:
            self.dist["max_l2"] = []
            self.dist["mean_l2"] = []
            self.dist["range_l2"] = []
            self.dist["max_fro"] = []
            self.dist["mean_fro"] = []
            self.dist["range_fro"] = []
            self.dist["max_fro_cos"] = []
            self.dist["mean_fro_cos"] = []
            self.dist["range_fro_cos"] = []
            self.dist["max_l1"] = []
            self.dist["mean_l1"] = []
            self.dist["max_inf"] = []
            self.dist["mean_inf"] = []
            self.dist["max_cos"] = []
            self.dist["mean_cos"] = []
            self.dist["range_cos"] = []
            self.dist["age"] = []
            self.dist["sex"] = []
            self.dist["laterality"] = []
            self.raw_dist["l2"] = []
            self.raw_dist["l1"] = []
            self.raw_dist["inf"] = []
            self.raw_dist["cos"] = []
            self.raw_dist["age"] = []
        if self.save_frontal:
            self.auc["overall"] = []
            self.auc["frontal"] = []
            if self.save_lateral:
                self.auc["lateral"] = []
            self.auc["index"] = []


    def __computeMetrics(self,pool,query):
        """
        Helper function to calculate cosine similarities or euclidean distances.

        Args:
        pool (Tensor): A tensor of shape (num_tasks, num_in_pool, embedding_dim)
        query (Tensor): A tensor of shape (num_tasks, num_in_query, embedding_dim)

        Return:
        metric (Tensor): A tensor of shape (num_tasks,num_in_query, num_in_pool)
        """

        if self.mode == "cosine":

            # Get norm of pool set: output [num_tasks, 1, num_in_pool]
            train_norm = torch.norm(pool,dim=2,keepdim=True)
            train_norm = torch.transpose(train_norm,1,2)

            # Get norm of test set: output [num_tasks, num_in_query, 1]
            test_norm = torch.norm(query,dim=2,keepdim=True)

            # Get denominator for calculating cosine similarity:
            # output [num_tasks, num_in_query, num_in_pool]
            deno = torch.matmul(test_norm, train_norm)

            # Get numerator for calculating cosine similarity:
            # output [num_tasks, num_in_query, num_in_pool]
            nume = torch.matmul(query,torch.transpose(pool,1,2))

            # Get cosine similarity: output [num_tasks, num_in_query, num_in_pool]
            metric = torch.div(nume,deno)

        else:

            #### reshape test set: output [num_tasks, num_in_query, 1, embedding_dim]
            Stest = torch.reshape(query,(query.shape[0],query.shape[1],1,query.shape[2]))

            #### reshape train set: output [num_tasks, 1, num_in_query, embedding_dim]
            Strain = torch.reshape(pool,(pool.shape[0], 1, pool.shape[1],pool.shape[2]))

            #### calculate pairwise differences: output [num_tasks, num_in_query, num_in_pool, embedding_dim]
            pairwise_diff = torch.sub(Stest,Strain)

            #### calculate L2-norm: output [num_tasks, num_in_query, num_in_pool]
            metric = torch.norm(pairwise_diff,p = 2, dim = 3)

        return metric

    def __getMask(self, pool_labels, label, mask = None):
        """
        compute the non-positive and non-negative masking.

        Args:
        pool_labels (LongTensor): A tensor of shape (num_tasks, num_in_pool) containing labels for pool Xrays
        label (string): can be "positive" or "negative"
        mask (Tensor): A tensor of shape (num_tasks, num_in_pool) indicating selected Xrays
        verbose (bool): print out to check correctness

        Return:
        non_mask (Tensor): A tensor of shape (num_tasks, 1, num_in_pool)
        num_ (Tensor): A tensor of shape (num_tasks, 1, 1)
        """

        # Get a mask for samples
        # output [num_tasks, num_in_pool]
        if mask is None:
            if label == "positive":
                not_mask = torch.logical_not(pool_labels == 1)
            else:
                not_mask = torch.logical_not(pool_labels == 0)
        else:
            if label == "positive":
                not_mask = torch.logical_or((mask == 0),(pool_labels == 0))
            else:
                not_mask = torch.logical_or((mask == 0),(pool_labels == 1))

        # Reshape the not_mask for broadcasting
        # output [num_tasks, 1, num_in_pool]
        not_mask = torch.reshape(not_mask,(not_mask.shape[0],1,not_mask.shape[1]))

        # Get the num_ for later calculation of mean
        # output [num_tasks, 1, 1]
        num_ = torch.sum((not_mask==False),dim=2,keepdim=True)

        # Check the num of samples is not 0
        if torch.any(torch.eq(num_,0)):
            print("WARNING! No "+label+" samples selected!!!!!!!")

        return not_mask, num_


    def __getAvg(self, metric,mask,num):
        """
        Calculate average of the metric given a mask and a number.

        Args:
        metric (Tensor): A tensor of shape (num_tasks,num_in_query,num_in_pool)
        mask (Tensor): A tensor of shape (num_tasks, 1, num_in_pool) to mask to 0
        num (Tensor): A tensor of shape (num_tasks, 1, 1) which is the total num to divide

        Return:
        average (Tensor): A tensor of shape (num_tasks, num_in_query) as the average of not_masked samples
        """

        # Mask metric to 0 for where mask == 1
        # output [num_tasks, num_in_query, num_in_pool]
        met_ = torch.masked_fill(metric,mask,0.0)

        # Get the sum of metrics for all non-zero samples
        # output [num_tasks, num_in_query, 1]
        sum_met = torch.sum(met_,dim=2,keepdim=True)

        # Get the average of metrics for all non-zero samples
        # output [num_tasks, num_in_query, 1]
        ave_met = torch.div(sum_met,num)

        # Handle possible dividing-by-0 cases
        # output [num_tasks, num_in_query, 1]
        ave_met = torch.masked_fill(ave_met,ave_met.isnan(),0.0)

        return ave_met


    def calculate_statistics(self,x,idx):
        emb = x.cpu().numpy()[:,:512]
        is_frontal = x.cpu().numpy()[:,514]
        idx_frontal = np.argwhere(is_frontal == 1).flatten()
        l2_met = pdist(emb,'euclidean')
        l1_met = pdist(emb,'minkowski',p=1.)
        inf_met = pdist(emb,'chebyshev')
        cos_met = pdist(emb,'cosine')
        fro_met = pdist(emb[idx_frontal,:],'euclidean')
        fro_cos_met = pdist(emb[idx_frontal,:],'cosine')
        self.dist["max_fro"].append(np.max(fro_met))
        self.dist["mean_fro"].append(np.mean(fro_met))
        self.dist["range_fro"].append(np.max(fro_met)-np.min(fro_met))
        self.dist["max_l2"].append(np.max(l2_met))
        self.dist["mean_l2"].append(np.mean(l2_met))
        self.dist["range_l2"].append(np.max(l2_met)-np.min(l2_met))
        self.dist["max_fro_cos"].append(np.max(fro_cos_met))
        self.dist["mean_fro_cos"].append(np.mean(fro_cos_met))
        self.dist["range_fro_cos"].append(np.max(fro_cos_met)-np.min(fro_cos_met))
        self.dist["max_l1"].append(np.max(l1_met))
        self.dist["mean_l1"].append(np.mean(l1_met))
        self.dist["max_inf"].append(np.max(inf_met))
        self.dist["mean_inf"].append(np.mean(inf_met))
        self.dist["max_cos"].append(np.max(cos_met))
        self.dist["mean_cos"].append(np.mean(cos_met))
        self.dist["range_cos"].append(np.max(cos_met)-np.min(cos_met))
        self.raw_dist["l2"].append(l2_met)
        self.raw_dist["l1"].append(l1_met)
        self.raw_dist["inf"].append(inf_met)
        self.raw_dist["cos"].append(cos_met)
        self.raw_dist["age"].append(x.cpu().numpy()[:,513])
        self.dist["sex"].append(np.mean(x.cpu().numpy()[:,512]))
        self.dist["laterality"].append(np.mean(x.cpu().numpy()[:,514]))
        self.dist["age"].append(np.mean(x.cpu().numpy()[:,513]))

    def get_wasserstein(self,otherPred,mean_age,std_age):
        wa_dist = {}
        wa_dist["l2"] = [wasserstein_distance(self.raw_dist["l2"][i], otherPred.raw_dist["l2"][i])
                         for i in range(len(self.raw_dist["l2"]))]
        wa_dist["l1"] = [wasserstein_distance(self.raw_dist["l1"][i], otherPred.raw_dist["l1"][i]) 
                         for i in range(len(self.raw_dist["l1"]))]
        wa_dist["inf"] = [wasserstein_distance(self.raw_dist["inf"][i], otherPred.raw_dist["inf"][i]) 
                          for i in range(len(self.raw_dist["inf"]))]
        wa_dist["cos"] = [wasserstein_distance(self.raw_dist["cos"][i], otherPred.raw_dist["cos"][i]) 
                          for i in range(len(self.raw_dist["cos"]))]
        wa_dist["age"] = [wasserstein_distance(self.raw_dist["age"][i], otherPred.raw_dist["age"][i]) 
                         for i in range(len(self.raw_dist["age"]))]
        wa_dist["actualAge"] = [wasserstein_distance(self.raw_dist["age"][i]*std_age+mean_age, 
                                                     otherPred.raw_dist["age"][i]*std_age+mean_age)
                                for i in range(len(self.raw_dist["age"]))]

        return wa_dist

    def get_auc(self,pred,true,is_frontal,index):
        y_true = true.cpu().numpy()
        y_pred = pred.cpu().numpy()
        fro = is_frontal.cpu().numpy()
        idx_frontal = np.argwhere(fro == 1).flatten()
        idx_lateral = np.argwhere(fro == 0).flatten()
        if len(np.argwhere(y_true[idx_frontal] == 1).flatten()) == 0 or len(np.argwhere(y_true[idx_frontal] == 1).flatten()) == len(idx_frontal):
            print("Warning! only one class in frontal!")
            print("%d in frontal"%(len(idx_frontal)))
        else:
            if self.save_lateral:
                if len(np.argwhere(y_true[idx_lateral] == 1).flatten()) == 0 or len(np.argwhere(y_true[idx_lateral] == 1).flatten()) == len(idx_lateral):
                    print("Warning! only one class in lateral!")
                    print("%d in lateral"%(len(idx_lateral)))
                else:
                    self.auc["lateral"].append(roc_auc_score(true[idx_lateral],pred[idx_lateral]))
            self.auc["frontal"].append(roc_auc_score(true[idx_frontal],pred[idx_frontal]))
            self.auc["overall"].append(roc_auc_score(true,pred))
            self.auc["index"].append(index)

    def forward_selected(self,selected, selected_labels, query, query_labels, reward = False, weight = None, mask = None):
        """ 
        Forward pass through selector but using only K selected X-rays from pool.

        Args:
        selected (Tensor): A tensor of shape (num_tasks, K, embedding_dim) which is K selected X-rays from pool
        selected_labels (LongTensor): A tensor of shape (num_tasks, K) containing labels for the selected X-rays
        query (Tensor): A tensor of test X-rays, shape (num_tasks, num_test_examples, embedding_dim)
        query_labels (LongTensor): A tensor of shape (num_tasks, num_test_examples) containing labels for query set
        weight (Tensor): A tensor of shape (num_tasks, num_in_pool, 2) for soft/weighted average

        Returns:
            score (tensor): auroc scores to query samples: [num of tasks, num of test samples], or:
            prediction (tensor): predictions to query samples: [num of tasks, num of test samples]
            num_positive: the number of selected positive Xrays
	"""

        if self.verbose:
            print("----------------------------")
            print("Selected size:")
            print(selected.shape)
            print("Query size:")
            print(query.shape)
            print("Selected labels:")
            print(selected_labels)
            print("Query labels:")
            print(query_labels)
            if mask is not None:
                print(mask)
            if weight is not None:
                print(weight)

        # Get the metric
        metric = self.__computeMetrics(selected,query)

        if weight is None:

            # Get a mask for samples that are not positive
            # output [num_tasks, 1, num_in_pool], [num_tasks,1,1]
            not_positive_mask, num_positive = self.__getMask(selected_labels,"positive",mask = mask)

            # Get a mask for samples that are not negative
            # output [num_tasks, 1, num_in_pool], [num_tasks,1,1]
            not_negative_mask, num_negative = self.__getMask(selected_labels,"negative",mask = mask)

            if self.save_metrics:
                num_pos = num_positive.cpu().numpy().flatten()
                num_neg = num_negative.cpu().numpy().flatten()
                if mask is None:
                    for i in range(selected.shape[0]):
                        self.calculate_statistics(selected[i,:,:],i)
                else:
                    for i in range(selected.shape[0]):
                        curr_mask = np.argwhere(mask.cpu().numpy()[i,:] == 1).flatten()
                        self.calculate_statistics(selected[i,curr_mask,:],i)

            # Compute average of metrics for positive selected Xrays
            # output [num_tasks, num_in_query, 1]
            ave_met_positive = self.__getAvg(metric,not_positive_mask,num_positive)

            # Compute average of metrics for negative selected Xrays
            # output [num_tasks, num_in_query, 1]
            ave_met_negative = self.__getAvg(metric,not_negative_mask,num_negative)

            if self.mode == "cosine":
                score = torch.sub(ave_met_positive,ave_met_negative)
            else:
                score = torch.sub(ave_met_negative,ave_met_positive)

            if self.save_frontal:
                for i in range(selected.shape[0]):
                    self.get_auc(score[i,:],query_labels[i,:],query[i,:,514],i)

            if reward:
                return torch.mean(score,dim = 2)
            else:
                return score

        # With weight, only support score calculation!!
        else:

            # The score has shape [num_tasks,num_in_query,2]
            score = torch.matmul(metric,weight)
            if self.mode == "cosine":
                score = torch.matmul(score,torch.Tensor([[[1.0],[-1.0]]]))
            else:
                score = torch.matmul(score,torch.Tensor([[[-1.0],[1.0]]])) 
            # The score has shape [num_tasks,num_in_query,1]
            return score

    def forward(self,x,weight = None):
        """
        Calculate predictions of query samples to each individual task in the batch.

        Args:
            x (dict): what is loaded through the dataloader
                1) x["pool"]: a batch of unlabeled X-rays: [num of tasks, num of training samples,feature dim]
                2) x["query"]: a batch of test X-rays: [num of tasks, num of test samples, feature dim]
                3) x["mask"]: a mask to indicate selected samples: [num of tasks, num of training samples]
                4) x["pool_labels"]: labels for 1): [num of tasks, num of training samples]
                5) x["query_labels"]: labels for 2): [num of tasks, num of test samples]

        Return:
            score (tensor): auroc scores to query samples: [num of tasks, num of test samples], or:
            prediction (tensor): predictions to query samples: [num of tasks, num of test samples]
            num_positive: the number of selected positive Xrays
        """

        return self.forward_selected(x['pool'],x['pool_labels'],x['query'],x['query_labels'],mask = x['mask'])

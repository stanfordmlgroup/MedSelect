import numpy as np
import pandas as pd
import constants
import argparse
from datasets.random_task_dataset import MEAN_AGE, STD_AGE
from scipy.stats import ttest_ind

best_models = {10:'model_epoch5_iter30',
               20:'model_epoch5_iter150',
               40:'model_epoch5_iter30',
               80:'model_epoch4_iter150',
               100:'model_epoch5_iter90',
               200:'model_epoch1_iter90',
               400:'model_epoch1_iter90'}

parser = argparse.ArgumentParser()
parser.add_argument('--k',dest='k',default=10,help='choose from 10,20,40,80,100')
arg = parser.parse_args()

arg.k = int(arg.k)

filer = f'/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/non_holdout/{arg.k}_stat.csv'
filerh = f"/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/holdout/{arg.k}_stat.csv"
filewar = f"/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/non_holdout/{arg.k}_wacomp.csv"
filewarh = f"/deep/group/activelearn/experiments/level1/meta_tst/random_baseline/cosine_auc/holdout/{arg.k}_wacomp.csv"
filec = f"/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/non_holdout/{arg.k}_stat.csv"
filech = f"/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/holdout/{arg.k}_stat.csv"
filewa = f"/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/non_holdout/{arg.k}_wacomp.csv"
filewah = f"/deep/group/activelearn/experiments/level1/meta_tst/cluster_baseline/cosine_auc/holdout/{arg.k}_wacomp.csv"
filel = f"/deep/group/activelearn/lstm_selector/k={arg.k}/ckpt/only_img/{best_models[arg.k]}_stat.csv"
filelh = f"/deep/group/activelearn/lstm_selector/k={arg.k}/ckpt/only_img/{best_models[arg.k]}_holdout_stat.csv"
filewal = f"/deep/group/activelearn/lstm_selector/k={arg.k}/ckpt/only_img/{best_models[arg.k]}_wacomp.csv"
filewalh = f"/deep/group/activelearn/lstm_selector/k={arg.k}/ckpt/only_img/{best_models[arg.k]}_holdout_wacomp.csv"

random = pd.read_csv(filer)
random_holdout = pd.read_csv(filerh)

random_rand = pd.read_csv(filewar)
random_randh = pd.read_csv(filewarh)

cluster = pd.read_csv(filec)
cluster_holdout = pd.read_csv(filech)

cluster_rand = pd.read_csv(filewa)
cluster_randh = pd.read_csv(filewah)

lstm = pd.read_csv(filel)
lstm_holdout = pd.read_csv(filelh)

lstm_rand = pd.read_csv(filewal)
lstm_randh = pd.read_csv(filewalh)

#print(data.head())

#print("-----label distributions-------")
#randnh = random["labeldist"].mean()
#randh = random_holdout["labeldist"].mean()
#clustnh = cluster["labeldist"].mean()
#clusth = cluster_holdout["labeldist"].mean()
#lstmnh = lstm["labeldist"].mean()
#lstmh = lstm_holdout["labeldist"].mean()
#pvalue_rand = ttest_ind(random["labeldist"],lstm["labeldist"])[1]
#pvalue_randh = ttest_ind(random_holdout["labeldist"],lstm_holdout["labeldist"])[1]
#pvalue_clust = ttest_ind(cluster["labeldist"],lstm["labeldist"])[1]
#pvalue_clusth = ttest_ind(cluster_holdout["labeldist"],lstm_holdout["labeldist"])[1]
#print("Non-Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f) \\\\"%(lstmnh,clustnh,pvalue_clust,randnh,pvalue_rand))
#print("Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f)"%(lstmh,clusth,pvalue_clusth,randh,pvalue_randh))
#print("\n")

print("-----sex distributions-------")
randnh = random["sex"].mean()
randh = random_holdout["sex"].mean()
clustnh = cluster["sex"].mean()
clusth = cluster_holdout["sex"].mean()
lstmnh = lstm["sex"].mean()
lstmh = lstm_holdout["sex"].mean()
pvalue_rand = ttest_ind(random["sex"],lstm["sex"])[1]
pvalue_randh = ttest_ind(random_holdout["sex"],lstm_holdout["sex"])[1]
pvalue_clust = ttest_ind(cluster["sex"],lstm["sex"])[1]
pvalue_clusth = ttest_ind(cluster_holdout["sex"],lstm_holdout["sex"])[1]
print("Non-Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f) \\\\"%(lstmnh,clustnh,pvalue_clust,randnh,pvalue_rand))
print("Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f)"%(lstmh,clusth,pvalue_clusth,randh,pvalue_randh))
print("\n")

print("-----lateral distributions-------")
randnh = random["laterality"].mean()
randh = random_holdout["laterality"].mean()
clustnh = cluster["laterality"].mean()
clusth = cluster_holdout["laterality"].mean()
lstmnh = lstm["laterality"].mean()
lstmh = lstm_holdout["laterality"].mean()
pvalue_rand = ttest_ind(random["laterality"],lstm["laterality"])[1]
pvalue_randh = ttest_ind(random_holdout["laterality"],lstm_holdout["laterality"])[1]
pvalue_clust = ttest_ind(cluster["laterality"],lstm["laterality"])[1]
pvalue_clusth = ttest_ind(cluster_holdout["laterality"],lstm_holdout["laterality"])[1]
print("Non-Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f) \\\\"%(lstmnh,clustnh,pvalue_clust,randnh,pvalue_rand))
print("Holdout & %.4f & %.4f (%.4f) & %.4f (%.4f)"%(lstmh,clusth,pvalue_clusth,randh,pvalue_randh))
print("\n")

print("-----Original Pairwise Distance: Non-Holdout-------")
randmean = random["mean_l2"].mean()
randstd = random["max_l2"].mean()
clustmean = cluster["mean_l2"].mean()
cluststd = cluster["max_l2"].mean()
lstmmean = lstm["mean_l2"].mean()
lstmstd = lstm["max_l2"].mean()
print("Mean & %.4f & %.4f & %.4f \\\\"%(randmean,clustmean,lstmmean))
print("Max & %.4f & %.4f & %.4f \\\\"%(randstd,cluststd,lstmstd))
print("\n")

print("-----Original Pairwise Distance: Holdout-------")
randmean = random_holdout["mean_l2"].mean()
randstd = random_holdout["max_l2"].mean()
clustmean = cluster_holdout["mean_l2"].mean()
cluststd = cluster_holdout["max_l2"].mean()
lstmmean = lstm_holdout["mean_l2"].mean()
lstmstd = lstm_holdout["max_l2"].mean()
print("Mean & %.4f & %.4f & %.4f \\\\"%(randmean,clustmean,lstmmean))
print("Max & %.4f & %.4f & %.4f \\\\"%(randstd,cluststd,lstmstd))
print("\n")

#print("-----Original Age: Non-Holdout-------")
randmean = (random["age"]*STD_AGE+MEAN_AGE).mean()
clustmean = (cluster["age"]*STD_AGE+MEAN_AGE).mean()
lstmmean = (lstm["age"]*STD_AGE+MEAN_AGE).mean()
pvalue_rand = ttest_ind(random["age"]*STD_AGE+MEAN_AGE,lstm["age"]*STD_AGE+MEAN_AGE)[1]
pvalue_clust = ttest_ind(cluster["age"]*STD_AGE+MEAN_AGE,lstm["age"]*STD_AGE+MEAN_AGE)[1]
print("Mean & %.4f & %.4f (%.4f) & %.4f (%.4f) \\\\"%(lstmmean,clustmean,pvalue_clust,randmean,pvalue_rand))
print("\n")

print("-----Original Age: Holdout-------")
randmean = random_holdout["age"].mean()
clustmean = cluster_holdout["age"].mean()
lstmmean = lstm_holdout["age"].mean()
print("Mean & %.4f & %.4f & %.4f \\\\"%(randmean,clustmean,lstmmean))
print("\n")

print("-----Wasserstein for Pairwise Distance: l2--------")
randnh = random_rand["l2"].mean()
randh = random_randh["l2"].mean()
clustnh = cluster_rand["l2"].mean()
clusth = cluster_randh["l2"].mean()
lstmnh = lstm_rand["l2"].mean()
lstmh = lstm_randh["l2"].mean()
print("Non-Holdout & %.4f & %.4f & %.4f \\\\"%(randnh,clustnh,lstmnh))
print("Holdout & %.4f & %.4f & %.4f"%(randh,clusth,lstmh))
print("\n")

print("-----Wasserstein for Pairwise Distance: age--------")
randnh = random_rand["actualAge"].mean()
randh = random_randh["actualAge"].mean()
clustnh = cluster_rand["actualAge"].mean()
clusth = cluster_randh["actualAge"].mean()
lstmnh = lstm_rand["actualAge"].mean()
lstmh = lstm_randh["actualAge"].mean()
print("Non-Holdout & %.4f & %.4f & %.4f \\\\"%(randnh,clustnh,lstmnh))
print("Holdout & %.4f & %.4f & %.4f"%(randh,clusth,lstmh))
print("\n")


UNLABELED_POOL_SIZE = 1000 #number of X-rays in unlabeled pool for each task
UNLABELED_POS_FRAC = 0.5   #portion of unlabeled pool X-rays which are positive for the condition
QUERY_SET_SIZE = 100       #number of X-rays in the labeled testing portion of each task
QUERY_POS_FRAC = 0.5       #portion of query set X-rays which are positive for the condition
NUM_META_TRAIN_TASKS = 10000
NUM_META_TEST_TASKS = 1000

HOLD_OUT = ["Atelectasis", "Edema"]
NON_HOLD_OUT = ["Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax",
                 "Enlarged Cardiomediastinum", "Cardiomegaly"]
ALL_COND = ["Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax",
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Atelectasis", "Edema"]

USE_ASL = False  # Whether to use Age Sex Laterality as input to selector

#Selector training parameters
LEARNING_RATE = 1e-4
K = [10, 20, 40, 80, 100, 200]  #how many X-rays to select from pool
NUM_EPOCHS = 5
BATCH_SIZE = 64

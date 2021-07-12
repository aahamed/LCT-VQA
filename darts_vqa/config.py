import torch

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# seed
SEED=10
# stats dir
ROOT_STATS_DIR='./experiment_data'

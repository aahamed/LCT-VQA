import torch

ROOT_STATS_DIR='./experiment_data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# options are 'fixed', 'differentiable'
ARCH_TYPE = 'fixed'

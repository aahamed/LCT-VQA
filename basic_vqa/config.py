'''
config.py
Holds configuration parameters for training
'''
import torch

# input directory for visual question answering
INPUT_DIR='../../data/vqa/inputs64'
# maximum length of question. the length in the VQA dataset = 26
MAX_QST_LEN=30
# maximum number of answers
MAX_NUM_ANS=10
# embedding size of feature vector
IMG_EMBED_SIZE=512
# embedding size of word
WORD_EMBED_SIZE=300
# number of layers of the RNN(LSTM)
LSTM_NUM_LAYERS=1
# hidden_size in the LSTM
LSTM_HIDDEN_SIZE=512
# learning rate for training
LEARNING_RATE=0.001
# momentum for sgd optimizer
MOMENTUM=0.99
# weight_decay for sgd optimizer
WEIGHT_DECAY=0
# period of learning rate decay
STEP_SIZE=10
# multiplicative factor of learning rate decay
GAMMA=0.1
# number of processes working on cpu 
NUM_WORKERS=8
# learning rate for arch encoding (darts)
ARCH_LEARNING_RATE=6e-4
# weight decay for arch encoding (darts)
ARCH_WEIGHT_DECAY=1e-3
# gradient clipping
GRAD_CLIP=5
# temperature for softmax
TEMPERATURE=0.1
# batch size
BATCH_SIZE=64
# number of epochs
NUM_EPOCHS=30
# fraction of training data to use ( for debugging )
TRAIN_PORTION=1.
# experiment name
EXP_NAME='default_exp'
# resume experiment <exp_name> from last checkpoint
RESUME=False
# seed
SEED=10
# stats dir
ROOT_STATS_DIR='./experiment_data'
# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# options are 'fixed', 'darts'
ARCH_TYPE = 'darts'
# report frequence
REPORT_FREQ = 10 if ARCH_TYPE == 'darts' else 100
# architecture update frequency
ARCH_UPDATE_FREQ = 2000
# architecture update frequency min
ARCH_UPDATE_FREQ_MIN = 100
# architecture update frequency decay
GAMMA_ARCH = 0.5
# skip stage 2
SKIP_STAGE2 = False

def update_config( args ):
    global BATCH_SIZE, NUM_EPOCHS, TRAIN_PORTION, \
            EXP_NAME, RESUME, NUM_WORKERS, ARCH_TYPE,\
            SKIP_STAGE2, ARCH_UPDATE_FREQ
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    TRAIN_PORTION = args.train_portion
    EXP_NAME = args.exp
    RESUME = args.resume
    NUM_WORKERS = args.num_workers
    ARCH_TYPE = args.arch_type
    SKIP_STAGE2 = args.skip_stage2
    ARCH_UPDATE_FREQ = args.arch_update_freq

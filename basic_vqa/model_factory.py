import torch.optim as optim
from torch.optim import lr_scheduler
from models_lct import VqaModel as LctVqaModel
from models import VqaModel
from constants import ARCH_TYPE
import config

def get_ef_model( dataset ):
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    model = LctVqaModel(
        embed_size=config.IMG_EMBED_SIZE,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=config.WORD_EMBED_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        hidden_size=config.LSTM_HIDDEN_SIZE)
    return model

def get_ef_optimizer( model ):
    optimizer = optim.Adam( model.parameters(), lr=config.LEARNING_RATE )
    return optimizer

def get_ef_scheduler( optimizer ):
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    return scheduler

def get_w_model( dataset ):
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    model = VqaModel(
        embed_size=config.IMG_EMBED_SIZE,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=config.WORD_EMBED_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        hidden_size=config.LSTM_HIDDEN_SIZE)
    return model

def get_w_optimizer( model ):
    optimizer = optim.Adam( model.parameters(), lr=config.LEARNING_RATE )
    return optimizer

def get_w_scheduler( optimizer ):
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    return scheduler

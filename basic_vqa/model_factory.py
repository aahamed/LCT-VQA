import torch.optim as optim
from torch.optim import lr_scheduler
from models_lct import VqaModel as LctVqaModel
from models import VqaModel
from constants import ARCH_TYPE

def get_ef_model( args, dataset ):
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    model = LctVqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size)
    return model

def get_ef_optimizer( model, args ):
    optimizer = optim.Adam( model.parameters(), lr=args.learning_rate )
    return optimizer

def get_ef_scheduler( optimizer, args ):
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    return scheduler

def get_w_model( args, dataset ):
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size)
    return model

def get_w_optimizer( model, args ):
    optimizer = optim.Adam( model.parameters(), lr=args.learning_rate )
    return optimizer

def get_w_scheduler( optimizer, args ):
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    return scheduler

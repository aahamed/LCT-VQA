import torch.optim as optim
from torch.optim import lr_scheduler
from vqa_model import VqaModel
from pcdarts.architect_vqa import Architect

def get_vqa_model( args, dataset ):
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    model = VqaModel( args.embed_size, qst_vocab_size,
        ans_vocab_size, args.word_embed_size, args.num_layers,
        args.hidden_size, args.arch_type )
    return model

def get_optimizer( args, model ):
    optimizer = optim.Adam( model.parameters(), lr=args.learn_rate )
    return optimizer

def get_scheduler( args, optimizer ):
    scheduler = lr_scheduler.StepLR( optimizer,
        step_size=args.step_size, gamma=args.gamma )
    return scheduler

def get_architect( args, model ):
    if args.arch_type == 'vgg':
        return None
    elif args.arch_type == 'darts':
        return Architect( model, args )
    else:
        raise Exception( f'Unrecognized arch_type: {args.arch_type}' )

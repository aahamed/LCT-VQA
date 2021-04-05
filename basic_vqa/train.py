import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models_lct import VqaModel
from pcdarts.architect import Architect
from itertools import cycle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    # import pdb; pdb.set_trace()
    log_dir = os.path.join( args.log_dir, args.exp )
    model_dir = os.path.join( args.model_dir, args.exp )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.model_dir, exist_ok=True)

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_portion=args.train_portion)

    dataset = data_loader['train'].dataset.dataset
    qst_vocab_size = dataset.qst_vocab.vocab_size
    ans_vocab_size = dataset.ans_vocab.vocab_size
    ans_unk_idx = dataset.ans_vocab.unk2idx
    # qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    # ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    # ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    start_epoch = 0

    # Load models if resume
    if args.resume:
        # import pdb; pdb.set_trace()
        load_path = os.path.join( model_dir, 'model.ckpt' )
        state_dict = torch.load( load_path )
        model.load_state_dict( state_dict[ 'model' ] )
        optimizer.load_state_dict( state_dict[ 'optimizer' ] )
        scheduler.load_state_dict( state_dict[ 'scheduler' ] )
        start_epoch = state_dict[ 'epoch' ]
    else:
        # ensure log_dir is empty
        # import pdb; pdb.set_trace()
        files = os.listdir( log_dir )
        if files:
            print( f'log dir: {log_dir} not empty. Please delete all files' +
                    ' in dir and rerun train command.' )
            exit( 1 ) 
    
    # import pdb; pdb.set_trace()
    # instantiate architect for architecture search
    architect = Architect(model.img_encoder.darts, args)

    for epoch in range(start_epoch, args.num_epochs):

        for phase in ['train', 'valid']:

            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size
            # batch_step_size = len(data_loader[phase])
            valid_queue_iter = None

            if phase == 'train':
                # scheduler.step()
                model.train()
                # setup validation queue for architecture search
                valid_queue_iter = cycle( iter( data_loader['valid'] ) )
            else:
                model.eval()

            optimizer.zero_grad()
            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                # optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label)

                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        # accumulate gradients for 32 batches
                        if ( batch_idx+1 ) % 32 == 0:
                            # STAGE 1: Update weights of ( E, F ) in encoder-decoder 
                            # model using training data
                            optimizer.step()
                            optimizer.zero_grad()

                            # STAGE 2: Update weights of VqaModel W using
                            # psuedo-generated qa data
                            # TODO

                            # STAGE 3: Architecture search
                            # Update architecture of E in encoder-decoder model 
                            # using validation data
                            batch_sample = next( valid_queue_iter )
                            val_image = batch_sample['image'].to(device)
                            val_question = batch_sample['question'].to(device)
                            val_label = batch_sample['answer_label'].to(device)
                            # import pdb; pdb.set_trace()
                            architect.step( image, question, label,
                                    val_image, val_question, val_label )
                        # END BATCH_IDX IF
                    # END TRAIN IF
                # END GRAD_ENABLED IF

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item()))
            
            # END DATALOADER FOR
            # udpate scheduler
            scheduler.step()
            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice
            # epoch_acc_exp1 = running_corr_exp1.double() / ( 
            #        batch_step_size * args.batch_size )
            # epoch_acc_exp2 = running_corr_exp2.double() / ( 
            #        batch_step_size * args.batch_size )

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            with open(os.path.join(log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
            torch.save({
                        'epoch': epoch+1, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       os.path.join(model_dir, 'model.ckpt'))
            # os.path.join(model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    parser.add_argument('--train_portion', type=float, default=1.,
                        help='fraction of training data to use ( for debugging )' )
    
    parser.add_argument('--exp', type=str, default='default_exp',
                        help='name of experiment' )

    parser.add_argument('--resume', action='store_true',
                        help='resume experiment <exp> from last checkpoint' )
    
    parser.add_argument('--arch_learning_rate', type=float, default=6e-4,
                        help='learning rate for arch encoding (darts)')
    
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        help='weight decay for arch encoding (darts)')
    
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping')

    args = parser.parse_args()

    main(args)

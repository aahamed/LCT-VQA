import os
import argparse
from experiment import Experiment

def main(args):
    exp = Experiment( args )
    exp.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    #parser.add_argument('--log_dir', type=str, default='./logs',
    #                    help='directory for logs.')

    #parser.add_argument('--model_dir', type=str, default='./models',
    #                    help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=512,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')
    
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='momentum for sgd optimizer')
    
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay for sgd optimizer')

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

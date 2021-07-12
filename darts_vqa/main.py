import os
import argparse
from experiment import Experiment

def main(args):
    exp = Experiment( args )
    exp.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--report_freq', type=int, default=100,
                        help='logging frequency')
    
    parser.add_argument('--arch_update_freq', type=int, default=100,
                        help='architecture update frequency')
    
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='max gradient value')
    
    parser.add_argument('--qst_only', action='store_true',
                        help='only train question generation' )
    
    parser.add_argument('--max_qst_len', type=int, default=30,
                        help='max question length')
    
    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='max number of answers to choose from')
    
    parser.add_argument('--learn_rate', type=float, default=0.001,
                        help='learning rate for gradient descent')
    
    parser.add_argument('--arch_learn_rate', type=float, default=6e-4,
                        help='learning rate for gradient descent')
    
    parser.add_argument('--arch_wt_decay', type=float, default=1e-3,
                        help='learning rate for gradient descent')
    
    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay')
    
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='scheduler decay rate')
    
    parser.add_argument('--embed_size', type=int, default=512,
                        help='Image embedding size')
    
    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='word embedding size')
    
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='LSTM hidden state size')
    
    parser.add_argument('--num_layers', type=int, default=1,
                        help='depth of LSTM')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size.')

    parser.add_argument('--train_portion', type=float, default=1.,
                        help='fraction of training data to use ( for debugging )' )
    
    parser.add_argument('--exp', type=str, default='default_exp',
                        help='name of experiment' )

    parser.add_argument('--resume', action='store_true',
                        help='resume experiment <exp> from last checkpoint' )
    
    parser.add_argument('--input_dir', type=str, default='../../data/vqa/inputs64',
                        help='vqa input dir' )
    
    parser.add_argument('--arch_type', type=str, default='vgg',
                        help='architecture of img encoder. Options are [vgg, darts]' )

    
    args = parser.parse_args()

    main(args)

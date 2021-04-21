import os
import argparse
from experiment import Experiment
from config import update_config

def main(args):
    update_config( args )
    exp = Experiment( args )
    exp.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--train_portion', type=float, default=1.,
                        help='fraction of training data to use ( for debugging )' )
    
    parser.add_argument('--exp', type=str, default='default_exp',
                        help='name of experiment' )

    parser.add_argument('--resume', action='store_true',
                        help='resume experiment <exp> from last checkpoint' )
    
    args = parser.parse_args()

    main(args)

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

    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of processes working on cpu.')

    parser.add_argument('--arch_type', type=str, default='fixed-vgg',
                        help='architecture type. Options: [ fixed-vgg, fixed-darts, darts ]')
    
    parser.add_argument('--arch_update_freq', type=int, default=1000,
                        help='architecture update frequency ( default: 1000 )')

    parser.add_argument('--skip_stage2', action='store_true',
                        help='Skip Stage2 of algorithm')
    
    parser.add_argument('--no_pretrain_enc', action='store_true',
                        help='Don\'t use a pretrained encoder')
    
    parser.add_argument('--no_img_enc', action='store_true',
                        help='Don\'t use an image encoder')
    
    parser.add_argument('--qst_only', action='store_true',
                        help='Don\'t use an image encoder')
    
    parser.add_argument('--darts_model_path', type=str, default='./pretrain/imagenet_model.pt',
                        help='path to pretrained darts model ( debugging )')
    
    parser.add_argument('--input_dir', type=str, default='../../data/vqa/inputs64',
                        help='input directory containing vqa data')
    
    args = parser.parse_args()

    main(args)

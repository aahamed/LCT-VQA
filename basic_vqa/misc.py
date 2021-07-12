import torch
import os
import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class VqaStruct( object ):
    '''
    Data structure to hold VQA data
    '''

    def __init__( self, input_dir, data_file='valid.npy' ):
        self.input_dir = input_dir
        self.data_file = data_file
        self.vqa_path = os.path.join( self.input_dir, self.data_file )
        self.vqa = np.load( self.vqa_path, allow_pickle=True )
        self.img_to_qst = defaultdict(list)
        self.build_img_to_qst()

    def build_img_to_qst( self ):
        '''
        Build dict mapping from image name to question tokens.
        Can be used for BLEU score calculation.
        '''
        for i in range( len( self.vqa ) ):
            entry = self.vqa[i]
            self.img_to_qst[ entry['image_name'] ].append( 
                entry['question_tokens'] )

    def qst_cnt_histogram( self ):
        hist = np.zeros( 1000, dtype=int )
        for k, v in self.img_to_qst.items():
            nq = len( v )
            hist[ nq ] += 1
        return hist

    def get_ref_qst( self, img_name ):
        ref_qst = self.img_to_qst[ img_name ]
        assert ref_qst
        return ref_qst

def num_correct( pred, multi_choice ):
        res = torch.stack([(ans == pred) for ans in multi_choice])
        res = res.any(dim=0).sum().item()
        return res

def num_correct_qst( qst_pred, qst ):
    '''
    Number of correct questions
    '''
    qst_pred = qst_pred.argmax( dim=2 )[:, :-1]
    qst = qst[:, 1:]
    not_match = ~( qst == qst_pred )
    err_cnt = not_match.sum( dim=1 )
    acc_0 = ( err_cnt == 0 ).sum().item()
    acc_3 = ( err_cnt <= 3 ).sum().item()
    acc_5 = ( err_cnt <= 5 ).sum().item()
    return acc_0, acc_3, acc_5

def BLEU4(ref_qst, pred_qst):
    return 100 * sentence_bleu( ref_qst, pred_qst,
            smoothing_function=SmoothingFunction().method1 )

def calc_bleu_scores( image_names, pred_qsts, qst_vocab, vqa_struct ):
    pred_qsts = [ qst_vocab.arr2qst(q).split() for q in pred_qsts ]
    N = len( image_names )
    b4 = 0
    for i in range( N ):
        name = image_names[i]
        ref_qst = vqa_struct.get_ref_qst( name )
        pred_qst = pred_qsts[i]
        b4 += BLEU4( ref_qst, pred_qst )
    return b4 / N

def _test_vqa_struct():
    input_dir = '../../data/vqa/inputs64'
    data_file = 'train.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    assert len( vqa_struct.img_to_qst ) == 82783
    data_file = 'valid.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    assert len( vqa_struct.img_to_qst ) == 40504

def _test_bleu4():
    input_dir = '../../data/vqa/inputs64'
    data_file = 'valid.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    import pdb; pdb.set_trace()
    img_name = 'COCO_val2014_000000262148'
    ref_qst = vqa_struct.get_ref_qst( img_name )
    pred_qst = [ 'what', 'is', 'the', 'man', 'holding', '?' ]
    bleu4 = BLEU4( ref_qst, pred_qst )

def main():
    # _test_vqa_struct()
    _test_bleu4()

if __name__ == '__main__':
    main()

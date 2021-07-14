import torch
import os
import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import text_helper

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
        self.img_to_qa = defaultdict(list)
        self.build_img_to_qst()
        self.build_img_to_qa()

    def build_img_to_qst( self ):
        '''
        Build dict mapping from image name to question tokens.
        Can be used for BLEU score calculation.
        '''
        for i in range( len( self.vqa ) ):
            entry = self.vqa[i]
            self.img_to_qst[ entry['image_name'] ].append( 
                entry['question_tokens'] )
    
    def build_img_to_qa( self ):
        '''
        Build dict mapping from image name to qst+ans tokens.
        Can be used for BLEU score calculation of unified model.
        '''
        for i in range( len( self.vqa ) ):
            entry = self.vqa[i]
            qst = entry['question_tokens']
            ans = np.random.choice( entry['valid_answers'] )
            qa = qst + ['<sep>'] + [ans]
            self.img_to_qa[ entry['image_name'] ].append( qa )

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

    def get_ref_qa( self, img_name ):
        ref_qa = self.img_to_qa[ img_name ]
        assert ref_qa
        return ref_qa

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

def calc_bleu_scores_unified( image_names, pred_qas,
        unified_vocab, vqa_struct ):
    pred_qas = [ unified_vocab.arr2qst(qa).split() for qa in pred_qas ]
    N = len( image_names )
    b4 = 0
    for i in range( N ):
        name = image_names[i]
        ref_qa = vqa_struct.get_ref_qa( name )
        pred_qa = pred_qas[i]
        b4 += BLEU4( ref_qa, pred_qa )
    return b4 / N

def extract_answer( qa_str, unified_vocab ):
    # convert idx to words
    qa_str = [ unified_vocab.word_list[ idx ] for idx in qa_str ]
    flag = False
    ans = []
    # extract words between <sep> and <end>
    for word in qa_str:
        if word == '<sep>':
            flag = True
            continue
        if word == '<end>':
            break
        if flag:
            ans.append( word )
    ans = ' '.join( ans )
    return ans

def unified_ans_acc( qa_gt, qa_pred, unified_vocab ):
    assert len( qa_gt ) == len( qa_pred )
    N = len( qa_gt )
    corr = 0
    for i in range( N ):
        ans_gt = extract_answer( qa_gt[i], unified_vocab )
        ans_pred = extract_answer( qa_pred[i], unified_vocab )
        if ans_gt == ans_pred:
            corr += 1
    return corr / N


def _test_vqa_struct():
    input_dir = '../../data/vqa/inputs64'
    data_file = 'train.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    assert len( vqa_struct.img_to_qst ) == 82783
    data_file = 'valid.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    assert len( vqa_struct.img_to_qst ) == 40504
    print( 'Test passed!' )

def _test_bleu4():
    input_dir = '../../data/vqa/inputs64'
    data_file = 'valid.npy'
    vqa_struct = VqaStruct( input_dir, data_file )
    img_name = 'COCO_val2014_000000262148'
    ref_qst = vqa_struct.get_ref_qst( img_name )
    pred_qst = [ 'what', 'is', 'the', 'man', 'holding', '?' ]
    bleu4 = BLEU4( ref_qst, pred_qst )
    ref_qa = vqa_struct.get_ref_qa( img_name )
    pred_qa = [ 'what', 'is', 'the', 'man', 'holding', '?',
            '<sep>', 'phone' ]
    bleu4 = BLEU4( ref_qa, pred_qa )
    print( 'Test passed!' )

def _test_extract_answer():
    input_dir = '../../data/vqa/inputs64'
    unified_vocab = text_helper.VocabDict(input_dir+'/vocab_unified.txt')
    qa_str = 'what color is the train ? <sep> red and black <end> <pad>'.split()
    qa_str = [ unified_vocab.word2idx( w ) for w in qa_str ]
    ans = extract_answer( qa_str, unified_vocab )
    assert ans == 'red and black'
    print( 'Test passed!' )

def _test_unified_ans_acc():
    input_dir = '../../data/vqa/inputs64'
    unified_vocab = text_helper.VocabDict(input_dir+'/vocab_unified.txt')
    qa_gt = [ 
        '<start> what color is the train ? <sep> red and black <end> <pad>',
        '<start> what is the man doing ? <sep> surfing in water <end> <pad>',
        '<start> what is the man doing ? <sep> skateboarding <end> <pad>'
    ]
    qa_gt = [ qa.split() for qa in qa_gt ]
    qa_pred = [
        'what color train ? <sep> red and black <end>',
        '<start> what man doing ? <sep> swimming in water <end> <pad>',
        '<start> what is the man ? <sep> skateboarding <end> <pad>'
    ]
    qa_pred = [ qa.split() for qa in qa_pred ]
    for i in range( len( qa_gt ) ):
        qa_gt[ i ] = [ unified_vocab.word2idx( w ) for w in qa_gt[ i ] ]
        qa_pred[ i ] = [ unified_vocab.word2idx( w ) for w in qa_pred[ i ] ]
    ans_acc = unified_ans_acc( qa_gt, qa_pred, unified_vocab )
    assert ans_acc == ( 2 / 3 )
    print( 'Test passed!' )

def main():
    _test_vqa_struct()
    _test_bleu4()
    _test_extract_answer()
    _test_unified_ans_acc()

if __name__ == '__main__':
    main()

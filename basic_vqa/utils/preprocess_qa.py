import json
import pickle
import re
import torch
import time
import numpy as np
import argparse
import h5py
import os
import text_helper


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(
    re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(
    r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def prepare_questions(questions_json):
    """
    Tokenize and normalize questions from a given question json in 
    the usual VQA format. Also return img_id.
    """
    qst_and_imgid = [ 
        (q['question'], q['image_id']) for q in questions_json['questions'] ]
    for question, img_id in qst_and_imgid:
        question = question.lower()[:-1]
        yield question.split(' '), img_id 


def prepare_answers(answers_json):
    """
    Normalize answers from a given answer json in the usual VQA format.
    """
    # The only normalization that is applied to both machine generated
    # answers as well as
    # ground truth answers is replacing most punctuation with space
    # (see [0] and [1]).
    # Since potential machine generated answers are just taken from most
    # common answers, applying the other
    # normalizations is not needed, assuming that the human answers are
    # already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/
    # 3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/
    # PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might
        # just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated
        # operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for ans_dict in answers_json['annotations']:
        answers = [ process_punctuation( a['answer'] ) for a in ans_dict['answers'] ]
        yield answers

def _encode_question(question, token_to_index, max_question_length=25):
    """ Turn a question into a vector of indices and a question length """
    # check that question length is max_question_length - 2
    # since we need to add <start> and <end> tokens
    assert len( question ) <= ( max_question_length - 2 )
    vec = torch.zeros(max_question_length).long()
    # set first index to <start> token
    vec[0] = token_to_index['<start>']
    for i, token in enumerate(question):
        index = token_to_index.get(token, 0)
        vec[i+1] = index
    vec[i+2] = token_to_index['<end>']
    # increase qst len by 1 for <start> token
    # <end> token is ignored
    return vec, len(question)+1

def _encode_answers(answers, answer_to_index):
    """ Turn an answer into a vector """
    # answer vec will be a vector of answer counts to determine
    # which answers will contribute to the loss.
    # this should be multiplied with 0.1 * negative log-likelihoods
    # that a model produces and then summed up
    # to get the loss that is weighted by how many humans gave that answer
    answer_vec = torch.zeros(len(answer_to_index))
    for answer in answers:
        index = answer_to_index.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec
    
def _check_integrity(questions, answers):
    """ Verify that we are using the correct data """
    qa_pairs = list(zip(questions['questions'],
        answers['annotations']))
    assert all(q['question_id'] == a['question_id'] 
            for q, a in qa_pairs), 'Questions not aligned with answers'
    assert all(q['image_id'] == a['image_id'] 
            for q, a in qa_pairs), 'Image id of question and answer don\'t match'
    assert questions['data_type'] == answers['data_type'], \
            'Mismatched data types'
    assert questions['data_subtype'] == answers['data_subtype'], \
            'Mismatched data subtypes'
    
def _find_answerable(answers):
    """
    Create a list of indices into questions that will have at
    least one answer that is in the vocab
    """
    answerable = []
    for i, a in enumerate(answers):
        answer_has_index = len(a.nonzero()) > 0
        # answer_has_index = sum(
        # [ answer_to_index.get(a, 0) for a in answers ] ) > 0
        # store the indices of anything that is answerable
        if answer_has_index:
            answerable.append(i)
    return answerable

def get_split_name( dirname ):
    '''
    Get split from dirname
    '''
    if 'train' in dirname:
        return 'train'
    if 'val' in dirname:
        return 'val'
    if 'test' in dirname:
        return 'test'
    return Exception( f'Unrecognized split: {dirname}' )

def process_qa( args, split, h5_fd ):
    # get qa paths for split
    ans_path = args.input_dir + \
            f'/Annotations/v2_mscoco_{split}_annotations.json'
    qst_path = args.input_dir + \
            f'/Questions/v2_OpenEnded_mscoco_{split}_questions.json'
    # get vocab
    vocab_ans_file = args.output_dir+'/vocab_answers.txt'
    vocab_ans = text_helper.VocabDict(vocab_ans_file)
    vocab_qst_file = args.output_dir+'/vocab_questions.txt'
    vocab_qst = text_helper.VocabDict(vocab_qst_file)
    
    # load json
    with open(qst_path, 'r') as fd:
        questions_json = json.load(fd)
    with open(ans_path, 'r') as fd:
        answers_json = json.load(fd)
    _check_integrity(questions_json, answers_json)
    num_qst = len( questions_json['questions'] )
    num_ans = len( answers_json['annotations'] )
        
    # vocab
    token_to_index = vocab_qst.word2idx_dict
    answer_to_index = vocab_ans.word2idx_dict
    ans_vocab_size = len( vocab_ans.word_list )
    assert token_to_index['<pad>'] == 0

    # qst, ans processing
    questions = prepare_questions(questions_json)
    answers = prepare_answers(answers_json)

    # get split name
    max_qst_len = 25
    split_name = get_split_name( split )
    group = h5_fd.create_group( split_name )
    # group.create_dataset( 'qst_id', ( num_qst, ), dtype=np.int32 )
    enc_qst_set = group.create_dataset( 'enc_qst',
            ( num_qst, max_qst_len ), dtype=np.int_ )
    qst_len_set = group.create_dataset( 'qst_len',
            ( num_qst, ), dtype=np.uint8 )
    enc_ans_set = group.create_dataset( 'enc_ans',
            ( num_ans, ans_vocab_size ), dtype=np.uint8 )
    img_id_set = group.create_dataset( 'img_id',
            ( num_qst, ), dtype=np.int32 )

    # import pdb; pdb.set_trace()
    for i, ( ( q, img_id ), a ) in enumerate( zip( questions, answers ) ):
        enc_qst, qst_len = _encode_question(q, token_to_index, max_qst_len)
        enc_ans = _encode_answers(a, answer_to_index)
        enc_qst_set[ i ] = enc_qst
        qst_len_set[ i ] = qst_len
        enc_ans_set[ i ] = enc_ans
        img_id_set[ i ] = img_id
        if (i+1) % 10000 == 0:
            print( f'processed [{i+1}/{num_qst}] qst-ans pairs' )

def main( args ):
    import pdb; pdb.set_trace()
    input_dir = args.input_dir
    output_dir = args.output_dir 
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join( output_dir, 'qst-ans.h5' )
    splits = [ 'train2014', 'val2014' ]
    
    with h5py.File( out_file, 'w', libver='latest' ) as fd:
        for split in splits:
            process_qa( args, split, fd )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='../../../data/vqa',
                        help='directory for inputs')
    
    parser.add_argument('--output_dir', type=str, default='../../../data/vqa/hdf5',
                        help='directory for outputs')
    
    args = parser.parse_args()

    main(args)

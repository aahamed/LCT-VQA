import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict


def make_vocab_questions(input_dir, output_dir):
    """Make dictionary for questions and save them into text file."""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    datasets = os.listdir(input_dir)
    for dataset in datasets:    
        with open(input_dir+'/'+dataset) as f:
            questions = json.load(f)['questions']
        set_question_length = [None]*len(questions)
        for iquestion, question in enumerate(questions):
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower())
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)
            set_question_length[iquestion] = len(words)
        question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    vocab_list.insert(2, '<start>')
    vocab_list.insert(3, '<end>')
   
    fname = output_dir + '/vocab_questions.txt'
    with open(fname, 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])
    
    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))


def make_vocab_answers(input_dir, n_answers, output_dir):
    """Make dictionary for top n answers and save them into text file."""
    answers = defaultdict(lambda: 0)
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir+'/'+dataset) as f:
            annotations = json.load(f)['annotations']
        for annotation in annotations:
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\s]", word):
                    continue
                answers[word] += 1
                
    answers = sorted(answers, key=answers.get, reverse=True)
    assert('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:n_answers-1] # '-1' is due to '<unk>'
    
    fname = output_dir + '/vocab_answers.txt'
    with open(fname, 'w') as f:
        f.writelines([w+'\n' for w in top_answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))
    print('Keep top %d answers into vocab' % n_answers)

def make_vocab_unified( output_dir ):
    '''Make text file for unified vocab for both qst and ans'''
    print( 'Make unified vocabulary' )
    unified_vocab_set = set()

    # add question vocab
    qst_file = output_dir + '/vocab_questions.txt'
    with open(qst_file) as f:
        for line in f:
            word = line.strip()
            unified_vocab_set.add( word )
    
    # add answer vocab
    ans_file = output_dir + '/vocab_answers.txt'
    with open(ans_file) as f:
        for line in f:
            # answer might have multiple words
            words = line.strip().split()
            unified_vocab_set.update( words )

    # remove special tokens
    special_tokens = [ '<pad>', '<unk>', '<start>', '<end>', '<sep>' ]
    for st in special_tokens:
        if st in unified_vocab_set:
            unified_vocab_set.remove( st )

    # sort
    unified_vocab_list = list( unified_vocab_set )
    unified_vocab_list.sort()
    
    # add back special tokens
    for i, st in enumerate( special_tokens ):
        unified_vocab_list.insert( i, st )
    
    # write unified vocab to txt file
    fname = output_dir + '/vocab_unified.txt'
    with open(fname, 'w') as f:
        f.writelines([w+'\n' for w in unified_vocab_list])

    assert len( unified_vocab_list ) == ( len( unified_vocab_set ) + 5 )
    print( f'Unified vocab len: {len(unified_vocab_list)}' )



def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_answers = args.n_answers
    make_vocab_questions(input_dir+'/Questions', output_dir)
    make_vocab_answers(input_dir+'/Annotations', n_answers, output_dir)
    make_vocab_unified(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../../data/vqa',
                        help='directory for input questions and answers')
    parser.add_argument('--output_dir', type=str, default='../../../data/vqa',
                        help='directory for output questions and answers')
    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab')
    args = parser.parse_args()
    main(args)

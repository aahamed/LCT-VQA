import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper


class VqaDataset(data.Dataset):
    '''
    Original VQA dataset
    Returns qst and ans seperately
    '''

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform


    def __getitem__(self, idx):
        
        # import pdb; pdb.set_trace()
        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans
        
        image_path = vqa[idx]['image_path']
        image_name = vqa[idx]['image_name']
        image = Image.open(image_path).convert('RGB')
        # padded with '<pad>' in 'ans_vocab'
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)
        qst2idc[0] = qst_vocab.word2idx('<start>')
        qst2idc[1:len(vqa[idx]['question_tokens'])+1] = \
                [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        qst2idc[len(vqa[idx]['question_tokens'])+1] = qst_vocab.word2idx('<end>')
        sample = {'image': image, 'question': qst2idc, 'image_path':image_path,
            'image_name':image_name}

        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            # for training
            sample['answer_label'] = ans2idx
            # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc = list([-1] * max_num_ans)
            # our model should not predict -1
            mul2idc[:len(ans2idc)] = ans2idc
            # for evaluation metric of 'multiple choice'
            sample['answer_multi_choice'] = mul2idc

        if transform:
            sample['image'] = transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.vqa)

class VqaDatasetUnified( VqaDataset ):
    '''
    Returns qst+ans in a concatenated string separated by <sep> token
    '''
    
    def __init__(self, input_dir, input_vqa,
            max_qst_length=30, max_num_ans=10, transform=None):
        super(VqaDatasetUnified, self).__init__(input_dir, input_vqa,
                max_qst_length, max_num_ans, transform)
        self.unified_vocab = text_helper.VocabDict(input_dir+'/vocab_unified.txt')

    def __getitem__(self, idx):
        vqa = self.vqa
        unified_vocab = self.unified_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans
        
        # get image
        image_path = vqa[idx]['image_path']
        image_name = vqa[idx]['image_name']
        image = Image.open(image_path).convert('RGB')

        # get qst
        qst_len = len(vqa[idx]['question_tokens'])
        qa2idc = np.array([unified_vocab.word2idx('<pad>')] * max_qst_length)
        qa2idc[0] = unified_vocab.word2idx('<start>')
        qa2idc[1:qst_len+1] = \
                [unified_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        
        # add <sep> token
        qa2idc[qst_len+1] = unified_vocab.word2idx('<sep>')

        # get ans
        ans = np.random.choice( vqa[idx]['valid_answers'] ).split()
        ans_len = len( ans )
        ans2idc = [unified_vocab.word2idx(w) for w in ans]

        # add ans to end of qst
        ptr = qst_len+2
        qa2idc[ptr:ptr+ans_len] = ans2idc

        # add <end> to end of ans
        ptr = ptr+ans_len
        qa2idc[ptr] =  unified_vocab.word2idx('<end>')

        # create sample
        sample = {'image': image, 'qa_str': qa2idc, 'image_name': image_name}
        
        if transform:
            sample['image'] = transform(sample['image'])

        return sample


def get_dataset( input_dir, input_vqa, max_qst_length,
        max_num_ans, transform, unified=False ):
    if unified:
        return VqaDatasetUnified(
            input_dir=input_dir,
            input_vqa=input_vqa,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform )
    else:
        return VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform )

def get_loader(input_dir, input_vqa_train, input_vqa_valid,
        max_qst_length, max_num_ans, batch_size, num_workers,
        train_portion, unified=False):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': get_dataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['train'],
            unified=unified ),
        'valid': get_dataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['valid'],
            unified=unified ) }

    # indices for subset
    data_idc = {
        phase: list( range( int( np.floor( 
            train_portion * len( vqa_dataset[phase] ) ) ) ) ) 
            for phase in [ 'train', 'valid' ] }

    vqa_subset = {
        phase: data.Subset( vqa_dataset[ phase ], data_idc[ phase ] )
        for phase in [ 'train', 'valid' ] }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_subset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader

def test( unified ):
    input_dir = '../../data/vqa/inputs64'
    max_qst_length = 30
    max_num_ans = 10
    batch_size = 4
    num_workers = 0
    train_portion = 0.25
    dataloader = get_loader(
        input_dir=input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers,
        train_portion=train_portion,
        unified=unified)
    print( 'len dataloader train:', len( dataloader['train'] ) )
    print( 'len dataloader valid:', len( dataloader['valid'] ) )
    import pdb; pdb.set_trace()
    train_loader = dataloader['train']
    batch_sample = next( iter( train_loader ) )
    if unified:
        unified_vocab = train_loader.dataset.dataset.unified_vocab
        qa_arr = batch_sample['qa_str'][ 0 ]
        qa_str = unified_vocab.arr2qst( qa_arr )
        print( 'qa_str:', qa_str )
    print( 'Test passed!' ) 

def main():
    test(unified=False)
    test(unified=True)

if __name__ == '__main__':
    main()

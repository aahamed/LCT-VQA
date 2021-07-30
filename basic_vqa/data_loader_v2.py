import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import h5py
from PIL import Image
from utils import text_helper


class VqaDataset(data.Dataset):
    '''
    Original VQA dataset
    Returns qst and ans seperately
    '''

    def __init__( self, input_dir, split, _transform=None ):
        self.input_dir = input_dir
        self._transform = _transform
        assert split in [ 'train', 'val' ]
        self.split = split
        self.h5_img_path = os.path.join( input_dir, 'images.h5' )
        self.h5_qa_path = os.path.join( input_dir, 'qst-ans.h5' )
        self.h5_img = None
        self.h5_qa = None
        self.num_qst = self.get_num_qst()
        self.img_id_to_h5_id = self.build_img_id_to_h5_id()
        # for backward compatability
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')

    def get_num_qst( self ):
        nq = 0
        with h5py.File( self.h5_qa_path, 'r' ) as fd:
            nq = len( fd[f'{self.split}/enc_qst'] )
        return nq

    def build_img_id_to_h5_id( self ):
        with h5py.File( self.h5_img_path, 'r' ) as fd:
            img_ids = fd[f'{self.split}/coco_ids'][()]
        img_id_to_h5_id = \
            { img_id : h5_id for h5_id, img_id in enumerate( img_ids ) }
        return img_id_to_h5_id

    def __getitem__( self, idx ):
        # import pdb; pdb.set_trace()
        if not self.h5_img:
            self.h5_img = h5py.File( self.h5_img_path, 'r')
        if not self.h5_qa:
            self.h5_qa = h5py.File( self.h5_qa_path, 'r' )
        enc_qst = self.h5_qa[f'{self.split}/enc_qst'][ idx ].astype( 'long' )
        qst_len = self.h5_qa[f'{self.split}/qst_len'][ idx ]
        enc_ans = self.h5_qa[f'{self.split}/enc_ans'][ idx ]
        img_id = self.h5_qa[f'{self.split}/img_id'][ idx ]
        h5_id = self.img_id_to_h5_id[ img_id ]
        img = Image.fromarray( self.h5_img[f'{self.split}/images'][h5_id] )
        if self._transform:
            img = self._transform( img )
        # answer_label, answer_multi_choice, image_name,
        # image_path are for backward compatability
        valid_answers = enc_ans.nonzero()[0]
        answer_label = self.ans_vocab.unk2idx
        if valid_answers.size > 0:
            answer_label = np.random.choice( valid_answers )
        answer_multi_choice = [-1] * 10
        answer_multi_choice[:len(valid_answers)] = valid_answers  
        image_name = f'COCO_{self.split}2014_{img_id:012}'
        image_path = f'{self.split}/images/{img_id}'
        sample = { 'question': enc_qst, 'image': img, 'qst_len': qst_len,
                'enc_ans': enc_ans, 'answer_label': answer_label,
                'image_name': image_name,
                'answer_multi_choice': answer_multi_choice,
                'image_path': image_path, 'image_id': img_id }
        return sample

    def __len__(self):
        return self.num_qst

def get_loader( input_dir, batch_size, num_workers, train_portion):

    transform = {
        phase: transforms.Compose(
            [ transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))
            ] ) 
        for phase in ['train', 'valid'] }

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            split='train',
            _transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            split='val',
            _transform=transform['valid'])}

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

def test():
    input_dir = '../../data/vqa/hdf5_64'
    img_size = 64
    batch_size = 4
    num_workers = 0
    train_portion = 0.25
    max_qst_len = 25
    ans_vocab_size = 1000
    dataloader = get_loader(
        input_dir=input_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_portion=train_portion)
    print( 'len dataloader train:', len( dataloader['train'] ) )
    print( 'len dataloader valid:', len( dataloader['valid'] ) )
    import pdb; pdb.set_trace()
    train_loader = dataloader['train']
    train_loader_iter = iter( train_loader )
    for i in range( 10 ):
        batch_sample = next( train_loader_iter )
        enc_qst = batch_sample[ 'question' ]
        enc_ans = batch_sample[ 'enc_ans' ]
        qst_len = batch_sample[ 'qst_len' ]
        img = batch_sample[ 'image' ]
        print( f'enc_qst.shape: {enc_qst.shape} enc_ans.shape: {enc_ans.shape} ' +
                f'qst_len.shape: {qst_len.shape} img.shape: {img.shape}')
        assert enc_qst.shape == ( batch_size, max_qst_len )
        assert enc_ans.shape == ( batch_size, ans_vocab_size )
        assert qst_len.shape == ( batch_size, )
        assert img.shape == ( batch_size, 3, img_size, img_size )
    

def main():
    test()

if __name__ == '__main__':
    main()

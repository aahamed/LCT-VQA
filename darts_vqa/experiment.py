import torch
import logging
import sys
import os
import random
import numpy as np
import factory
import config
from torch import nn
from data_loader import get_loader
from misc import num_correct, num_correct_qst, \
    VqaStruct, calc_bleu_scores
from itertools import cycle

class Experiment( object ):

    def __init__( self, args ):
        import pdb; pdb.set_trace()
        self.name = args.exp
        self.exp_dir = os.path.join( config.ROOT_STATS_DIR, self.name )
        self.args = args
        self.arch_type = args.arch_type
        self.arch_update_freq = args.arch_update_freq
        self.qst_only = args.qst_only
        self.grad_clip = args.grad_clip
        self.report_freq = args.report_freq

        # set seed
        seed = config.SEED if config.SEED else random.randint(0, 1e5)
        torch.manual_seed( seed )
        random.seed( seed )
        np.random.seed( seed )

        # get dataloader
        self.data_loader = get_loader(
            input_dir=args.input_dir,
            input_vqa_train='train.npy',
            input_vqa_valid='valid.npy',
            max_qst_length=args.max_qst_len,
            max_num_ans=args.max_num_ans,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_portion=args.train_portion,
            unified=args.unified)

        self.qst_vocab = self.data_loader['train'].dataset.dataset.qst_vocab
        self.ans_vocab = self.data_loader['valid'].dataset.dataset.ans_vocab
        self.vqa_struct = VqaStruct( args.input_dir, data_file='valid.npy' )

        # exp params
        self.epochs = args.num_epochs
        self.current_epoch = 0
        
        # init criterion, model, optimizer, scheduler, architect
        self.criterion = nn.CrossEntropyLoss()
        self.vqa_model = factory.get_vqa_model(args,
                self.data_loader['train'].dataset.dataset)
        self.optimizer = factory.get_optimizer(args, self.vqa_model)
        self.scheduler = factory.get_scheduler(args, self.optimizer)
        self.architect = factory.get_architect(args, self.vqa_model)

        self.init_model()

        # stats
        self.train_loss = []
        self.train_ans_acc = []
        self.val_loss = []
        self.val_ans_acc = []
        self.val_b4 = []

        self.load_experiment()
        self.log( f'seed is: {seed}' )
        self.log( f'args: {args}' )

    
    def setup_logger( self ):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.log( f'Exp Name: {self.name}\n\n' )
    
    # Loads the experiment data if exists to resume training from last saved
    # checkpoint.
    def load_experiment(self):
        os.makedirs(config.ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.exp_dir):
            if not self.args.resume:
                # if resume option is not specified
                # check to make sure exp_dir is empty
                files = os.listdir( self.exp_dir )
                if len(files) > 1:
                    print( f'exp dir: {self.exp_dir} not empty. ' +
                            'Please delete all files' +
                            ' in dir and rerun train command.' )
                    import pdb; pdb.set_trace()
                    exit( 1 )
            else:
                # resume training from last checkpoint
                self.load_model()
                self.load_stats()
        else:
            os.makedirs(self.exp_dir)
        self.setup_logger()

    def log( self, log_str ):
        logging.info( log_str )

    def init_model( self ):
        self.criterion.to( config.DEVICE )
        self.vqa_model.to( config.DEVICE )

    def run( self ):
        for epoch in range( self.current_epoch, self.epochs ):
            self.log( f'Starting Epoch: {epoch+1}' )
            if self.arch_type == 'darts':
                self.log( f'genotype: {self.vqa_model.genotype()}' )
            self.current_epoch = epoch
            self.train()
            self.val()
            self.scheduler.step()
            self.save_model()
            self.save_stats()
        self.val()
    
    def evaluate_gen_qst( self, batch_sample ):
        '''
        Helper routine to evaluate generated questions
        '''
        self.vqa_model.eval()
        # import pdb; pdb.set_trace()
        image = batch_sample['image'].to(config.DEVICE)
        question = batch_sample['question']
        answer = batch_sample['answer_label']
        image_path = batch_sample['image_path']
        # ground truth question and answers
        qst = [ self.qst_vocab.arr2qst( q ) for q in question ]
        ans = [ self.ans_vocab.idx2word( a ) for a in answer ]
        
        # generated question-answer
        with torch.no_grad():
            gen_question, gen_answer = self.vqa_model.generate( image )
        gen_answer = torch.argmax( gen_answer, 1 )
        gen_qst = [ self.qst_vocab.arr2qst( q ) for q in gen_question ]
        gen_ans = [ self.ans_vocab.idx2word( a ) for a in gen_answer ]

        n = min( 4, len( image ) )
        self.log( 'Evaluating question answer pairs' )
        for i in range( n ):
            self.log( f'image path:{image_path[i]}' )
            self.log( f'ground truth qst: {qst[i]} ans: {ans[i]}' )
            self.log( f'generated qst: {gen_qst[i]} ans: {gen_ans[i]}' )

    def train( self ):
        self.vqa_model.train()
        total_loss = 0
        ans_corr = 0
        num_batches = len( self.data_loader['train'] )
        dataset = self.data_loader['train'].dataset
        ans_unk_idx = dataset.dataset.ans_vocab.unk2idx
        N = len( dataset )
        valid_queue_iter = cycle( iter( self.data_loader['valid'] ) )
        lr = self.scheduler.get_lr()[0]
        #  import pdb; pdb.set_trace()

        for batch_idx, batch_sample in enumerate( self.data_loader['train'] ):
            # get training data
            image = batch_sample['image'].to(config.DEVICE)
            question = batch_sample['question'].to(config.DEVICE)
            label = batch_sample['answer_label'].to(config.DEVICE)
            multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

            if self.arch_type == 'darts' and \
                    ( batch_idx % self.arch_update_freq == 0 ):
                # STAGE1: architecture update
                batch_sample = next( valid_queue_iter )
                val_image = batch_sample['image'].to(config.DEVICE)
                val_question = batch_sample['question'].to(config.DEVICE)
                val_label = batch_sample['answer_label'].to(config.DEVICE)
                # import pdb; pdb.set_trace()
                self.architect.step( image, question, label,
                        val_image, val_question, val_label, lr )

            # STAGE2: weight update
            self.optimizer.zero_grad()
            ans_out, qst_out = self.vqa_model(image, question)
            ans_loss = self.criterion(ans_out, label)
            qst = question[:, 1:].flatten()
            qst_pred = qst_out[:, :-1].flatten(end_dim=1)
            qst_loss = self.criterion(qst_pred, qst)
            if self.qst_only:
                loss = qst_loss
            else:
                loss = ans_loss + qst_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.vqa_model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
           
            # calculate accuracy
            _, pred = torch.max(ans_out, dim=1)
            pred[ pred==ans_unk_idx ] = -9999
            ans_corr += num_correct( pred.cpu(), multi_choice )

            if batch_idx % self.report_freq == 0:
                self.log( '| TRAIN SET | STAGE2 | ' + 
                f'EPOCH [{self.current_epoch+1:02d}/{self.epochs:02d}] ' +
                f'Step [{batch_idx:04d}/{num_batches:04d}] ' +
                f'Loss: {loss.item():.4f}' )

        avg_loss = total_loss / num_batches
        ans_acc = ans_corr / N
        self.train_loss.append( avg_loss )
        self.train_ans_acc.append( ans_acc )

        self.log( f'| TRAIN_SET | EPOCH [{self.current_epoch+1:02d}/' +
                f'{self.epochs:02d}] Loss: {avg_loss:.4f} ' +
                f'Ans-acc: {ans_acc:.4f} ' )
       
        # check generated question quality
        self.evaluate_gen_qst( batch_sample )

    def val( self ):
        self.vqa_model.eval()
        total_loss = 0
        ans_corr = 0
        total_b4 = 0
        dataset = self.data_loader['valid'].dataset
        N = len(dataset)
        ans_unk_idx = dataset.dataset.ans_vocab.unk2idx
        num_batches = len( self.data_loader['valid'] )

        with torch.no_grad():
            for batch_idx, batch_sample in enumerate( self.data_loader['valid'] ):
                # get training data
                image = batch_sample['image'].to(config.DEVICE)
                question = batch_sample['question'].to(config.DEVICE)
                label = batch_sample['answer_label'].to(config.DEVICE)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                image_name = batch_sample['image_name']
                
                # get validation loss
                ans_out, qst_out = self.vqa_model(image, question)
                ans_loss = self.criterion(ans_out, label)
                qst = question[:, 1:].flatten()
                qst_pred = qst_out[:, :-1].flatten(end_dim=1)
                qst_loss = self.criterion(qst_pred, qst)
                if self.qst_only:
                    loss = qst_loss
                else:
                    loss = ans_loss + qst_loss
                total_loss += loss.item() 
            
                # calculate accuracy
                pred = torch.argmax(ans_out, dim=1)
                pred[ pred==ans_unk_idx ] = -9999
                ans_corr += num_correct( pred.cpu(), multi_choice )

                # bleu score
                pred_qst, pred_ans = self.vqa_model.generate( image )
                b4 = calc_bleu_scores( image_name, pred_qst,
                    self.qst_vocab, self.vqa_struct )
                total_b4 += b4

                if batch_idx % self.report_freq == 0:
                    self.log( '| VAL SET | ' + 
                    f'EPOCH [{self.current_epoch+1:02d}/{self.epochs:02d}] ' +
                    f'Step [{batch_idx:04d}/{num_batches:04d}] ' +
                    f'Loss: {loss.item():.4f} ' + 
                    f'BLEU4: {b4:.4f}' )
        
        # print stats
        avg_loss = total_loss / num_batches
        avg_b4 = total_b4 / num_batches
        ans_acc = ans_corr / N
        self.val_loss.append( avg_loss )
        self.val_ans_acc.append( ans_acc )
        self.val_b4.append( avg_b4 )

        self.log( f'| VAL_SET | EPOCH [{self.current_epoch+1:02d}/' +
                f'{self.epochs:02d}] Loss: {avg_loss:.4f} ' +
                f'Ans acc: {ans_acc:.4f} ' +
                f'BLEU4: {avg_b4:.4f}' )

    
    def save_model( self ):
        save_path = os.path.join( self.exp_dir, 'vqa_model.pt' )
        vqa_model_dict = self.vqa_model.state_dict()
        opt_dict = self.optimizer.state_dict()
        sched_dict = self.scheduler.state_dict()
        epoch = self.current_epoch+1
        torch.save( {
            'vqa_model': vqa_model_dict,
            'opt': opt_dict,
            'sched': sched_dict,
            'epoch': epoch }, save_path )
        save_arch_path = os.path.join( self.exp_dir, 'arch_par.pt' )
        self.vqa_model.save_arch_parameters( save_arch_path )

    def load_model( self ):
        load_path = os.path.join( self.exp_dir, 'vqa_model.pt' )
        state_dict = torch.load( load_path )
        self.vqa_model.load_state_dict( state_dict['vqa_model'] )
        self.optimizer.load_state_dict( state_dict['opt'] )
        self.scheduler.load_state_dict( state_dict['sched'] )
        self.current_epoch = state_dict['epoch']
        load_arch_path = os.path.join( self.exp_dir, 'arch_par.pt' )
        self.vqa_model.load_arch_parameters( load_arch_path )

    def save_stats( self ):
        save_path = os.path.join( self.exp_dir, 'stats.pt' )
        stat_dict = {
            'train_loss': self.train_loss,
            'train_ans_acc': self.train_ans_acc,
            'val_loss': self.val_loss,
            'val_ans_acc': self.val_ans_acc,
            'val_b4': self.val_b4,
            'args': self.args,
        }
        epoch = self.current_epoch+1
        torch.save( stat_dict, save_path )
    
    def load_stats( self ):
        load_path = os.path.join( self.exp_dir, 'stats.pt' )
        stat_dict = torch.load( load_path )
        self.train_loss = stat_dict['train_loss']
        self.train_ans_acc = stat_dict['train_ans_acc']
        self.val_loss = stat_dict['val_loss']
        self.val_ans_acc = stat_dict['val_ans_acc']
        self.val_b4 = stat_dict['val_b4']


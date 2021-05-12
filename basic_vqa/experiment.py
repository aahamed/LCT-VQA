import torch
import logging
import sys
import os
import config
import random
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch import nn
from data_loader import get_loader
from model_factory import *
from architect_factory import get_architect
from itertools import cycle
from file_utils import *
from plot import plot_loss_acc
from models_base import softXEnt

def num_correct( pred, multi_choice ):
        res = torch.stack([(ans == pred) for ans in multi_choice])
        res = res.any(dim=0).sum().item()
        return res

class Experiment( object ):

    def __init__( self, args ):
        import pdb; pdb.set_trace()
        # self.args = args
        self.name = config.EXP_NAME
        self.exp_dir = os.path.join( config.ROOT_STATS_DIR, self.name )
        
        # set seed
        seed = config.SEED if config.SEED else random.randint(0, 1e5)
        torch.manual_seed( seed )
        random.seed( seed )
        np.random.seed( seed )

        # get dataloaders for training and validation
        self.data_loader = get_loader(
            input_dir=config.INPUT_DIR,
            input_vqa_train='train.npy',
            input_vqa_valid='valid.npy',
            max_qst_length=config.MAX_QST_LEN,
            max_num_ans=config.MAX_NUM_ANS,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            train_portion=config.TRAIN_PORTION)

        self.qst_vocab = self.data_loader['train'].\
                dataset.dataset.qst_vocab
        self.ans_vocab = self.data_loader['train'].\
                dataset.dataset.ans_vocab

        # setup experiment params
        self.epochs = config.NUM_EPOCHS
        self.current_epoch = 0
        self.best_model = None

        # init criterion, models, optimizers, schedulers
        # ef_* corresponds to encoder-decoder model
        self.criterion = nn.CrossEntropyLoss()
        self.ef_model = get_ef_model(
                self.data_loader['train'].dataset.dataset )
        self.ef_optimizer = get_ef_optimizer( self.ef_model )
        self.ef_scheduler = get_ef_scheduler( self.ef_optimizer )
        self.train_ef_loss = []
        self.train_ef_acc = []
        self.val_ef_loss = []
        self.val_ef_acc = []

        self.w_model = get_w_model(
                self.data_loader['train'].dataset.dataset )
        self.w_optimizer = get_w_optimizer( self.w_model )
        self.w_scheduler = get_w_scheduler( self.w_optimizer )
        self.train_w_loss = []
        self.train_w_acc = []


        # instantiate architect for architecture search
        self.architect = get_architect(self.ef_model, self.w_model,
                self.ef_optimizer, self.w_optimizer)
        self.arch_update_freq = config.ARCH_UPDATE_FREQ

        self.init_model()

        self.load_experiment()
        self.log( f'seed is: {seed}' )

    
    def setup_logger( self ):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.log( f'Exp Name: {self.name}\n\n' )
    
    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def load_experiment(self):
        os.makedirs(config.ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.exp_dir):
            if not config.RESUME:
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
                self.read_stats()
        else:
            os.makedirs(self.exp_dir)
        self.setup_logger()

    def init_model( self ):
        self.criterion.to( config.DEVICE )
        self.ef_model.to( config.DEVICE )
        self.w_model.to( config.DEVICE )

    def set_arch_update_freq( self ):
        self.arch_update_freq = int( 
                config.ARCH_UPDATE_FREQ * ( 
                    config.GAMMA_ARCH ** self.current_epoch ) )
        self.arch_update_freq = max( self.arch_update_freq,
                config.ARCH_UPDATE_FREQ_MIN )
        self.log( f'architecture update freq: {self.arch_update_freq}' )

    def run( self ):
        for epoch in range( self.current_epoch, self.epochs ):
            self.log( f'Starting Epoch: {epoch+1}' )
            if config.ARCH_TYPE == 'pcdarts':
                self.log( f'genotype: {self.ef_model.genotype()}' )
            self.current_epoch = epoch
            self.set_arch_update_freq()
            self.train()
            self.val()
            self.ef_scheduler.step()
            if not config.SKIP_STAGE2:
                self.w_scheduler.step()
            self.save_model()
            self.record_stats()

    def evaluate_gen_qst( self, batch_sample ):
        self.ef_model.eval()
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
            gen_question, gen_answer = self.ef_model.generate( image )
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
        self.ef_model.train()
        self.w_model.train()
        ef_loss = 0
        ef_corr_1 = 0
        ef_corr_2 = 0
        w_loss = 0
        w_corr = 0
        dataset = self.data_loader['train'].dataset
        N = len(dataset)
        batch_step_size = len( self.data_loader['train'] )
        ans_unk_idx = dataset.dataset.ans_vocab.unk2idx
        valid_queue_iter = cycle( iter( self.data_loader['valid'] ) )
        lr = self.ef_scheduler.get_lr()[0]
        # import pdb; pdb.set_trace()
        
        for batch_idx, batch_sample in enumerate( self.data_loader['train'] ): 
            # get training data
            image = batch_sample['image'].to(config.DEVICE)
            question = batch_sample['question'].to(config.DEVICE)
            label = batch_sample['answer_label'].to(config.DEVICE)
            multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
            
            if config.ARCH_TYPE == 'pcdarts' and \
                    ( batch_idx % self.arch_update_freq == 0 ):
                # STAGE 3: Architecture Search
                # Update architecture of E in encoder-decoder model 
                # using validation data
                batch_sample = next( valid_queue_iter )
                val_image = batch_sample['image'].to(config.DEVICE)
                val_question = batch_sample['question'].to(config.DEVICE)
                val_label = batch_sample['answer_label'].to(config.DEVICE)
                # import pdb; pdb.set_trace()
                self.architect.step( image, question, label,
                        val_image, val_question, val_label, lr )
            
            # STAGE 1: Update weights of encoder-decoder model
            # using training data
            self.ef_optimizer.zero_grad()
            ans_out, qst_out = self.ef_model(image, question)
            _, ef_pred1 = torch.max(ans_out, 1)
            _, ef_pred2 = torch.max(ans_out, 1)
            ans_loss = self.criterion(ans_out, label)
            qst = question[:, 1:].flatten()
            qst_out = qst_out[:, :-1].flatten(end_dim=1)
            qst_loss = self.criterion(qst_out, qst)
            loss = ans_loss + qst_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.ef_model.parameters(), config.GRAD_CLIP)
            self.ef_optimizer.step()
            
            # Evaluation metric of 'multiple choice'
            # ef_pred1: our model prediction to '<unk>' IS accepted as the answer.
            # ef_pred2: our model prediction to '<unk>' is NOT accepted as the answer.
            ef_pred2[ef_pred2 == ans_unk_idx] = -9999
            ef_loss += loss.item()
            ef_corr_1 += num_correct( ef_pred1.cpu(), multi_choice )
            ef_corr_2 += num_correct( ef_pred2.cpu(), multi_choice )
            if batch_idx % config.REPORT_FREQ == 0:
                self.log( ('| TRAIN SET | STAGE1 | Epoch [{:02d}/{:02d}], ' + 
                        'Step [{:04d}/{:04d}], EF-Loss: {:.4f}')
                          .format(self.current_epoch+1, self.epochs,
                              batch_idx, int(batch_step_size), loss.item()))
            
            if not config.SKIP_STAGE2:
                # generate pseudo qa tests
                pseudo_qst, pseudo_ans = self.ef_model.generate( image )
                pseudo_ans = F.softmax( pseudo_ans, dim=1 )
                # STAGE 2: Train W vqa model using the
                # pseudo qa tests generated by EF
                self.w_optimizer.zero_grad()
                w_out = self.w_model( image, pseudo_qst )
                loss = softXEnt( w_out, pseudo_ans )
                loss.backward()
                nn.utils.clip_grad_norm_(self.w_model.parameters(), config.GRAD_CLIP)
                self.w_optimizer.step()
                w_pred = torch.argmax( w_out, 1 )
                pseudo_pred = torch.argmax( pseudo_ans, 1 )
                w_corr += ( w_pred == pseudo_pred ).sum().item()
                w_loss += loss.item()
                if batch_idx % config.REPORT_FREQ == 0:
                    self.log( ( '| TRAIN SET | STAGE2 | Epoch [{:02d}/{:02d}], ' +
                        'Step [{:04d}/{:04d}], W-Loss: {:.4f}' )
                        .format(self.current_epoch+1, self.epochs,
                            batch_idx, int(batch_step_size), loss.item()) )
            

        # Print the average loss and accuracy in an epoch.
        ef_loss = ef_loss / batch_step_size
        ef_acc_1 = ef_corr_1 / N
        ef_acc_2 = ef_corr_2 / N
        w_loss = w_loss / batch_step_size
        w_acc = w_corr / N
        self.train_ef_loss.append( ef_loss )
        self.train_ef_acc.append( ef_acc_2 )
        self.train_w_loss.append( w_loss )
        self.train_w_acc.append( w_acc )

        self.log( f'| TRAIN SET | Epoch [{self.current_epoch+1:02d}/' + 
                f'{self.epochs:02d}], EF-Loss: {ef_loss:.4f} ' + 
                f'EF-Acc: {ef_acc_2:.4f}, ' +
                f'W-Loss: {w_loss:.4f}, ' +
                f'W-Acc: {w_acc:.4f}' )

        self.evaluate_gen_qst( batch_sample )
            

    def val( self ):
        self.ef_model.eval()
        self.w_model.eval()
        running_loss = 0
        ef_corr_1 = 0
        ef_corr_2 = 0
        dataset = self.data_loader['valid'].dataset
        N = len(dataset)
        batch_step_size = len( self.data_loader['valid'] )
        ans_unk_idx = dataset.dataset.ans_vocab.unk2idx

        with torch.no_grad():
            for batch_idx, batch_sample in enumerate( self.data_loader['valid'] ):
                # get training data
                image = batch_sample['image'].to(config.DEVICE)
                question = batch_sample['question'].to(config.DEVICE)
                label = batch_sample['answer_label'].to(config.DEVICE)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                
                # get validation loss
                output, _ = self.ef_model(image, question) # [batch_size, ans_vocab_size=1000]
                _, ef_pred1 = torch.max(output, 1)  # [batch_size]
                _, ef_pred2 = torch.max(output, 1)  # [batch_size]
                loss = self.criterion(output, label)
            
                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                ef_pred2[ef_pred2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                ef_corr_1 += num_correct( ef_pred1.cpu(), multi_choice )
                ef_corr_2 += num_correct( ef_pred2.cpu(), multi_choice )
                # print minibatch stats
                if batch_idx % 100 == 0:
                    self.log( ('| VALID SET | Epoch [{:02d}/{:02d}], ' + 
                            'Step [{:04d}/{:04d}], Loss: {:.4f}')
                              .format(self.current_epoch+1, self.epochs,
                                  batch_idx, int(batch_step_size), loss.item()))
        
        # Print the average loss and accuracy in an epoch.
        ef_loss = running_loss / batch_step_size
        ef_acc_1 = ef_corr_1 / N
        ef_acc_2 = ef_corr_2 / N
        self.val_ef_loss.append( ef_loss )
        self.val_ef_acc.append( ef_acc_2 )

        self.log( f'| VALID SET | Epoch [{self.current_epoch+1:02d}/' + 
                f'{self.epochs:02d}], Loss: {ef_loss:.4f} ' + 
                f'Acc(Exp1): {ef_acc_1:.4f}, ' +
                f'Acc(Exp2): {ef_acc_2:.4f}' )

    def log( self, log_str ):
        logging.info( log_str )

    def read_stats( self ):
        fname = 'train_ef_loss.txt'
        self.train_ef_loss = read_file_in_dir( self.exp_dir, fname )
        fname = 'train_ef_acc.txt'
        self.train_ef_acc = read_file_in_dir( self.exp_dir, fname )
        fname = 'val_ef_loss.txt'
        self.val_ef_loss = read_file_in_dir( self.exp_dir, fname )
        fname = 'val_ef_acc.txt'
        self.val_ef_acc = read_file_in_dir( self.exp_dir, fname )
        fname = 'train_w_loss.txt'
        self.train_w_loss = read_file_in_dir( self.exp_dir, fname )
        fname = 'train_w_acc.txt'
        self.train_w_acc = read_file_in_dir( self.exp_dir, fname )

    def record_stats( self ):
        fname = 'train_ef_loss.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.train_ef_loss )
        fname = 'train_ef_acc.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.train_ef_acc )
        fname = 'val_ef_loss.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.val_ef_loss )
        fname = 'val_ef_acc.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.val_ef_acc )
        fname = 'train_w_loss.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.train_w_loss )
        fname = 'train_w_acc.txt'
        write_to_file_in_dir( self.exp_dir, fname, self.train_w_acc )
        self.plot_stats()

    def plot_stats( self ):
        fname = os.path.join( self.exp_dir, 'ef_train_loss_acc.png' )
        prefix = 'EF Training'
        plot_loss_acc( self.train_ef_loss, self.train_ef_acc, prefix, fname )
        fname = os.path.join( self.exp_dir, 'ef_val_loss_acc.png' )
        prefix = 'EF Validation'
        plot_loss_acc( self.val_ef_loss, self.val_ef_acc, prefix, fname )
        fname = os.path.join( self.exp_dir, 'w_train_loss_acc.png' )
        prefix = 'W Training'
        plot_loss_acc( self.train_w_loss, self.train_w_acc, prefix, fname )
    
    def save_model( self ):
        save_path = os.path.join( self.exp_dir, 'ef_model.pt' )
        ef_model_dict = self.ef_model.state_dict()
        ef_opt_dict = self.ef_optimizer.state_dict()
        ef_sched_dict = self.ef_scheduler.state_dict()
        epoch = self.current_epoch+1
        torch.save( {
            'ef_model': ef_model_dict,
            'ef_opt': ef_opt_dict,
            'ef_sched': ef_sched_dict,
            'epoch': epoch }, save_path )
        save_path = os.path.join( self.exp_dir, 'w_model.pt' )
        w_model_dict = self.w_model.state_dict()
        w_opt_dict = self.w_optimizer.state_dict()
        w_sched_dict = self.w_scheduler.state_dict()
        torch.save( {
            'w_model': w_model_dict,
            'w_opt': w_opt_dict,
            'w_sched': w_sched_dict,
            'epoch': epoch }, save_path )
    
    def load_model( self ):
        load_path = os.path.join( self.exp_dir, 'ef_model.pt' )
        state_dict = torch.load( load_path )
        self.ef_model.load_state_dict( state_dict['ef_model'] )
        self.ef_optimizer.load_state_dict( state_dict['ef_opt'] )
        self.ef_scheduler.load_state_dict( state_dict['ef_sched'] )
        load_path = os.path.join( self.exp_dir, 'w_model.pt' )
        state_dict = torch.load( load_path )
        self.w_model.load_state_dict( state_dict['w_model'] )
        self.w_optimizer.load_state_dict( state_dict['w_opt'] )
        self.w_scheduler.load_state_dict( state_dict['w_sched'] )
        self.current_epoch = state_dict['epoch']

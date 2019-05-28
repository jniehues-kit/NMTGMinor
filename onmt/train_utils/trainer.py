from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
import os
import random 
import numpy as np
from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner
from onmt.ModelConstructor import init_model_parameters
from onmt.utils import checkpoint_paths
from torch.distributions import Categorical
import torch.nn.functional as F



class BaseTrainer(object):
    
    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):
        
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1)
        
        self.loss_function = loss_function
        self.start_time = 0
        
    def run(self, *args,**kwargs):
        
        raise NotImplementedError    
    
    def eval(self, data):
        
        raise NotImplementedError
        
    def to_variable(self, data):

        for i, t in enumerate(data):
            if self.cuda:
                data[i] = Variable(data[i].cuda())
            else:
                data[i] = Variable(data[i])

        return data
            

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                   'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.' )
            grads.append(p.grad.data)
        return grads
        
    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)
        self.optim = onmt.Optim(opt)
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda()
        
        self.optim.set_parameters(self.model.parameters())

    def save(self, epoch, valid_ppl, batch_order=None, iteration=-1):
        
        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()
                
        #  drop a checkpoint
        checkpoint = {
                'model': model_state_dict,
                'dicts': dicts,
                'opt': opt,
                'epoch': epoch,
                'iteration' : iteration,
                'batch_order' : batch_order,
                'optim': optim_state_dict
        }
        
        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)
        
        # check te save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print (" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def eval(self, data):
        total_loss = 0
        total_words = 0
                
        batch_order = data.create_order(random=False)
        self.model.eval()
        """ PyTorch semantics: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):

                batch = data.next()[0]

                if(self.cuda):
                    batch.cuda()
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                if(self.opt.sample_target_order):
                    self.sample_target_order(batch)


                outputs = self.model(batch)
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.Constants.PAD)
                outputs['tgt_mask'] = tgt_mask

                if self.opt.predict_position == "relative":
                    loss_dict = self.loss_function(outputs, targets, model=self.model,
                                                   backward=False,pos_targets=batch.get('mapping'))
                else:
                    loss_dict = self.loss_function(outputs, targets, model=self.model,
                                               backward=False)
                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):
        
        opt = self.opt
        train_data = self.train_data
        
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            train_data.shuffle()

        # Shuffle mini batch order.
        
        if resume:
            train_data.batch_order = batch_order
            train_data.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batch_order = train_data.create_order()
            iteration = 0

        total_loss, total_words = 0, 0
        pos_loss_data,ce_loss_data = 0,0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        n_samples = len(train_data)
        
        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        
        for i in range(iteration, n_samples):

            curriculum = (epoch < opt.curriculum)

            batch = train_data.next(curriculum=curriculum)[0]
            if(self.cuda):
                batch.cuda()
            
            oom = False
            try:

                if(self.opt.sample_target_order):
                    #print ("Before:",batch.get("target_input").t()[0])
                    #print ("Before:",batch.get("target_output").t()[0])
                    self.sample_target_order(batch)
                    #print ("After:",batch.get("target_input").t()[0])
                    #print ("After:",batch.get("target_output").t()[0])
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                outputs = self.model(batch)

                targets = batch.get('target_output')
                tgt_inputs = batch.get('target_input')

                batch_size = batch.size

                tgt_mask = targets.data.ne(onmt.Constants.PAD)
                outputs['tgt_mask'] = tgt_mask
                
                normalizer = 1

                if self.opt.predict_position == "relative":
                    loss_dict = self.loss_function(outputs, targets, model=self.model,
                                                   backward=True,pos_targets=batch.get('mapping'))
                    pos_loss_data += loss_dict['pos_data']
                    ce_loss_data += loss_dict['ce_data']
                else:
                    loss_dict = self.loss_function(outputs, targets, model=self.model,
                                               backward=True, normalizer=normalizer)
                loss_data = loss_dict['data']

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise e        
                
            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size
                
                counter = counter + 1 
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size
                
                # We only update the parameters after getting gradients from n mini-batches
                # simulating the multi-gpu situation
                # if counter == opt.virtual_gpu:
                # if counter >= opt.batch_size_update:
                
                if num_accumulated_words >= opt.batch_size_update * 0.95:
                    grad_denom = 1
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words
                    # Update the parameters.
                    self.optim.step(grad_denom=grad_denom)
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)
                        
                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        
                        self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)

                num_words = tgt_size
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                optim = self.optim

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                           "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                          (epoch, i+1, len(train_data),
                           math.exp(report_loss / report_tgt_words),
                           optim.getLearningRate(),
                           optim._step,
                           report_src_words/(time.time()-start),
                           report_tgt_words/(time.time()-start),
                           str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))
                    if(self.opt.predict_position == "relative"):
                        print(("POS Loss %6.2f, CE LOSS %6.2f") % (pos_loss_data,ce_loss_data))

                    report_loss, report_tgt_words = 0, 0
                    report_src_words = 0
                    start = time.time()

        return total_loss / total_words


    def sample_target_order(self,batch):
        target = batch.get("target_output")
        batch_size = batch.size
        voc_size = self.dicts['tgt'].size()


        history_mask = torch.zeros([batch_size,target.size(0)], dtype=torch.uint8)
        shuffle = []
        log_probs = torch.ones([batch_size, voc_size], dtype=torch.float64)
        if(self.cuda):
            log_probs = log_probs.cuda()
            history_mask = history_mask.cuda()
        probs = torch.zeros(target.t().size()).type_as(log_probs)

        for i in range(target.size(0)):

 
            # get word probabiliy distribution
            if (self.opt.sample_target_distribution == "uniform"):
                log_probs.fill_(1)
            else:
                # sample only on position, take uniform here
                log_probs.fill_(1)

            #Probability only for normal words
            log_probs.narrow(1,0,onmt.Constants.voc_start).fill_(-float('inf'))

            #filter to target vocabulary
            word_probs = log_probs.gather(1,target.t())


            #check if already at end of sentence or eos -> all prob to current poisition
            mask =  (target[i].eq(onmt.Constants.PAD) +  target[i].eq(onmt.Constants.EOS)).unsqueeze(1).expand(-1,word_probs.size(1))
            oneHot = torch.zeros(1,target.size(0)).type_as(history_mask)
            oneHot[0,i] = 1

            if(self.cuda):
                mask = mask.cuda()

            word_probs.masked_fill_(mask,-float('inf'))
            word_probs.masked_fill_(mask*oneHot.expand(batch_size,-1),1)

            #add position distribution
            if (self.opt.sample_target_distribution == "outsideInside"):
                probs.fill_(-float('inf'))
                if(i % 2== 0):
                    probs.narrow(1,int(i/2),1).fill_(0)
                else:
                    length = target.ne(onmt.Constants.PAD).sum(0)
                    index = (length -2 - int(i/2)).clamp_(0,target.size(0)).unsqueeze(1)
                    probs.scatter_(1,index,0)
                #mask eos and padding
                m = target[i].eq(onmt.Constants.PAD) + target[i].eq(onmt.Constants.EOS)
                probs.masked_fill_(m.unsqueeze(1).expand(-1,probs.size(1)),0)
                word_probs += probs

            word_probs.masked_fill_(history_mask,-float('inf'))

            m = Categorical(probs=F.softmax(word_probs,dim=-1))
            sample = m.sample().unsqueeze(1)
            #print(sample.t())
            shuffle.append(sample.t())
            #ones = torch.ones(sample.size()).type_as(pad_tensor)
            history_mask.scatter_(1,sample,1)

            #index and direction


            #
        #print(shuffle[0])
        mapping=torch.cat(shuffle)
        new_target = target.gather(0,mapping)
        batch.tensors["target_output"] = new_target
        target_input = batch.tensors["target_input"]
        target_input[1:,:] = new_target[:-1,:]
        batch.tensors["target_input"] = target_input
        mapping = mapping.t().float()/(batch.get('tgt_length').float().unsqueeze(1).expand(-1,target.size(0)) - 2)
        batch.tensors["mapping"] = mapping
        batch.tensors["mapping_input"] = torch.cat([torch.zeros(mapping.size(0),1).fill_(-1).type_as(mapping),mapping[:,:-1]],1)

    def run(self, save_file=None):
        
        opt = self.opt
        model = self.model
        optim = self.optim
        
        # Try to load the save_file
        checkpoint = None
        if save_file:
            checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)
        
        if checkpoint is not None:
            print('Loading model and optim from checkpoint at %s' % save_file)
            self.model.load_state_dict(checkpoint['model'])
            
            if not opt.reset_optim:
                self.optim.load_state_dict(checkpoint['optim'])
                if 'batch_order' in checkpoint:
                    batch_order = checkpoint['batch_order']
                    iteration = checkpoint['iteration'] + 1
                else:
                    batch_order = None
                    iteration = 0
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))
                resume=True  
            else:
                batch_order = None
                iteration = 0
                resume=False

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume=False
        
        valid_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        
        self.start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                                 batch_order=batch_order,
                                                 iteration=iteration)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            self.save(epoch, valid_ppl)
            batch_order = None
            iteration = None
            resume = False
        
        
    
    
    

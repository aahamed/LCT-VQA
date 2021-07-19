import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import config
import logging
from torch.autograd import Variable

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class ArchitectLct(object):
    '''
    Architect for 3 stage Learning by Creating
    question answering Tests
    '''

    def __init__(self, ef_model, w_model,
            ef_optimizer, w_optimizer):
        self.ef_model = ef_model
        self.w_model = w_model
        self.ef_momentum = 0
        self.ef_weight_decay = 0
        # optimizer for architect
        self.optimizer = torch.optim.Adam(self.ef_model.arch_parameters(),
            lr=config.ARCH_LEARNING_RATE, betas=(0.5, 0.999),
            weight_decay=config.ARCH_WEIGHT_DECAY)
        self.ef_optimizer = ef_optimizer
        self.w_optimizer = w_optimizer

    def step(self, img_train, qst_train, label_train,
            img_valid, qst_valid, label_valid,
            ef_lr, w_lr):
        # import pdb; pdb.set_trace()
        self.ef_optimizer.zero_grad()
        self.w_optimizer.zero_grad()
        self.optimizer.zero_grad()
        # do second order approximation (unrolling) unconditionally
        # since first order approximation not possible
        self._backward_step_unrolled(img_train, qst_train,
                label_train, img_valid, qst_valid, label_valid,
                ef_lr, w_lr)
        self.optimizer.step()

    def _backward_step_unrolled(self, img_train, qst_train,
        label_train, img_valid, qst_valid, label_valid, ef_lr,
        w_lr):
        # unroll ef_model with one step gd using training data
        unrolled_ef_model = self._compute_unrolled_model(img_train,
                qst_train, label_train, ef_lr, self.ef_optimizer, 
                self.ef_model, self.ef_model._loss)
        # generate pseudo qa dataset using unrolled_ef_model
        pseudo_qst, pseudo_ans = unrolled_ef_model.generate(img_train)
        pseudo_ans = F.softmax(pseudo_ans / config.TEMPERATURE, dim=1)
        # unroll w_model with one step gd using pseudo qa data
        unrolled_w_model = self._compute_unrolled_model_2(img_train,
                qst_train, label_train, pseudo_qst, pseudo_ans, w_lr,
                self.w_optimizer, self.w_model, self.w_model._soft_loss,
                exp_zero_grad=36) 
        # compute grad w' vector
        unrolled_loss = unrolled_w_model._loss(img_valid, qst_valid,
                label_valid)
        grad_wprime = self._calc_grad(unrolled_loss,
                unrolled_w_model.parameters, exp_zero_grad=36)
        # compute kappa
        def pseudo_qa_fn():
            # generate pseudo qa dataset using unrolled_ef_model
            pseudo_qst, pseudo_ans = unrolled_ef_model.generate(img_train)
            pseudo_ans = F.softmax(pseudo_ans / config.TEMPERATURE, dim=1)
            return pseudo_qst, pseudo_ans
        def qa_fn():
            return qst_train, label_train
        kappa = self._hessian_vector_product_2(grad_wprime, img_train,
            qa_fn, pseudo_qa_fn, self.w_model, 
            self.w_model._soft_loss, unrolled_ef_model.parameters,
            exp_zero_grad=2)
        # compute gamma
        gamma = self._hessian_vector_product(kappa, img_train,
                qa_fn, self.ef_model,
                self.ef_model._loss, self.ef_model.arch_parameters,
                exp_zero_grad=0)
        # update gradients of arch_parameters
        for v, g in zip(self.ef_model.arch_parameters(), gamma):
          if v.grad is None:
            v.grad = Variable(g.data * ef_lr * w_lr)
          else:
            v.grad.data.copy_(g.data * ef_lr * w_lr)

        # log unrolled loss
        logging.info('| TRAIN SET | STAGE3 | W\'-Val-Loss: {:.4f}'
                .format(unrolled_loss.item()))

    def _compute_unrolled_model(self, img, qst, label,
            eta, optimizer, model, loss_fn, exp_zero_grad=0,
            weight_decay=0, momentum=0):
        '''
        Compute unrolled model

        exp_zero_grad: number of model parameters with gradient
        expected to be 0. For ef_model this should be 0 and
        for w_model this should be 36 due to the pretrained
        vgg img encoder. ( For debugging )
        '''
        loss = loss_fn(img, qst, label)
        theta = _concat(model.parameters()).data
        try:
            moment = _concat(optimizer.state[v]['momentum_buffer'] 
                    for v in model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        grads = self._calc_grad(loss, model.parameters, exp_zero_grad)
        dtheta = _concat(grads).data + weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
                theta.sub(eta, moment+dtheta), model)
        return unrolled_model
    
    def _compute_unrolled_model_2(self, img, qst, label,
            pseudo_qst, pseudo_label, eta, optimizer, model,
            loss_fn, exp_zero_grad=0, weight_decay=0, momentum=0):
        '''
        Compute unrolled model

        exp_zero_grad: number of model parameters with gradient
        expected to be 0. For ef_model this should be 0 and
        for w_model this should be 36 due to the pretrained
        vgg img encoder. ( For debugging )
        '''
        loss = loss_fn(img, qst, label, pseudo_qst, pseudo_label)
        theta = _concat(model.parameters()).data
        try:
            moment = _concat(optimizer.state[v]['momentum_buffer'] 
                    for v in model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        grads = self._calc_grad(loss, model.parameters, exp_zero_grad)
        dtheta = _concat(grads).data + weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
                theta.sub(eta, moment+dtheta), model)
        return unrolled_model
  
    def _construct_model_from_theta(self, theta, model):
        model_new = model.new()
        model_dict = model.state_dict()

        params, offset = {}, 0
        for k, v in model.named_parameters():
          v_length = np.prod(v.size())
          params[k] = theta[offset: offset+v_length].view(v.size())
          offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to( config.DEVICE )

    def _calc_grad(self, loss, param_fn,
            exp_zero_grad=0):
        '''
        Calculate gradient of loss w.r.t.
        to parameters returned by param_fn

        param_fn: generator yielding params
        Keeping this as a function since we need
        to iterate through generator twice ( once
        in torch.autograd )
        '''
        grads = list( torch.autograd.grad(loss, param_fn(),
                allow_unused=True) )
        num_zero_grad = 0
        params = param_fn()
        for i, p in enumerate( params ):
            if grads[i] is None:
                grads[i] = torch.zeros_like(p)
                num_zero_grad += 1
            else:
                assert grads[i].shape == p.shape
        assert num_zero_grad == exp_zero_grad
        return grads
    
    def _hessian_vector_product(self, vector, img,
            qa_fn, model, loss_fn, param_fn,
            r=1e-2, exp_zero_grad=0):
        '''
        Calculate hessian vector product using finite
        difference approx.

        qa_fn: function returning qa pair. This is needed
        to avoid having to set retain_graph=true since
        we have to backprop through the qa pair multiple
        times.
        '''
        R = r / _concat(vector).norm()
        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)
        qst, ans = qa_fn()
        loss = loss_fn(img, qst, ans)
        grads_p = self._calc_grad(loss, param_fn, exp_zero_grad)

        for p, v in zip(model.parameters(), vector):
            p.data.sub_(2*R, v)
        qst, ans = qa_fn()
        loss = loss_fn(img, qst, ans)
        grads_n = self._calc_grad(loss, param_fn, exp_zero_grad)

        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_2( self, vector, img,
            qa_fn, pseudo_qa_fn, model, loss_fn, param_fn,
            r=1e-2, exp_zero_grad=0 ):
        '''
        hessian vector prod for formulation 2
        '''
        R = r / _concat(vector).norm()
        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)
        qst, ans = qa_fn()
        pseudo_qst, pseudo_ans = pseudo_qa_fn()
        loss = loss_fn(img, qst, ans, pseudo_qst, pseudo_ans)
        grads_p = self._calc_grad(loss, param_fn, exp_zero_grad)

        for p, v in zip(model.parameters(), vector):
            p.data.sub_(2*R, v)
        qst, ans = qa_fn()
        pseudo_qst, pseudo_ans = pseudo_qa_fn()
        loss = loss_fn(img, qst, ans, pseudo_qst, pseudo_ans)
        grads_n = self._calc_grad(loss, param_fn, exp_zero_grad)

        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


import torch
import numpy as np
import torch.nn as nn
import config
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model):
    # self.network_momentum = args.momentum
    # self.network_weight_decay = args.weight_decay
    self.network_momentum = 0
    self.network_weight_decay = 0
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=config.ARCH_LEARNING_RATE, betas=(0.5, 0.999),
        weight_decay=config.ARCH_WEIGHT_DECAY)

  def _compute_unrolled_model(self, img, qst, label, eta, network_optimizer):
    loss = self.model._loss(img, qst, label)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    loss.backward()
    grad_model = [v.grad.data for v in self.model.parameters()]
    dtheta = _concat(grad_model).data + self.network_weight_decay * theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, img_train, qst_train, label_train,
          img_valid, qst_valid, label_valid, 
          eta=None, network_optimizer=None, unrolled=True):
    # import pdb; pdb.set_trace()
    self.optimizer.zero_grad()
    if unrolled:
        # assert False and 'unrolled not supported'
        # self._backward_step_unrolled( input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        self._backward_step_unrolled( img_train, qst_train,
                label_train, img_valid, qst_valid, label_valid,
                eta, network_optimizer )
    else:
        self._backward_step( img_valid, qst_valid, label_valid)
    self.optimizer.step()

  def _backward_step(self, img_valid, qst_valid, label_valid):
    loss = self.model._loss(img_valid, qst_valid, label_valid)
    loss.backward()

  def _backward_step_unrolled(self, img_train, qst_train,
          label_train, img_valid, qst_valid, label_valid,
          eta, network_optimizer ):
    
    unrolled_model = self._compute_unrolled_model(img_train, qst_train, label_train,
            eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(img_valid, qst_valid, label_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, img_train,
            qst_train, label_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, img, qst, label, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(img, qst, label)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(img, qst, label)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

from FrEIA.framework import *
from FrEIA.modules import *

import config as c

nodes = [InputNode(c.ndim_y + c.ndim_z, name='input')]

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, c.hidden_layer_sizes), nn.ReLU(),
                         nn.Linear(c.hidden_layer_sizes,  c_out))

for i in range(c.N_blocks):
    nodes.append(Node(nodes[-1].out0,
                  GLOWCouplingBlock,
                  {'subnet_constructor':subnet_fc, 'clamp':c.exponent_clamping},
                  name='coupling_{}'.format(i)))

    if c.use_permutation:
        nodes.append(Node(nodes[-1].out0,
                      PermuteRandom,
                      {'seed':i}, name='permute_{}'.format(i)))

nodes.append(OutputNode([nodes[-1].out0], name='output'))

model = ReversibleGraphNet(nodes, verbose=c.verbose_construction)
model.to(c.device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)

gamma = (c.final_decay)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

def optim_step():
    #for p in params_trainable:
        #print(torch.mean(torch.abs(p.grad.data)).item())
    optim.step()
    optim.zero_grad()

def scheduler_step():
    #weight_scheduler.step()
    pass

def save(name):
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    #try:
    #    optim.load_state_dict(state_dicts['opt'])
    #except ValueError:
    #    print('Cannot load optimizer for some reason or other')

"""
神经网络程序
"""

import torch
import torch.nn as nn


# 全连接神经网络
class fully_connected(nn.Module):
    
    def __init__(self, units, layers, in_dim, out_dim, activation='ReLu'):
        super(fully_connected, self).__init__()
        
        self.activation = {'ReLU': nn.ReLU(), 'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh()}[activation]
        
        self.in_layer = [nn.Linear(in_dim, units), self.activation]
        self.hidden_layer_lst = [nn.Linear(units, units), self.activation] * (layers - 2)
        self.feature_layer = [nn.Linear(units, out_dim)]
        
        self.fcn = nn.ModuleList(self.in_layer + self.hidden_layer_lst + self.feature_layer)
        
    def forward(self, x, z):
        X = torch.concat([x, z], dim=1)
        for h in self.fcn:
            X = h(X)
        return X











"""
RealNVP Code For Solving 2-d PDE Inverse Problems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0")



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



class WeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim, weight_norm=True, scale=False):
        
        super(WeightNormFC, self).__init__()

        if weight_norm:
            self.dense = nn.utils.weight_norm(nn.Linear(in_dim, out_dim))
            
            if not scale:
                self.dense.weight_g.data = torch.ones_like(self.dense.weight_g.data)
                self.dense.weight_g.requires_grad = False    # freeze scaling
        else:
            self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.dense(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, weight_norm):
        """Initializes a ResidualBlock.

        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            WeightNormFC(dim, dim, weight_norm),
            nn.ReLU(),
            WeightNormFC(dim, dim, weight_norm),
            nn.ReLU(),
            WeightNormFC(dim, dim, weight_norm))
    

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.res_block(x)    
    
    
class ResidualModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, weight_norm, hps):
        """Initializes a ResidualModule.

        Args:
            in_dim: number of input features.
            dim: number of features in residual blocks.
            out_dim: number of output features.
            res_blocks: number of residual blocks to use.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualModule, self).__init__()
        self.res_blocks = hps.res_blocks
        self.skip = hps.skip
        
        if self.res_blocks > 0:
            self.in_block = WeightNormFC(in_dim, dim, weight_norm)
            self.core_block = nn.ModuleList(
                [ResidualBlock(dim, weight_norm) 
                for _ in range(self.res_blocks)])
            self.out_block = nn.Sequential(
                nn.ReLU(),
                WeightNormFC(dim, out_dim, weight_norm))
            
            if self.skip:
                self.in_skip = WeightNormFC(dim, dim, weight_norm)
                self.core_skips = nn.ModuleList(
                    [WeightNormFC(dim, dim, weight_norm) for _ in range(self.res_blocks)])
     

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        assert self.res_blocks > 0, "ResBlock的数量应该为正整数!"
        x = self.in_block(x)
        if self.skip:
            out = self.in_skip(x)
        for i in range(len(self.core_block)):
            x = self.core_block[i](x)
            if self.skip:
                out = out + self.core_skips[i](x)
        if self.skip:
            x = out
        return self.out_block(x)

    
    
class DenseModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, weight_norm):
        
        super(DenseModule, self).__init__()
        
#         self.block = nn.Sequential(WeightNormFC(in_dim, dim, weight_norm),
#                                    nn.ReLU(),
#                                    WeightNormFC(dim, int(0.5 * dim), weight_norm),
#                                    nn.ReLU(),
#                                    WeightNormFC(int(0.5 * dim), int(0.5 * dim), weight_norm),
#                                    nn.ReLU(),
#                                    WeightNormFC(int(0.5 * dim), dim, weight_norm),
#                                    WeightNormFC(dim, out_dim, weight_norm))
        self.block = nn.Sequential(WeightNormFC(in_dim, dim, weight_norm),
                                   nn.ReLU(),
                                   WeightNormFC(dim, dim, weight_norm),
                                   nn.ReLU(),
                                   WeightNormFC(dim, out_dim, weight_norm))


    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.block(x)

    
class AbstractCoupling(nn.Module):
    def __init__(self, mask_config, hps):
        """
        Initializes an AbstractCoupling.
        """
        super(AbstractCoupling, self).__init__()
      
        self.weight_norm = hps.weight_norm
        self.coupling_bn = hps.coupling_bn
        self.mask_config = mask_config

    def batch_stat(self, x):
        """Compute (spatial) batch statistics.

        Args:
            x: input minibatch.
        Returns:
            batch mean and variance.
        """
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.mean((x - mean)**2, dim=0, keepdim=True)
        return mean, var

    
class ChannelwiseAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        
        super(ChannelwiseAffineCoupling, self).__init__(mask_config, hps)
        
#         self.s_net = ResidualModule(in_out_dim//2, mid_dim, in_out_dim//2, self.weight_norm, hps)
#         self.t_net = ResidualModule(in_out_dim//2, mid_dim, in_out_dim//2, self.weight_norm, hps)
        self.s_net = DenseModule(in_out_dim//2, mid_dim, in_out_dim//2, self.weight_norm)
        self.t_net = DenseModule(in_out_dim//2, mid_dim, in_out_dim//2, self.weight_norm)
        
#         self.in_bn = nn.BatchNorm1d(in_out_dim//2)
#         self.out_bn = nn.BatchNorm1d(in_out_dim//2, affine=False)
        
    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, D] = list(x.size())
        
        if self.mask_config:
            (on, off) = x.split(D//2, dim=1)
        else:
            (off, on) = x.split(D//2, dim=1)
        
#         off = self.in_bn(off)
        
        if reverse == False:
            # Forward
            self.scale = self.s_net(off)
            self.trans = self.t_net(off)
            
            on = (torch.exp(0.636 *2* torch.atan(self.scale))) * on + self.trans
            log_det_J = torch.sum(0.636 *2* torch.atan(self.scale), dim=1)
            
            if self.coupling_bn:
                mean, var = self.batch_stat(on)
#                 on = self.out_bn(on)
                on = (on - mean) / torch.sqrt(var + 1e-5)
                log_det_J = log_det_J - 0.5 * torch.sum(torch.log(var + 1e-5), dim=1)

        else:
            # Reverse
            self.scale = self.s_net(off)
            self.trans = self.t_net(off)
            
            log_det_J = torch.zeros(off.shape[0]).to(x.device)
            
            if self.coupling_bn:
                mean, var = self.batch_stat(on)
#                 on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
                on = on * torch.sqrt(var + 1e-5) + mean
                log_det_J = 0.5 * torch.sum(torch.log(var + 1e-5), dim=1)
            
            on = (on - self.trans) / (torch.exp(0.636 *2* torch.atan(self.scale)))
            log_det_J = log_det_J - torch.sum(0.636 *2* torch.atan(self.scale), dim=1)
        
        if self.mask_config:
            x = torch.cat((on, off), 1)
        else:
            x = torch.cat((off, on), 1)
        
        return x, log_det_J
    
    
class ChannelwiseCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        """Initializes a ChannelwiseCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseCoupling, self).__init__()

        self.coupling = ChannelwiseAffineCoupling(in_out_dim, mid_dim, mask_config, hps)


    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

    
class Permute_data(nn.Module):
    '''
    Args:
    x: input (B x C x H x W)
    To permute the data channel-wise. This operation called during both the training and testing.
    '''
    def __init__(self, in_out_dim, seed):
        super(Permute_data, self).__init__()
        # fixed seed
        np.random.seed(seed)
        self.Permute_data = np.random.permutation(in_out_dim)
        np.random.seed()
        Permute_sample = np.zeros((self.Permute_data.shape))
        for i, j in enumerate(self.Permute_data):
            Permute_sample[j] = i
        self.Permute_sample = Permute_sample

    def forward(self, x, reverse=False):
        if reverse == False:
            y = x[:, self.Permute_data]
            return y
        else:
            y = x[:, self.Permute_sample]
            return y
        
    
class RealNVP(nn.Module):
    def __init__(self, in_out_dim, hps):
        """Initializes a RealNVP.

        Args:
            datainfo: information of dataset to be modeled.
            hps: the set of hyperparameters.
        """
        
        super(RealNVP, self).__init__()
        self.hps = hps
        mid_dim = hps.base_dim
        self.in_out_dim = in_out_dim
        self.res_blocks = hps.res_blocks
        self.skip = hps.skip

        # Channelwise Coupling Layers
        self.s1_chan = self.channelwise_combo(in_out_dim, mid_dim, hps)
        
        
    def channelwise_combo(self, in_out_dim, mid_dim, hps):
        """Construct a combination of channelwise coupling layers.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            hps: the set of hyperparameters.
        Returns:
            A combination of channelwise coupling layers.
        """
        return nn.ModuleList([
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps)])

    def forward(self, x, reverse=False):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        log_det_J = torch.zeros(x.size()[0]).to(x.device)

        if reverse == False:
            for i in range(len(self.s1_chan)):
                x, inc = self.s1_chan[i](x)
                log_det_J = log_det_J + inc
#                 x = self.s1_perm[i](x)
        else:            
            for i in reversed(range(len(self.s1_chan))):
#                 x = self.s1_perm[i](x, reverse=True)
                x, inc = self.s1_chan[i](x, reverse=True)
                log_det_J = log_det_J + inc
                
        return x, log_det_J
        
    def weight_reg(self, x):
        
        weight_scale = None
        for name, param in self.named_parameters():
            param_name = name.split('.')[-1]
            if param_name in ['weight_g', 'scale'] and param.requires_grad:
                if weight_scale is None:
                    weight_scale = torch.pow(param, 2).sum()
                else:
                    weight_scale = weight_scale + torch.pow(param, 2).sum()
        return weight_scale

            

        

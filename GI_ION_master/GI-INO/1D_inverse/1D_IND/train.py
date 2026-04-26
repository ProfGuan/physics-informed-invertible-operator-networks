from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as distributions

from FrEIA.framework import *
from FrEIA.modules import *

import config as c

import losses
import model
import monitoring




assert c.train_loader and c.test_loader, "No data loaders supplied"


def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)


def loss_max_likelihood(out_x,jac_back):
    x_samples = out_x
    jac = jac_back
    neg_log_likeli =( 0.5  * torch.sum((x_samples[:, :c.ndim_x])**2, 1)
                     +0.5 / c.add_y_noise**2 * torch.sum((x_samples[:, c.ndim_x:c.ndim_x+c.ndim_y])**2, 1)
                     +0.5 / c.add_pad_noise**2 * torch.sum((x_samples[:, c.ndim_x+c.ndim_y:])**2, 1)
                     - jac )

    return torch.mean(neg_log_likeli)

def loss_forward_fit(out, y):
    l_forw_fit =  losses.l2_fit(out[:, -c.ndim_y:], y[:, -c.ndim_y:])
    return l_forw_fit

def loss_back_fit(out_x, x):
    l_back_fit =  losses.l2_fit(out_x[:,:c.ndim_x], x[:,:c.ndim_x])
    return l_back_fit


def log_prob_gauss(x,sigema):
    gauss = distributions.Normal(
    torch.tensor(0).to(c.device), torch.tensor(sigema).to(c.device))
    return gauss.log_prob(x).sum(dim=1).reshape(1,-1)

def log_prob_uniform(x, lb, ub, k=200):
    lb = torch.tensor(lb).to(x.device)
    ub = torch.tensor(ub).to(x.device)
    # 计算对数概率密度
    term1 = (torch.tanh(k * (x - lb)) + 1) / 2
    term2 = (torch.tanh(k * (ub - x)) + 1) / 2
    log_gate = torch.log(term1) + torch.log(term2)
    return torch.sum(log_gate - torch.log(ub - lb),dim=1)

def normal_dist(out_z,z_star):
    latent = distributions.Normal(
    torch.tensor(0).to(c.device), torch.tensor(1).to(c.device))
    return torch.sum(latent.log_prob(out_z) - latent.log_prob(z_star), dim=1).reshape(1,-1)


def log_Px(out_x_xpad):
    out_x = out_x_xpad[:, 0:c.ndim_x]
    out_nosie = out_x_xpad[:, c.ndim_x:c.ndim_x+c.ndim_y]
    out_x_pad = out_x_xpad[:, c.ndim_x+c.ndim_y:]
    log_p_x= log_prob_gauss(out_nosie,1)
    log_p_noise = log_prob_gauss(out_nosie,1*c.add_y_noise)
    log_p_xpad = log_prob_gauss(out_x_pad,1*c.add_pad_noise)

    return  log_p_noise +log_p_xpad +log_p_x

#     
def independence_loss(out_x, log_det_J, out_x_star, log_det_J_star,out_z,z_star):
      
    return torch.square(log_Px(out_x) + log_det_J - log_Px(out_x_star) - log_det_J_star - normal_dist(out_z,z_star)).mean()

def similarity_loss(out_x, log_det_J, out_x_star, log_det_J_star):
    
    return torch.square(log_Px(out_x) - log_Px(out_x_star)).mean()

def train_epoch(i_epoch, test=False):

    if not test:
        model.model.train()
        loader = c.train_loader

    if test:
        model.model.eval()
        loader = c.test_loader
        nograd = torch.no_grad()
        nograd.__enter__()


    batch_idx = 0
    loss_history = []
    total_loss_history = []

    for x, y in loader:

        if batch_idx > c.n_its_per_epoch:
            break

        batch_losses = []
        batch_losses_weighted = []
        batch_idx += 1
        x, y = Variable(x).to(c.device), Variable(y).to(c.device)
        while y.dim() > 2:
            y=y.squeeze(1) 
  
        noise = c.add_y_noise * noise_batch(c.ndim_y)
 
        if c.add_y_noise > 0:
            y += noise
        x = torch.cat((x, noise), dim=1)
        if c.ndim_pad_x:
            x = torch.cat((x, c.add_pad_noise * noise_batch(c.ndim_pad_x)), dim=1)
        if c.ndim_pad_zy:
            y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)

        z = noise_batch(c.ndim_z)
        y = torch.cat((z, y), dim=1)


        out_z_y = model.model(x)[0]
        out_x_xpad ,jac_back = model.model(y, rev=True)


        reversed= model.model(out_z_y, rev=True)
        rec_x_xpad = reversed[0]

        rec_log_det_J = reversed[1]

        out_z, out_y = out_z_y[:, 0:c.ndim_z], out_z_y[:, c.ndim_z:(c.ndim_z+c.ndim_y)]

        z_star = noise_batch(c.ndim_z)

        y_star = torch.cat((z_star, out_y), dim=1)
        y_simi = torch.cat((out_z, y[:,-c.ndim_y:]), dim=1)
    
        if c.ndim_pad_zy:
            y_simi = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y_simi), dim=1)
            y_star = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y_star), dim=1)
        rec_x_xpad_simi,rec_log_det_J_simi = model.model(y_simi, rev=True)
        rec_x_xpad_star,rec_log_det_J_star = model.model(y_star, rev=True)

        # out_x_star, out_xpad_star = out_x_xpad_star[:, 0:c.ndim_x], out_x_xpad_star[:, c.ndim_x:(c.ndim_x+c.ndim_y)]


        if c.train_max_likelihood:
            loss_ml = loss_max_likelihood(out_x_xpad,jac_back)
            batch_losses.append(loss_ml)
            batch_losses_weighted.append(c.lambd_max_likelihood *loss_ml)   
            
        if c.train_forward_fit:
            loss_forwardfit = loss_forward_fit(out_z_y, y)
            batch_losses.append(loss_forwardfit)
            batch_losses_weighted.append(c.lambd_fit_forw*loss_forwardfit)  
            
     
        if c.train_independence_loss:
            loss_ind1=independence_loss(rec_x_xpad, rec_log_det_J, rec_x_xpad_star, rec_log_det_J_star,out_z,z_star)
            loss_ind2=independence_loss(rec_x_xpad, rec_log_det_J, rec_x_xpad_simi, rec_log_det_J,out_z,out_z)
            loss_ind=loss_ind1+loss_ind2
            batch_losses.append(loss_ind)
            batch_losses_weighted.append(c.lambd_independence_loss*loss_ind)

              

            
        l_total = sum(batch_losses_weighted)
        if c.show_total_loss:
            batch_losses.append(l_total)
  
        
        loss_history.append([l.item() for l in batch_losses])
        total_loss_history.append(l_total.item())

        if not test:
            l_total.backward()
            # 梯度裁剪：按范数裁剪（关键步骤）
            nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=2.0)
            model.optim_step()
           
        if test:

            if c.test_time_functions:
                out_x_xpad = model.model(y, rev=True)
                for f in c.test_time_functions:
                    f(out_x_xpad, out_z_y, x, y)

            nograd.__exit__(None, None, None)

    return np.mean(loss_history, axis=0), np.array(total_loss_history).mean()

def main():
    monitoring.restart()
    
    try:
        monitoring.print_config()
        t_start = time()
        optimal_loss = 1e+10
        best_epoch = 0
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * c.pre_lowlevel

            train_losses, total_train_loss = train_epoch(i_epoch)
            test_losses, total_test_loss = train_epoch(i_epoch, test=True)
            if c.show_test_loss:
                monitoring.show_loss(np.concatenate([train_losses, test_losses]))
            else:
                monitoring.show_loss(train_losses)
            model.scheduler_step()
            
            if total_test_loss < optimal_loss:
                optimal_loss = total_test_loss
                best_epoch = i_epoch
                model.save(c.filename_out)
            # if i_epoch-best_epoch>500:
            #     print(f"Early stopping at epoch {i_epoch+1}")
            #     break

#             if (i_epoch % 50 == 0):
#                 print('Saving results at iteration:'+str(i_epoch))
#                 model.save(c.filename_out + '_' + str(i_epoch))

    except KeyboardInterrupt:
        model.save(c.filename_out + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
        model.save(c.filename_out)
        

if __name__ == "__main__":
    main()

import os
import torch
from torch import nn
from torch.nn.functional import relu, elu
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import numpy as np
from torch.utils.data import DataLoader
import math
import time
import opt_einsum as oe
def construct_cg(l_in,l_filter,l_out):
    from sympy.physics.quantum.cg import CG
    from sympy.physics.wigner import wigner_3j
    from sympy import S
    cg = np.zeros((2*l_in+1, 2*l_filter+1, 2*l_out+1))

    for idx_m_in, m_in in enumerate(range(-l_in, l_in+1)):
        for idx_m_filter, m_filter in enumerate(range(-l_filter, l_filter+1)):
            for idx_m_out, m_out in enumerate(range(-l_out, l_out+1)):
                cg[idx_m_in, idx_m_filter,idx_m_out] =CG(l_in, m_in, l_filter, m_filter, l_out, m_out).doit()
    return torch.from_numpy(cg.astype(np.float32))

def grad_spherical(neighbor_grad): # shperical term (x,y,z,-> shprical cood)
    factor = (math.pi)**(-0.5)  # 1/sqrt(pi)
    batch_size = neighbor_grad.size(0) # 8   r.shape = 8,4,4,3
    num_p = neighbor_grad.size(1) # p
    
    grad_Y1 = torch.zeros( (batch_size,  num_p, 3, 1), device = neighbor_grad.device, dtype=neighbor_grad.dtype )  #(tensor(8,4,4,3))
    grad_Y2 = torch.zeros( (batch_size,  num_p, 3, 1), device = neighbor_grad.device, dtype=neighbor_grad.dtype )  #(tensor(8,4,4,3))

    grad_norm = torch.linalg.norm(neighbor_grad, axis= -2, keepdim = True)

    ##real

    grad_Y1[:,:,1,:] = math.sqrt(3/4)*factor*(neighbor_grad[:,:,1,:])
    grad_Y1[:,:,2,:] = math.sqrt(3/4)*factor*(neighbor_grad[:,:,2,:])
    grad_Y1[:,:,0,:] = math.sqrt(3/4)*factor*(neighbor_grad[:,:,0,:])

    ##complex

    #grad_Y1[:,:,0,:] = math.sqrt(3/8)*factor*(neighbor_grad[:,:,0,:])
    #grad_Y1[:,:,1,:] = math.sqrt(3/4)*factor*(neighbor_grad[:,:,2,:])
    #grad_Y1[:,:,2,:] =-math.sqrt(3/8)*factor*(neighbor_grad[:,:,0,:])

    #grad_Y2[:,:,0,:] =-math.sqrt(3/8)*factor*(neighbor_grad[:,:,1,:])
    #grad_Y2[:,:,2,:] =-math.sqrt(3/8)*factor*(neighbor_grad[:,:,1,:])

    mask = neighbor_mask
    grad_Y1 = grad_Y1.masked_fill( mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )
    grad_Y2 = grad_Y1.masked_fill( mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )

    grad_Y1 = grad_Y1 / (grad_norm + 1e-9)
    grad_Y2 = grad_Y2 / (grad_norm + 1e-9)

    #return grad_Y1, grad_Y2
    return grad_Y1

class RBF(nn.Module):
    def __init__(self, rbf_low=0.0, rbf_high=5.0, rbf_count=250, requires_grad=False, exp=False, gamma=None, layernorm=False, eps=1e-6 ) :
        super().__init__()
        if exp:
            self.centers = nn.Parameter(torch.exp( torch.from_numpy( (np.mgrid[rbf_low:np.log(rbf_high):1j*rbf_count].astype(np.float32) ))) , requires_grad=requires_grad )
        else:
            self.centers = nn.Parameter( torch.from_numpy( (np.mgrid[rbf_low:rbf_high:1j*rbf_count].astype(np.float32) )) , requires_grad=requires_grad )
        if (gamma is None):    
            self.gamma   = nn.Parameter( 1.0/(self.centers**2) , requires_grad=requires_grad)
        else: 
            self.gamma   = gamma 

        self.layernorm = layernorm            
        self.eps = eps
        self.requires_grad= requires_grad
    def forward(self, inp ):
        size= np.ones( len(inp.size()), dtype=np.int64 )
        size[-1] =-1
        out =  torch.exp ( -self.gamma * ( inp - self.centers.reshape( tuple(size.tolist()) ) )**2 ) # (batch-size, num_ponits, num_points, rbf_count)

        if(self.layernorm):
            mean = torch.mean(out, dim=-1, keepdims=True)
            std  = torch.std (out, dim=-1, keepdims=True)
            return (out-mean) / (std+self.eps)
        return out            
    def __str__(self):
        np.set_printoptions(2,threshold=10)
        string= f"RBF kernel with {self.gamma} and {self.centers.squeeze().detach().cpu().numpy()} (requires_grad={self.requires_grad})"
        np.set_printoptions()
        return string


class TensorFieldLayer(nn.Module):
    def __init__(self, l_in, l_filter, l_out, n_feature_in, n_feature_out, n_feature_radial ):
        # l_in: angular momentum for input data
        # l_filter: angular momentum for spherical harmonics
        # l_out: angular momentum for output data
        super().__init__()
        self.l_in, self.l_filter, self.l_out = l_in, l_filter, l_out
        self.n_feature_in, self.n_feature_out = n_feature_in, n_feature_out

        self.radial_embedding = nn.Sequential( nn.Linear( n_feature_radial, n_feature_in),
                                               nn.ReLU(),
                                               #nn.GELU(),
                                               nn.Linear( n_feature_in , n_feature_in),
                                             )
        self.self_interact = nn.Linear(n_feature_in, n_feature_out, bias=False) # self-interaction
        self.bias          = nn.Linear(1,1)
        # fix weight as 1 (only use bias)
        with torch.no_grad():
            self.bias.weight= nn.Parameter(torch.ones_like(self.bias.weight), requires_grad=False)
            #self.linear2.weight.require_grad =False

        self.coef = nn.Parameter(construct_cg(l_in, l_filter, l_out), requires_grad=False)
        assert torch.sum(torch.abs(self.coef))>1e-6, f"All cg coefficients are 0, please check l_in({l_in}),l_filter({l_filter}),l_out({l_out})"

    def spherical(self, r): # shperical term (x,y,z,-> shprical cood)
        # r, normalized positions, (batch_size, num_points, num_points, 3)
        # return (batch_size, num_points, num_points, 2*l_filter+1)
        # please refer https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_spherical_harmonics.py
        import math
        factor = (math.pi)**(-0.5)  # 1/sqrt(pi)
        batch_size = r.size(0) # 8   r.shape = 8,4,4,3
        num_p = r.size(1) # p
        num_q = r.size(2) # q
        Y1 = torch.zeros( (batch_size,  num_p, num_q, 2*self.l_filter+1), dtype=r.dtype, device = r.device )  
        Y2 = torch.zeros( (batch_size,  num_p, num_q, 2*self.l_filter+1), dtype=r.dtype, device = r.device )  
        if (self.l_filter==0):
            Y1 = Y1+0.5* factor
        elif(self.l_filter==1):
            Y1[:,:,:,0] = math.sqrt(3/8)*factor*(r[:,:,:,0])
            Y1[:,:,:,1] = math.sqrt(3/4)*factor*(r[:,:,:,2])
            Y1[:,:,:,2] =-math.sqrt(3/8)*factor*(r[:,:,:,0])

            Y2[:,:,:,0] =-math.sqrt(3/8)*factor*(r[:,:,:,1])
            Y2[:,:,:,2] =-math.sqrt(3/8)*factor*(r[:,:,:,1])
        elif(self.l_filter==2):
            Y1[:,:,:,0] = 0.25*math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,0]-r[:,:,:,1]*r[:,:,:,1])
            Y1[:,:,:,1] = 0.5 *math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,2])
            Y1[:,:,:,2] = 0.25*math.sqrt(5)   *factor * (3*r[:,:,:,2]**2 -1)
            Y1[:,:,:,3] =-0.5 *math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,2])
            Y1[:,:,:,4] = 0.25*math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,0]-r[:,:,:,1]*r[:,:,:,1])

            Y2[:,:,:,0] =-0.5 *math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,1])
            Y2[:,:,:,1] =-0.5 *math.sqrt(15/2)*factor * (r[:,:,:,1] * r[:,:,:,2])
            Y2[:,:,:,3] =-0.5 *math.sqrt(15/2)*factor * (r[:,:,:,1] * r[:,:,:,2])
            Y2[:,:,:,4] = 0.5 *math.sqrt(15/2)*factor * (r[:,:,:,0] * r[:,:,:,1])
        elif(self.l_filter==3):
            pass
        else:
            raise NotImplementedError(f'The spherical hamonics for the given l_filter ({self.l_filter}) is not implemented')

        index = (torch.linalg.norm(r, dim=-1)==0.0)

        Y1[index]=1.0
        Y2[index]=1.0
        return Y1, Y2

    def forward(self,rbf, unit_vector, inp1, inp2, mask):
        # pos (batch_size, num_points, 3)
        # inp (batch_size, num_points, 2*l_in+1, n_feature_in)
        # return (batch_size, num_points, 2*l_out+1, n_feature_out)
        with torch.no_grad():
            Y1, Y2 = self.spherical(unit_vector)
            Y1 = Y1.masked_fill( mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )
            Y2 = Y2.masked_fill( mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )
        radial = self.radial_embedding(rbf)
        radial = radial.masked_fill( mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )
       
#        torch.cuda.synchronize()
#        st = time.time()
#        out1= torch.einsum('abc,npqb,npqf,npaf->nqcf', self.coef, Y1, radial, inp1) - torch.einsum('abc,npqb,npqf,npaf->nqcf', self.coef, Y2, radial, inp2)
#        out2= torch.einsum('abc,npqb,npqf,npaf->nqcf', self.coef, Y2, radial, inp1) + torch.einsum('abc,npqb,npqf,npaf->nqcf', self.coef, Y1, radial, inp2)
#        torch.cuda.synchronize()
#        et = time.time()
#        print('    einsum',et-st)
#        
#        torch.cuda.synchronize()
#        st = time.time()
        out1= oe.contract('abc,npqb,npqf,npaf->nqcf', self.coef, Y1, radial, inp1) - oe.contract('abc,npqb,npqf,npaf->nqcf', self.coef, Y2, radial, inp2)
        out2= oe.contract('abc,npqb,npqf,npaf->nqcf', self.coef, Y2, radial, inp1) + oe.contract('abc,npqb,npqf,npaf->nqcf', self.coef, Y1, radial, inp2)
#        torch.cuda.synchronize()
#        et = time.time()
#        print('opt_einsum',et-st)
#        exit(-1)
        norm = torch.linalg.norm(torch.stack([out1, out2], dim=0),dim=[0,-2],keepdim=True)[0] # keep only -2 dim 
#        print('out1', out1)
#        print('y1', Y1.shape)
#        print('radial', radial.shape)
#        print('inp1', inp1.shape)
        out1 = out1 / (norm+1e-9)
        out2 = out2 / (norm+1e-9)
#        print('out/n', out1)         
        out1 = self.self_interact(out1)
        out2 = self.self_interact(out2)
#        print('out_int', out1) 
        nonlinear =  relu( self.bias( torch.linalg.norm(torch.stack([out1, out2], dim=0), dim=[0,-2] ).unsqueeze(-1) ).transpose(-1,-2) )
        out1 = out1*nonlinear
        out2 = out2*nonlinear
#        print('out_non',out1)
        return out1, out2

    def __str__(self):
        return f"TensorFieldLayer with l_values({self.l_in}, {self.l_filter}, {self.l_out}) and n_features ({self.n_feature_in}, {self.n_feature_out}) )"


class MyModel(pl.LightningModule):
    def __init__(self, list_l_inout, list_l_filter, list_n_feature, learning_rate,\
                       num_in_feature, num_hid_feature, num_out_feature, rbf_kernel = RBF()  
                 ):
        from copy import deepcopy
        from itertools import product
        super(MyModel, self).__init__()
        # list_l_inout  = [ ([0,1],[0,1,2]) , ([0,1,2], [0]) ]
        # list_l_filter   = [0,1,2,3]
        # list_n_feature= [512, 512, 56 ]
        assert len(list_l_inout) > 0
        assert len(list_l_inout) == len(list_n_feature)-1

        self.rbf_kernel = rbf_kernel
        rbf_count = rbf_kernel.centers.size(-1)

        self.list_l_inout   = deepcopy(list_l_inout)
        self.list_l_filter  = deepcopy(list_l_filter)
        self.list_n_feature = deepcopy(list_n_feature)

        self.list_layers = nn.ModuleList([])
        for idx_layer, (list_l_in, list_l_out) in enumerate(self.list_l_inout):
            layers = {}
            for l_in, l_out in product(list_l_in, list_l_out):
                for l_filter in self.list_l_filter:
                    try:
                        # nn.ModuleDict does not support tuple key therefore, typecasting is applied
                        layers[f'{l_in},{l_filter},{l_out}'] =  TensorFieldLayer(l_in, l_filter, l_out, \
                                                                                 self.list_n_feature[idx_layer], \
                                                                                 self.list_n_feature[idx_layer+1], rbf_count)
                    except AssertionError:
                        # The case that all CG coefficients are 0 will be skipped
                        continue
            self.list_layers.append(nn.ModuleDict(layers))


        self.inp_rbf_r = RBF(rbf_low = -15.0, rbf_high = 5.0, rbf_count = num_in_feature, gamma=None, layernorm=True)
        self.inp_rbf_c = RBF(rbf_low = -15.0, rbf_high = 5.0, rbf_count = num_in_feature, gamma=None, layernorm=True)

        self.grad_rbf_r = nn.Linear(1, num_out_feature, bias = False)
        self.grad_rbf_c = nn.Linear(1, num_out_feature, bias = False)


        self.readout = nn.Sequential( 
                                     nn.Linear(2*num_out_feature, num_out_feature),
                                     nn.ReLU(),
                                     #nn.Linear(num_out_feature, 2)
                                     nn.Linear(num_out_feature, 1)
                                    )

        self.learning_rate = learning_rate
        self.eps = 1e-8

        self.recover_f = lambda out, ke: torch.exp( torch.nn.functional.softplus(out) )* ke
        #self.recover_f = lambda out, ke: torch.exp( torch.abs(out) )* ke
        #self.recover_f = lambda out, ke: torch.exp( (out)**2 )* ke

    def step(self, batch):
        inp, kinetic_e, kinetic_v, grid, neighbor_inp, neighbor_grad, neighbor_lapla, neighbor_veff, neighbor_grid, neighbor_mask = batch
        batch_size = inp.size(0)
        max_neighbor = neighbor_mask.size(1)

        with torch.no_grad():
            #print(max_neighbor, neighbor_grad.size())
            TF_factor = (3*np.pi**2 )
            ke_w  = (1/8) * (torch.linalg.norm(inp[:,1:4], dim = 1))**2 / inp[:,0]
#            ke_tf = 627.5095 * (3/10)*(TF_factor**(2/3))*(inp[:,0]**(5/3))
#            ke_w  = 627.5095 * (1/8) * (torch.linalg.norm(inp[:,1:4], dim = 1))**2 / inp[:,0]
#            d_tf  = 627.5095 * (1/2) * (TF_factor**(2/3))*(inp[:,0]**(2/3))
#            d_w   = 627.5095 * (((1/8) * (torch.linalg.norm(inp[:,1:4], dim = 1))**2 / (inp[:,0]**2)) - ((1/4) * inp[:,-1] /  (inp[:,0]**2)))
#            kinetic_e = 627.5095 * kinetic_e
            

            R = (grid[:,:3].unsqueeze(1) - neighbor_grid[:,:,:3]).unsqueeze(2)    # (batch_size, 1, num_neighbor_points, 3)
            D = torch.linalg.norm(R, ord=2, dim=-1, keepdim=True) # (batch_size, 1, num_neighbor_points, 1)
            rbf = self.rbf_kernel(D)
            unit_vector = ( R/(D+self.eps)).masked_fill(D < self.eps, 0.0 )
            # clear out meaningless values correponding dummy points

            rbf = rbf.masked_fill( neighbor_mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )
            unit_vector = unit_vector.masked_fill( neighbor_mask.unsqueeze(-1).unsqueeze(-1)==False, 0 )

        #grad_norm_inp = self.inp_rbf_r(torch.linalg.norm(neighbor_grad, axis= -2, keepdim = True))
        neighbor_grad_norm = torch.linalg.norm(neighbor_grad, axis= -2, keepdim = True)

        neighbor_inp = neighbor_inp.unsqueeze(2) # add angular dimension  (L=0 case)
        neighbor_lapla = neighbor_lapla.unsqueeze(2) # add angular dimension  (L=0 case)
        #torch.cuda.synchronize()
#        st = time.time()
        neighbor_inp_data1 =  (self.inp_rbf_r(torch.log10(neighbor_inp)))
        neighbor_inp_data2 =  (self.inp_rbf_c(torch.log10(neighbor_grad_norm)))
        #torch.cuda.synchronize()
#        et = time.time()
#        print('neighbor_inp_embedding1',et-st)
#        st = time.time()
        neighbor_inp_data3 =  self.grad_rbf_r( (neighbor_grad/ (self.eps + neighbor_grad_norm)) ) # L=1
        neighbor_inp_data4 =  self.grad_rbf_c( (neighbor_grad/ (self.eps + neighbor_grad_norm)) ) # L=1
#        print('n_inp_i', neighbor_inp)
#        print('n_inp_n', grad_norm_inp)
        layer_inp = {0: (neighbor_inp_data1, neighbor_inp_data2), 1: (neighbor_inp_data3, neighbor_inp_data4)}
        for idx_layer, layers in enumerate(self.list_layers):
            layer_out = {}
            for key, layer in layers.items():
#                st = time.time()
                # key string is splitted and casted to int
                l_in, l_filter, l_out = tuple( [ int(l) for l in key.split(',') ] )
                current_layer_out1, current_layer_out2 = layer( rbf, unit_vector, layer_inp[l_in][0], layer_inp[l_in][1], neighbor_mask )
               	
                if (idx_layer!=len(self.list_layers)-1):
                    current_layer_out1 =  current_layer_out1 + layer_inp[l_in][0]
                    current_layer_out2 =  current_layer_out2 + layer_inp[l_in][1] 
                try:
                    layer_out[l_out] = (layer_out[l_out][0] + current_layer_out1, layer_out[l_out][1] + current_layer_out2)
                except KeyError:
                    layer_out[l_out] = (current_layer_out1, current_layer_out2)
#                torch.cuda.synchronize()
#                et = time.time()
#                print(f'{key} layer: ',et-st)

            for l_out in layer_out.keys():
                layer_out[l_out] =  (layer_out[l_out][0] / len(layers), layer_out[l_out][1]/len(layers) )
            layer_inp = layer_out

        out_data = layer_out
        out_data= torch.cat([out_data[0][0],out_data[0][1]], dim=-1)
        
#        st = time.time()
        out = (self.readout(out_data))
        return self.recover_f(out.reshape(-1), ke_w.reshape(-1))

    def training_step(self, batch, datch_idx):
        inp, kinetic_e, kinetic_v, grid, neighbor_inp, neighbor_grad, neighbor_lapla, neighbor_veff, neighbor_grid, neighbor_mask = batch
        batch_size = inp.size(0)
        max_neighbor = neighbor_mask.size(1)
        
        pred = self.step(batch)
        
        grid_w = grid[:,-1]
        
        sum_weight_loss = torch.sum(torch.abs(grid_w * ( pred.reshape(-1) - kinetic_e.reshape(-1) )))*627.5095
        return_val = {'loss':sum_weight_loss, 'loss_sum':sum_weight_loss}
        
        return return_val
    def training_epoch_end(self, outputs):
        self.log('train_loss', sum([loss['loss_sum'] for loss in outputs]), prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
#        inp, kinetic_e, kinetic_v, grid, neighbor_data, neighbor_veff, neighbor_grid, neighbor_mask = batch
        inp, kinetic_e, kinetic_v, grid, neighbor_inp, neighbor_grad, neighbor_lapla, neighbor_veff, neighbor_grid, neighbor_mask = batch
        batch_size = inp.size(0)
        max_neighbor = neighbor_mask.size(1)
        
        pred = self.step(batch)
        
        grid_w = grid[:,-1]

        sum_weight_loss = torch.sum(torch.abs(grid_w * ( pred.reshape(-1) - kinetic_e.reshape(-1))))*627.5095 
        return_val = {'loss':sum_weight_loss, 'loss_sum':sum_weight_loss}
        
        return return_val
    def validation_epoch_end(self, outputs):
        self.log('valid_loss', sum([loss['loss_sum'] for loss in outputs]), prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inp, kinetic_e, kinetic_v, grid, neighbor_inp, neighbor_grad, neighbor_lapla, neighbor_veff, neighbor_grid, neighbor_mask = batch
        
        batch_size = neighbor_inp.size(0)
        max_neighbor = neighbor_inp.size(1)
        
        pred = self.step(batch).reshape(-1)
        grid_w = grid[:,-1].reshape(-1)

        sum_weight_loss = torch.sum(torch.abs(grid_w * ( pred - kinetic_e.reshape(-1) )))
        return_val = {'loss':sum_weight_loss, 'loss_sum':sum_weight_loss}
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6) } }

#   #deftraining_epoch_end(self, valid_step_outputs):
#    def validation_epoch_end(self, valid_step_outputs):
#        out_losses = torch.stack( [ item[0] for item in valid_step_outputs]).sum(dim=0)
#       assert len(out_losses)==2
#
#        out_losses = self.all_gather(out_losses)
#        if(len(out_losses.size())==2):
#            out_losses = out_losses.sum(0)
#            assert len(out_losses)==2
#        self.log('sum0_v',  out_losses[0] , prog_bar=True, logger=True, on_step=False, on_epoch=True)
#        self.log('sum1_v',  out_losses[1] , prog_bar=True, logger=True, on_step=False, on_epoch=True)
#        return
#
#    def predict_step(self,batch, batch_idx):
#    assert torch.logical_not(torch.any(torch.isnan(f_inp(inp)) )), "f_inp nan"+str(inp)+str(f_inp(inp))
#
#    embed_inp = self.inp_feature(f_inp(inp), f_inp(neighbor_inp), neighbor_d, neighbor_mask)
#
#    pred_out = f_inv_out( self.readout_inp( embed_inp ) )
#    
#    assert torch.logical_not(torch.any(torch.isnan(pred_out) )), "f_out nan"+str(pred_out)+str(self.readout_inp( embed_inp ))
#
#    np.save('pred-'+str(self.global_rank)+'-'+str(batch_idx)+'.npy', pred_out.detach().cpu().numpy())
#    if(len(batch)==8):
#        out_loss    = torch.sum(torch.abs( (out-pred_out)*grid_weight ), 0 )
#    #self.log('out0_p',  out_loss[0] , prog_bar=True, logger=True, on_step=False, on_epoch=True)
#    #self.log('out1_p',  out_loss[1] , prog_bar=True, logger=True, on_step=False, on_epoch=True)
#        return pred_out, out_loss
#    return pred_out



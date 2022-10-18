import torch
from torch.utils.data import Dataset
import numpy as np
import math
from tqdm import trange
from functools import partial
from multiprocessing import Pool
from time import time
import h5py
def set_neighbor( idx_grid, idx_mol, grid,  d_sample_neighbor):
    return np.linalg.norm( grid[idx_mol][idx_grid,:3].reshape(1,3) - grid[idx_mol][:,:3], axis=-1 ) < d_sample_neighbor

class DB(Dataset):
    def __init__(self, filename, indices, d_sample_neighbor = 3, precision=64):
        import h5py
        #from torch.nn.utils.rnn import pad_sequence

        if precision==16:
            self.precision = torch.half
        elif precision==32:
            self.precision = torch.float32
        elif precision==64:
            self.precision = torch.float64
        self.filename = filename
        st = time()
        f=h5py.File(self.filename,'r')
        print(time()-st, ' loading h5 file')
        st = time()
        self.indices = indices
        self.num_mol = len(self.indices)
        self.d_sample_neighbor = d_sample_neighbor

        self.n_grid = [0]+[ len(f[str(i).zfill(6)]['grid']) for i in self.indices ]
        self.cumsum = np.cumsum(self.n_grid)
        print(self.cumsum[-1])
        grid = [ f[str(i).zfill(6)]['grid'][:] for i in self.indices]
        #rho  = [ f[str(i).zfill(6)]['rho'][:]  for i in self.indices]
        #veff = [ -1*( f[str(i).zfill(6)]['vh'][:]+f[str(i).zfill(6)]['vxc'][:]+ f[str(i).zfill(6)]['vz'][:] ) for i in indices ]

        self.list_idx_neighbor = []
        self.list_n_neighbor = []
        with Pool () as p:
            for idx_mol in range(self.num_mol):
                local_set_neighbor = partial(set_neighbor, grid=grid, idx_mol=idx_mol, d_sample_neighbor=self.d_sample_neighbor)

                idx_neighbor = list(p.map(local_set_neighbor, range(self.n_grid[1+idx_mol]) ) )
                self.list_idx_neighbor+= idx_neighbor
                self.list_n_neighbor  += list( p.map( sum, idx_neighbor) )

            self.max_neighbor = max(self.list_n_neighbor)
            #for i, getitem in enumerate([getitem1, getitem2,getitem3, getitem4, getitem5, getitem6, getitem7, getitem8, getitem9, getitem10]):
            #    print(i, 'start')
            #    local_get_item = partial(getitem, cumsum=self.cumsum, veff=self.veff, grid=self.grid, rho=self.rho, list_idx_neighbor=self.list_idx_neighbor, max_neighbor=self.max_neighbor, precision=self.precision)
            #    self.data.append( p.map(local_get_item, range(self.cumsum[-1])) )

        assert len(self.list_idx_neighbor)==self.cumsum[-1], f'size should be identical ({len(self.list_n_neighbor)}, {self.cumsum[-1]})'
        print('max: ',self.max_neighbor)
        print(time()-st, ' db construction ')
            #for idx_grid in trange(self.n_grid[1+idx_mol]):
            #    distances = np.linalg.norm( self.grid[idx_mol][idx_grid,:3].reshape(1,3) - self.grid[idx_mol][:,:3], axis=-1 )
            #    self.list_idx_neighbor.append(  distances<self.d_sample_neighbor )
        f.close()
    def __len__(self):
        return self.cumsum[-1]

#    def __getitem__(self,idx): return tuple([ item[idx] for item in self.data ])

    def __getitem__(self,idx):
        idx_mol  = int( np.sum(self.cumsum<=idx) ) -1 # index for self.indices
        idx_grid = idx-self.cumsum[idx_mol]
        idx_mol_ = str(self.indices[idx_mol]).zfill(6)

        with h5py.File(self.filename,'r') as f:
            veff =  -1*( f[idx_mol_]['vh'][:]+f[idx_mol_]['vxc'][:]+ f[idx_mol_]['vz'][:] )
            grid = f[idx_mol_]['grid'][:]
            rho  = f[idx_mol_]['rho'][:]
        #distances = np.linalg.norm( np.expand_dims(grid[idx_grid,:3],0) - grid[:,:3], axis=-1 )
        #idx_neighbor = distances<self.d_sample_neighbor
        idx_neighbor = self.list_idx_neighbor[idx]
        num_neighbor = self.list_n_neighbor[idx]
        #rho = (den, den/x, den/y, den/z, laplacian, tau)

        neighbor_inp =  np.pad( rho[:-1, idx_neighbor].T.reshape((-1,5)),    ((0,self.max_neighbor-num_neighbor), (0,0)), constant_values=0.0)
        neighbor_veff=  np.pad( veff[idx_neighbor].reshape((-1,1)),          ((0,self.max_neighbor-num_neighbor), (0,0)), constant_values=0.0)
        neighbor_grid=  np.pad( grid[idx_neighbor, :].reshape((-1,4)),       ((0,self.max_neighbor-num_neighbor), (0,0)), constant_values=0.0)
        #print(torch.from_numpy(rho[:-1,idx_grid].reshape((-1,1)) ).type(self.precision).size())
        
#        return  torch.from_numpy(rho[:-1,idx_grid].reshape((-1,1)) ),\
#                torch.from_numpy(rho[ -1,idx_grid].reshape((-1,1))),\
#                torch.from_numpy(veff[idx_grid].reshape((-1,1))),\
#                torch.from_numpy(grid[idx_grid, :]),\
#                torch.from_numpy(neighbor_inp),\
#                torch.from_numpy(neighbor_veff),\
#                torch.from_numpy(neighbor_grid),\
#                torch.cat( [torch.ones(num_neighbor, dtype=torch.bool), torch.zeros(self.max_neighbor-num_neighbor, dtype=torch.bool) ] )
#
#
        return  torch.from_numpy(rho[:-1,idx_grid].reshape((-1,1)) ).type(self.precision),\
                torch.from_numpy(rho[ -1,idx_grid].reshape((-1,1))).type(self.precision),\
                torch.from_numpy(veff[idx_grid].reshape((-1,1))).type(self.precision),\
                torch.from_numpy(grid[idx_grid, :]).type(self.precision),\
                torch.from_numpy(neighbor_inp).type(self.precision)[:,0].unsqueeze(-1),\
                torch.from_numpy(neighbor_inp).type(self.precision)[:,1:4].unsqueeze(-1),\
                torch.from_numpy(neighbor_inp).type(self.precision)[:,-2].unsqueeze(-1),\
                torch.from_numpy(neighbor_veff).type(self.precision),\
                torch.from_numpy(neighbor_grid).type(self.precision),\
                torch.cat( [torch.ones(num_neighbor, dtype=torch.bool), torch.zeros(self.max_neighbor-num_neighbor, dtype=torch.bool) ] )
                #torch.from_numpy(neighbor_veff).type(self.precision).unsqueeze(-1),\


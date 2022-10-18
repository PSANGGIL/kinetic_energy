import os
import torch
import h5py
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from   pytorch_lightning.loggers import CSVLogger
from model import TensorFieldLayer
from model import MyModel
from model import RBF
if __name__=="__main__":
    from copy import deepcopy
    import os
    import h5py
    import pickle
    import numpy as np
    #from db2 import DB, DB_fast
    #from db2 import DB
    from db3 import DB
    #from db4 import DB
    #from db import DB, my_collate_fn
    from torch.utils.data import DataLoader
    from functools import partial
    from argparse import ArgumentParser
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import seed_everything
    from glob import glob
    from time import time 
    from pytorch_lightning.trainer.supporters import CombinedLoader
    
    import random
    torch.manual_seed(12956134)
    torch.cuda.manual_seed_all(15)
#    train_db = DB('./f2_1.h5', [1], 0.000001 )
#    train_db = DB('./f2_0.h5', [0], 0.5 )
#    valid_db = DB('./f2_1.h5', [1], 0.5 )
#    valid_db = DB('./f2_0.h5', [0], 0.000001 )
#    with open('./valid_db_9.pkl', 'rb') as f:
#       train_db =     pickle.load(f)#
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_in_feature", type=int,  default=1024)
    parser.add_argument('--num_hid_feature', type=int, default=1024)
    parser.add_argument('--num_out_feature', type=int, default=1024)
    #parser.add_argument('--accelerator', type=str, default='gpu')
    #parser.add_argument('--devices', type=str, default='gpu')
    #parser.add_argument('--gpus', type=str, default='1')
#    parser.add_argument('--multiplying_train_batch', type=int, default=1)
#    parser.add_argument('--test_ratio', type=float, default=0.1)
#    parser.add_argument('--valid_ratio', type=float, default=0.1)
#    parser.add_argument('--dropout', type=float, default=0.1)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    np.random.seed(2)
    if(len(glob('indices*txt'))==1):
        indices = np.loadtxt( glob('indices*txt')[0] ).astype(int)
    else:
        indices = list(range(1,801) )
        np.random.shuffle( indices )
        np.savetxt( 'indices.txt', indices)
    
#    test_db_length  = int(len(indices)*args.test_ratio)
#    valid_db_length = int(len(indices)*args.valid_ratio)
#    train_db_length = len(indices)-test_db_length-valid_db_length
#    train_indices = indices[:train_db_length]
#    valid_indices = indices[train_db_length:-test_db_length]
#    test_indices  = indices[-test_db_length:]



#test 
##########################################################

    num_workers = 12
    #indices = np.arange(1,801)
    N_train     = 20 
    N_valid     = 10
    indices = np.arange(1,N_train+N_valid+1)
    np.random.shuffle( indices )
    train_indices  = indices[:N_train]
    valid_indices  = indices[N_train:N_train+N_valid]
    
    train_indices  = [7]
    valid_indices  = [1]
    
    print(train_indices)
    print(valid_indices)
##########################################################


    st = time()

    with open('./train_db_mol7_local.pkl', 'rb') as f:
        train_db =     pickle.load(f)

    with open('./valid_db_mol1_local.pkl', 'rb') as f:
        valid_db =     pickle.load(f)


#    train_db = DB('../../lda_light.h5', train_indices, 0.000001)
#    valid_db = DB('../../lda_light.h5', valid_indices, 0.000001)
#    train_db = DB('./f2_0.h5', [0], 0.1 )
#    valid_db = DB('./f2_1.h5', [1], 0.1 )
#    valid_db = DB('./f2_0.h5', [0], 0.1 )
#
#
#    indices1  = np.where( np.array(valid_db.list_n_neighbor) ==1 )[0]
#    indices2 = list(set(list(range(len(valid_db)))) - set(indices1.tolist()))
#    
#    valid_db_1= torch.utils.data.Subset(deepcopy(valid_db), indices1)
#    valid_db_1.dataset.max_neighbor=1
#    valid_db_2= torch.utils.data.Subset(deepcopy(valid_db), indices2)
#    
#    train_dataloader_1 = DataLoader(valid_db_1, batch_size = args.batch_size,   shuffle=True,  num_workers=num_workers,  drop_last=True, persistent_workers=False)
#    train_dataloader_2 = DataLoader(valid_db_2, batch_size = args.batch_size,   shuffle=True,  num_workers=num_workers,  drop_last=True, persistent_workers=False)
#
#    combined_loader = CombinedLoader({"valid_db_1" : train_dataloader_1, "valid_db_2" : train_dataloader_2}, mode="max_size_cycle")
#    
#
    # pickle dump 
###############################################
#    with open('train_db_mol7_local.pkl', 'wb') as f:
#       pickle.dump(train_db,f, protocol=4)    
#    with open('valid_db_mol1_local.pkl', 'wb') as f:
#       pickle.dump(valid_db,f, protocol=4)
    et = time()
    print(et-st,'s train.py')
    
    #train_dataloader = DataLoader(train_db, batch_size = args.batch_size,   shuffle=False,  num_workers=num_workers,  drop_last=True, persistent_workers=False)
    train_dataloader = DataLoader(train_db, batch_size = args.batch_size, shuffle=True, num_workers=num_workers,                  persistent_workers=False)
    valid_dataloader = DataLoader(valid_db, batch_size = args.batch_size, shuffle=False, num_workers=num_workers,                  persistent_workers=False)
#    test_dataloader = DataLoader(test_db, batch_size = args.batch_size, shuffle=False, num_workers=num_workers,                  persistent_workers=False)

    checkpoint_callback = ModelCheckpoint(
                                          monitor='valid_loss',
                                           save_top_k = 1, save_last=True)

    args.callbacks=[checkpoint_callback]

    csv_logger = CSVLogger('lightning_csv_logs', name=f'{args.batch_size}-{args.num_in_feature}-{args.learning_rate:.5E}')
    
    rbf_kernel = RBF(exp=True, layernorm=True)
     
    model = MyModel( [ ([0,1], [0,1]), ([0,1],[0,1]), ([0,1],[0]) ], [0,1], \
                     [ args.num_in_feature, args.num_hid_feature, args.num_hid_feature, args.num_out_feature], args.learning_rate, args.num_in_feature, args.num_hid_feature, args.num_out_feature, rbf_kernel)
    #    PATH = '/data/psanggil/model_0805/lightning_csv_logs/64-1024-1.00000E-05/version_5/checkpoints/epoch=19-step=64560.ckpt'
#
#    model =  model.load_from_checkpoint(PATH, list_l_inout=  [ ([0,1], [0,1]), ([0,1],[0,1]), ([0,1],[0]) ],
#                                              list_l_filter = [0,1],
#                                              list_n_feature=[ args.num_in_feature, args.num_hid_feature, args.num_hid_feature, args.num_out_feature],
#                                              learning_rate = args.learning_rate,
#                                              num_in_feature = args.num_in_feature,
#                                              num_hid_feature = args.num_hid_feature,
#                                              num_out_feature=args.num_out_feature,
#                                              rbf_kernel=rbf_kernel)    

#    print(model.summarize)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger=csv_logger
#    trainer.fit(model, train_dataloader)
    
    trainer.fit(model, train_dataloader, valid_dataloader)
    #trainer.fit(model, combined_loader)
#    trainer.test(model,train_dataloader)

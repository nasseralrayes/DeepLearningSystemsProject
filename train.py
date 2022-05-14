import os 
import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import torchio


import gc
torch.cuda.empty_cache()
import datetime

from torch.utils.data import Dataset, DataLoader

import argparse
from src.data.torch_utils import MonkeyEyeballsDataset
from src.models.from_scratch import resnet_for_multimodal_regression as resnet

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--labels', default='data/monkey_data.csv', metavar='DF',
    help='path to ICP/IOP dataframe')
parser.add_argument('--scans', default='data/torch_standardized', metavar='DIR',
    help='path to dataset folder')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument('--lr', default=3e-4, type=float, metavar='LR',
    help='initial learning rate')
parser.add_argument('--batch', default=4, type=int, metavar='BATCH',
    help='number of samples per mini-batch')
parser.add_argument('--pretrain_model', default=None, type=str,
    help='Model filepath to warm start on')

def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 12345
    torch.manual_seed(seed)

    def train(dataloader_train, 
              dataloader_val, 
              model, 
              optimizer, 
              scheduler, 
              val_interval,
              loss=nn.MSELoss(reduction='sum'), 
              total_epochs=100):

        # settings
        batches_per_epoch = len(dataloader_train)
        print('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

        if device == 'cuda':
          loss = loss.to(device)
        
        model.train()

        temp_train = []
        temp_val = []
        
        train_loss_epoch = []
        val_loss_epoch = []

        for epoch in tqdm(range(total_epochs)):

            print('Start epoch {}'.format(epoch))
            
            for batch_id, batch_data in enumerate(dataloader_train):

                # getting data batch
                icp = batch_data['icp'].float().unsqueeze(1).cuda()
                iop = batch_data['iop'].float().cuda()
                scan = batch_data['scan'].float().cuda()

                if device == 'cuda': 
                    scan = scan.to(device)

                # standardize input
                icp = (icp - 15) / 11 
                iop = (iop - 22) / 13

                optimizer.zero_grad()

                # add fake channel dimension as 5-D input is expected
                preds = model(scan.unsqueeze(1), iop)
                
                # calculating loss
                loss_value = loss(preds, icp)
                loss_value.backward()                
                optimizer.step()
                
                temp_train.append(loss_value.item())
                
                # get validation loss
                if batch_id == len(dataloader_train)-1:
                    train_loss_epoch.append(np.mean(temp_train[:]))
                    temp_train.clear()
                    
                    model.eval()

                    print('')
                    print('Validating...')
                
                    for batch_id_val, batch_data_val in enumerate(dataloader_val):
                        icp_val = batch_data_val['icp'].float().unsqueeze(1).cuda()
                        iop_val = batch_data_val['iop'].float().cuda()
                      
                        scan_val = batch_data_val['scan'].float().cuda()

                        icp_val = (icp_val - 15) / 11
                        iop_val = (iop_val -22)/ 13

                        if device == 'cuda': 
                            scan_val = scan_val.to(device)

                        preds_val = model(scan_val.unsqueeze_(1),iop_val)
                                
                        loss_value_val = loss(preds_val, icp_val)
                        temp_val.append(loss_value_val.item())
                        
                    val_loss_epoch.append(np.mean(temp_val[:]))
                    temp_val.clear()
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    model.train()
        
            print('current train losses =', train_loss_epoch)
            print('current validation losses =', val_loss_epoch)
            
        print('Finished training')

        print('final train losses =', train_loss_epoch)
        print('final validation losses =', val_loss_epoch)

    labels = pd.read_csv(args.labels)
    labels = labels[labels['torch_present'] & ~labels['icp'].isnull() & ~labels['iop'].isnull() & labels['icp'] > 0] 
    labels['icp'] = labels['icp'].astype('float')
    labels['iop'] = labels['iop'].astype('float')

    train_labels = labels[(labels['monkey_id'] != 14) & (labels['monkey_id'] != 9)]
    val_labels = labels[(labels['monkey_id'] == 14) or (labels['monkey_id'] == 9)]

    # transform
    transform = torchio.Compose([
        torchio.RandomFlip(axes=2, p=0.5),
        torchio.RandomAffine(
            degrees=(0, 0, 10),
            translation=1
        ),
        torchio.RandomBlur(1, p=0.2),
        torchio.RandomNoise(mean=0,std=1),
        torchio.RandomGamma(),
        torchio.RandomAffine(
            scales=(1.2, 1.5)
        )
    ])

    med_train = MonkeyEyeballsDataset(args.scans, train_labels, transform=transform)
    med_val = MonkeyEyeballsDataset(args.scans, val_labels)

    dataloader_train = DataLoader(med_train, batch_size=args.batch, shuffle=True,pin_memory=True, num_workers=2) 
    dataloader_val = DataLoader(med_val, batch_size=args.batch, shuffle=False)

    model = resnet.resnet50(sample_input_D=128, sample_input_H=128, sample_input_W=512).cuda()

    net_dict = model.state_dict() 

    if args.pretrain_model is not None:
        print ('loading pretrained model {}'.format(args.pretrain_model))
        pretrain = torch.load(args.pretrain_model)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

    OPTIMIZER = torch.optim.Adamax(model.parameters(), lr=args.lr)
    SCHEDULER = lr_scheduler.ExponentialLR(OPTIMIZER, gamma=0.99)
    LOSS = nn.MSELoss(reduction='mean')

    train(dataloader_train=dataloader_train, 
          dataloader_val=dataloader_val,
          model=model, 
          optimizer=OPTIMIZER, 
          scheduler=SCHEDULER, 
          total_epochs=args.epochs, 
          val_interval=10,
          loss=LOSS)

if __name__ == '__main__':
    main()
import os
import random
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

# utils
from utils.load_data import data_load, model_dataset
from utils.loss import Loss_fun
from model.burgers_nets import *

# all seeds
def set_seed(seed: int = 2024):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# get_gpu_mem_info
def get_gpu_mem_info(gpu_id=0):
    """
    get gpu memory info, unit:MB
    :param gpu_id: device ID
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} does not exist!'.format(gpu_id))
    else:
        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print('total gpu memory:', total, 'used gpu memory:', used, 'free gpu memory:', free )


# train
def train_one_epoch(model, device, data_loader, optimizer, criterion, args):
    model.train()
    total_loss = 0
    init_step = 5
    train_step = args.train_step
    
    for batch_idx, (data, phy) in enumerate(data_loader):
        if train_step > data.shape[1]:
            train_step = data.shape[1]    
        u_0 = data[:,:init_step,...].to(device)
        real_data = data[:,init_step:train_step,...].to(device)
        phy = phy.to(device)
            
        optimizer.zero_grad()
        # forward
        u_t1 = model(u_0, phy=phy, step=real_data.shape[1])

        
        loss_t  = criterion.relative_loss(u_t1, real_data)
        loss_t.backward()
        optimizer.step()
        total_loss = total_loss + loss_t.item()
        del u_0, u_t1, real_data
    return_loss = total_loss / len(data_loader)
    return return_loss

# val
def val_one_epoch(model, device, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    init_step = 5
    last_step = args.test_step
    with torch.no_grad():
        for batch_idx, (data, phy) in enumerate(data_loader):    
            u_0 = data[:,:init_step,...].to(device)
            real_data = data[:,init_step:last_step,...].to(device)
            phy = phy.to(device)
                
            # forward
            u_t1 = model(u_0, phy=phy, step=real_data.shape[1])

            u_t1 = u_t1*(data_max-data_min)+data_min
            real_data = real_data*(data_max-data_min)+data_min

            loss_t  = criterion.relative_loss(u_t1, real_data)
            loss_t = loss_t.detach().cpu()
            total_loss = total_loss + loss_t.item()
            del u_0, u_t1, real_data
    return_loss = total_loss / len(data_loader)
    return return_loss

# test_per_point
def test_accumulative_error(model, device, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    init_step = 5
    last_step = args.test_step
    losses=[]
    u_all = []
    gt_all = []
    with torch.no_grad():
        for batch_idx, (data, phy) in enumerate(data_loader):
            u_0 = data[:,:init_step,...].to(device)
            real_data = data[:,init_step:last_step,...].to(device)
            phy = phy.to(device)
                
            # forward
            u_t1 = model(u_0, phy=phy, step=real_data.shape[1])

            u_t1 = u_t1*(data_max-data_min)+data_min
            real_data = real_data*(data_max-data_min)+data_min

            loss_t  = criterion.point_relative_loss(u_t1, real_data)
            loss_t = loss_t.detach().cpu()
            u_t1 = u_t1.detach().cpu().numpy()
            real_data = real_data.detach().cpu().numpy()

            losses.append(loss_t)
            u_all.append(u_t1)
            # gt_all.append(real_data)
    return_loss = np.zeros([len(losses), len(losses[0])])
    pred_data = u_all[0].copy()
    for i in range(len(losses)):
        return_loss[i] = losses[i]
        if i > 0:
            pred_data = np.concatenate([pred_data, u_all[i]], axis=0)

    return np.mean(return_loss, axis=0), pred_data


def main(args, train_loader, val_loader, test_loader):
    if args.gpu == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device(args.gpu)

    assert args.model == 'res' or 'cno' or 'unet' or 'lstm' or 'fno' or 'percnn' or 'ppnn' or 'papm'
    if args.model == 'res':
        model = DilResNet(mode=args.mode).to(device)
    if args.model == 'cno':
        model = CNO_model(mode=args.mode).to(device)
    if args.model == 'unet':
        model = UNet_model(mode=args.mode).to(device)
    if args.model == 'lstm':
        model = ConvLSTM(mode=args.mode).to(device)
    if args.model == 'fno':
        model = fno_model(mode=args.mode).to(device)
    if args.model == 'percnn':
        model = percnn_model().to(device)
    if args.model == 'ppnn':
        model = ppnn_model().to(device)
    if args.model == 'papm':
        model = papm_model().to(device)

    if args.pre_trained_weight_path:
        if os.path.exists(args.pre_trained_weight_path):
            model.load_state_dict(torch.load(args.pre_trained_weight_path))
            print('Successfully load weight!')
        else:
            print('Weight doesn\'t exist!')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2, eta_min=2e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.98, patience=7, verbose=True)
    criterion = Loss_fun()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    # train and validation
    num_epochs = args.epoches
    test_interval = args.test_interval
    loss_best = np.inf
    loss_train = np.zeros(num_epochs)
    loss_test = np.zeros(num_epochs)
    learning_rate = []
    train_time_per_epoch = np.zeros(num_epochs)
    test_time_per_epoch = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        if epoch % test_interval == 0 or epoch == num_epochs-1:
            val_start_time = time.time()
            val_loss = val_one_epoch(model, device, val_loader, criterion, args)
            val_end_time = time.time()

        start_time = time.time()
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion, args)
        end_time = time.time()
            
        loss_train[epoch] = train_loss
        loss_test[epoch] = val_loss
        learning_rate.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        
        training_time = end_time - start_time
        testing_time = val_end_time - val_start_time
        train_time_per_epoch[epoch] = training_time
        test_time_per_epoch[epoch] = testing_time
        if (epoch+1) % 5 == 0 or epoch == 0 or (epoch+1) == num_epochs:
            print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Test Loss: {val_loss:.4e}")
            print(f"time per epoch/s:{training_time:.2f}")

        if val_loss < loss_best:
            loss_best = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.weight_path)

        # update learning rate
        scheduler.step(epoch)
    # draw
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_train, linewidth=1, color='blue', label='train')
    plt.semilogy(loss_test, linewidth=0.5, color='orange', label='test')
    plt.legend(loc=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(best_epoch, loss_best, marker='*', markersize=10, color='r')
    plt.title("epoch %d best valloss:%0.5f" % (best_epoch, loss_best), fontsize=20)
    plt.savefig(args.loss_path)
    plt.close('all')

    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate, linewidth=1, color='blue', label='lr')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.savefig(args.lr_path)
    plt.close('all')

    # save train figs
    train_time_per_epoch = np.mean(train_time_per_epoch)
    test_time_per_epoch = np.mean(test_time_per_epoch)

    note = open(args.recording_path, mode = 'w')
    note.write('total_params:'+str(total_params)+'\n')
    note.write('average_train_time/s:'+str(train_time_per_epoch)+'\n')
    note.write('average_test_time/s:'+str(test_time_per_epoch)+'\n')
    if args.test_accumulative_error:
         model.load_state_dict(torch.load(args.weight_path))
         acc_error, pred_data = test_accumulative_error(model, device, test_loader, criterion, args)
         note.write('test_accumulative_error:'+str(acc_error))
         np.save(args.pred_data_save_path, pred_data)
    note.close()

        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--loss_path", type=str, default='./loss_fun.png', help='The path to save loss function')
    parse.add_argument("--recording_path", type=str, default='./record.txt', help='The path to save train time and test information')
    parse.add_argument("--lr_path", type=str, default='./learning_rate.png', help='The path to save learning curve rate')
    parse.add_argument("--train_bs", type=int, default=16, help='The batch size of training data')
    parse.add_argument("--val_bs", type=int, default=8, help='The batch size of testing data')
    parse.add_argument("--epoches", type=int, default=1)
    parse.add_argument("--seed", type=int, default=2024)
    parse.add_argument("--train_step", type=int, default=50, help='The number of train steps for model training, must be a multiple of 5')
    parse.add_argument("--test_step", type=int, default=100, help='The number of input last model timestep, Must be a multiple of 5')
    parse.add_argument("--test_interval", type=int, default=10, help='The interval of epoches for testing')
    parse.add_argument("--train_ratio", type=float, default=1, help='The ration of selection for training data')
    parse.add_argument("--file_link", type=str, default='./datasets/burgers_500_101_2_64_64.h5', help='Data path')
    parse.add_argument("--model", type=str, default='unet',
                       help='The network, must be one of: res, cno, unet, lstm, fno, percnn, ppnn, papm')
    parse.add_argument("--mode", type=str, default='rollout',
                       help='The train strategy, must be single_step or rollout')
    parse.add_argument("--shuffle", type=bool, default=False, help='Whether to shuffle the training dataset')
    parse.add_argument("--test_accumulative_error", type=bool, default=True, 
                       help='Whether to test accumulative error on test dataset')
    parse.add_argument("--weight_path", type=str, default='./weights/papm.pth')
    parse.add_argument("--gpu", type=str, default='cuda:0', help='The device to run models, if using cpu, input \'cpu\'')
    parse.add_argument("--pre_trained_weight_path", type=str, default=None, help='If you have pre-trained model,add path')
    parse.add_argument("--pred_data_save_path", type=str, default='pred_data.npy', help='The path to save pred data')
    args = parse.parse_args()
    print(args)
    
    # set seed
    set_seed(args.seed)
    
    # data_load
    X_train, X_val, X_test, phy= data_load(args)
    in_ch = X_val.shape[2]
    print('num of channels:', in_ch)
    print('train_shape:', X_train.shape,'val_shape:', X_val.shape, 'test_shape:', X_test.shape)

    # Normalize
    # For better convergence, we did not use inverse normalization when calculating losses during training,
    # but used it during testing
    data_max = max(np.max(X_train),np.max(X_val),np.max(X_test))
    data_min = min(np.min(X_train),np.min(X_val),np.min(X_test))
    X_train = (X_train-data_min)/(data_max-data_min)
    X_val = (X_val-data_min)/(data_max-data_min)
    X_test = (X_test-data_min)/(data_max-data_min)
    
    # data_set
    train_dataset = model_dataset(X_train, phy)
    val_dataset = model_dataset(X_val, phy)
    test_dataset = model_dataset(X_test, phy)

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size= args.train_bs, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size= args.val_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size= args.val_bs, shuffle=False)

    main(args, train_loader, val_loader, test_loader)

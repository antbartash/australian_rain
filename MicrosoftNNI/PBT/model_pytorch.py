DEFAULT_DIR = ''


import argparse
import logging

import numpy as np
import random
import os
import nni
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

np.random.seed(42)
random.seed(42)

logger = logging.getLogger('PBT')




### LOADING DATA ###
def load_data():
    data_dir = ''
    X_train = torch.load(os.path.join(data_dir, 'X_train.pt')).detach().numpy()
    X_valid = torch.load(os.path.join(data_dir, 'X_valid.pt')).detach().numpy()
    X_test = torch.load(os.path.join(data_dir, 'X_test.pt')).detach().numpy()
    y_train = torch.load(os.path.join(data_dir, 'y_train.pt')).to(torch.float64).detach().numpy()
    y_valid = torch.load(os.path.join(data_dir, 'y_valid.pt')).to(torch.float64).detach().numpy()
    y_test = torch.load(os.path.join(data_dir, 'y_test.pt')).to(torch.float64).detach().numpy()
    return X_train, X_valid, X_test, y_train, y_valid, y_test

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()



### DATALOADERS ###
def dataloader(X, y, args):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32)) 
    loader = DataLoader(dataset, batch_size=2**args['batch_size_power'])
    return loader


### MODEL ###
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(X_train.shape[1], 512)
        self.act0 = nn.LeakyReLU(negative_slope=0.3)
    def forward(self, x):
        x = self.fc0(x)
        x = self.act0(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        

def eval(X, y, model):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    pred_logits = model.eval()(X).detach()
    y_pred = torch.argmax(pred_logits, dim=1)
    y_true = torch.argmax(y, dim=1)
    return f1_score(y_true.detach().numpy(),
                    y_pred.detach().numpy(),
                    average='micro')


def save_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

def load_checkpoint(checkpoint_path):
    model_state_dict = torch.load(checkpoint_path)
    return model_state_dict


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainloader = dataloader(X_train, y_train, args)
    # testloader = dataloader(X_test, y_test, args)
    # validloader = dataloader(X_valid, y_valid, args) 

    model = Net().to(device)

    save_checkpoint_dir = args['save_checkpoint_dir']
    save_checkpoint_path = os.path.join(save_checkpoint_dir, 'model.pth')
    load_checkpoint_path = os.path.join(args['load_checkpoint_dir'], 'model.pth')

    if os.path.isfile(load_checkpoint_path):
        model_state_dict = load_checkpoint(load_checkpoint_path)
        logger.info('test:', load_checkpoint_path)
        logger.info(type(model_state_dict))
        model.load_state_dict(model_state_dict)

    optimizer = torch.optim.Adamax(model.parameters(), 
                                   lr=args['lr'],
                                   #betas=(args['beta1'], args['beta2'])
                                   )


    # epochs is perturbation interval
    for epoch in range(1, args['epochs']+1):
        train(args, model, device, trainloader, optimizer, epoch)
        train_score = eval(X_train, y_train, model)
        valid_score = eval(X_valid, y_valid, model)
        metrics = {
            'train_f1-score': train_score,
            'valid_f1-score': valid_score
        }

        if epoch < args['epochs']:
            # report intermediate result
            nni.report_intermediate_result(metrics)
            logger.debug('metrics %g', metrics)
            logger.debug('Pipe send intermediate result done.')    
        else:
            # report final result
            nni.report_final_result(valid_score)
            logger.debug('Final valid f1-score %g', valid_score)
            logger.debug('Send final result done')

    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
    save_checkpoint(model, save_checkpoint_path)


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch PBT BERT_base')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size_power', type=int, default=8, metavar='N',
                        help='input batch size for training is 2**batch_size_power (default: 2**8)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--beta1', type=float, default=0.9, metavar='beta1',
    #                     help='beta1 (default: 0.9)')
    # parser.add_argument('--beta2', type=float, default=0.999, metavar='beta2',
    #                     help='beta2 (default: 0.999)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_checkpoint_dir', type=str, default=DEFAULT_DIR,
                        help='where to save checkpoint of this trial')
    parser.add_argument('--load_checkpoint_dir', type=str, default=DEFAULT_DIR,
                        help='where to load the model')
    
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters from tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
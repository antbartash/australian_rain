DEFAULT_DIR = ''


import argparse
import logging

import numpy as np
import random
import os
import nni
import torch
import tensorflow as tf
from tensorflow import keras
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
    loader = tf.data.Dataset.from_tensor_slices((X, y)).batch(2**args['batch_size_power'])
    return loader


### MODEL ###
def get_model():
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation='softmax')
    ])
    return model



def train(args, model, train_loader, valid_loader, optimizer):
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=[keras.metrics.F1Score(average='micro')]
    )
    history = model.fit(
        train_loader, epochs=1, 
        validation_data=valid_loader,
        verbose=1
    )
        

def eval(X, y, model):
    return model.evaluate(X, y)[1]


def save_checkpoint(model, checkpoint_path):
    model.save_weights(checkpoint_path)

def load_checkpoint(model, checkpoint_path):
    model.load_weights(checkpoint_path)
    return model


def main(args):
    tf.random.set_seed(args['seed'])

    trainloader = dataloader(X_train, y_train, args)
    testloader = dataloader(X_test, y_test, args)
    validloader = dataloader(X_valid, y_valid, args) 

    model = get_model()

    save_checkpoint_dir = args['save_checkpoint_dir']
    save_checkpoint_path = os.path.join(save_checkpoint_dir, 'model.h5')
    load_checkpoint_path = os.path.join(args['load_checkpoint_dir'], 'model.h5')

    if os.path.isfile(load_checkpoint_path):
        model = load_checkpoint(model, load_checkpoint_path)
        logger.info('test:', load_checkpoint_path)
        # logger.info(type(model_state_dict))

    optimizer = keras.optimizers.Adamax(learning_rate=args['lr'])


    # epoch is perturbation interval
    for epoch in range(1, args['epochs']+1):
        train(args, model, trainloader, validloader, optimizer)
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
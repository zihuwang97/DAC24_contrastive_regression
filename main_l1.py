import torch 
import torch.nn as nn
from torch.utils.data import Dataset

import argparse
import numpy as np

from train_supcr import encoder_trainer, predictor_trainer, whole_trainer


class circuit_data(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def main(args):
    device = 'cuda:1'
    # loading data
    data = np.loadtxt(args.data_path)
    label = np.loadtxt(args.label_path)
   
    data = torch.as_tensor(data, dtype=torch.float)
    label = torch.as_tensor(label, dtype=torch.float)
    
    training_size = 100
    train_data = data[:training_size]
    test_data = data[training_size:]
    train_label = label[:training_size][:,:]
    test_label = label[training_size:][:,:]
    target_size = train_label.size(-1)

    dataset_train = circuit_data(train_data, train_label)
    dataset_test = circuit_data(test_data, test_label)

    # training
    trainer = whole_trainer(num_epoch=args.epochs, batch_size=args.batch_size, 
                            learning_rate=args.lr, save_freq=10, 
                            device=device, embed_size=args.embed_size, 
                            input_feature_num=args.input_feature_num, target_size=target_size)
    
    trainer.train(dataset_train,dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=5e-5, type=float, help='initial encoder learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--embed_size', default=128, type=int, help='number of embedding features')
    parser.add_argument('--epochs', default=5000, type=int, help='number of epochs for predictor training')
    parser.add_argument('--print_freq', default=500, type=int, help='frequency of printing training log & saving model')
    parser.add_argument('--dataset', type=int, default=14, choices=[12,14,18])
    args = parser.parse_args()

    if args.dataset == 12:
        args.data_path = 'circuit_data/12/x_data.dat'
        args.label_path = 'circuit_data/12/y_data.dat'
    elif args.dataset == 14:
        args.data_path = 'circuit_data/14/x_data.dat'
        args.label_path = 'circuit_data/14/y_data.dat'
    else:
        args.data_path = 'circuit_data/18/x_data.dat'
        args.label_path = 'circuit_data/18/y_data.dat'
    args.input_feature_num = args.dataset

    main(args)





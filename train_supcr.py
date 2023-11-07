import os
import time
import datetime
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from models import encoder, predictor, whole_model
from util import mkdir
from data_aug import scale_2_stage, random_scale_2_stage


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


def SupCR_loss(rep, labels, temp=4.0, thre=0.7):
    bsz = rep.size(0)
    # calculate label distance and sort
    labels = labels.view(bsz,-1)
    # label_dist = torch.abs(labels - labels.T)
    label_dist = torch.cdist(labels, labels)
    label_dist_sorted, indices = torch.sort(label_dist, -1, descending=True)

    # calculate logits and sort them w.r.t. label distance
    scores = rep @ rep.T
    logits = torch.exp(scores / temp)
    logits_sorted = torch.scatter(torch.zeros_like(logits).cuda(rep.device), 
                                  -1, torch.argsort(indices), logits)

    # generate mask out long distance positives
    mask = torch.where(label_dist_sorted <= thre, 1.0, 0.0).cuda(rep.device)
    elig_pos_count = torch.clamp(torch.sum(mask, -1), min=1.0)

    # TODO: Weight pos and neg pairs in loss
    # weights = 

    # loss
    neg_sum = torch.cumsum(logits_sorted, -1)
    loss = torch.log(logits_sorted[:,1:] / neg_sum[:,:-1])
    loss = torch.where(mask[:,1:] == 1, loss, 0.0) # mask out long distance positives
    loss = torch.sum(loss, -1) / elig_pos_count
    # loss = torch.sum(loss, -1) / (bsz - 1)
    return - torch.sum(loss) / bsz


class encoder_trainer():
    def __init__(self, num_epoch, batch_size, temp, learning_rate, save_freq, device, embed_size, input_feature_num):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.temperature = temp
        self.lr = learning_rate
        self.save_freq = save_freq
        self.device = device
        self.embed_size = embed_size
        self.input_feature_num = input_feature_num

    def train(self, dataset_train, augment):
        model = encoder(input_feature_num=self.input_feature_num, hdn_size=self.embed_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))

        ### training
        ## Phase 1: representation learning
        # making saving directory
        save_path = os.path.join(os.getcwd(), 'train_log')
        today = datetime.date.today()
        formatted_today = today.strftime('%y%m%d')
        now = time.strftime("%H:%M:%S")
        save_path = os.path.join(save_path, formatted_today + now)
        mkdir(save_path)

        for epoch in range(self.num_epochs):
            model.train()
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)

                # random scale
                # sample_, labels_ = scale_2_stage(sample, labels, 'cmrr')
                sample_, labels_ = random_scale_2_stage(sample, labels)
                sample = torch.cat((sample, sample_), 0)
                labels = torch.cat((labels, labels_), 0)

                rep = model(sample) 
                rep = nn.functional.normalize(rep, dim=-1)

                # Add data augmentation to data
                # if augment == 'yes':
                #     g_noise = torch.normal(mean=torch.zeros_like(sample), std=5e-3*torch.ones_like(sample)).to(self.device)
                #     noisy_sample = sample + g_noise
                #     noisy_sample = torch.clamp(noisy_sample, min=0.0, max=1.0)
                #     noisy_rep = model(noisy_sample)
                #     noisy_rep = nn.functional.normalize(noisy_rep, dim=-1)
                #     labels = labels.repeat(2,1)
                #     rep = torch.cat([rep, noisy_rep], 0)

                loss = SupCR_loss(rep, labels, temp=self.temperature)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % self.save_freq == 0:
                print('[%d]  loss: %.3f' % (epoch + 1, loss))
                model_state = model.state_dict()
                file_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save(model_state, file_save_path)
        return file_save_path


class predictor_trainer():
    def __init__(self, num_epoch, batch_size, learning_rate, save_freq, pretrained, device, embed_size, input_feature_num, target_size):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.lr = learning_rate
        self.save_freq = save_freq
        self.pretrained = pretrained  # pretrained model path
        self.device = device
        self.embed_size = embed_size
        self.input_feature_num = input_feature_num
        self.target_size = target_size

    def train(self, dataset_train, dataset_test):
        # load the pre-trained encoder
        enc = encoder(input_feature_num=self.input_feature_num, hdn_size=self.embed_size)
        state_dict = torch.load(self.pretrained, map_location="cpu")
        enc.load_state_dict(state_dict)
        enc.to(self.device)

        model = predictor(embed_size=self.embed_size, target_size=self.target_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        testloader = DataLoader(dataset_test, batch_size=1024,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))
        criterion = nn.MSELoss()
        eval_criterion = nn.L1Loss()

        ### training
        ## Phase 2: training regressor
        # making saving directory
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    rep = enc(sample) 
                pred = model(rep)
                loss = criterion(labels, pred)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # testing model
            if (epoch + 1) % 300 == 0:
                model.eval()
                error = 0
                for step, (sample,labels) in enumerate(testloader, 0):
                    sample = sample.to(self.device)
                    labels = labels.to(self.device)
                    rep = enc(sample) 
                    pred = model(rep)
                    error += eval_criterion(labels, pred)
                    
                        # rand_num = random.randint(0,len(labels)-10)
                        # print(labels[rand_num:rand_num+5], pred.view(-1)[rand_num:rand_num+5], error)
                        # _, idx = torch.topk(torch.abs(labels - pred.view(-1)), 5)
                        # print(labels[idx], pred.view(-1)[idx], error)
                print(error/step)


class whole_trainer():
    def __init__(self, num_epoch, batch_size, learning_rate, save_freq, device, embed_size, input_feature_num, target_size):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.lr = learning_rate
        self.save_freq = save_freq
        self.device = device
        self.embed_size = embed_size
        self.input_feature_num = input_feature_num
        self.target_size = target_size

    def train(self, dataset_train, dataset_test):
        # load the pre-trained encoder
        model = whole_model(input_size=self.input_feature_num, embed_size=self.embed_size, target_size=self.target_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        testloader = DataLoader(dataset_test, batch_size=1024,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))
        criterion = nn.MSELoss()
        eval_criterion = nn.L1Loss()

        ### training
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)
                pred = model(sample)
                loss = criterion(labels,pred)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # if (epoch + 1) % self.save_freq == 0:
            #     print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                
            # testing model
            if (epoch + 1) % 500 == 0:
                model.eval()
                error = 0
                for step, (sample,labels) in enumerate(testloader, 0):
                    sample = sample.to(self.device)
                    labels = labels.to(self.device)
                    pred = model(sample)
                    error += eval_criterion(labels, pred)
                    # neg_count = torch.where(labels<=0, 1.0, 0.0)
                    # neg_count = torch.sum(neg_count)
                    
                        # rand_num = random.randint(0,len(labels)-10)
                        # print(labels[rand_num:rand_num+5], pred.view(-1)[rand_num:rand_num+5], error, neg_count)
                        # _, idx = torch.topk(torch.abs(labels - pred.view(-1)), 5)
                        # print(labels[idx], pred.view(-1)[idx], error)
                print(error/step)



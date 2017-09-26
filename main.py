from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loading import Protein_feat66_Dataset
from model import BiGRU

if torch.cuda.is_available():
    Variable = Variable.cuda()

relative_path = '../../DL4Bio/feat66_tensorflow/'
trainList_addr = 'data/trainList'
validList_addr = 'data/validList'
testList_addr = 'data/testList'

batch_size = 16
input_size = 66
max_seq_length = 300
hidden_size = 100
num_layers = 3
num_classes = 8
num_epochs = 2
print_every = 10
learning_rate = 0.001

train_dataset = Protein_feat66_Dataset(relative_path, trainList_addr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

valid_dataset = Protein_feat66_Dataset(relative_path, validList_addr, max_seq_length=683)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

def evaluate(rnn, dataloader):
    correct = 0
    total = 0
    for i, (name, features, labels, masks, seq_len) in enumerate(dataloader):
        # print("masks:", masks.size(), masks)
        # print("seq_len:", seq_len.size(), seq_len)
        features = Variable(features)
        labels = labels.view(-1)
        masks = masks.view(-1)
        masks_np = masks.numpy()
        ones = np.count_nonzero(masks_np)
        # print("Nonzeros in masks", ones)
        labels = labels.type(torch.LongTensor)
        outputs = rnn(features)
        _, predicted = torch.max(outputs, 1)
        matched = torch.eq(predicted.data, labels)
        matched = torch.mul(matched.type(torch.FloatTensor), masks)

        correct += matched.sum()
        total += masks.sum()
        # print("correct prediction: {}; total prediction: {}".format(correct, total))
        # print("sum comparison:", masks.sum(), seq_len.sum())
    accuracy = correct / total
    return accuracy


rnn = BiGRU(input_size, hidden_size, num_layers, num_classes, batch_size)
accuracy = evaluate(rnn, valid_loader)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (name, features, masks, labels, seq_len) in enumerate(train_loader):
        
        features = Variable(features)
        labels = labels.view(-1)
        labels = Variable(labels.type(torch.LongTensor))
        # print("labels size: {}".format(labels.size()))
        optimizer.zero_grad()
        outputs = rnn(features)
        # print("outputs size: {}".format(outputs.size()))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            accuracy = evaluate(rnn, valid_loader)
            print("Epoch: {}, iteration: {}, loss: {}, validation_accuracy: {}".format(epoch, i, loss.data[0], accuracy))
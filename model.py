from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, bidirectional=True):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                          dropout=0, bidirectional=self.bidirectional)
        self.linear1 = nn.Linear(self.hidden_size * (self.bidirectional + 1), self.hidden_size // 2)
        self.linear2 = nn.Linear(self.hidden_size // 2, self.num_classes)
        self.softmax = nn.Softmax()

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.linear1(output)
        output = self.linear2(output)
        output = output.view(-1, self.num_classes)
        output = self.softmax(output)
        return output

    

if __name__ == '__main__':
    batch_size = 16
    input_size = 66
    max_seq_length = 300
    hidden_size = 100
    num_layers = 3
    num_classes = 8
    num_epochs = 2
    print_every = 10

    rnn = BiGRU(input_size, hidden_size, num_layers, num_classes, batch_size)
    input = Variable(torch.randn(batch_size, max_seq_length, input_size), requires_grad=True)
    labels = Variable(torch.LongTensor(batch_size * max_seq_length,))
    print(labels.size())
    outputs = rnn(input)
    print(outputs.size())
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(loss)
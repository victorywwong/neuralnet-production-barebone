import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # model parameters
        self.output_class_num = 1
        self.in_channels = 1
        self.out_channels = 36
        self.kernel_heights = [1,2,3,5]
        self.keep_probab = 0.1
        self.vocab_size = 120000 # how many unique words to use (i.e num rows in embedding vector)
        self.embed_dim = 300 # how big is each word vector

        output_class_num = self.output_class_num
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_heights = self.kernel_heights
        keep_probab = self.keep_probab
        vocab_size = self.vocab_size
        embed_dim = self.embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (K, embed_dim)) for K in kernel_heights])
        self.dropout = nn.Dropout(keep_probab)
        self.fc1 = nn.Linear(len(kernel_heights)*out_channels, output_class_num)

    # get model params:

    def get_model_parameters(self):
        return {
            'output_class_num': self.output_class_num,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_heights': self.kernel_heights,
            'keep_probab': self.keep_probab,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        }

    # forward

    def forward(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
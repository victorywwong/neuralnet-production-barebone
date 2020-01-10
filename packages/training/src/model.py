import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        
        """
		Demo CNN Classifier
        Arguments
		---------
		output_class_num (C) : 1 = (pos)
		in_channels (Ci) : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
		out_channels (Co) : Number of output channels after convolution operation performed on the input matrix
		kernel_heights (Ks) : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
		keep_probab : Probability of retaining an activation node during dropout operation
		vocab_size : Size of the vocabulary containing unique words
		embed_dim (D) : Embedding dimension of word embeddings
		embedding_matrix : Embedding Matrix
        --------
		
		"""
        output_class_num = args.class_num
        in_channels = 1
        out_channels = args.kernel_num
        kernel_heights = args.kernel_sizes
        keep_probab = 1.0 - args.dropout
        vocab_size = args.embed_num
        embed_dim = args.embed_dim
        embedding_matrix = args.embedding_matrix
        
        # kernel_heights = [1,2,3,5]
        # out_channels = 36
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (K, embed_dim)) for K in kernel_heights])
        self.dropout = nn.Dropout(keep_probab)
        self.fc1 = nn.Linear(len(kernel_heights)*out_channels, output_class_num)

    # forward

    def forward(self, x):
        """
		The idea for this Convolutional Neural Netwok Classification example is very simple. We perform convolution operation on the embedding matrix 
		whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
		We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
		and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
		to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
		
		Parameters
		----------
		x: 

		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		logits.size() = (batch_size, output_size)
		
		"""
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
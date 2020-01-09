import numpy as np
import torch
import torch.nn.functional as F
from torchtext import data

import spacy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en')
TEXT = data.Field(tokenize = 'spacy', batch_first = True)


def prediction_result(model, sentence, output_class_num, in_channels, out_channels, kernel_heights, keep_probab, vocab_size, embed_dim, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
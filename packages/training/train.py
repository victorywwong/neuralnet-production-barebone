import os
import argparse
import torch
import torch.nn
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data_loader import MyDataset
import copy
import time
from src.utils import sigmoid

from src.data_loader import load_and_prec, load_glove, load_para
from src.model import Net


parser = argparse.ArgumentParser(description='CNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 5]')
parser.add_argument('-batch-size', type=int, default=512, help='batch size for training [default: 512]')
parser.add_argument('-split-num', type=int, default=5, help='number of splits')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-debug', type=bool, default=False, help='whether or not in debug mode')
# model
parser.add_argument('-dropout', type=float, default=0.9, help='the probability for dropout [default: 0.9]')
parser.add_argument('-vocab-size', type=float, default=120000, help='size of the vocabulary containing unique words [default: 120000]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='1,2,3,5', help='comma-separated kernel size to use for convolution')
# preprocessing
parser.add_argument('-max-length', type=int, default=70, help='maximum length of sentence [default: 70]')
# options
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# others
parser.add_argument('-seed', type=int, default=10, help='seed number [default: 10]')

args = parser.parse_args()

n_splits = args.split_num
batch_size = args.batch_size
n_epochs = args.epochs 
SEED = args.seed
vocab_size = args.vocab_size # how many unique words to use (i.e num rows in embedding vector)
maxlen = args.max_length # max number of words in a question to use
debug = args.debug

# n_splits = 5
# batch_size = 512
# n_epochs = 5 
# SEED = 10
# vocab_size = 120000 # how many unique words to use (i.e num rows in embedding vector)
# maxlen = 70 # max number of words in a question to use
# debug = 0

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def train(model, train_loader, optimizer, loss_fn, scheduler, clip, feats, kfold_X_features):

    model.train()

    avg_loss = 0.  
    for i, (x_batch, y_batch, index) in enumerate(train_loader):
        if feats:       
            f = kfold_X_features[index]
            y_pred = model([x_batch,f])
        else:
            y_pred = model(x_batch)

        if scheduler:
            scheduler.batch_step()

        # Compute and print loss.
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
        
    return avg_loss
    
def evaluate(model, valid_loader, loss_fn, valid_preds_fold, feats, kfold_X_valid_features):
    
    model.eval()
    
    avg_val_loss = 0.
    for i, (x_batch, y_batch,index) in enumerate(valid_loader):
        if feats:
            f = kfold_X_valid_features[index]            
            y_pred = model([x_batch,f]).detach()
        else:
            y_pred = model(x_batch).detach()
        
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]
    
        
    return avg_val_loss, valid_preds_fold

def test(model, test_loader, train_preds, valid_idx, test_preds, valid_preds_fold, test_preds_fold, splits, feats, test_features):
    
    for i, (x_batch,) in enumerate(test_loader):
        if feats:
            f = test_features[i * batch_size:(i+1) * batch_size]
            y_pred = model([x_batch,f]).detach()
        else:
            y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        
    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

    return test_preds

def checkpoint(model, fold, epoch):
    model_out_path = "model_fold_{}epoch_{}.pth".format(fold,epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def run(x_train,y_train,features,x_test, model_obj, feats = False,clip = True):
    seed_everything(SEED)
    avg_losses_f = []
    avg_val_losses_f = []
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(x_train)))
    # matrix for the predictions on the test set
    test_preds = np.zeros((len(x_test)))
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(i*1000+i)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        if feats:
            kfold_X_features = features[train_idx.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
            test_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        
        model = copy.deepcopy(model_obj)
        if args.snapshot is not None:
            print('\nLoading model from {}...'.format(args.snapshot))
            cnn.load_state_dict(torch.load(args.snapshot))
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.lr)
        
        ################################################################################################
        scheduler = False
        ###############################################################################################

        train = MyDataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))
        valid = MyDataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
        steps = 0
        best_step = 0
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            start_time = time.time()
            avg_loss = train(model, train_loader, optimizer, loss_fn, scheduler, clip, feats, kfold_X_features)
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((len(x_test)))
            avg_val_loss, valid_preds_fold = evaluate(model, valid_loader, loss_fn, valid_preds_fold, feats, kfold_X_valid_features)
            steps += 1

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_step = steps
                if args.save_best:
                    checkpoint(model, i+1, epoch)
            else:
                if steps - best_step >= args.early_stop:
                    print('early stop by {} steps.'.format(steps))
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss) 
        # predict all samples in the test set batch per batch
        test_preds = test(model, test_loader, train_preds, valid_idx, test_preds, valid_preds_fold, test_preds_fold, splits, feats, test_features)


    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))
    return train_preds, test_preds

# always call this before training for deterministic results
seed_everything()

x_train, x_test, y_train, word_index = load_and_prec(SEED, vocab_size, maxlen, debug)
x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

if debug:
    paragram_embeddings = np.random.randn(120000,300)
    glove_embeddings = np.random.randn(120000,300)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
    args.embedding_matrix = embedding_matrix
else:
    glove_embeddings = load_glove(vocab_size, word_index)    
    paragram_embeddings = load_para(vocab_size, word_index)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)   
    args.embedding_matrix = embedding_matrix
train_preds , test_preds = run(x_train,y_train,None,x_test,Net(args), feats = False, clip=False)
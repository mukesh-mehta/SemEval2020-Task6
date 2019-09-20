import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import config
from utils import tokenizer




class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x 

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            if self.y_vars is not None: # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)

class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, embedding_vector, emb_dim=100,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vector))
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds

def train(train, val, model_out_path, device, epochs = 10, vectors="glove.6B.100d"):

    print("Training on : {}, Validating on :{}".format(train, val))
    # Compute class weight
    train_df = pd.read_csv(config.TASK1["Folds"]+"/"+train, sep="\t")
    class_weights = torch.FloatTensor([compute_class_weight('balanced', [0,1], train_df['has_def'].values)[1]]).to(device)
    del train_df #delete df

    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    tv_datafields = [("text", TEXT), ("has_def", LABEL),
                 ("filename", None)]

    trn, vld = TabularDataset.splits(
               path=config.TASK1["Folds"],
               train=train,validation=val,
               format='tsv',
               skip_header=True, 
               fields=tv_datafields)

    TEXT.build_vocab(trn, vld, vectors=vectors, max_size=20000, min_freq=10)
    # get train and val iterator
    train_iter, val_iter = BucketIterator.splits(
                     (trn, vld), # we pass in the datasets we want the iterator to draw data from
                     batch_sizes=(512,  512),
                     device=torch.device(device), # if you want to use the GPU, specify the GPU number here
                     sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
                     sort_within_batch=False,
                     repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
                    )

    train_dl = BatchWrapper(train_iter, 'text', ['has_def'])
    valid_dl = BatchWrapper(val_iter, 'text', ['has_def'])
    
    model = SimpleBiLSTMBaseline(100, TEXT.vocab.vectors, emb_dim=100).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() # turn on training mode
        train_preds = []
        train_truth = []
        for x, y in tqdm(train_dl):
            opt.zero_grad()

            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()
            train_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
            train_truth.extend(y.cpu().numpy())
            running_loss += loss.item() * x.size(0)
            
        epoch_loss = running_loss / len(trn)
        
        # calculate the validation loss for this epoch
        val_loss = 0.0
        val_preds = []
        val_truth = []
        model.eval() # turn on evaluation mode
        for x, y in valid_dl:
            preds = model(x)
            loss = loss_func(preds, y)
            val_loss += loss.item() * x.size(0)
            val_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
            val_truth.extend(y.cpu().numpy())

        val_loss /= len(vld)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
        print("classification report Train")
        train_preds = np.where(np.array(train_preds)<0.5, 0, 1).flatten()
        print(classification_report(train_truth, train_preds))
        print("classification report Validation")
        val_preds = np.where(np.array(val_preds)<0.5, 0, 1).flatten()
        print(classification_report(val_truth, val_preds))
    return

def train_kfold(model_out_path, num_folds = 5, epochs = 1, vectors = "glove.6B.100d", device = "cpu"):
    #create log file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = open("log_{}.log".format(timestr),"w")
    old_stdout = sys.stdout
    sys.stdout = log_file

    for fold in range(num_folds):
        print("-"*10, "Fold number: {}".format(fold),  "-"*30)
        train("train_{}.csv".format(fold),
            "val_0.csv".format(fold),
            model_out_path+"model_{}.pth".format(fold),
            device,
            epochs = epochs)
    sys.stdout = old_stdout
    log_file.close()
    return

if __name__ == '__main__':
    train_kfold("", num_folds = 5, epochs = 1, vectors = "glove.6B.100d", device = "cpu")
import sys
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

import config
from utils import tokenizer
from preprocess import create_data
from model import SimpleLSTMBaseline, DeepMoji

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

def get_iterators(train, val, test, device, vectors="glove.6B.100d"):
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, pad_token="<PAD>", include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False)

    tv_datafields = [("text", TEXT), ("has_def", LABEL),
                 ("filename", None)]

    trn, vld, tst = TabularDataset.splits(
               path=config.TASK1["Folds"],
               train=train, validation=val, test=test,
               format='tsv',
               skip_header=True, 
               fields=tv_datafields)

    TEXT.build_vocab(trn, vld, vectors=vectors, max_size=20000, min_freq=10)

    train_iter, val_iter, test_iter = BucketIterator.splits(
                     (trn, vld, tst), # we pass in the datasets we want the iterator to draw data from
                     batch_sizes=(512,  512, 512),
                     device=torch.device(device), # if you want to use the GPU, specify the GPU number here
                     sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
                     sort_within_batch=True,
                     repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
                    )
    return train_iter, val_iter, test_iter, TEXT

def train(train, val, test, model_out_path, device, epochs = 10, vectors="glove.6B.100d"):
    print("Training on : {}, Validating on :{}".format(train, val))
    # Compute class weight
    train_df = pd.read_csv(config.TASK1["Folds"]+"/"+train, sep="\t")
    class_weights = torch.FloatTensor([compute_class_weight('balanced', [0,1], train_df['has_def'].values)[1]]).to(device)
    del train_df #delete df

    # Get iterators and Vocab_instance
    train_iter, val_iter, test_iter, TEXT = get_iterators(train, val, test, device, vectors=vectors)

    train_dl = BatchWrapper(train_iter, 'text', ['has_def'])
    valid_dl = BatchWrapper(val_iter, 'text', ['has_def'])
    test_dl = BatchWrapper(test_iter, 'text', ['has_def'])
    
    # model = SimpleLSTMBaseline(100, TEXT.vocab.vectors, emb_dim=100).to(device)
    model = DeepMoji(vocab_size=len(TEXT.vocab), embedding_dim=100, hidden_state_size=512,
                    num_layers=2,
                    output_dim=1,
                    dropout=0.5,
                    bidirectional=True,
                    pad_idx=TEXT.vocab.stoi["<PAD>"]
                ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.BCEWithLogitsLoss()

    # Best score
    best_score = 0.0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() # turn on training mode
        train_preds = []
        train_truth = []
        for x, y in tqdm(train_dl):
            opt.zero_grad()

            preds = model(x[0], x[1]) # x[0] is text sequence, x[1] is len of sequence
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()
            train_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
            train_truth.extend(y.cpu().numpy())
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_dl)
        
        # evaluate on validation set
        val_loss, val_preds, val_truth = evaluate(valid_dl, model, loss_func, device)

        train_preds = np.where(np.array(train_preds)<0.5, 0, 1).flatten()
        train_fscore = f1_score(train_truth, train_preds)
        val_fscore = f1_score(val_truth, val_preds)
        print('Epoch: {}, Training Loss: {:.4f}, Training f-score {:.4f}, Validation Loss: {:.4f}, Validation f-score {:.4f}'.format(
            epoch, epoch_loss, train_fscore, val_loss, val_fscore))
        print("classification report Train")
        print(classification_report(train_truth, train_preds))
        print("classification report Validation")
        print(classification_report(val_truth, val_preds))
        if val_fscore > best_score:
            best_score = val_fscore
            torch.save(model.state_dict(), model_out_path)
            print("Saving model with best_score {}".format(best_score))

    test_loss, test_preds, test_truth = evaluate(test_dl, model, loss_func, device, checkpoint = model_out_path)
    print("Test Loss: {:.4f}".format(test_loss))
    print("classification report Test")
    print(classification_report(test_truth, test_preds))
    return


def evaluate(loader, model, loss_func, device, checkpoint=None):
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
    val_loss = 0.0
    val_preds = []
    val_truth = []
    model.eval() # turn on evaluation mode
    for x, y in loader:
        preds = model(x[0], x[1])
        loss = loss_func(preds, y)
        val_loss += loss.item()
        val_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
        val_truth.extend(y.cpu().numpy())

    val_loss /= len(loader)
    val_preds = np.where(np.array(val_preds)<0.5, 0, 1).flatten()
    return val_loss, val_preds, val_truth


def train_kfold(num_folds, epochs, vectors = "glove.6B.100d", model_out_path=config.TASK1["Model_outpath"], device = "cpu"):
    #create log file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = open("log_{}.log".format(timestr),"w")
    old_stdout = sys.stdout
    sys.stdout = log_file

    print("Num folds {}, Epochs {}, Embeddings {}".format(num_folds, epochs, vectors))
    # create train and val data for torchtext format
    create_data(config.TASK1["Train"], config.TASK1["Folds"])
    create_data(config.TASK1["Dev"], config.TASK1["Folds"]+"task1_dev.csv", test=True)

    for fold in range(num_folds):
        print("-"*10, "Fold number: {}".format(fold),  "-"*30)
        train("train_{}.csv".format(fold),
            "val_0.csv".format(fold),
            "task1_dev.csv", 
            model_out_path+"model_{}.pth".format(fold),
            device,
            epochs = epochs)
    sys.stdout = old_stdout
    log_file.close()
    return

if __name__ == '__main__':
    Fire(train_kfold)
    # python train.py --num_folds=10 --epochs=20 --vectors="glove.6B.100d" --device="cuda"
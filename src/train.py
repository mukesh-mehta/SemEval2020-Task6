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
from preprocess import create_data
from model import SimpleLSTMBaseline, DeepMoji
from loader import BatchWrapper, get_iterators
from loss import f1_loss

def train(train, val, test, model_out_path, device, epochs = 10, vectors="glove.6B.300d"):
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
    model = DeepMoji(embedding_vector=TEXT.vocab.vectors,
                    vocab_size=len(TEXT.vocab),
                    embedding_dim=300,
                    hidden_state_size=512,
                    num_layers=2,
                    output_dim=1,
                    dropout=0.5,
                    bidirectional=True,
                    pad_idx=TEXT.vocab.stoi["<PAD>"]
                ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-2)
    # loss_func = nn.BCEWithLogitsLoss()

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
            # loss = loss_func(preds, y)
            loss = f1_loss(preds, y)
            loss.backward()
            opt.step()
            train_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
            train_truth.extend(y.cpu().numpy())
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_dl)
        
        # evaluate on validation set
        val_loss, val_preds, val_truth = evaluate(valid_dl, model, f1_loss, device) #change loss here

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

    test_loss, test_preds, test_truth = evaluate(test_dl, model, f1_loss, device, checkpoint = model_out_path)#change loss here
    test_fscore = f1_score(test_truth, test_preds)
    print("Test Loss: {:.4f}, Test F1-score {:.4f}".format(test_loss, test_fscore))
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


def train_kfold(num_folds, epochs, vectors = "glove.6B.300d", model_out_path=config.TASK1["Model_outpath"], device = "cpu"):
    #create log file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = open("log_{}.log".format(timestr),"w")
    old_stdout = sys.stdout
    sys.stdout = log_file

    print("Num folds {}, Epochs {}, Embeddings {}".format(num_folds, epochs, vectors))
    # create train and val data for torchtext format
    create_data(config.TASK1["Train"], config.TASK1["Folds"], num_fold = num_folds)
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
    # python train.py --num_folds=10 --epochs=20 --vectors="glove.6B.300d" --device="cuda"
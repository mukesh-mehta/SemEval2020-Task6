import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config
from loader import task2_iterators
from model import BiLstm_Crf
from preprocess import create_data_task2

import os
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from fire import Fire
import warnings
warnings.filterwarnings("ignore")

def train(model, iterator, optimizer, device=None):
    model.train()
    model.inference = False
    
    epoch_loss = 0.0 
    
    for batch in tqdm(iterator):
        inp = batch.text
        target = batch.labels
        
        optimizer.zero_grad()
        
        loss = model(inp, target)
        # crf loss
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, device=None):
    model.eval()
    
    epoch_loss = 0.0
    predictions = []
    true_labels = []
    model.inference = True
    for batch in tqdm(iterator):
        inp = batch.text
        target = batch.labels
                
        out, loss = model(inp, target)
        # out = [seq_len, batch_size]
        # crf loss
        
        predictions.extend(out.contiguous().view(-1).cpu().tolist())
        true_labels.extend(target.contiguous().view(-1).cpu().tolist())
                
        epoch_loss += loss.item()
    
    f1 = f1_score(true_labels, predictions, average="macro")
        
    return epoch_loss / len(iterator), f1

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(train_file, val_file, epochs, vectors, device, model_outpath):
    print(train_file, val_file, epochs, vectors, device, model_outpath)
    # create train and val dataset
    create_data_task2(config.TASK2["Train_deft"], os.path.join(config.TASK2['csv_files'], train_file), test=False)
    create_data_task2(config.TASK2["val_deft"], os.path.join(config.TASK2['csv_files'], val_file), test=False)
    
    train_iterator, val_iterator, TEXT, LABELS = task2_iterators(train_file, val_file, device, vectors)
    print(len(train_iterator), len(val_iterator))
    model = BiLstm_Crf(
        TEXT.vocab.vectors,
        vocab_size=len(TEXT.vocab), 
        embedding_dim=300, hidden_dim=512, 
        output_dim=len(LABELS.vocab), 
        num_layers=2, bidirectional=False
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    model.inference = False
    VAL_LOSS = 1e10
    
    for epoch in range(epochs):
        train_loss = train(model, train_iterator, optimizer)
        val_loss, val_f1 = evaluate(model, val_iterator)
        
        if VAL_LOSS > val_loss:
            VAL_LOSS = val_loss
            torch.save(model.state_dict(), 'bilstm-ner-crf-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')
        print(f'Val. F1 Score is : {val_f1:.2f}')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    Fire(train_model)
    # python task2_train.py --train_file="train.csv" --val_file="val.csv" --epochs=20 --vectors="glove.6B.300d" --device="cuda" --model_outpath="task2_model.pt"
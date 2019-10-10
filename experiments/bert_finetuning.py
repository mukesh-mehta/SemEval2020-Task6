from transformers import BertTokenizer
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')

MAX_LEN = 100 # max is 512 for BERT
import numpy as np

class text_dataset(Dataset):
    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        
    def __getitem__(self,index):
        
        tokenized = tokenizer.tokenize(self.X[index])
        
        if len(tokenized) > MAX_LEN : tokenized = tokenized[:MAX_LEN]
            
        ids = tokenizer.convert_tokens_to_ids(tokenized)
            
        ids = torch.tensor(ids + [0] * (MAX_LEN - len(ids)))
        
        labels = [torch.from_numpy(np.array(self.y[index]))]
      
        return ids, labels[0]
    
    def __len__(self):
        return len(self.X)
        
from transformers import BertConfig
from transformers import BertModel
import torch.nn as nn
import torch

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        
class BertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        
num_labels = 1
model = BertForSequenceClassification(num_labels).cpu()
# model = torch.nn.DataParallel(model)

import pandas as pd
tr = pd.read_csv("../deft_corpus/data/Task1_folds/train_0.csv", sep="\t").head(1000)
print(tr.shape)
val = pd.read_csv("../deft_corpus/data/Task1_folds/val_0.csv", sep="\t").head(100)
print(val.shape)

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.BCEWithLogitsLoss(weight=torch.tensor([1.4880]).double())

epoch=1
for e in range(epoch):
    train_loss = []
    train_preds = []
    train_truth = []
    model.train()
    for x, y in tqdm(torch.utils.data.DataLoader(text_dataset(tr["text"].values, tr['has_def'].values), batch_size=16, num_workers=12)):
#         print(x.shape, y.shape)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred.reshape(-1).double(), y.double())
        
        train_preds.extend(y_pred.cpu().data.numpy())
        train_truth.extend(y.cpu().data.numpy())
        
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())
        
    val_loss = []
    val_preds = []
    val_truth = []
    model.eval()
    for x, y in tqdm(torch.utils.data.DataLoader(text_dataset(val["text"].values, val['has_def'].values), batch_size=16, num_workers=12)):
        y_pred = model(x)
        
        val_preds.extend(y_pred.cpu().data.numpy())
        val_truth.extend(y.cpu().data.numpy())
        
        loss = loss_func(y_pred.reshape(-1).double(), y.double())
        val_loss.append(loss.item())
print("Epoch: %d, Train loss: %.3f,  Val loss: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))

train_preds = np.where(np.array(train_preds)<0.5, 0, 1).flatten()
val_preds = np.where(np.array(val_preds)<0.5, 0, 1).flatten()
print("Train acc: %.3f, Val acc: %.3f" %(accuracy_score(train_truth,train_preds),accuracy_score(val_truth,val_preds)))
print("Classification report" , classification_report(val_truth,val_preds))
print("Confusion report" , confusion_matrix(val_truth,val_preds))
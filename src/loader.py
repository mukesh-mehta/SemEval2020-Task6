import torch

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

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

def get_iterators(train, val, test, device, batch_size=256, vectors="glove.6B.100d"):
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

    TEXT.build_vocab(trn, vld, vectors=vectors, max_size=20000, min_freq=2)

    train_iter, val_iter, test_iter = BucketIterator.splits(
                     (trn, vld, tst), # we pass in the datasets we want the iterator to draw data from
                     batch_sizes=(batch_size,  batch_size, batch_size),
                     device=torch.device(device), # if you want to use the GPU, specify the GPU number here
                     sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
                     sort_within_batch=True,
                     repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
                    )
    return train_iter, val_iter, test_iter, TEXT

def task2_iterators(train, val, device, vectors, batch_size=256):
    TEXT = Field(sequential=True, tokenize=None, init_token="<sos>", eos_token="<eos>")
    LABELS = Field(sequential=True, init_token="<sos>",eos_token="<eos>", is_target=True)

    tv_datafields = [("text", TEXT), ("labels", LABELS)]
    trn, vld = TabularDataset.splits(
                   path=config.TASK2['data'], # the root directory where the data lies
                   train=train,validation=val,
                   format='csv',
                   skip_header=True,
                   fields=tv_datafields)
    TEXT.build_vocab(trn,vld, vectors="glove.6B.300d")
    LABELS.build_vocab(trn,vld)
    train_iterator, val_iterator = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(batch_size, batch_size), device=device, sort_within_batch=False, sort_key=lambda x: len(x.text)
    )
    return train_iterator, val_iterator, TEXT, LABELS
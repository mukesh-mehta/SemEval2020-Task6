from config import Task1_finetune_config
import pandas as pd
import csv

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import (WEIGHTS_NAME, BertConfig,
                            AlbertConfig,
                            AlbertForSequenceClassification,
                            AlbertTokenizer,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import math
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report
import config as CONFIG
from preprocess import create_data_task1

config = Task1_finetune_config
MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
            'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
            'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        }

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def load_and_cache_examples(examples, tokenizer, evaluate=False, no_cache=False):
    """
    Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

    Utility function for train() and eval() methods. Not intended to be used directly.
    """

#     process_count = self.args['process_count']

    tokenizer = tokenizer
    output_mode = 'classification'
#     args=self.args

    if not os.path.isdir(config['cache_dir']):
        os.mkdir(config['cache_dir'])

    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(config['cache_dir'], f"cached_{mode}_{config['model_type']}_{config['max_seq_length']}_binary")

    features = convert_examples_to_features(examples,
                                    tokenizer,
                                    label_list=config['label_list'],
                                    max_length=config['max_seq_length'],
                                    output_mode=output_mode,
                                    pad_on_left=bool(config['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    pad_token_segment_id=4 if config['model_type'] in ['xlnet'] else 0,
    )

#         if not no_cache:
#             torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train_model(model, train_dataloader):
    device = 'cuda'
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in range(int(config['num_train_epochs'])):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[3]}
            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if config['model_type'] != 'distilbert':
                inputs['token_type_ids'] = batch[2] if config['model_type'] in ["bert", "xlnet", "albert"] else None  
            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if config['gradient_accumulation_steps'] > 1:
                loss = loss / config['gradient_accumulation_steps']

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config['save_steps'] > 0 and global_step % config['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        config['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
        print("Epoch ", _, global_step, tr_loss / global_step)
    return model

def evaluate(model, eval_dataloader):
    output_mode = "classification"
    device = "cuda"
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[3]}
            # XLM, DistilBERT and RoBERTa don't use segment_ids
#             preds.extend(batch[3])
            if config['model_type'] != 'distilbert':
                inputs['token_type_ids'] = batch[2] if config['model_type'] in ["bert", "xlnet", "albert"] else None  
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    return preds, out_label_ids

if __name__ == '__main__':
    if not os.path.exists(CONFIG.TASK1["Folds"]):
        os.mkdir(CONFIG.TASK1["Folds"])
    create_data_task1(CONFIG.TASK1["Train"], CONFIG.TASK1["Folds"])
    create_data_task1(CONFIG.TASK1["Dev"], CONFIG.TASK1["Folds"]+"/task1_dev.csv", test=True)

    config_class, model_class, tokenizer_class = MODEL_CLASSES['albert']
    tokenizer = tokenizer_class.from_pretrained('albert-base-v2')
    # update config
    config['model_name'] = 'albert'
    config['model_type'] = 'albert'

    # train model
    train_df = pd.read_csv("../deft_corpus/data/Task1_folds/train_0.csv", sep="\t")[['text','has_def']]
    train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))]


    train_dataset = load_and_cache_examples(train_examples, tokenizer, evaluate=False, no_cache=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['train_batch_size'])
    t_total = len(train_dataloader) // config['gradient_accumulation_steps'] * config['num_train_epochs']

    model = model_class.from_pretrained('albert-base-v2', num_labels=2)
    model.to('cuda')
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = math.ceil(t_total * config['warmup_ratio'])
    config['warmup_steps'] = warmup_steps if config['warmup_steps'] == 0 else config['warmup_steps']

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)

    model = train_model(model, train_dataloader)

    # evaluate model
    eval_df = pd.read_csv("../deft_corpus/data/Task1_folds/task1_dev.csv", sep="\t")[['text','has_def']]
    eval_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))]
    eval_dataset = load_and_cache_examples(eval_examples, tokenizer, evaluate=False, no_cache=False)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config['eval_batch_size'])
    preds, out_label_ids = evaluate(model, eval_dataloader)
    # dev_submission = eval_df
    # dev_submission['has_def'] = preds
    # dev_submission.to_csv("dev_tsk1_sub.deft", sep='\t', index=False)
    print(classification_report(out_label_ids, preds))

    for file in os.listdir("../deft_corpus/data/Task1/dev"):
        eval_text = []
        truth = []
        with open("../deft_corpus/data/Task1/dev/"+file, 'r') as eval_file:
            csv_reader = csv.reader(eval_file, delimiter = "\t")
            [(eval_text.append(row[0]), truth.append(int(row[1]))) for row in csv_reader]
        # eval_df = pd.read_csv("../deft_corpus/data/Task1/dev/"+file, sep="\t", names=['text', 'has_def'])
        eval_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(eval_text, truth))]
        eval_dataset = load_and_cache_examples(eval_examples, tokenizer, evaluate=False, no_cache=False)
        # eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=config['eval_batch_size'])
        preds, out_label_ids = evaluate(model, eval_dataloader)
        with open("/home/mukesh/Desktop/SemEval_Task1_Submission/"+file, 'w') as out_file:
            csv_writer = csv.writer(out_file, delimiter = '\t')
            for tx, label in zip(eval_text,preds):
                csv_writer.writerow([tx, label])
    
    for file in os.listdir("../deft_corpus/data/test_files/subtask_1"):
        eval_text = []
        truth = []
        with open("../deft_corpus/data/test_files/subtask_1/"+file, 'r') as eval_file:
            csv_reader = csv.reader(eval_file, delimiter = "\t")
            [(eval_text.append(row[0]), truth.append(0)) for row in csv_reader]
        # eval_df = pd.read_csv("../deft_corpus/data/Task1/dev/"+file, sep="\t", names=['text', 'has_def'])
        eval_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(eval_text, truth))]
        eval_dataset = load_and_cache_examples(eval_examples, tokenizer, evaluate=False, no_cache=False)
        # eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=config['eval_batch_size'])
        preds, out_label_ids = evaluate(model, eval_dataloader)
        with open("/home/mukesh/Desktop/SemEval_Submission/task1/"+file, 'w') as out_file:
            csv_writer = csv.writer(out_file, delimiter = '\t')
            for tx, label in zip(eval_text,preds):
                csv_writer.writerow([tx, label])
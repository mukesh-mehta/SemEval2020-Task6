CUDA_LAUNCH_BLOCKING=1
from torch.nn import CrossEntropyLoss
from config import Task2_finetune_config as config
from config import Task2_labels as label_list
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
import math
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from utils import parse_deft, submission_task2
from preprocess import create_data_task2

pad_token_label_id = CrossEntropyLoss().ignore_index

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=pad_token_label_id,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label[0]=='O' or label[0]=='I':
#                 print([label_map[label]]+[label_map[label]]*(len(word_tokens) - 1))
                label_ids.extend([label_map[label]]+[label_map[label]]*(len(word_tokens) - 1))
            elif label[0]=='B':
#                 print([label_map[label]]+[label_map["I"+label[1:]]]*(len(word_tokens) - 1))
                label_ids.extend([label_map[label]]+[label_map["I"+label[1:]]]*(len(word_tokens) - 1))
                
        

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
        
#         print(len(label_ids), max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features

def load_and_cache_examples(examples, tokenizer, labels, pad_token_label_id):
    features = convert_examples_to_features(examples, labels, config['max_seq_length'], tokenizer,
                                            cls_token_at_end=bool(config['model_type'] in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if config['model_type'] in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(config['model_type'] in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(config['model_type'] in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if config['model_type'] in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def train_model(model, train_dataloader):
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    device='cuda'
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(config['num_train_epochs']):
    #     epoch_iterator = tqdm(train_dataloader)
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if config['model_type'] in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
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
    print(global_step, tr_loss / global_step)
    return model

def evaluate(model, eval_dataloader, labels):
    device = "cuda"
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if config['model_type'] in ["bert", "xlnet"] else None,
                      # XLM and RoBERTa don"t use segment_ids
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }
    return results, preds_list, out_label_list

if __name__ == '__main__':
    MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    config_ = config_class.from_pretrained('bert-base-uncased',
                                              num_labels=len(label_list))
    model = model_class.from_pretrained('bert-base-uncased', config=config_)
    model.to('cuda')
    
    # train_df = pd.read_csv("train.csv",sep = ",")
    train_df = create_data_task2("../deft_corpus/data/deft_files/train")
    train_df.dropna(inplace=True)
    train_examples = [InputExample(i, str(text).split(" "), str(label).split(" ")) for i, (text, label) in enumerate(zip(train_df['text'].values, train_df['labels'].values))]
    train_dataset = load_and_cache_examples(train_examples, tokenizer, label_list, pad_token_label_id)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=4)
    t_total = len(train_dataloader) // config['gradient_accumulation_steps'] * config['num_train_epochs']


    eval_df = create_data_task2("../deft_corpus/data/deft_files/dev")
    eval_df.dropna(inplace=True)
    eval_examples = [InputExample(i, str(text).split(" "), str(label).split(" ")) for i, (text, label) in enumerate(zip(eval_df['text'].values, eval_df['labels'].values))]
    eval_dataset = load_and_cache_examples(eval_examples, tokenizer, label_list, pad_token_label_id)
    eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=4)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=t_total)
    model=train_model(model, train_dataloader)
    results, preds_list, out_label_list = evaluate(model, eval_dataloader, label_list)
    print(results)
    for file in os.listdir("../deft_corpus/data/deft_files/dev"):
        print(file)
        eval_df = pd.DataFrame.from_records(parse_deft("../deft_corpus/data/deft_files/dev/"+file))
        eval_examples = [InputExample(i, str(text).split(" "), str(label).split(" ")) for i, (text, label) in enumerate(zip(eval_df['text'].values, eval_df['labels'].values))]
        eval_dataset = load_and_cache_examples(eval_examples, tokenizer, label_list, pad_token_label_id)
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=4)
        results, preds_list, out_label_list = evaluate(model, eval_dataloader, label_list)
        eval_df['labels'] = [" ".join(preds) for preds in preds_list]
        submission_task2(eval_df[['text', 'filename', 'start', 'end', 'labels']], "/home/mukesh/Desktop/SemEval_Task1_Submission/task_2_"+file)


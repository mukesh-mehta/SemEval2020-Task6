import os

DATA_FOLDER = "../deft_corpus/data/"

TASK1 = {
    "Train": os.path.join(DATA_FOLDER, "Task1/train/"),
    "Dev": os.path.join(DATA_FOLDER, "Task1/dev/"),
    "Folds": os.path.join(DATA_FOLDER, "Task1_folds/"),
    "Model_outpath": "/media/mukesh/36AD331451677000/semevalTask1Model/"
}

TASK2 = {
    "Train_deft": os.path.join(DATA_FOLDER,"deft_files/train"),
    "val_deft": os.path.join(DATA_FOLDER,"deft_files/dev"),
    "csv_files": os.path.join(DATA_FOLDER, "Task2")
}

Task1_finetune_config = {
            'output_dir': 'outputs/',
            'cache_dir': 'cache_dir',
            'max_seq_length': 150,
            'train_batch_size': 12,
            'gradient_accumulation_steps': 1,
            'eval_batch_size': 8,
            'num_train_epochs': 1,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.06,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,

            'logging_steps': 50,
            'save_steps': 1000,

            'overwrite_output_dir': False,
            'reprocess_input_data': False,
            'label_list':[0,1]
        }

Task2_labels = [
'O',
'I-Definition',
'I-Term',
'I-Secondary-Definition',
'B-Term',
'B-Definition',
'I-Definiti-frag',
'I-Qualifier',
'I-Alias-Term',
'B-Alias-Term',
'B-Secondary-Definition',
'I-Referential-Definition',
'B-Referential-Definition',
'B-Qualifier',
'B-Referential-Term',
'I-Referential-Term',
'B-Definiti-frag',
'I-Ordered-Definition',
'I-Ordered-Term',
'B-Te-frag',
'B-Ordered-Definition',
'B-Ordered-Term',
'I-Te-frag',
'B-Alias-Te-frag',
'I-Alias-Te-frag'
]

Task2_finetune_config ={
    "max_seq_length": 125,
    "model_type": "bert",
    "num_train_epochs":4,
    "gradient_accumulation_steps":1,
    'learning_rate':5e-5,
    'adam_epsilon':1e-8,
    'warmup_steps':0,
    'weight_decay':0.0,
    'max_grad_norm':1.0
}
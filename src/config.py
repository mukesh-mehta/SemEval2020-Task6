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

Task2_labels = ['B-Referential-Definition',
 'I-Referential-Term',
 'I-Alias-Term',
 'I-Qualifier',
 'B-Ordered-Term',
 'B-Ordered-Definition',
 'B-Referential-Term',
 'O',
 'B-Qualifier',
 'I-Term-frag',
 'I-Definition',
 'I-Definition-frag',
 'I-Referential-Definition',
 'I-Term',
 'B-Secondary-Definition',
 'I-Ordered-Definition',
 'B-Alias-Term',
 'I-Ordered-Term',
 'B-Definition',
 'B-Term-frag',
 'B-Definition-frag',
 'I-Secondary-Definition',
 'B-Term',
 'B-Alias-Term-frag']

Task2_finetune_config ={
    "max_seq_length": 125,
    "model_type": "bert",
    "num_train_epochs":1,
    "gradient_accumulation_steps":1,
    'learning_rate':5e-5,
    'adam_epsilon':1e-8,
    'warmup_steps':0,
    'weight_decay':0.0,
    'max_grad_norm':1.0
}
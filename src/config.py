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
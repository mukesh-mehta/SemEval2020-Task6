import pandas as pd
import config
import os
from sklearn.model_selection import StratifiedKFold

from utils import get_text_labels, parse_deft

def create_data_task1(folder_path, out_path, num_fold = 5, test=False):
    master_df = pd.DataFrame()
    for i, file in enumerate(os.listdir(folder_path)):
        temp_df = pd.read_csv(os.path.join(folder_path, file), names=['text', 'has_def'], sep="\t")
        temp_df['filename'] = str(file)#.split(".")[0].split("_")[1]
        master_df = pd.concat([master_df, temp_df])
        del temp_df
    if not test:
        master_df['length'] = master_df['text'].map(len)
        master_df = master_df[master_df['length']>40]
        master_df.drop(columns=['length'])
        # Initialize stratified k-fold
        print(master_df.shape)
        skf = StratifiedKFold(n_splits=num_fold, random_state=22)
        for i, (train_idx, test_idx) in enumerate(skf.split(master_df['text'], master_df["has_def"])):
            train_df = master_df.iloc[train_idx]
            val_df = master_df.iloc[test_idx]
            train_df.to_csv("{}/train_{}.csv".format(out_path, i), index=False, sep="\t")
            val_df.to_csv("{}/val_{}.csv".format(out_path, i), index=False, sep="\t")
    else:
        master_df.to_csv(out_path, sep="\t", index=False)
    return

def create_data_task2(folder_path, output_path, test=False):
    all_data = []
    for files in os.listdir(folder_path):
        all_data.extend(parse_deft(os.path.join(folder_path, files)))
    pd.DataFrame.from_records(all_data).to_csv(output_path, index=False)
    print("output_path", len(all_data), pd.DataFrame.from_records(all_data).shape)
    return


if __name__ == '__main__':
    if not os.path.exists(config.TASK1["Folds"]):
        os.mkdir(config.TASK1["Folds"])
    create_data_task1(config.TASK1["Train"], config.TASK1["Folds"])
    create_data_task1(config.TASK1["Dev"], config.TASK1["Folds"]+"/task1_dev.csv", test=True)
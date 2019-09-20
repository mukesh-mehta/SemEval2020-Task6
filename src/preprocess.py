import pandas as pd
import config
import os
from sklearn.model_selection import StratifiedKFold

def create_data(folder_path, out_path, num_fold = 5, test=False):
    master_df = pd.DataFrame()
    for i, file in enumerate(os.listdir(folder_path)):
        temp_df = pd.read_csv(os.path.join(folder_path, file), names=['text', 'has_def'], sep="\t")
        temp_df['filename'] = str(file).split(".")[0].split("_")[1]
        master_df = pd.concat([master_df, temp_df])
    if not test:
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=num_fold)
        for i, (train_idx, test_idx) in enumerate(skf.split(master_df['text'], master_df["has_def"])):
            train_df = master_df.iloc[train_idx]
            val_df = master_df.iloc[test_idx]
            train_df.to_csv("{}/train_{}.csv".format(out_path, i), index=False, sep="\t")
            val_df.to_csv("{}/val_{}.csv".format(out_path, i), index=False, sep="\t")
    else:
        master_df.to_csv(out_path, sep="\t", index=False)
    return

if __name__ == '__main__':
    if not os.path.exists(config.TASK1["Folds"]):
        os.mkdir(config.TASK1["Folds"])
    create_data(config.TASK1["Train"], config.TASK1["Folds"])
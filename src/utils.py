import pandas as pd

def create_data(folder_path):
    df = pd.DataFrame()
    for i, file in enumerate(os.listdir(folder_path)):
        temp_df = pd.read_csv(os.path.join(folder_path, file), names=['text', 'has_def'], sep="\t")
        temp_df['filename'] = str(file).split(".")[0].split("_")[1]
        df = pd.concat([df, temp_df])
    return df.to_json(orient='records')
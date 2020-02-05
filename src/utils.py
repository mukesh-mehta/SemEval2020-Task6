import spacy
import string

spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits)).strip()
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_seq(text):
    return text.split(" ")

def get_text_labels(sequence_tags):
    # TOKEN TXT_SOURCE_FILE START_CHAR END_CHAR TAG TAG_ID ROOT_ID RELATION
    text = [data[0].strip() for data in sequence_tags]
    tags = [data[4].strip() for data in sequence_tags]
    start = [data[2].strip() for data in sequence_tags]
    end = [data[3].strip() for data in sequence_tags]
    f_name = [data[1].strip() for data in sequence_tags]
    return {"text":" ".join(text), "filename":" ".join(f_name), "labels": " ".join(tags), "start": " ".join(start), "end": " ".join(end)}

def parse_deft(deft_file):
    with open(deft_file, 'r') as deft:
        all_text = deft.read()
    all_sequences = []
    for lines in all_text.split("\n\n"):
        if lines != '\n':
            sents = []
            for token_data in lines.split("\n"):
                if len(token_data.split("\t"))==8 and (token_data.split("\t")[4].strip()[0] in ["B", "I", "O"]):
                    sents.append(token_data.split("\t"))
            all_sequences.append(get_text_labels(sents))
    return all_sequences

def submission_task2(df, out_file):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.split())
    with open(out_file, 'w') as f:
        for tok, file, start, end, label in df.values:
            for meta in zip(tok, file, start, end, label):
                f.write("\t".join(meta)+'\n')
            f.write("\n\n")
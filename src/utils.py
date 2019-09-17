import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

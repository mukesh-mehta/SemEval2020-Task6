import spacy
import string

spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits)).strip()
    return [tok.text for tok in spacy_en.tokenizer(text)]

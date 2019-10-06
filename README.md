# SemEval2020-Task6


This repository aims to solve ............

There are three subtasks

Subtask 1: Sentence Classification

	Given a sentence, classify whether or not it contains a definition. This is the traditional definition extraction task.

Subtask 2: Sequence Labeling

	Label each token with BIO tags according to the corpus' tag specification (see Data page).

Subtask 3: Relation Classification

	Given the tag sequence labels, label the relations between each tag according to the corpus' relation specification

### Task1

There are *16745* samples for training and ..... samples for test. Which is spread across .... num of topics and have the distributions as follows.

Overall Distribution

|   | has def  |no def   | Total |
|---|---|---|---|
| Train|  5627 | 11118  |16745 |
|  Val |   534| 277  | 811 |
| Test |   |   | |

##### Approach:
1. Training data is present in different text files which are processed to convert in '.csv' format.
2. Divided the data in 10 folds.
3. Torch-text and pytorch framework is used with custom preprocessing and tokenizer.
4. Experiments are done with pre-trained embeddings and Sequence classification network with dynamic length of sentence.

#### Model
##### Baseline
1. The Baseline model is a *Deepmoji model* model.

```python
	model = DeepMoji(embedding_vector=TEXT.vocab.vectors,
    	                vocab_size=len(TEXT.vocab),
        	            embedding_dim=300,
            	        hidden_state_size=256,
                	    num_layers=2,
                    	output_dim=1,
                    	dropout=0.5,
                    	bidirectional=True,
                    	pad_idx=TEXT.vocab.stoi["<PAD>"]
                	).to(device)	
```

2. Model was trained using *Adam* optimizer for 20 epochs with 256 batch size and class weight.

##### Results
Following is the classification report.



Training Loss 1.0857, Training f-score 0.6150, Validation Loss 1.1003, Validation f-score 0.6354

classification report Train

|              |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|         0.0  |     0.80    |  0.79  |    0.80   |   9720|
|         1.0  |     0.61    |  0.62  |    0.62   |   5063|
|    accuracy  |             |        |    0.73   |  14783|
|   macro avg  |     0.70    |  0.71  |    0.71   |  14783|
|weighted avg  |     0.73    |  0.73  |    0.73   |  14783|

classification report Validation (10% of total training data)

|              |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|         0.0  |     0.84    |  0.69  |    0.75   |   1081|
|         1.0  |     0.55    |  0.75  |    0.64   |   563 |
|    accuracy  |             |        |    0.71   |   1644|
|   macro avg  |     0.70    |  0.72  |    0.70   |   1644|
|weighted avg  |     0.74    |  0.71  |    0.71   |   1644|


Val set Loss: 1.0481, Val set F1-score 0.6585

|              |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|         0.0  |     0.86    |  0.69  |    0.77   |   534|
|         1.0  |     0.57    |  0.78  |    0.66   |   277 |
|    accuracy  |             |        |    0.72   |   811|
|   macro avg  |     0.71    |  0.74  |    0.71   |   811|
|weighted avg  |     0.76    |  0.72  |    0.73   |   811|



Following is the behavior of loss function with number of epochs.

![png] (https://github.com/mukesh-mehta/SemEval2020-Task6/blob/master/Loss_avg.png?raw=true)

![png] (https://github.com/mukesh-mehta/SemEval2020-Task6/blob/master/F-Score_avg.png?raw=true)

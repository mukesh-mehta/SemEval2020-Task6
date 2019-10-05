# SemEval2020-Task6


This repository aims to solve ............

There are three subtasks

1. Task1
2. Task2
3. Task3

### Task1

There are *16745* samples for training and ..... samples for test. Which is spread across .... num of topics and have the distributions as follows.

Overall Distribution

|   | has def  |no def   | Total |
|---|---|---|---|
| Train|  5627 | 11118  |16745 |
|  Val |   |   | |
| Test |   |   | |

##### Approach:
1. Training data is present in different text files which are processed to convert in '.csv' format.
2. Divided the data in 5 folds.
3. Torch-text and pytorch framework is used with custom preprocessing and tokenizer.
4. Experiments are done with pre-trained embeddings and Sequence classification network with dynamic length of sentence.

#### Model
##### Baseline
1. The Baseline model is a *LSTM* model with *100* hidden units and *2* linear layers with *glove.6B.100d* embeddings.
2. Model was trained using *Adam* optimizer for 10 epochs with 128 batch size and class weight.

##### Results
Following is the classification report.

Training Loss: 0.4737, Validation Loss: 0.2037

classification report Train

|              |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|         0.0  |     0.82    |  0.86  |    0.84   |   8894|
|         1.0  |     0.69    |  0.61  |    0.65   |   4501|
|    accuracy  |             |        |    0.78   |  13395|
|   macro avg  |     0.75    |  0.74  |    0.75   |  13395|
|weighted avg  |     0.77    |  0.78  |    0.78   |  13395|

classification report Validation (20% of total training data)

|              |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|         0.0  |     0.87    |  0.63  |    0.73   |   2224|
|         1.0  |     0.53    |  0.82  |    0.64   |   1126|
|    accuracy  |             |        |    0.69   |   3350|
|   macro avg  |     0.70    |  0.73  |    0.69   |   3350|
|weighted avg  |     0.76    |  0.69  |    0.70   |   3350|

FOllowing is the behavior of loss function with number of epochs.

![png] (https://github.com/mukesh-mehta/SemEval2020-Task6/blob/master/Loss_avg.png)

![png] (https://github.com/mukesh-mehta/SemEval2020-Task6/blob/master/F-Score_avg.png)

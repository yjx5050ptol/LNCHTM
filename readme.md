This is the official implementation for LNCHTM (Lifelong Hierarchical Topic Modeling via Non-parametric Word Embedding Clustering, ECML-PKDD 2024).

The structure of this repository is as follows:

LNCHTM

- code (includes all source codes)
  -- dataset.ini (the information about datasets)
  -- experiment.ini (experiment settings for training and evaluation)
  -- embedding.py (class to obtain word embeddings)
  -- KnowledgeBase.py (class to implement the KB)
  -- NonparametricClustering.py (class to apply AP algorithm)
  -- preprocess.py (code for preprocessing the original datasets)
  -- representativeness.py (class to evaluate the word representativeness)
  -- LNCHTM.py (LNCHTM implementation and runner)
  -- evaluation.py (experiment evaluation implementation)
  -- utils.py 

- dataset (stores the original datasets which can be obtained through links in the paper)
- processed\_dataset (stores our processed datasets, which are for training and evaluation)
- prepared (stores the pre-obtained word co-occurrence matrices)
- knowledgebase (stores the KBs for different experiment settings and chunks)
- glove (stores the GloVe embedding files that can be used by torchtext.vocab)



For running LNCHTM on a dataset, you need to choose an existing experiment setting (such as LNCHTM_20News), or you can offer a new experiment setting following the format in 'experiment.ini', and then use the shell 'run_and_eval.sh':

```
bash run_and_eval.sh LNCHTM_20News
```

( PS: If you want to run the catastrophic forgetting test, you can add a code line to LNCHTM.py to call the function catastrophic_forgetting() .)


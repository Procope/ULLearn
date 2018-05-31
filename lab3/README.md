## Evaluating sentence representation

In lab 2, we implemented and trained three different models to learn the word embeddings: Skip-gram, Bayesian skip-gram,
and EmbedAlign. We now compare the obtained word representations using SentEval, a Facebook evaluation toolkit for
evaluating the quality of sentence embeddings through a diverse set of downstream _transfer_ tasks. 

To run the SentEval tasks, simply use the following scripts:
- `senteval_skipgram.py` 
- `senteval_embedalign.py`
- `senteval_glove.py`
- `senteval_fasttext.py`

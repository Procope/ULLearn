from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
#import data
# data.py is part of Senteval and it is used for loading word2vec style files

import tensorflow as tf
import logging
from collections import defaultdict
import dill
import gensim

# Import repos
sys.path.insert(0, 'SentEval')
sys.path.insert(0, 'dgm4nlp')
import senteval
import dgm4nlp

# Paths
PATH_TO_DATA = 'SentEval/data'
PATH_TO_MODEL_DIR = 'models/glove/'
PATH_TO_MODEL = 'vectors50.txt'
EMBED_DIMS = 100

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#





# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': '',
                   'usepytorch': False,
                   'kfold': 10,
                   'model': None}
# made dictionary a dotdict
params_senteval = dotdict(params_senteval)
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}


def prepare(params, samples):
    """
    In this example we are going to load a tensorflow model,
    we open a dictionary with the indices of tokens and the computation graph
    """

    params.model = {}

    with open(PATH_TO_MODEL_DIR + PATH_TO_MODEL, 'r') as f_in:
        lines = f_in.readlines()

    for line in lines:
        split_line = line.split()
        word = split_line[0]
        vector = [float(r) for r in split_line[1:]]

        if len(vector) != EMBED_DIMS:
            print('> 100 dims')
            continue
        params.model[word] = np.array(vector)

    return


def batcher(params, batch):
    """
    At this point batch is a python list containing sentences. Each sentence is a list of tokens (each token a string).
    The code below will take care of converting this to unique ids that EmbedAlign can understand.

    This function should return a single vector representation per sentence in the batch.
    In this example we use the average of word embeddings (as predicted by EmbedAlign) as a sentence representation.

    In this method you can do mini-batching or you can process sentences 1 at a time (batches of size 1).
    We choose to do it 1 sentence at a time to avoid having to deal with masking.

    This should not be too slow, and it also saves memory.
    """

    # if a sentence is empty, dot is set to be the only token.
    # you can change it into NULL dependening on your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        word_embeds = [params.model[word] for word in sent if word in params.model]

        if len(word_embeds) == 0:
            word_embeds = [params.model['.']]

        sent_vec = np.mean(word_embeds, axis=0)

        # check if there is any NaN in vector (they appear sometimes when there's padding)
        if np.isnan(sent_vec.sum()):
            sent_vec = np.nan_to_num(sent_vec)

        embeddings.append(sent_vec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


if __name__ == "__main__":
    # define paths
    # path to senteval data
    # note senteval adds downstream into the path
    params_senteval.task_path = PATH_TO_DATA

    # we use 10 fold cross validation
    params_senteval.kfold = 10
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression),
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'STS16',
                      'SST5', 'TREC', 'MRPC', 'SICKEntailment',
                      'Depth', 'BigramShift', 'Tense', 'SubjNumber']

    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)

    with open('output/glove50_results.txt', 'w') as f_out:
        print(results, file=f_out)

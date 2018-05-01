import numpy as np
import random
import torch
from collections import defaultdict

def read_corpus(corpus_path, threshold, n_sentences):

    with open(corpus_path, 'r') as f:
        if n_sentences:
            corpus = f.readlines()[:n_sentences]
        else:
            corpus = f.readlines()

    tokenized_corpus = [sentence.split() for sentence in corpus]

    vocabulary = []
    counter = defaultdict(int)

    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
            counter[token] += 1

    print('Number of discarded word types:', len([w for w in vocabulary if counter[w] <= threshold]))

    vocabulary = [w for w in vocabulary if counter[w] > threshold]

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (w, idx) in word2idx.items()}

    return tokenized_corpus, word2idx, idx2word


def create_skipgrams(tokenized_corpus,
                     word2idx,
                     window_size,
                     batch_size):

    triplets = []
    V = len(word2idx)

    for sentence in tokenized_corpus:

        sentence_ids = [word2idx[w] for w in sentence if w in word2idx.keys()]

        for center_word in range(len(sentence_ids)):

            for offset in range(-window_size, window_size + 1):
                context_word = center_word + offset

                # handle beginning and end of sentence and discard center word
                if context_word < 0 or context_word >= len(sentence_ids) or center_word == context_word:
                    continue

                context_word_id = sentence_ids[context_word]

                # negative samples: draw from unigram distribution to the 3/4th power
                neg_context_ids = torch.LongTensor(np.random.randint(0, V, 2*window_size))

                triplets.append((sentence_ids[center_word], context_word_id, neg_context_ids))

    # shuffle
    triplets = random.sample(triplets, len(triplets))

    n_batches = len(triplets) // batch_size
    cutoff = len(triplets) - n_batches * batch_size

    if cutoff > 0:
        triplets = triplets[:-cutoff]

    batches = [triplets[x : x+batch_size] for x in range(0, len(triplets), batch_size)]

    batches_new = []

    for batch in batches:
        center_id_batch       = [triplet[0] for triplet in batch]
        pos_context_id_batch  = [triplet[1] for triplet in batch]
        neg_context_ids_batch = [triplet[2] for triplet in batch]

        batches_new.append((center_id_batch, pos_context_id_batch, neg_context_ids_batch))

    return batches_new

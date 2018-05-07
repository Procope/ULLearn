import numpy as np
import random
import torch
from collections import defaultdict
import pickle


def read_corpus(corpus_path, word_limit=10000, n_sentences=None):

    with open(corpus_path, 'r') as f:
        if n_sentences:
            corpus = f.readlines()[:n_sentences]
        else:
            corpus = f.readlines()

    tokenized_corpus = [sentence.split() for sentence in corpus]

    # vocabulary = []
    counter = defaultdict(int)

    for sentence in tokenized_corpus:
        for token in sentence:
            # if token not in vocabulary:
            #     vocabulary.append(token)
            counter[token] += 1

    sorted_counter_items = sorted(counter.items(), key=lambda t: t[1], reverse=True)
    vocabulary = [w for (w, c) in sorted_counter_items[:word_limit]]
    vocabulary.insert(0, '-UNK-')
    vocabulary.insert(1, '-PAD-')

    # print('Number of discarded word types:', len([w for w in vocabulary if counter[w] <= threshold]))
    # vocabulary = [w for w in vocabulary if counter[w] > threshold]

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (w, idx) in word2idx.items()}

    return tokenized_corpus, word2idx, idx2word


def create_skipgrams(tokenized_corpus,
                     word2idx,
                     window_size,
                     batch_size):

    triplets = []
    V = len(word2idx)

    with open("stop_words_en.txt", "r") as f:
        stop_words = list(map(str.strip, f.readlines()))

    for i, sentence in enumerate(tokenized_corpus, start=1):

        if i % 50000 == 0:
            print('{} sentences processed.'.format(i))

        sentence_ids = [word2idx[w] if (w in word2idx.keys() and w not in stop_words) else word2idx['-UNK-'] for w in sentence]
        # print(sentence_ids)
        #sentence_ids = [word2idx[w] for w in sentence if w in word2idx.keys()]

        for center_word in range(len(sentence_ids)):

            for offset in range(-window_size, window_size + 1):
                context_word = center_word + offset

                # handle beginning and end of sentence and discard center word
                if context_word < 0 or context_word >= len(sentence_ids) or center_word == context_word:
                    continue

                context_word_id = sentence_ids[context_word]

                # negative samples: draw from unigram distribution to the 3/4th power
                neg_context_ids = torch.LongTensor(np.random.randint(0, V, 2 * window_size))
                triplets.append((sentence_ids[center_word], context_word_id, neg_context_ids))

    # shuffle
    triplets = random.sample(triplets, len(triplets))

    n_batches = len(triplets) // batch_size
    cutoff = len(triplets) - n_batches * batch_size

    if cutoff > 0:
        triplets = triplets[:-cutoff]

    batches = [triplets[x: x + batch_size] for x in range(0, len(triplets), batch_size)]

    batches_new = []

    for batch in batches:
        center_id_batch = [triplet[0] for triplet in batch]
        pos_context_id_batch = [triplet[1] for triplet in batch]
        neg_context_ids_batch = [triplet[2] for triplet in batch]

        batches_new.append((center_id_batch, pos_context_id_batch, neg_context_ids_batch))

    return batches_new


def create_parallel_batches(tokenized_corpus_l1, tokenized_corpus_l2,
                            word2idx_l1, word2idx_l2,
                            batch_size):

    V_l1, V_l2 = len(word2idx_l1), len(word2idx_l2)

    batches_l1 = create_monolingual_batches(tokenized_corpus_l1, word2idx_l1, batch_size)
    batches_l2 = create_monolingual_batches(tokenized_corpus_l2, word2idx_l2, batch_size)

    return batches_l1, batches_l2


def create_monolingual_batches(tokenized_corpus, word2idx, batch_size):

    with open("stop_words_en.txt", "r") as f:
        stop_words = list(map(str.strip, f.readlines()))

    batches = []
    batch = []
    batch_max_len = 0

    pad_idx = word2idx['-PAD-']

    for i, sentence in enumerate(tokenized_corpus, start=1):

        sentence = [w if (w in word2idx.keys()
                          and w not in stop_words
                          )
                    else '-UNK-'
                    for w in sentence]

        sentence_ids = [word2idx[w] for w in sentence]

        sent_len = len(sentence_ids)
        if sent_len > batch_max_len:
            batch_max_len = sent_len

        batch.append(sentence_ids)

        if i % batch_size == 0:
            for sent in batch:
                offset = batch_max_len - len(sent)
                sent += [pad_idx] * offset

            batches.append(torch.LongTensor(batch))
            # reset for next batch
            batch = []
            batch_max_len = 0

        if i % 50000 == 0:
            print('{} sentences processed.'.format(i))

    return batches


# SKIPGRAM ###########################################################
corpus, word2idx, _ = read_corpus('data/europarl/training.en', n_sentences=500)
data = create_skipgrams(corpus, word2idx, 5, 100)

pickle.dump(data, open("skipgram-europarl-en-5w-100btc-500.p", "wb"))
pickle.dump(word2idx, open("w2i-skipgram-europarl-en-500.p", "wb"))
# pickle.dump(idx2word, open("i2w-skipgram-europarl-en.p", "wb" ))
######################################################################


# EMBEDALIGN ####################################################################################
# corpus_en, word2idx_en, _ = read_corpus('data/europarl/training.en')
# corpus_fr, word2idx_fr, _ = read_corpus('data/europarl/training.fr')

# batches = create_parallel_batches(corpus_en, corpus_fr, word2idx_en, word2idx_fr, batch_size=100)

# pickle.dump(batches, open("embedalign-europarl-100btc.p", "wb" ))
# pickle.dump(word2idx_en, open("w2i-europarl-en.p", "wb" ))
# pickle.dump(word2idx_fr, open("i2wc-europarl-fr.p", "wb" ))
#################################################################################################

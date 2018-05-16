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
    # idx2word = {idx: w for (w, idx) in word2idx.items()}

    return tokenized_corpus, word2idx, counter


def create_skipgrams(tokenized_corpus,
                     word2idx,
                     counter,
                     window_size,
                     batch_size):

    triplets = []
    V = len(word2idx)

    with open("stop_words_en.txt", "r") as f:
        stop_words = list(map(str.strip, f.readlines()))

    vocab, freqs = map(list,zip(*counter.items()))
    freqs = np.array(freqs)
    freqs = freqs ** (3/4)
    freqs = freqs / np.sum(freqs)

    for i, sentence in enumerate(tokenized_corpus, start=1):

        if i % 50000 == 0:
            print('{} sentences processed.'.format(i))

        sentence_ids = [word2idx[w]
                        if (w in word2idx.keys()
                            and w not in stop_words)
                        else word2idx['-UNK-']
                        for w in sentence
                        ]

        for center_word in range(len(sentence_ids)):

            for offset in range(-window_size, window_size + 1):
                context_word = center_word + offset

                # handle beginning and end of sentence and discard center word
                if context_word < 0 or context_word >= len(sentence_ids) or center_word == context_word:
                    continue

                context_word_id = sentence_ids[context_word]

                # negative samples: draw from unigram distribution to the 3/4th power
                neg_context = np.random.choice(vocab, (2 * window_size), replace=False, p=freqs)
                neg_context_ids = torch.tensor([
                                               word2idx[w] if
                                               w in word2idx
                                               else 0
                                               for w in neg_context
                                               ])

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


def create_BSG_data(tokenized_corpus,
                     word2idx,
                     counter,
                     window_size,
                     batch_size):

    tuples = []
    V = len(word2idx)

    with open("stop_words_en.txt", "r") as f:
        stop_words = list(map(str.strip, f.readlines()))

    for i, sentence in enumerate(tokenized_corpus, start=1):

        if i % 50000 == 0:
            print('{} sentences processed.'.format(i))

        sentence_ids = [word2idx[w]
                        if (w in word2idx.keys()
                            and w not in stop_words)
                        else word2idx['-UNK-']
                        for w in sentence
                        ]

        for center_word in range(len(sentence_ids)):

            context_ids = [sentence_ids[pos]
                            if (pos >= 0 and
                                pos < len(sentence_ids)
                                and center_word != sentence_ids[pos])
                            else 0
                            for pos in range(-window_size, window_size + 1)
                            ]
            tuples.append((sentence_ids[center_word], context_ids))

    # shuffle
    tuples = random.sample(tuples, len(tuples))

    n_batches = len(tuples) // batch_size
    cutoff = len(tuples) - n_batches * batch_size

    if cutoff > 0:
        tuples = tuples[:-cutoff]

    batches = [tuples[x: x + batch_size] for x in range(0, len(tuples), batch_size)]
    batches_new = []

    for batch in batches:
        center_id_batch = [tupl[0] for tupl in batch]
        context_id_batch = [tupl[1] for tupl in batch]
        batches_new.append((center_id_batch, context_id_batch))

    return batches_new

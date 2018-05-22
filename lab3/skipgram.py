import logging
import tempfile
import time
from collections import defaultdict, Counter
import gensim
from itertools import chain


FREQUENCY_THRESHOLD = 1
EMBED_DIMS = 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLING = 5
NUM_EPOCHS = 50
STOP_WORDS = False


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

with open('../lab2/data/europarl/training.en', 'r') as corpus_f:
    corpus = [line.strip() for line in corpus_f.readlines()]

if STOP_WORDS:
    with open('../lab2/stop_words_en.txt', 'r') as f_in:
        stop_words = [line.strip() for line in f_in.readlines()]


print('Tokenize corpus...')
start = time.time()
if STOP_WORDS:
    tokenized_corpus = [[token for token in sentence.split()
                        if token not in stop_words]
                        for sentence in corpus]
else:
    tokenized_corpus = [[token for token in sentence.split()]
                        for sentence in corpus]

print("Done in {} seconds".format(time.time() - start))


model = gensim.models.Word2Vec(tokenized_corpus,
                               sg=1, # Skip-gram
                               size=EMBED_DIMS,
                               window=WINDOW_SIZE,
                               negative=NEGATIVE_SAMPLING,
                               min_count=FREQUENCY_THRESHOLD,
                               max_vocab_size=None,
                               workers=4,
                               iter=1)

model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=NUM_EPOCHS)

model.save('skipgram-{}-w{}-fr{}-ns{}-ep{}.embs'.format(EMBED_DIMS,
                                                        WINDOW_SIZE,
                                                        FREQUENCY_THRESHOLD,
                                                        NEGATIVE_SAMPLING,
                                                        NUM_EPOCHS))

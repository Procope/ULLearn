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
NUM_EPOCHS = [10, 20, 30]
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

for num_epochs in NUM_EPOCHS:
    model = gensim.models.FastText(tokenized_corpus,
                                   min_count=FREQUENCY_THRESHOLD,
                                   sg=1,  # Skip-gram
                                   size=100,
                                   window=5,
                                   max_vocab_size=None,
                                   word_ngrams=1,  # this makes it FastText (0 -> word2vec)
                                   workers=4,
                                   negative=5,
                                   iter=num_epochs)

    model.save('models/fasttext/fasttext-{}-w{}-fr{}-ns{}-ep{}.embs'.format(EMBED_DIMS,
                                                            WINDOW_SIZE,
                                                            FREQUENCY_THRESHOLD,
                                                            NEGATIVE_SAMPLING,
                                                            num_epochs))

import numpy as np
import torch
import argparse
import pickle
import random
from torch.autograd import Variable

from EmbedAlign import EmbedAlign
from utils.preprocess import read_corpus, create_parallel_batches

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=100, help='Word vector dimensionality')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--n_batches', type=int, default=None, help='Number of training batches')
parser.add_argument('--context', action='store_true', default=True, help='Encode words with their context')


args = parser.parse_args()
embed_dim = args.dims
batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr
num_batches = args.n_batches
with_context = args.context
num_sentences = batch_size * num_batches

# output_path = args.save
print('Embedding dimensionality: {}'.format(embed_dim))
if with_context:
    print('Encoding words with context.')
print('Batch size: {}'.format(batch_size))
print('Number of sentence pairs: {}'.format(batch_size * num_batches))
print('{} epochs. Initial learning rate: {}'.format(num_epochs, lr))

print('--- Load data ---')

corpus_en, word2idx_en, _ = read_corpus('data/europarl/training.en', n_sentences=num_sentences)
corpus_fr, word2idx_fr, _ = read_corpus('data/europarl/training.fr', n_sentences=num_sentences)

batches = create_parallel_batches(corpus_en,
                                  corpus_fr,
                                  word2idx_en,
                                  word2idx_fr,
                                  batch_size)

pickle.dump(batches, open("embedalign-europarl-{}btc-{}.p".format(batch_size, num_sentences), "wb" ))
pickle.dump(word2idx_en, open("w2i-europarl-en-{}btc-{}.p".format(batch_size, num_sentences), "wb" ))
pickle.dump(word2idx_fr, open("w2i-europarl-fr-{}btc-{}.p".format(batch_size, num_sentences), "wb" ))

# with open('models/w2i-europarl-en-100btc-5000.p', 'rb') as f_in:
#     word2idx_en = pickle.load(f_in)

# with open('models/w2i-europarl-fr-100btc-5000.p', 'rb') as f_in:
#     word2idx_fr = pickle.load(f_in)

# with open('models/embedalign-europarl-100btc-5000.p', 'rb') as f_in:
#     batches_en, batches_fr = pickle.load(f_in)

vocab_size_en = len(word2idx_en)
vocab_size_fr = len(word2idx_fr)

model = EmbedAlign(vocab_size_en, vocab_size_fr, embed_dim, with_context)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

batches = list(zip(batches[0], batches[1]))
random.shuffle(batches)
if num_batches:
    batches = batches[:num_batches]


print('--- Train ---')

# Train
for epoch in range(1, num_epochs+1):

    overall_loss, overall_acc_l1, overall_acc_l2 = 0, 0, 0
    model.train()

    for batch_en, batch_fr in batches:

        batch_en = Variable(batch_en, requires_grad=False)
        batch_fr = Variable(batch_fr, requires_grad=False)

        optimizer.zero_grad()

        loss, acc_l1, acc_l2 = model(batch_en, batch_fr)

        overall_loss += loss.item()
        overall_acc_l1 += acc_l1.item()
        overall_acc_l2 += acc_l2.item()

        loss.backward()
        optimizer.step()

    print('Loss at epoch {}: {}'.format(epoch, overall_loss / epoch))

torch.save(model.state_dict(), 'EmbedAlignModel-{}btc-{}lr-{}ep-{}.p'.format(batch_size,
                                                                             lr[2:],
                                                                             num_epochs,
                                                                             num_sentences))

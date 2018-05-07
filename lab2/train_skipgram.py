import numpy as np
import torch
import argparse
from torch.autograd import Variable

from Skipgram import Skipgram
from preprocess import read_corpus, create_skipgrams


parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=100, help='Word vector dimensionality')
parser.add_argument('--window', type=int, default=5, help='One-sided window size')
parser.add_argument('--batch', type=int, default=100, help='Number of batches')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--n_batches', type=int, default=50, help='Number of batches.')

args = parser.parse_args()
embed_dim = args.dims
window_size = args.window
batch_size = args.batch
num_epochs = args.epochs
lr = args.lr
num_batches = args.n_batches

print('Embedding dimensionality: {}'.format(embed_dim))
print('Window size: {}'.format(window_size))
print('Batch size: {}'.format(batch_size))
print('Number of sentence pairs: {}'.format(batch_size * num_batches))
print('Number of epochs: {}'.format(num_epochs))
print('Initial learning rate: {}'.format(lr))


# corpus, word2idx, idx2word = read_corpus('data/europarl/training.en', n_sentences = batch_size * num_batches)
# data = create_skipgrams(corpus, word2idx, window_size, batch_size)

with open('w2i-skipgram-europarl-en-2000.p', 'rb') as f_in:
    word2idx = pickle.load(f_in)

with open('skipgram-europarl-en-5w-100btc-2000.p', 'rb') as f_in:
    data = pickle.load(f_in)

V = len(word2idx)


model = Skipgram(V, embed_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train
for epoch in range(1, num_epochs+1):
    overall_loss = 0
    model.train()
    optimizer.zero_grad()

    for batch in data:

        center_id       = torch.LongTensor(batch[0])
        pos_context_id  = torch.LongTensor(batch[1])
        neg_context_ids = torch.stack(batch[2])

        loss = model(center_id, pos_context_id, neg_context_ids)

        overall_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    # if epoch % 10 == 0:
    print('Loss at epoch {}: {}'.format(epoch, overall_loss / epoch))


# Write embeddings to file
embeddings = model.input_embeds.weight

with open('skipgram-europarl-en-{}w-{}btc-{}.txt'.format(window_size, batch_size, num_batches*batch_size), 'w') as f_out:
    for idx in range(embeddings.size()[0]):
        word = idx2word[idx]

        embed = embeddings[idx, :]
        embed = str(list(embed.data.numpy()))
        embed = embed[1:-1]

        print('{} {}'.format(word, embed), file=f_out)

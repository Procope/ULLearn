import numpy as np
import torch
import argparse

from torch.autograd import Variable

from Skipgram import Skipgram
from preprocess import read_corpus, create_skipgrams


parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=100, help='Word vector dimensionality')
parser.add_argument('--window', type=int, default=2, help='One-sided window size')
parser.add_argument('--batch', type=int, default=100, help='Number of batches')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--test', type=int, default=100, help='Number of sentences to consider for testing')

args = parser.parse_args()
embed_dim = args.dims
window_size = args.window
batch_size = args.batch
num_epochs = args.epochs
lr = args.lr
n_sent = args.test


corpus, word2idx, idx2word, V = read_corpus('data/europarl/training.en', n_sent)
data = create_skipgrams(corpus, word2idx, window_size, batch_size)

model = Skipgram(V, embed_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)



for epoch in range(num_epochs):
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

    if epoch % 10 == 0:
        print(f'Loss at epoch {epoch}: {overall_loss/len(data)}')

embeddings = model.input_embeds.weight

print(embeddings)

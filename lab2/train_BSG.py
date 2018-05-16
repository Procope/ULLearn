import numpy as np
import torch
import argparse
from torch.autograd import Variable
import pickle
from BayesianSG import BayesianSG
from preprocess import read_corpus, create_BSG_data


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
print('Number of sentences: {}'.format(batch_size * num_batches))
print('Number of epochs: {}'.format(num_epochs))
print('Initial learning rate: {}'.format(lr))


print("Load data.")
corpus, word2idx, counter = read_corpus('data/europarl/training.en', n_sentences=batch_size * num_batches)
data = create_BSG_data(corpus, word2idx, counter, window_size, batch_size)
V = len(word2idx)


print('Train.')
model = BayesianSG(V, embed_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
for epoch in range(1, num_epochs + 1):
    overall_loss = 0
    model.train()

    for batch in data:

        center_id = torch.LongTensor(batch[0])
        context_ids = torch.LongTensor(batch[1])

        optimizer.zero_grad()
        loss = model(center_id, context_ids)

        overall_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('Loss at epoch {}: {}'.format(epoch, overall_loss))
    # print(model.input_embeds.weight[:3])

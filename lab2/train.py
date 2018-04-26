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
parser.add_argument('--save', type=str, default='skipgram-embeds.txt', help='Path of the output text file containing embeddings')
parser.add_argument('--threshold', type=int, default=5, help='Discard words occurring less than threshold times')

args = parser.parse_args()
embed_dim = args.dims
window_size = args.window
batch_size = args.batch
num_epochs = args.epochs
lr = args.lr
n_sent = args.test
output_path = args.save


corpus, word2idx, idx2word = read_corpus('data/europarl/training.en', args.threshold, n_sent)
data = create_skipgrams(corpus, word2idx, window_size, batch_size)
V = len(word2idx)

model = Skipgram(V, embed_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train
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


# Write embeddings to file
embeddings = model.input_embeds.weight

with open(output_path, 'w') as f_out:
    for idx in range(embeddings.size()[0]):
        word = idx2word[idx]

        embed = embeddings[idx, :]
        embed = str(list(embed.data.numpy()))
        embed = embed[1:-1]

        print('{} {}'.format(word, embed), file=f_out)

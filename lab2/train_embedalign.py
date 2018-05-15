import numpy as np
import torch
import argparse
import pickle
from random import shuffle
from torch.autograd import Variable

from EmbedAlign import EmbedAlign
from preprocess import read_corpus, create_parallel_batches

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=100, help='Word vector dimensionality')
parser.add_argument('--batch', type=int, default=100, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--test', type=int, default=None, help='Number of sentences to consider for testing')
parser.add_argument('--n_batches', type=int, default=None, help='Number of training batches')
parser.add_argument('--context', action='store_true', default=False, help='Encode words with their context')
# parser.add_argument('--save', type=str, default='skipgram-embeds.txt', help='Path of the output text file containing embeddings')


args = parser.parse_args()
embed_dim = args.dims
batch_size = args.batch
num_epochs = args.epochs
lr = args.lr
num_batches = args.n_batches
with_context = args.context

# output_path = args.save
print('Embedding dimensionality: {}'.format(embed_dim))
if with_context:
    print('Encoding words with context.')
print('Batch size: {}'.format(batch_size))
print('Number of sentence pairs: {}'.format(batch_size * num_batches))
print('{} epochs. Initial learning rate: {}'.format(num_epochs, lr))

print('--- Load data ---')

# corpus_en, word2idx_en, _ = read_corpus('data/europarl/training.en', n_sentences=args.test)
# corpus_fr, word2idx_fr, _ = read_corpus('data/europarl/training.fr', n_sentences=args.test)

# batches_en, batches_fr = create_parallel_batches(corpus_en, corpus_fr, word2idx_en, word2idx_fr, batch_size=batch_size)

with open('models/w2i-europarl-en.p', 'rb') as f_in:
    word2idx_en = pickle.load(f_in)

with open('models/w2i-europarl-fr.p', 'rb') as f_in:
    word2idx_fr = pickle.load(f_in)

with open('models/embedalign-europarl-100btc.p', 'rb') as f_in:
    batches_en, batches_fr = pickle.load(f_in)

vocab_size_en = len(word2idx_en)
vocab_size_fr = len(word2idx_fr)

model = EmbedAlign(vocab_size_en,
                  vocab_size_fr,
                  embed_dim,
                  with_context)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


batches = list(zip(batches_en, batches_fr))
shuffle(batches)
len(batches_en)

if num_batches:
    batches = batches[:num_batches]


print('--- Train ---')

# Train
for epoch in range(1, num_epochs+1):

    overall_loss = 0
    model.train()

    for batch_en, batch_fr in batches:

        batch_en = Variable(batch_en, requires_grad=False)
        batch_fr = Variable(batch_fr, requires_grad=False)

        optimizer.zero_grad()

        loss = model(batch_en, batch_fr)

        overall_loss += loss.item()
        loss.backward()
        optimizer.step()

    # if epoch % 5 == 0:
    print('Loss at epoch {}: {}'.format(epoch, overall_loss / epoch))


torch.save(model.state_dict(), 'EmbedAlignModel-{}.p'.format(num_batches))


# # Write embeddings to file
# embeddings = model.input_embeds.weight

# with open(output_path, 'w') as f_out:
#     for idx in range(embeddings.size()[0]):
#         word = idx2word[idx]

#         embed = embeddings[idx, :]
#         embed = str(list(embed.data.numpy()))
#         embed = embed[1:-1]

#         print('{} {}'.format(word, embed), file=f_out)

import numpy as np
import torch
import argparse
import pickle
from random import shuffle
from torch.autograd import Variable

from EmbedAlign import EmbedAlign
from utils.preprocess import read_corpus, create_parallel_batches

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=100, help='Word vector dimensionality')
parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
parser.add_argument('--n_batches', type=int, default=11, help='Number of training batches')
parser.add_argument('--context', action='store_true', default=True, help='Encode words with their context')
parser.add_argument('--path', type=str, help='Path of EmbedAlign model')

args = parser.parse_args()
embed_dim = args.dims
batch_size = args.batch_size
num_batches = args.n_batches
with_context = args.context
model_path = args.path
if model_path is None:
    raise ValueError('Provide path of EmbedAlign model: --path <model.path>')

VOCAB_SIZE = 10000 + 2

# output_path = args.save
print('Embedding dimensionality: {}'.format(embed_dim))
if with_context:
    print('Encoding words with context.')
print('Batch size: {}'.format(batch_size))
print('Number of sentence pairs: {}'.format(batch_size * num_batches))

print('--- Load data ---')


corpus_en, word2idx_en, _ = read_corpus('data/wa/test.en')
corpus_fr, word2idx_fr, _ = read_corpus('data/wa/test.fr')

batches_en, batches_fr = create_parallel_batches(corpus_en,
                                                 corpus_fr,
                                                 word2idx_en,
                                                 word2idx_fr,
                                                 batch_size)

state_dict = torch.load(model_path)
V_l1 = state_dict['affine_l1.bias'].size()[0]
V_l2 = state_dict['affine_l2.bias'].size()[0]

model = EmbedAlign(V_l1, V_l2, embed_dim, with_context)
model.load_state_dict(state_dict)

batches = list(zip(batches_en, batches_fr))

if num_batches:
    batches = batches[:num_batches]

overall_acc_l1, overall_acc_l2 = 0, 0

model.eval()

for batch_en, batch_fr in batches:

    batch_en = Variable(batch_en, requires_grad=False)
    batch_fr = Variable(batch_fr, requires_grad=False)

    _, acc_l1, acc_l2 = model(batch_en, batch_fr)

    overall_acc_l1 += acc_l1.item()
    overall_acc_l2 += acc_l2.item()

print(overall_acc_l1 / len(batches))
print(overall_acc_l2 / len(batches))

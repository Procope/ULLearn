import numpy as np
import argparse
import pickle

from substitution_scores import skipgram_scores, kl_scores
from retrieve_vectors import retrieve_skipgram_vectors, retrieve_BSG_vectors, retrieve_embedalign_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='skipgram or embedalign')
parser.add_argument('--w2i', type=str, help='The path of the word2index dictionary')
parser.add_argument('--model_path', type=str, help='The model path')
parser.add_argument('--threshold', type=int, default=10,
                    help='If we have less than threshold candidates, skip this target word')
parser.add_argument('--skipgram_mode', type=str, default='cosine',
                    help='[cosine, add, baladd, mult, balmult]')

args = parser.parse_args()
model = args.model.lower()
model_path = args.model_path
w2i_path = args.w2i

if model not in ['skipgram', 'bsg', 'embedalign']:
    raise ValueError('No such model as {}'.format(model))


# Fixed paths
LST_GOLD_CANDIDATES = 'data/lst/lst.gold.candidates'
LST_TEST_PREPROCESSED = 'data/lst/lst_test.preprocessed'


# Load candidates for each target word
with open(LST_GOLD_CANDIDATES, 'r') as f:
    lines = map(str.strip, f.readlines())

candidates = {}
for line in lines:
    target, rest = line.split('.', maxsplit=1)
    pos_tag, rest = rest.split('::', maxsplit=1)
    alternatives = rest.split(';')
    candidates[target] = alternatives


# Retrieve embeddings
if model == 'skipgram':
    with open(w2i_path, 'rb') as f:
        word2idx = pickle.load(f)

    target2embeds, skip_count = retrieve_skipgram_vectors(model_path, candidates, args.threshold)
else:

    if model == 'embedalign':
        retrieve_fn = retrieve_embedalign_vectors
    elif model == 'bsg':
        retrieve_fn = retrieve_BSG_vectors

    with open(w2i_path, 'rb') as f:
        word2idx = pickle.load(f)

    # Just to increase readability
    target2locs, target2scales, target2str, target2sentIDs, targe2alt, skip_count = retrieve_fn(model_path,
                                                                                                LST_TEST_PREPROCESSED,
                                                                                                candidates,
                                                                                                word2idx,
                                                                                                args.threshold)

print('{} examples were skipped.'.format(skip_count))


# Compute scores and print output
with open('lst.out', 'w') as f_out:
    skipped_entries = 0

    if model == 'embedalign' or model == 'bsg':
        for target in target2locs.keys():
            for locs_matrix, scales_matrix, target_str, sentID, alt in zip(target2locs[target],
                                                                           target2scales[target],
                                                                           target2str[target],
                                                                           target2sentIDs[target],
                                                                           targe2alt[target]):

                # Print preamble
                print('RANKED\t{} {}'.format(target_str, sentID), file=f_out, end='')


                # Sort alternatives by their scores
                scores = kl_scores(locs_matrix, scales_matrix)
                words_and_scores = list(zip(alt, scores))
                words_and_scores.sort(key=lambda t: t[1], reverse=False)

                # Write ranked alternatives and their scores to file
                for w, s in words_and_scores:
                    print('\t{} {}'.format(w, s), file=f_out, end='')

                print(file=f_out)  # conclude file with new line


    elif model == 'skipgram':
        with open(LST_TEST_PREPROCESSED, 'r') as f_in:
            lines = list(map(str.strip, f_in.readlines()))

        with open(model_path, 'r') as f_in:
            embeds_file = list(map(str.strip, f_in.readlines()))

            # retrieve embedding dimensions from file
            first_line = embeds_file[0]
            _, vector_str = first_line.split(" ", maxsplit=1)
            embed_dims = len(vector_str.split(", "))

            embeddings = np.empty((len(embeds_file), embed_dims))
            for line in embeds_file:
                term, vector_str = first_line.split(" ", maxsplit=1)
                vector = np.array(vector_str.split(", "))
                embeddings[word2idx[term]] = vector

        for line in lines:
            target, sent_id, target_position, sentence = line.split('\t')

            target_word = target.split('.')[0]
            alternatives = candidates[target_word]
            alternative_ids = [word2idx[w] for w in alternatives if w in word2idx.keys()]

            context_ids = [word2idx[w] for w in sentence.split() if w in word2idx.keys()]
            context_embeds = np.array([embeddings[i] for i in context_ids])

            try:
                embed_matrix = target2embeds[target_word]
            except KeyError:
                continue

            # Print preamble
            print('RANKED\t{} {}'.format(target, sent_id), file=f_out, end='')

            # Sort alternatives by their scores
            scores = skipgram_scores(embed_matrix, context_embeds, args.skipgram_mode)
            words_and_scores = list(zip(alternatives, scores))
            words_and_scores.sort(key=lambda t: t[1], reverse=True)

            # Write ranked alternatives and their scores to file
            for w, s in words_and_scores:
                print('\t{} {}'.format(w, s), file=f_out, end='')

            print(file=f_out)  # conclude file with new lines

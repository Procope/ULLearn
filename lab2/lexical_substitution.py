import numpy as np
import argparse
import pickle

from substitution_scores import skipgram_scores, kl_scores
from retrieve_vectors import retrieve_skipgram_vectors, retrieve_BSG_vectors, retrieve_embedalign_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='skipgram or embedalign')
parser.add_argument('--threshold', type=int, default=10, help='If we have less than threshold candidates, skip this target word')
parser.add_argument('--skipgram_mode', type=str, default='cosine', help='[cosine, add, baladd, mult, balmult]')
args = parser.parse_args()

model = args.model.lower()
skipgram_mode = args.skipgram_mode


with open('data/lst/lst.gold.candidates', 'r') as f:
    lines = map(str.strip, f.readlines())

candidates = {}
for line in lines:
    target, rest = line.split('.', maxsplit=1)
    pos_tag, rest = rest.split('::', maxsplit=1)
    alternatives = rest.split(';')
    candidates[target] = alternatives



if model == 'embedalign':
    with open('w2i-europarl-en-100btc-5000.p', 'rb') as f:
        word2idx = pickle.load(f)

    target2locs, target2scales, target2strings, target2sentIDs, target2alternatives, skip_count = retrieve_embedalign_vectors(
                                                                                  'EmbedAlignModel-50.p',
                                                                                  'data/lst/lst_test.preprocessed',
                                                                                  candidates,
                                                                                  word2idx,
                                                                                  args.threshold)

elif model == 'skipgram':
    with open('models/w2i-skipgram-europarl-en-5w-100btc-5000.p', 'rb') as f:
        word2idx = pickle.load(f)

    target2embeds, skip_count = retrieve_skipgram_vectors('models/skipgram-europarl-en-5w-100btc-5000.txt',
                                                          candidates,
                                                          args.threshold)

elif model == 'bsg':
    with open('w2i-bsg-europarl-en-100btc-1000.p', 'rb') as f:
        word2idx = pickle.load(f)

    target2locs, target2scales, target2strings, target2sentIDs, target2alternatives, skip_count = retrieve_BSG_vectors('BSGModel-100btc-1000.p', 'data/lst/lst_test.preprocessed',
                                                                                  candidates,
                                                                                  word2idx,
                                                                                  args.threshold)

else:
    raise ValueError()

print('{} examples were skipped.'.format(skip_count))


with open('lst.out', 'w') as f_out:
    skipped_entries = 0

    if model == 'embedalign' or model == 'bsg':
        for target in target2locs.keys():
            for locs_matrix, scales_matrix, target_string, sentID, alternatives in zip(target2locs[target],
                                                                                      target2scales[target],
                                                                                      target2strings[target],
                                                                                      target2sentIDs[target],
                                                                                      target2alternatives[target]):

                scores = kl_scores(locs_matrix, scales_matrix)

                # Print preamble
                print('RANKED\t{} {}'.format(target_string, sentID), file=f_out, end='')



                # Sort alternatives by their scores
                words_and_scores = list(zip(alternatives, scores))
                words_and_scores.sort(key=lambda t: t[1], reverse=False)

                # Write ranked alternatives and their scores to file
                for w, s in words_and_scores:
                    print('\t{} {}'.format(w, s), file=f_out, end='')
                print(file=f_out)


    elif model == 'skipgram':
        with open('data/lst/lst_test.preprocessed', 'r') as f_in:
            lines = list(map(str.strip, f_in.readlines()))

        with open('/Users/mario/Downloads/skipgram-europarl-en-5w-100btc-5000.txt', 'r') as f_in:
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

            # Score alternatives
            scores = skipgram_scores(embed_matrix, context_embeds, skipgram_mode)

            # Print preamble
            print('RANKED\t{} {}'.format(target, sent_id), file=f_out, end='')

            # Sort alternatives by their scores
            words_and_scores = list(zip(alternatives, scores))
            words_and_scores.sort(key=lambda t: t[1], reverse=True)

            # Write ranked alternatives and their scores to file
            for w, s in words_and_scores:
                print('\t{} {}'.format(w, s), file=f_out, end='')
            print(file=f_out)

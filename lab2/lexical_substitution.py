import numpy as np
import pickle
import torch
from torch.nn import Softplus
from sklearn.preprocessing import normalize
import argparse

def skipgram_scores(embeds, context, mode='cosine'):
    '''
    Scoring functions from:
    Melamud, Oren, Omer Levy, and Ido Dagan.
    "A simple word embedding model for lexical substitution."
    Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing. 2015.
    '''
    t, alternatives = embeds[0], embeds[1:]

    if mode == 'cosine':
        scores = alternatives @ t
        scores = scores.tolist()

    elif mode == 'add':
            scores = [
                (a @ t + np.sum(c @ a)) / (c_len + 1)
            for a
            in alternatives
    ]

    elif mode == 'baladd':
        scores = [
                (c_len * (a @ t) + np.sum(c @ a)) / (2 * c_len)
                for a
                in alternatives
        ]

    elif mode == 'mult':
        scores = [
                (((u @ v + 1) / 2) * np.prod((c @ a + 1) / 2)) ** (1 / (c_len + 1))
                for a
                in alternatives
        ]

    elif mode == 'balmult':
        scores = [
                ((((u @ v + 1) / 2) ** c_len) * np.prod((c @ a + 1) / 2)) ** (2 * c_len)
                for a
                in alternatives
        ]

    elif mode == 'test':
        return range(len(alternatives))
    else:
        raise ValueError('Mode: [add, baladd, mult, balmult, test]')

    return scores


def kl_div(s0, s1, m_0, m_1):
    # u,l are cov matrices
    # m_1 and m_2 are mean vectors
    kl = 0.5 * (np.trace(np.matmul(np.linalg.inv(s1), s0)) \
            + np.matmul(np.matmul(np.transpose(m_1 - m_0), np.linalg.inv(s1)), (m_1 - m_0)) - s0.shape[0] \
            + np.log(np.linalg.det(s1) / np.linalg.det(s0)))
    return kl


def embedalign_scores(embeds_means, embeds_vars):

    t_mean = embeds_means[0]
    t_cov_matrix = np.diag(embeds_vars[0])

    alternatives_means = list(embeds_means[1:])
    alternatives_cov_matrices = [np.diag(v) for v in embeds_vars[1:]]

    # scoring
    scores = [
            kl_div(alternatives_cov_matrices[a],
                   t_cov_matrix,
                   alternatives_means[a],
                   t_mean)
            for a
            in range(len(alternatives_means))
    ]

    return scores


def retrieve_skipgram_vectors(model_path, candidates_dict, threshold):
        with open(model_path, 'r') as f_in:
            embed_file = map(str.strip, f_in.readlines())

        word2embed = {}
        for line in embed_file:
            line = line.split()
            word = line[0]
            embed = np.array([float(x[:-1]) for x in line[1:]])
            word2embed[word] = embed

        for e in word2embed.values():
            e /= np.linalg.norm(e)

        target2embeds = {}

        skip_count = 0
        for target, alternatives in candidates_dict.items():
            embeds = []
            alternative_count = 0

            if target not in word2embed:
                skip_count += 1
                continue
            else:
                embeds.append(word2embed[target])

            for w in alternatives:
                try:
                    embeds.append(word2embed[w])
                    alternative_count += 1
                except KeyError:
                    continue

            if alternative_count > threshold:
                target2embeds[target] = np.array(embeds)
            else:
                skip_count += 1

        return target2embeds, skip_count


def retrieve_embedalign_vectors(model_path, candidates_dict, word2index, threshold):
    model = torch.load(model_path)
    target2means = {}
    target2vars = {}

    # Retrieve parameters
    embeddings = model['embeddings.weight']
    mean_W = model['inference_net.affine1.weight']
    var_W = model['inference_net.affine2.weight']
    mean_b = model['inference_net.affine1.bias']
    var_b = model['inference_net.affine2.bias']
    softplus = Softplus()

    skip_count = 0
    for target, alternatives in candidates_dict.items():
        try:
            target_id = word2index[target]
        except KeyError:
            # target word not in dictionary, skip it
            skip_count += 1
            continue

        alternative_count = 0
        alternative_ids = []

        for a in alternatives:
            try:
                alternative_ids += [word2index[a]]
                alternative_count += 1
            except KeyError:
                # alternative word not in dictionary
                pass

        if alternative_count < threshold:
            skip_count += 1
            continue

        embeds = [embeddings[w] for w in [target_id] + alternative_ids]
        embeds = torch.stack(embeds)

        mean_vecs = embeds @ torch.t(mean_W) + mean_b
        var_vecs = embeds @ torch.t(var_W) + var_b
        var_vecs = softplus(var_vecs)

        target2means[target] = mean_vecs.numpy()
        target2vars[target] = var_vecs.numpy()

    return target2means, target2vars, skip_count



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='skipgram or embedalign')
    parser.add_argument('--threshold', type=int, default=10, help='If we have less than threshold candidates, skip this target word')
    args = parser.parse_args()

    model = args.model.lower()

    with open('w2i-europarl-en.p', 'rb') as f:
        word2idx = pickle.load(f)

    with open('data/lst/lst.gold.candidates', 'r') as f:
        lines = map(str.strip, f.readlines())

    candidates = {}
    for line in lines:
        target, rest = line.split('.', maxsplit=1)
        pos_tag, rest = rest.split('::', maxsplit=1)
        alternatives = rest.split(';')
        candidates[target] = alternatives


    with open('data/lst/lst_test.preprocessed', 'r') as f:
        lines = map(str.strip, f.readlines())


    if model == 'embedalign':
        target2means, target2vars, skip_count = retrieve_embedalign_vectors('EmbedAlignModel-10.p',
                                                                            candidates,
                                                                            word2idx,
                                                                            args.threshold)
    elif model == 'skipgram':
        target2embeds, skip_count = retrieve_skipgram_vectors('skipgram-embeds-100.txt',
                                                              candidates,
                                                              args.threshold)
    elif model == 'bsg':
        raise NotImplementedError()
    else:
        raise ValueError()

    print('{} examples were skipped.'.format(skip_count))



    skipped_entries = 0
    with open('lst.out', 'w') as f_out:
        for line in lines:
            target, sent_id, target_position, sentence = line.split('\t')

            target_word = target.split('.')[0]
            sentence = sentence.split()


            # sentence_ids = [word2idx[w] for w in sentence if w in word2idx.keys()]
            alternatives = candidates[target_word]
            alternative_ids = [word2idx[w] for w in alternatives if w in word2idx.keys()]

            if model == 'embedalign':
                try:
                    mean_matrix = target2means[target_word]
                    var_matrix = target2vars[target_word]
                except KeyError:
                    skipped_entries += 1
            elif model == 'skipgram':
                try:
                    embed_matrix = target2embeds[target_word]
                except KeyError:
                    skipped_entries += 1
                    continue

            # Score alternatives
            if model == 'embedalign':
                scores = embedalign_scores(mean_matrix, var_matrix)
            elif model == 'skipgram':
                scores = skipgram_scores(embed_matrix, sentence, 'cosine')
            #

            # Print preamble
            print('RANKED\t{} {}'.format(target, sent_id), file=f_out, end='')

            # Sort alternative by their scores
            words_and_scores = list(zip(alternatives, scores))
            words_and_scores.sort(key=lambda t: t[1], reverse=False)  # True for skipgram

            # Write ranked alternatives and their scores to file
            for w, s in words_and_scores:
                print('\t{} {}'.format(w, s), file=f_out, end='')
            print(file=f_out)

    print("{} entries have been skipped.".format(skipped_entries))

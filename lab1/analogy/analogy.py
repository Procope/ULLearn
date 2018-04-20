import os
import argparse
import numpy as np
from sklearn.preprocessing import normalize

import sys
sys.path.insert(0, 'utils')
from utils import read_word_embeds


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="One of 'deps', 'bow2', or 'bow5'.")
parser.add_argument('--test', action='store_true', default=False, help="Use a few analogy queries")

args = parser.parse_args()


def analogy_score(filepath, embeddings, w2i, model):
    """
    Args:
        filepath: the path of the .txt file containing analogies
        embeddings: the embedding matrix
        w2i: a dictionary mapping words to (row) indices of the embedding matrix
    Return:
        accuracy
        Mean Reciprocal Rank
        n_analogies: the number of analogy queries in the given analogy class
    """
    embeddings = normalize(embeddings)  # l2-normalize
    i2w = {i:w for w,i in w2i.items()}

    n_analogies = 0
    accuracy_count = 0
    reciprocal_ranks = []

    class_name = filepath.split('/')[-1]  # remove folder path
    class_name = class_name[:-4] # remove .txt extension

    with open(filepath, "r") as f_in, open("analogy/analogy-results-{}/analogies-{}.txt".format(model, class_name), "w") as f_out:

        lines = f_in.readlines()
        for line in lines:
            if line[0] == ":":
                scores = []
                continue

            list_line = line.strip().split()

            a, b, c, d = map(str.lower, list_line[:4])

            try:
                a_idx, b_idx, c_idx, d_idx = w2i[a], w2i[b], w2i[c], w2i[d]
            except KeyError:
                continue

            n_analogies += 1
            potential_embed = embeddings[b_idx] - embeddings[a_idx] + embeddings[c_idx]

            similarities = embeddings @ potential_embed
            sorted_word_indices = [idx for _, idx in sorted(zip(similarities, w2i.values()), reverse=True)]

            # discard the input question words during this search (Mikolov et al., 2013)
            for idx in [a_idx, b_idx, c_idx]:
                sorted_word_indices.remove(idx)

            rank_target = sorted_word_indices.index(d_idx) + 1

            d_hat = [i2w[i] for i in sorted_word_indices[:5]]
            print("{} : {} = {} : {} -> {}".format(a, b, c, d, d_hat), file=f_out)

            if rank_target == 1:
                accuracy_count += 1

            reciprocal_ranks.append(1 / rank_target)


    mean_rec_rank = 1 / n_analogies * sum(reciprocal_ranks)
    accuracy = accuracy_count / n_analogies

    return accuracy, mean_rec_rank, n_analogies


def analogy_task(folder_path, embeddings, word2index, model):
    """
    Args:
        folder_path: path of the folder containing analogy queries
        embeddings: the embedding matrix
        word2index: a dictionary mapping words to (row) indices of the embedding matrix
    Return:
        mean_acc: the accuracy of this embedding model
        mean_MRR: the Mean Reciprocal Rank of this embedding model
        accuracies: (dict) accuracies for all analogy classes
        MRRs: (dict) MRR values for all analogy classes
    """
    accuracies, MRRs, sizes = {}, {}, {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        filepath = "{}/{}".format(folder_path, filename)
        class_name = filename[:-4]  # remove .txt extension from name

        # compute accuracy and MRR for this analogy class
        accuracy, mean_rec_rank, n_analogies = analogy_score(filepath, embeddings, word2index, model)

        # store metrics for this analogy class
        accuracies[class_name] = accuracy
        MRRs[class_name]       = mean_rec_rank
        sizes[class_name]      = n_analogies

        print(class_name, accuracy, mean_rec_rank)

    # Now compute the overall metrics
    mean_acc, mean_MRR, total_size = 0, 0, 0
    for key, acc in accuracies.items():
        MRR, size = MRRs[key], sizes[key]

        mean_acc += size * acc
        mean_MRR += size * MRR

        total_size += size

    mean_acc /= total_size
    mean_MRR /= total_size

    return mean_acc, mean_MRR, accuracies, MRRs


embeddings, w2i = read_word_embeds("models/{}.words.bz2".format(args.model))

if args.test:
    mean_acc, mean_MRR, accuracies, MRRs = analogy_task('analogy/analogy-questions-test', embeddings, w2i, args.model)
else:
    mean_acc, mean_MRR, accuracies, MRRs = analogy_task('analogy/analogy-questions', embeddings, w2i, args.model)

print('Overall')
print(mean_acc, mean_MRR)


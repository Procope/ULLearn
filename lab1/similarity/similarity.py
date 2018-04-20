import numpy as np
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, 'utils')
from utils import read_word_embeds, get_MEN, get_simlex, cosine_similarity


def similarity_correlation(pairs, similarity_scores, embeddings, word2index):
    """
    Returns pearson and spearman correlation coefficient between human judgement
    and ranking defined by cosine similarity of embeddings

    pairs: list of pairs of words
    similarity_scores: the corresponding human similarity judgement scores of pairs
    embeddings: the embedding matrix
    word2index: a dictionary mapping words to (row) indices of the embedding matrix
    """

    scores = list(similarity_scores)
    not_in_embedds_count = 0
    cosine_sims = []
    for idx, pair in enumerate(pairs):
        w1, w2 = pair

        try:
            cosine_sim = cosine_similarity(embeddings[word2index[w1]], embeddings[word2index[w2]])
            cosine_sims.append(cosine_sim)
        except KeyError:
            not_in_embedds_count += 1
            scores.pop(idx)

    # correlations
    pearson_r, p_pearson = pearsonr(cosine_sims, scores)
    spearman_r, p_spearman = spearmanr(cosine_sims, scores)

    return pearson_r, spearman_r, p_pearson, p_spearman, not_in_embedds_count



pairs_men, sim_men = get_MEN()
pairs_simlex, sim_simlex = get_simlex()

for model in ['deps', 'bow2', 'bow5']:

    embeddings, word2index = read_word_embeds("models/{}.words.bz2".format(model))

    for dataset in ['SIM-LEX', 'MEN']:
        print(model, dataset)

        if dataset == 'MEN':
            pearson_r, spearman_r, p_pearson, p_spearman, words_not_in_embeds = similarity_correlation(pairs_men,
                                                                                                       sim_men,
                                                                                                       embeddings,
                                                                                                       word2index)
        else:
            pearson_r, spearman_r, p_pearson, p_spearman, words_not_in_embeds = similarity_correlation(pairs_simlex,
                                                                                                       sim_simlex,
                                                                                                       embeddings,
                                                                                                       word2index)

        print(pearson_r, spearman_r, p_pearson, p_spearman, words_not_in_embeds)

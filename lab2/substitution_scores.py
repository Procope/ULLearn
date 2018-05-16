import torch
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence


def skipgram_scores(embeds, context, mode='cosine'):
    '''
    Scoring functions from:
    Melamud, Oren, Omer Levy, and Ido Dagan.
    "A simple word embedding model for lexical substitution."
    Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing. 2015.
    '''
    t, alternatives = embeds[0], embeds[1:]
    c = context
    c_len = len(context)


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
                (((t @ a + 1) / 2) * np.prod((c @ a + 1) / 2)) ** (1 / (c_len + 1))
                for a
                in alternatives
        ]

    elif mode == 'balmult':
        scores = [
                ((((t @ a + 1) / 2) ** c_len) * np.prod((c @ a + 1) / 2)) ** (2 * c_len)
                for a
                in alternatives
        ]

    elif mode == 'test':
        return range(len(alternatives))
    else:
        raise ValueError('Mode: [add, baladd, mult, balmult, test]')

    return scores


def multivariate_normal_kl(scale0, scale1, loc0, loc1):
    cov0 = torch.diagflat(torch.tensor(scale0 ** 2))
    cov1 = torch.diagflat(torch.tensor(scale1 ** 2))

    d0 = MultivariateNormal(torch.tensor(loc0), covariance_matrix=cov0)
    d1 = MultivariateNormal(torch.tensor(loc1), covariance_matrix=cov1)

    return kl_divergence(d0, d1)


def kl_scores(embeds_locs, embeds_scales):
    t_loc = embeds_locs[0]
    t_scale = embeds_scales[0]

    alternatives_locs = list(embeds_locs[1:])
    alternatives_scales = list(embeds_scales[1:])

    scores = [
            multivariate_normal_kl(
                                   alternatives_scales[a],
                                   t_scale,
                                   alternatives_locs[a],
                                   t_loc)
            for a
            in range(len(alternatives_locs))
    ]

    return scores

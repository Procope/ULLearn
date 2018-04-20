import numpy as np
from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [20, 20]

import sys
sys.path.insert(0, 'utils')
from utils import read_word_embeds, read_noun_list

def viz_vectors(data, labels, method, model_name):
    """
    method: "PCA" or "tsne"

    """
    if method == "tsne":
        tsne = TSNE(n_components=2, random_state=0)
        data_2d = tsne.fit_transform(data)
    elif method == "pca":
        pca = PCA(n_components=2, random_state=0)
        data_2d = pca.fit_transform(data)


    for i in range(len(data_2d[:, 0])):
        plt.scatter(data_2d[i, 0],data_2d[i, 1])
        plt.annotate(labels[i],
                     xy=(data_2d[i, 0], data_2d[i, 1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    try:
        plt.savefig('{}/{}.pdf'.format(method, model_name))
    except FileNotFoundError:
        plt.savefig('clustering/{}/{}.pdf'.format(method, model_name))

    plt.close()


def k_means(data, words, k):
    """
    k: k for kmeans

    returns a dictionary with words belonging to the k differen clusters

    """

    kmeans_model = KMeans(n_clusters=k, random_state=0)
    labels = kmeans_model.fit_predict(data)

    clusters = defaultdict(list)

    for l in range(len(labels)):
        clusters[str(labels[l])]+=[words[l]]

    return clusters


def clusters_to_file(clusters, filename):
    try:
        f_out = open('clusters/{}'.format(filename), 'w')
    except FileNotFoundError:
        f_out = open('clustering/clusters/{}'.format(filename), 'w')

    for c in clusters.keys():
        print(c, file=f_out)
        for noun in clusters[c]:
            print(noun, file=f_out)

    f_out.close()



for model in ['deps', 'bow2', 'bow5']:
    try:
        embeddings, word2index = read_word_embeds("../models/{}.words.bz2".format(model))
    except FileNotFoundError:
        embeddings, word2index = read_word_embeds("models/{}.words.bz2".format(model))

    data, labels = read_noun_list(embeddings, word2index)

    print('Model: {}'.format(model))

    # viz_vectors(data, labels, "pca", model)
    # viz_vectors(data, labels, "tsne", model)

    print('K-means clustering.')
    for K in [5, 10, 20, 50, 100, 200, 400]:
        clusters = k_means(data, labels, K)
        clusters_to_file(clusters, '{}-{}'.format(model, K))

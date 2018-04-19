import bz2
import numpy as np


# creating dictionary for word embeddings
def read_word_embeds(filename):
    """
    Args:
        filename: a .bz2 file containing word embeddings
    Return:
        embeddings: the embedding (numpy) matrix
        word2idx: a dictionary mapping words to (row) indices of the embedding matrix
    """

    embed = []
    count = 0
    word2idx = dict()
    embeddings = []

    print('Reading embeddings from {} ...'.format(filename), end=' ')
    with bz2.open(filename, "rt") as bz_file:

        for line in bz_file:
            list_line = list(line.split())
            word = list_line[0]
            vector = list(map(float, list_line[1:]))

            embeddings.append(vector)
            word2idx[word] = count
            count += 1
    print('Done.')
    return np.array(embeddings), word2idx


def cosine_similarity(u, v):
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    return np.dot(u, v) / (u_norm * v_norm)


def get_simlex():
    # reading in SimLex-999 similarity ratings
    pairs_simlex = []
    sim_simlex = []

    try:
        file = open("SimLex-999/SimLex-999.txt", "r")
    except FileNotFoundError:
        file = open("similarity/SimLex-999/SimLex-999.txt", "r")

    lines = file.readlines()[1:]
    file.close()

    for line in lines:
        list_line = line.strip().split("\t")

        if len(list_line) >= 4:
            pairs_simlex.append([list_line[0], list_line[1]])
            sim_simlex.append(list_line[3])
        else:
            print('Invalid line:', line)


    sim_simlex = list(map(float, sim_simlex))

    return pairs_simlex, sim_simlex


def get_MEN():
    # reading in MEN similary ratings
    pairs_MEN = []
    sim_MEN = []

    try:
        file = open("MEN/MEN_dataset_natural_form_full", "r")
    except FileNotFoundError:
        file = open("similarity/MEN/MEN_dataset_natural_form_full", "r")

    lines = file.readlines()
    file.close()

    for line in lines:
        list_line = line.strip().split()

        if len(list_line) == 3:
            pairs_MEN.append([list_line[0],list_line[1]])
            sim_MEN.append(list_line[2])
        else:
            print('Invalid line:', line)

    sim_MEN = list(map(float, sim_MEN))

    return pairs_MEN, sim_MEN


def read_noun_list(embeddings, w2i, n=0):
    data = []
    labels = []

    try:
        file = open("2000_nouns_sorted.txt", "r")
    except FileNotFoundError:
        file = open("clustering/2000_nouns_sorted.txt", "r")

    if n <= 0:
        lines = file.readlines()
    elif n > 0:
        lines = file.readlines()[:n]
    file.close()

    for line in lines:
        word = line.strip()

        if word in w2i.keys():
            data.append(embeddings[w2i[word]])
            labels.append(word)

    return data, labels

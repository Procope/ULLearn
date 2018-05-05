import numpy as np
import pickle


def skipgram_scores(embeds, target, alternatives, context, mode='cosine'):

    t = embeds[target]
    alternatives = np.array([embeds[a] for a in alternatives])
    c_len = len(context)
    c = np.array([embeds[w] for w in context])

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
                  (c_len * (a @ t) + np.sum(c @ a)) / (2 *c_len)
                  for a
                  in alternatives
                  ]

    elif mode == 'mult':
        scores = [
                  ( ((u @ v + 1) / 2) * np.prod((c @ a + 1) / 2) ) ** (1 / (c_len + 1))
                  for a
                  in alternatives
                  ]

    elif mode == 'balmult':
        scores = [
                  ( (((u @ v + 1) / 2) ** c_len) * np.prod((c @ a + 1) / 2) ) ** (2 * c_len)
                  for a
                  in alternatives
                  ]

    elif mode == 'test':
        return range(len(alternatives))
    else:
        raise ValueError('Mode: [add, baladd, mult, balmult, test]')




if __name__ == "__main__":

    with open('w2i-europarl-en.p', 'rb') as f:
        word2idx = pickle.load(f)

    with open('data/lst/lst.gold.candidates', 'r') as f:
        lines = map(str.strip, f.readlines())

    candidates = {}

    for line in lines:
        target,      rest = line.split('.' , maxsplit=1)
        pos_tag,     rest = rest.split('::', maxsplit=1)
        alternatives      = rest.split(';' )

        candidates[target] = alternatives


    with open('data/lst/lst_test.preprocessed', 'r') as f:
        lines = map(str.strip, f.readlines())

    skipped_entries = 0
    with open('lst.out', 'w') as f_out:
        for line in lines:
            target, sent_id, target_position, sentence = line.split('\t')

            target_word = target.split('.')[0]
            sentence = sentence.split()

            try:
                target_id = word2idx[target_word]
            except KeyError:
                skipped_entries += 1
                continue
            # sentence_ids = [word2idx[w] for w in sentence if w in word2idx.keys()]

            alternatives = candidates[target_word]
            alternative_ids = [word2idx[w] for w in alternatives if w in word2idx.keys()]

            scores = skipgram_scores(None, target_id, alternative_ids, sentence, 'test')

            print('RANKED\t{} {}'.format(target, sent_id), file=f_out, end='')

            words_and_scores = list(zip(alternatives, scores))
            words_and_scores.sort(key=lambda t: t[1], reverse=True)

            for w, s in words_and_scores:
                print('\t{} {}'.format(w, s), file=f_out, end='')
            print(file=f_out)


    print("{} entries have been skipped.".format(skipped_entries))

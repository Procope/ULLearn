import torch
import numpy as np
from torch.nn import Softplus, ReLU
from collections import defaultdict


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


def retrieve_embedalign_vectors(model_path, task_path, candidates_dict, word2index, threshold):
    model = torch.load(model_path)

    # Retrieve parameters
    embeddings = model['embeddings.weight']
    mean_W = model['inference_net.affine1.weight']
    var_W = model['inference_net.affine2.weight']
    mean_b = model['inference_net.affine1.bias']
    var_b = model['inference_net.affine2.bias']
    softplus = Softplus()

    with open(task_path, 'r') as f_in:
        lines = f_in.readlines()

    target2means        = defaultdict(list)
    target2vars         = defaultdict(list)
    target2strings      = defaultdict(list)
    target2sentIDs      = defaultdict(list)
    target2alternatives = defaultdict(list)

    skip_count = 0
    for line in lines:
        target, sentID, target_position, context = line.split('\t')
        target_word = target.split('.')[0]

        context_ids = [word2index[w] for w in context.split() if w in word2index]  # might be empty
        try:
            target_id = word2index[target_word]
        except KeyError:
            # target word not in dictionary, skip it
            skip_count += 1
            continue

        alternatives = candidates_dict[target_word]
        alternative_count = 0
        good_alternatives = []
        alternative_ids = []

        for a in alternatives:
            try:
                alternative_ids += [word2index[a]]
                good_alternatives += [a]
                alternative_count += 1
            except KeyError:
                # alternative word not in dictionary
                pass

        if alternative_count < threshold:
            skip_count += 1
            continue

        context_embeds = torch.stack([embeddings[i] for i in context_ids])
        context_avg = torch.mean(context_embeds, dim=0)
        context_avg = context_avg.repeat(alternative_count+1, 1)
        context_avg = torch.tensor(context_avg)

        embeds = [embeddings[w] for w in [target_id] + alternative_ids]
        embeds = torch.stack(embeds)

        h = torch.cat((embeds, context_avg), dim=1)

        mean_vecs = h @ torch.t(mean_W) + mean_b
        var_vecs = h @ torch.t(var_W) + var_b
        var_vecs = softplus(var_vecs)

        target2means[target].append(mean_vecs.numpy())
        target2vars[target].append(var_vecs.numpy())
        target2strings[target].append(target)
        target2sentIDs[target].append(sentID)
        target2alternatives[target].append(good_alternatives)

    return target2means, target2vars, target2strings, target2sentIDs, target2alternatives, skip_count


def retrieve_BSG_vectors(model_path, task_path, candidates_dict, word2index, threshold):
    model = torch.load(model_path)

    # Retrieve parameters
    embeddings = model['embeddings.weight']
    encoder_W = model['encoder.weight']
    encoder_b = model['encoder.bias']
    affine_loc_W = model['affine_loc.weight']
    affine_scale_W = model['affine_scale.weight']
    affine_loc_b = model['affine_loc.bias']
    affine_scale_b = model['affine_scale.bias']
    softplus = Softplus()
    relu = ReLU()

    with open(task_path, 'r') as f_in:
        lines = f_in.readlines()

    target2locs         = defaultdict(list)
    target2scales       = defaultdict(list)
    target2strings      = defaultdict(list)
    target2sentIDs      = defaultdict(list)
    target2alternatives = defaultdict(list)

    skip_count = 0
    for line in lines:
        target, sentID, target_position, context = line.split('\t')
        target_word = target.split('.')[0]

        context_ids = [word2index[w] for w in context.split() if w in word2index]  # might be empty
        try:
            target_id = word2index[target_word]
        except KeyError:
            # target word not in dictionary, skip it
            skip_count += 1
            continue

        alternatives = candidates_dict[target_word]
        alternative_count = 0
        good_alternatives = []
        alternative_ids = []

        for a in alternatives:
            try:
                alternative_ids += [word2index[a]]
                good_alternatives += [a]
                alternative_count += 1
            except KeyError:
                # alternative word not in dictionary
                pass

        if alternative_count < threshold:
            skip_count += 1
            continue

        center_embeds = torch.stack([embeddings[w] for w in [target_id] + alternative_ids])  # [a+1,d]
        center_embeds = center_embeds.unsqueeze(1)  # [a+1,1,d]
        center_embeds = center_embeds.repeat(1, len(context_ids), 1)  # [a+1,c,d]

        context_embeds = torch.stack([embeddings[i] for i in context_ids])  # [c,d]
        context_embeds = context_embeds.unsqueeze(0)  # [1,c,d]
        context_embeds = context_embeds.repeat(len(alternative_ids)+1, 1, 1)  # [a+1,c,d]

        encoder_input = torch.cat((center_embeds, context_embeds), dim=2)  # [a+1,c,2d]

        # Encode context-aware representations
        h = relu(encoder_input @ torch.t(encoder_W) + encoder_b)  # [a+1,c,2d]
        h = torch.sum(h, dim=1)  # [a+1,2d]

        # Inference step
        loc_vecs = h @ torch.t(affine_loc_W) + affine_loc_b  # [a+1,d]
        scale_vecs = softplus(h @ torch.t(affine_scale_W) + affine_scale_b)  # [a+1,d]

        target2locs[target].append(loc_vecs.numpy())
        target2scales[target].append(scale_vecs.numpy())
        target2strings[target].append(target)
        target2sentIDs[target].append(sentID)
        target2alternatives[target].append(good_alternatives)

    return target2locs, target2scales, target2strings, target2sentIDs, target2alternatives, skip_count

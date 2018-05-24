import re
import numpy as np

transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2',
                      'SST5', 'TREC', 'MRPC', 'SICKEntailment',
                      'Depth', 'BigramShift', 'Tense', 'SubjNumber']
s

models = ['skipgram', 'embedalign', 'glove50', 'fasttext']

accs = {}

for model in models:
    with open('output/23may/' + model + '_results.txt', 'r') as f_in:
        line = f_in.readlines()[0]

    line = re.sub("'|{|}|:|,", "", line)
    split_line = line.split()

    accs[model] = {}

    split_line = iter(split_line)
    for string in split_line:
        if string in transfer_tasks:

            if string == 'MRPC': rng = 10
            else: rng = 8

            for i in range(rng):
                next_str = next(split_line)
                if next_str == 'acc':
                    accs[model][string] = float(next(split_line))
                    break

# Compute average accuracy
for model in models:
    model_accs = np.array([float(x) for x in accs[model].values()])
    accs[model]['avg'] = np.mean(model_accs)

print("\\begin{table}")
print("    \\begin{tabular}{l|c|c|c|c}")
print("    ~              & \\textbf{Skip-Gram} & \\textbf{EA} & \\textbf{GloVe} & \\textbf{FastText} \\\\")
print('    \\hline')

for task in transfer_tasks:
    task_str = task
    if task == 'SICKEntailment': task_str = 'SICK-E'
    if task == 'Depth': task_str = 'TreeDepth'

    print("    {}  &  {:04.2f}  &  {:04.2f}  &  {:04.2f}  &  {:04.2f}  \\\\"
          .format(task_str, *[accs[x][task] for x in models]))

print("    \\end{tabular}")
print("\\end{table}")

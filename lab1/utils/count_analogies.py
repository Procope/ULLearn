import os
import numpy as np
tot_analogies = 0

for filename in os.listdir('analogy-questions'):
    if not filename.endswith(".txt"):
        continue

    filepath = "analogy-questions/{}".format(filename)
    class_name = filename[:-4]

    with open(filepath, "r") as f_in:
        n_analogies = len(list(f_in.readlines()))

        tot_analogies += n_analogies

    print(class_name, n_analogies)

print(tot_analogies)

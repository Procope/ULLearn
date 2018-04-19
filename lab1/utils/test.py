import bz2
# creating dictionary for word embeddings
filename = "bow2.words.bz2"
embedd = []
with bz2.open(filename, "rt") as bz_file:
    for line in bz_file:
        line.split('-')
        embedd.append(line)

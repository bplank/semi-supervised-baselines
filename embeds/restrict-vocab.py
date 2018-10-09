import sys
from collections import Counter

## get all types from gweb sancl labeled + unlabeled data
file_path="tokens.txt"
data = set()
with open(file_path, 'rb') as f:
    for line in f:
        word = line.decode('utf-8','ignore').strip()
        data.add(word)

def load_embeddings_file(file_name, sep=" ",lower=False, vocab=None):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name, errors='ignore', encoding='utf-8'):
        try:
            fields = line.strip().split(sep)
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            if lower:
                word = word.lower()
            if word in vocab:
                emb[word] = vec # only use words which are in vocab
        except ValueError:
            print("Error converting: {}".format(line))

    print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
    return emb

def save_embeds(emb, out_filename):
    OUT = open(out_filename,"w")
    for word in emb.keys():
        wembeds_expression = emb[word]
        OUT.write("{} {}\n".format(word," ".join([str(x) for x in wembeds_expression])))
    OUT.close()

print("vocab of size loaded: ", len(data))
embeds = load_embeddings_file("glove.6B.100d.txt", vocab=data)

save_embeds(embeds, "glove.6B.100d.restr.txt")

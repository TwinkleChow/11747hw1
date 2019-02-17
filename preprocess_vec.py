import io
from collections import defaultdict
import pickle
import numpy as np

def load_vectors(fname,w2i):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in w2i:
        	data[tokens[0]] = map(float, tokens[1:])
    return data

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
nwords = len(w2i)
# load embedding
vecs = load_vectors('crawl-300d-2M.vec',w2i)
emb_size=300
emb = np.random.uniform(-0.25,0.25,(nwords,emb_size))
for w in w2i:
	if w in vecs:
		emb[w2i[w],:]= np.array(vecs[w])
np.savetxt('emb.txt',emb)


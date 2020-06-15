import numpy as np
import nltk
from nltk.corpus import stopwords
import sklearn
import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords
from scipy import spatial

model = api.load("glove-wiki-gigaword-50")

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def cosine_distance(a,b):
    a = remove_stopwords(a)
    b = remove_stopwords(b)
    a_avg = None
    b_avg = None
    for w in a.split(" "):
        a_avg = a_avg + model[w] if a_avg else model[w]
    for w in b.split(" "):
        b_avg = b_avg + model[w] if b_avg else model[w]
    a_avg /= len(a.split(" "))
    b_avg /= len(b.split(" "))
    return (1 - spatial.distance.cosine(a_avg, b_avg))





def task_loss(decoded):
    pairs = decoded.split("<|endoftext|>")
    for p in pairs:
        a,b = p.split("[EOS]")
        dist = cosine_distance(a,b)
        blue = bleu(a,b)




import os
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import utils
import torchtext.vocab as vocab
import numpy as np

# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

cache_dir = "../glove"


class Ltaxo_Embedding(object):
    def __init__(self, option, dim, min_count = 1):
        self.option = option
        if option == "word2vec":
            self.embedding = Word2Vec(vector_size = dim, sg = 1, workers = 24, min_count = min_count)
        elif option == "glove":
            self.embedding = vocab.GloVe(name = '6B', dim = dim, cache = cache_dir)
        elif option == "fasttext":
            self.embedding = FastText(vector_size = dim, sg = 1, workers = 24, min_count = min_count)
    
    def construct(self, corpus):
        if self.option == "word2vec":
            self.embedding.build_vocab(corpus)
            self.embedding.train(corpus, total_examples = self.embedding.corpus_count, epochs = self.embedding.epochs)
        elif self.option == "fasttext":
            self.embedding.build_vocab(corpus)
            self.embedding.train(corpus, total_examples = self.embedding.corpus_count, epochs = self.embedding.epochs)

    def fit_corpus(self, corpus):
        if self.option == "word2vec":
            self.embedding.build_vocab(corpus, update = True)
            self.embedding.train(corpus, total_examples = self.embedding.corpus_count, epochs = self.embedding.epochs)
        elif self.option == "fasttext":
            self.embedding.build_vocab(corpus, update = True)
            self.embedding.train(corpus, total_examples = self.embedding.corpus_count, epochs = self.embedding.epochs)

    def get_embedding(self, word):
        if self.option == "word2vec":
            emb = self.embedding.wv[word]
        elif self.option == "glove":
            emb = self.embedding.vectors[self.embedding.stoi[word]]
        elif self.option == "fasttext":
            emb = self.embedding.wv[word]
        return np.array(emb)

    def get_word(self, idx):
        if self.option == "word2vec":
            return self.embedding.wv.index_to_key[idx]
        elif self.option == "glove":
            return self.embedding.itos[idx]
        elif self.option == "fasttext":
            return self.embedding.wv.index_to_key[idx]

    def gather_bad_ids(self, dict):
        bad_ids = []
        if self.option == "glove":
            for i in range(len(dict)):
                if not dict[i] in self.embedding.stoi:
                    bad_ids.append(i)
        return bad_ids
    
    def save(self, fname):
        if self.option == "word2vec":
            self.embedding.save(fname)
        elif self.option == "fasttext":
            self.embedding.save(fname)
        
    def load(self, fname):
        if self.option == "word2vec":
            self.embedding = Word2Vec.load(fname)
        elif self.option == "fasttext":
            self.embedding = FastText.load(fname)


    

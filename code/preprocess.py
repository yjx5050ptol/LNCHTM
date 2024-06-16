import os
from gensim import corpora
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim.downloader
from gensim import corpora
import string
import numpy as np
import configparser as cp
import sys
import re
from tqdm import tqdm
import scipy
from sklearn.feature_extraction.text import CountVectorizer
import embedding
import pickle
import gzip
import csv

# first time you need to download the followings
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

RANDOM_SEED = 90

def load_origin_file(file_name):
    words = []
    docs = open(file_name, encoding = "UTF-8", errors = "ignore")
    lines = docs.readlines()
    for line in lines:
        if line == "\n" :#or line.startswith("From:") or line.startswith("Subject:"):
            continue
        tokens = process_line(line)
        words.extend(tokens)
    docs.close()
    return words

def process_line(line):
    line = line.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    line = line.lower()
    tokens = nltk.word_tokenize(line)
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = lemma_words(tagged_tokens)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    alpha = re.compile('^[a-zA-Z_]+$')
    tokens = [token if alpha.match(token) else "_" for token in tokens]
    tokens = [token if len(token) >= 3 else "_" for token in tokens]
    tokens = [token for token in tokens if token != "_"]
    return tokens

def load_total_files(doc_dir):
    f = open(doc_dir, "r")
    f_l = open(doc_dir + "_label", "r")
    total_words = [line.strip().split(" ") for line in f.readlines()]
    total_labels = [line.strip() for line in f_l.readlines()]
    f.close()
    f_l.close()
    return total_words, total_labels

def lemma_words(tagged_tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = []
    for word, tag in tagged_tokens:
        if tag.startswith('NN'):
            word_lematizer =  wordnet_lemmatizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'): 
            word_lematizer =  wordnet_lemmatizer.lemmatize(word, pos='v')   # v代表动词
        elif tag.startswith('JJ'): 
            word_lematizer =  wordnet_lemmatizer.lemmatize(word, pos='a')   # a代表形容词
        elif tag.startswith('R'): 
            word_lematizer =  wordnet_lemmatizer.lemmatize(word, pos='r')   # r代表代词
        else: 
            word_lematizer =  wordnet_lemmatizer.lemmatize(word)
        tokens.append(word_lematizer)
    return tokens

def shuffle_chunks(doc_num, chunk_num):
    chunk_record = {}
    np.random.seed(RANDOM_SEED)
    shuffled_indices = np.random.permutation(doc_num)
    single_chunk_size = int(doc_num / chunk_num)
    chunk_remains = doc_num % chunk_num
    for i in range(chunk_num - 1):
        if i < chunk_remains:
            chunk_record[i] = shuffled_indices[i * single_chunk_size + i : (i + 1) * single_chunk_size + i + 1]
        else:
            chunk_record[i] = shuffled_indices[i * single_chunk_size + chunk_remains : (i + 1) * single_chunk_size + chunk_remains]
    chunk_record[chunk_num - 1] = shuffled_indices[(chunk_num - 1) * single_chunk_size + chunk_remains : ]
    return chunk_record

def divide_tvt(doc_num, train_ratio, val_ratio, test_ratio):
    np.random.seed(RANDOM_SEED)
    shuffled_indices = np.random.permutation(doc_num)
    train_num = int(train_ratio * doc_num)
    val_num = int(val_ratio * doc_num)
    test_num = doc_num - train_num - val_num
    train_indices = shuffled_indices[0 : train_num]
    val_indices = shuffled_indices[train_num : train_num + val_num]
    test_indices = shuffled_indices[train_num + val_num : ]
    assert len(train_indices) == train_num and len(val_indices) == val_num and len(test_indices) == test_num
    return train_indices, val_indices, test_indices
    
def save_file(total_words, total_labels, doc_indices, save_dir, with_label = False):
    f = open(save_dir, "w")
    for idx in doc_indices:
        words = total_words[idx]#load_origin_file(total_files[idx])
        if with_label:
            f.write(total_labels[idx] + "\t")
        f.write(" ".join(words) + "\n")
    f.close()
    
def load_origin_snippets(total_files, labels):
    total_words = []
    label_dict = {}
    for idx, label in enumerate(labels):
        label_dict[label] = idx
    label_record = []
    for file in total_files:
        f = open(file)
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                continue
            line = line.strip().split(" ")
            label = line[-1]
            line.pop(-1)
            line = " ".join(line)
            words = process_line(line)
            if label not in label_dict:
                print(line)
            label_record.append(label_dict[label])
            total_words.append(words)
    return total_words, label_record

def save_labels(total_labels, doc_indices, save_dir, is_digit = False):
    f_l = open(save_dir, "w")
    label_dict = corpora.Dictionary([total_labels])
    for idx in doc_indices:
        if is_digit:
            f_l.write(str(label_dict.token2id[total_labels[idx]]) + "\n")
        else:
            f_l.write(total_labels[idx] + "\n")
    f_l.close()

def load_total(total_files):
    total_words = []
    total_size = len(total_files)
    pbar = tqdm(total = total_size, desc = "Load Origin Files")
    for file_name in total_files:
        total_words.append(load_origin_file(file_name))
        pbar.update(1)
    return total_words

def save_total(total_words, label_record, doc_dir, vocab_dir, vocab_size):  
    filter_dict = corpora.Dictionary(total_words)
    filter_dict.filter_extremes(no_below = 5, no_above = 0.95, keep_n = vocab_size)
    filter_dict.save_as_text(vocab_dir)
    f = open(doc_dir, "w")
    f_l = open(doc_dir + "_label", "w")
    f_l_id = open(doc_dir + "_label_id", "w")
    for idx,words in enumerate(total_words):
        word_idxes = filter_dict.doc2idx(words)
        filter_words = []
        for word_idx in word_idxes:
            if word_idx >= 0:
                filter_words.append(filter_dict[word_idx])
        if len(filter_words) == 0:
            print(total_words[idx], idx)
            continue
        f.write(" ".join(filter_words) + "\n")
        f_l.write(labels[label_record[idx]] + "\n")
        f_l_id.write(str(label_record[idx]) + "\n")
    f.close()

    #f_l.close()

def save_vocab(filter_dict, vocab_dir):
    words = [filter_dict[idx] for idx in filter_dict]
    f = open(vocab_dir, "w")
    f.write("\n".join(words) + "\n")
    f.close()

def save_ECRTM(total_words, filter_dict, train_indices, test_indices, save_dir):
    total_docs = [" ".join(words) for words in total_words]
    vectorizer = CountVectorizer(analyzer="word", lowercase = False)
    vectorizer.fit(total_docs)
    for i in range(len(filter_dict)):
        if filter_dict[i] not in vectorizer.vocabulary_:
            print(filter_dict[i])
    train_docs = [total_docs[idx] for idx in train_indices]
    test_docs = [total_docs[idx] for idx in test_indices]
    train_bow_matrix = vectorizer.transform(train_docs)        
    test_bow_matrix = vectorizer.transform(test_docs)
    scipy.sparse.save_npz(os.path.join(save_dir, "train_bow.npz"), train_bow_matrix)
    scipy.sparse.save_npz(os.path.join(save_dir, "test_bow.npz"), test_bow_matrix)
    vocab = [filter_dict[idx] for idx in filter_dict]
    word_embeddings = make_word_embeddings(vocab)
    scipy.sparse.save_npz(os.path.join(save_dir, 'word_embeddings.npz'), word_embeddings)

def make_word_embeddings(vocab):
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))
    num_found = 0
    for i, word in enumerate(tqdm(vocab, desc="===>making word embeddings")):
        try:
            key_word_list = glove_vectors.index_to_key
        except:
            key_word_list = glove_vectors.index2word

        if word in key_word_list:
            word_embeddings[i] = glove_vectors[word]
            num_found += 1

    print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(word_embeddings)

def load_origin_grolier(bow_file, vocab_file):
    f_v = open(vocab_file, "r")
    vocabs = [line.strip() for line in f_v.readlines()]
    vocab_size = len(vocabs)
    f_d = open(bow_file, "r")
    docs = [line.strip().split(",") for line in f_d.readlines()]
    total_words = []
    label_record = []
    for doc in docs:
        if len(doc) > 0:
            words = [vocabs[int(word) - 1] for word in doc if (word.isdigit() and int(word) < vocab_size)]
            total_words.append(words)
            label_record.append(0)
    return total_words, label_record

def save_bow(total_words, filter_dict, doc_indices, save_dir):
    f = open(save_dir, "w")
    for doc_idx in doc_indices:
        bow = filter_dict.doc2bow(total_words[doc_idx])
        f.write(" ".join(["{}:{}".format(i,j) for (i,j) in bow]) + "\n")
      
def save_TaxoGen(total_words, vocab_dir, embedding_dir, doc_dir):     
    embedding_model = embedding.Ltaxo_Embedding("glove", 300)
    filter_dict = corpora.Dictionary.load_from_text(vocab_dir)
    bad_ids = embedding_model.gather_bad_ids(filter_dict)
    filter_dict.filter_tokens(bad_ids = bad_ids)
    
    with open(embedding_dir, "w") as f:
        for idx in filter_dict:
            word = filter_dict[idx]
            word_embedding = embedding_model.get_embedding(word)
            line = [str(val) for val in word_embedding]
            line.insert(0, word)
            f.write(" ".join(line) + "\n")      
    
    with open(doc_dir, "w") as f:
        for idx,words in enumerate(total_words):
            word_idxes = filter_dict.doc2idx(words)
            filter_words = []
            for word_idx in word_idxes:
                if word_idx >= 0:
                    filter_words.append(filter_dict[word_idx])
            if len(filter_words) == 0:
                print(total_words[idx])
                continue
            f.write(" ".join(filter_words) + "\n")




            
if __name__ == "__main__":
    work_directory = os.path.dirname(os.path.abspath(__file__))
    data_ini = sys.argv[1]
    data_config = cp.ConfigParser()
    data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding = 'utf-8')
    chunk_num = int(data_config.get(data_ini, "chunk_num"))
    vocab_size = int(data_config.get(data_ini, "vocab_size"))
    dataset_dir = os.path.join("../dataset", data_config.get(data_ini, "dataset_dir"))
    labels = data_config.get(data_ini, 'label').split(',')
    filename_prefix = data_config.get(data_ini, "filename_prefix")
    save_dir = os.path.join("../processed_dataset", data_ini)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_record = []
    total_files = []
    if data_ini == "20News":
        for (label_idx, label) in enumerate(labels):
            cur_label_dir = os.path.join(dataset_dir, label)
            file_list = os.listdir(cur_label_dir)
            for file_name in file_list:
                total_files.append(os.path.join(cur_label_dir, file_name))
                label_record.append(label_idx)
    elif data_ini == "IMDB":
        for div in ["train", "test"]:
            for (label_idx, label) in enumerate(labels):
                cur_label_dir = os.path.join(dataset_dir, div, label)
                file_list = os.listdir(cur_label_dir)
                for file_name in file_list:
                    total_files.append(os.path.join(cur_label_dir, file_name))
                    label_record.append(label_idx)
    elif data_ini == "Grolier":
        total_words, label_record = load_origin_grolier(os.path.join(dataset_dir, "grolier15276.csv"), os.path.join(dataset_dir, "grolier15276_words.txt"))

    vocab_dir = os.path.join(save_dir, "vocab_dict")
    doc_dir = os.path.join(save_dir, "total")
    
    
    if os.path.exists(vocab_dir):
        filter_dict = corpora.Dictionary.load_from_text(vocab_dir)
    else:
        if data_ini in ["20News", "IMDB"]:
            total_words = load_total(total_files)
        save_total(total_words, label_record, doc_dir, vocab_dir, vocab_size)
        filter_dict = corpora.Dictionary.load_from_text(vocab_dir)
    print(len(filter_dict))
    save_vocab(filter_dict, os.path.join(save_dir, "vocab"))
    total_words, total_labels = load_total_files(doc_dir)
    
    save_total(total_words, label_record, doc_dir, vocab_dir, vocab_size)
    filter_dict = corpora.Dictionary.load_from_text(vocab_dir)
    total_words, total_labels = load_total_files(doc_dir)

    chunk_record = shuffle_chunks(len(total_labels), chunk_num)
    for i in range(chunk_num):
        save_file(total_words, total_labels, chunk_record[i], os.path.join(save_dir, filename_prefix + "_" + str(i)))
        save_labels(total_labels, chunk_record[i], os.path.join(save_dir, filename_prefix + "_" + str(i) + "_label"))

    train_indices, val_indices, test_indices = divide_tvt(len(total_labels), 0.48, 0.12, 0.4)
    merge_indices = list(train_indices.copy())
    merge_indices.extend(val_indices)

    save_file(total_words, total_labels, train_indices, os.path.join(save_dir, "train_wl"), with_label = True)
    save_file(total_words, total_labels, val_indices, os.path.join(save_dir, "valid_wl"), with_label = True)
    save_file(total_words, total_labels, test_indices, os.path.join(save_dir, "test_wl"), with_label = True)
    save_file(total_words, total_labels, train_indices, os.path.join(save_dir, "train"), with_label = False)
    save_file(total_words, total_labels, val_indices, os.path.join(save_dir, "valid"), with_label = False)
    save_file(total_words, total_labels, test_indices, os.path.join(save_dir, "test"), with_label = False)
    

    save_labels(total_labels, train_indices, os.path.join(save_dir, "train_labels_id"), is_digit = True)
    save_labels(total_labels, val_indices, os.path.join(save_dir, "val_labels_id"), is_digit = True)
    save_labels(total_labels, test_indices, os.path.join(save_dir, "test_labels_id"), is_digit = True)
    save_labels(total_labels, train_indices, os.path.join(save_dir, "train_labels"), is_digit = False)
    save_labels(total_labels, val_indices, os.path.join(save_dir, "val_labels"), is_digit = False)
    save_labels(total_labels, test_indices, os.path.join(save_dir, "test_labels"), is_digit = False)
    
    save_ECRTM(total_words, filter_dict, merge_indices, test_indices, save_dir)
    save_file(total_words, total_labels, merge_indices, os.path.join(save_dir, "train_valid"), with_label = False)
    save_labels(total_labels, merge_indices, os.path.join(save_dir, "train_valid_labels_id"), is_digit = True)
    
    save_bow(total_words, filter_dict, merge_indices, os.path.join(save_dir, "train.feat"))
    save_bow(total_words, filter_dict, test_indices, os.path.join(save_dir, "test.feat"))
    
    save_TaxoGen(total_words, vocab_dir, os.path.join(save_dir, "embeddings.txt"), os.path.join(save_dir, "papers.txt"))
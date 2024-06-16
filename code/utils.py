import os
import numpy as np

def vector_distance(v1, v2, metric = "cosine"):
        if metric == "l2":
            return np.linalg.norm(v1 - v2, ord = 2)
        elif metric == "cosine":
            return 1 - (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
def normalize(v):
    v_min = v.min()
    v_max = v.max()
    return (v - v_min) / (v_max - v_min)

def load_corpus(file_name:str):
    words = []
    docs = open(file_name, encoding = "UTF-8", errors = "ignore")
    lines = docs.readlines()
    for line in lines:
        word_list = [line.strip().split(" ")]
        words += word_list
    docs.close()
    return words

def save_topics(topics, file_name):
    f = open(file_name, "w")
    for topic in topics:
        f.write(" ".join(topic) + "\n")
    f.close()

def load_topics(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    topics = [line.strip().split(" ") for line in lines]
    f.close()
    return topics

def load_relation(file_name):
    relation = np.load(file_name)
    return relation

def save_relation(arr, file_name):
    np.save(file_name, arr)
    return


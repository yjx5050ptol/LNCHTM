import os
import sys
import utils
import numpy as np

max_depth = 3


def print_hierarchy(idx, depth, topics, relations, output):
    output.write("\t" * depth + "[{}]\n".format(" ".join(topics[depth][idx])))   
    if depth < 2:
        for child in np.argwhere(relations[depth][idx] == 1):
            print_hierarchy(child[0], depth + 1, topics, relations, output)
    return

if __name__ == "__main__":
    dataset = sys.argv[1]
    chunk = sys.argv[2]
    res_dir = os.path.join("../result/LNCHTM_{}".format(dataset), "chunk_{}".format(chunk))
    topics = [utils.load_topics(os.path.join(res_dir,"topics_{}.txt".format(i))) for i in range(max_depth)]
    relations = [utils.load_relation(os.path.join(res_dir,"relation_{}.npy".format(i))) for i in range(max_depth - 1)]
    output = open(os.path.join(res_dir, "hierarchy.txt"), "w")
    for idx in range(len(topics[0])):
        print_hierarchy(idx, 0, topics, relations, output)
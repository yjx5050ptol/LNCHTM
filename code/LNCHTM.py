import configparser as cp
import os
import sys
import utils
import numpy as np

#from sklearnex import patch_sklearn, unpatch_sklearn, config_context
#patch_sklearn()

from gensim import corpora
import gensim
import embedding
import NonparametricClustering
import evaluation
import KnowledgeBase
import representativeness
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

import warnings
warnings.filterwarnings('ignore')

postprocessed_dir = "../processed_dataset/"
res_dir = "../result/"
prepared_dir = "../prepared/"


class LNCHTM(object):
    def __init__(self, exp_ini):
        work_directory = os.path.dirname(os.path.abspath(__file__))
        exp_config = cp.ConfigParser()
        data_config = cp.ConfigParser()
        exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding = 'utf-8')
        data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding = 'utf-8')

        self.exp_setting = exp_ini
        self.dataset_name = exp_config.get(exp_ini, "dataset")
        self.clustering_option = exp_config.get(exp_ini, "clustering")
        self.embedding_option = exp_config.get(exp_ini, "embedding")
        self.embedding_dim = int(exp_config.get(exp_ini, "embedding_dim"))
        #self.min_count = int(exp_config.get(exp_ini, "min_count"))
        self.n_layers = int(exp_config.get(exp_ini, "n_layers"))
        self.top_ns = exp_config.get(exp_ini, "top_ns").split(",")
        self.top_ns = [int(val) for val in self.top_ns]
        self.top_n = max(self.top_ns)
        #self.vocab_size = int(exp_config.get(exp_ini, "vocab_size"))
        self.co_factor = float(exp_config.get(exp_ini, "co_factor"))
        self.rep_threshold = float(exp_config.get(exp_ini, "rep_threshold"))
        self.with_kb = int(exp_config.get(exp_ini, "with_kb"))

        self.chunks = int(data_config.get(self.dataset_name, "chunk_num"))
        self.filename_prefix = data_config.get(self.dataset_name, "filename_prefix")

        self.embedding = embedding.Ltaxo_Embedding(self.embedding_option, self.embedding_dim)
        self.clustering = NonparametricClustering.NpClustering(self.clustering_option)
        self.kb = KnowledgeBase.KnowledgeBase(self.chunks // 3)

        self.vocab_size_list = np.zeros(self.chunks, dtype = int)
        
        self.eval_model = evaluation.Topic_Evaluation(self.exp_setting)
                
    def train(self):
        if self.with_kb:
            #get the proper value of kb_threshold
            self.with_kb = 0
            self.train_chunk(0)
            res = self.eval_chunk(0)
            self.kb_threshold = res["c_npmi"] * 1.2
            print("KB threshold is set to {}".format(self.kb_threshold))
            #warm up
            self.with_kb = 1
            self.train_chunk(0)
        for chunk in range(self.chunks):
            self.train_chunk(chunk)
            
    def catastrophic_forgetting(self):
        for kb_idx in range(self.chunks):
            for chunk in range(self.chunks):
                self.train_chunk(chunk, cf = kb_idx)
            
    def split_into_children(self, samples, labels, word_idxes, topic_idx):
        n_children = np.max(labels) + 1
        samples_list = [[] for i in range(n_children)]
        word_idxes_list = [[] for i in range(n_children)]
        topic_idx_list = [topic_idx + "_" + str(i) for i in range(n_children)]
        for (i, label) in enumerate(labels):
            samples_list[label].append(samples[i])
            word_idxes_list[label].append(word_idxes[i])
        return samples_list, word_idxes_list, topic_idx_list

    def extract_topic_words(self, samples, center, word_idxes, dict):
        distances = np.array([utils.vector_distance(sample, center) for sample in samples])
        sorted_words_idxes = distances.argsort()[: self.top_n]
        top_words_idxes = [word_idxes[i] for i in sorted_words_idxes]
        top_words = [dict[idx] for idx in top_words_idxes]
        return top_words, top_words_idxes

    def generate_doc_word(self, chunk_dir, chunk_dict, stop_words):
        # doc_word = np.zeros(shape = (len(chunk_corpus), len(chunk_dict)))
        # for (i, doc) in enumerate(chunk_corpus):
        #     doc_idx = chunk_dict.doc2idx(doc)
        #     for word_idx in doc_idx:
        #         if word_idx == -1:
        #             continue
        #         doc_word[i][word_idx] = 1
        # return doc_word
        #vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase=False, stop_words = stop_words)
        origin_vocab = [chunk_dict[idx] for idx in range(len(chunk_dict))]
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase = True, vocabulary = origin_vocab, stop_words = stop_words)
        doc_word = vectorizer.fit_transform(open(chunk_dir, "r")).toarray()
        vocab = vectorizer.get_feature_names_out()
        for i, v in enumerate(origin_vocab):
            if vocab[i] != v:
                print(v, vocab[i])
        return doc_word

    def generate_cooccurrence(self, doc_word):
        (doc_size, word_size) = doc_word.shape
        cooccurrence_mat = np.zeros(shape = (word_size, word_size), dtype = float)
        pbar = tqdm(total = word_size, desc = "Co-occurrence Preparing", leave = True)
        for i in range(word_size):
            flag_i = doc_word[:, i] > 0
            #cooccurence_mat[i][i] = np.sum(flag_i) / doc_size
            for j in range(i + 1, word_size):
                flag_j = doc_word[:, j] > 0
                cooccurrence_mat[i][j] = np.sum(flag_i * flag_j) #/ doc_size
                cooccurrence_mat[j][i] = cooccurrence_mat[i][j]
            pbar.update(1)
        pbar.close()
        return cooccurrence_mat

    def generate_affinity(self, samples):
        # n_samples = len(samples)
        # pbar = tqdm(total = n_samples, desc = "Affinity Calculating", leave = False, delay = 5)
        # affinity = np.zeros((n_samples, n_samples), dtype = float)
        # for i in range(n_samples):
        #     for j in range(i + 1, n_samples):
        #         affinity[i][j] = 1.0 - utils.vector_distance(samples[i], samples[j], metric = "cosine")
        #         affinity[j][i] = affinity[i][j]
        #     pbar.update(1)
        # pbar.close()
        # return affinity
        return cosine_similarity(samples)

    def filter_samples(self, samples, word_idxes, top_words_idxes):
        top_words_idxes.sort()
        for i,idx in enumerate(top_words_idxes):
            samples.pop(idx - i)
            word_idxes.pop(idx - i)
        return samples, word_idxes

    def extract_cooccurrence(self, cooccurence, word_idxes):
        word_size = len(word_idxes)
        if word_size == cooccurence.shape[0]:
            return cooccurence
        else:
            cur_cooccurence = np.zeros(shape = (word_size, word_size), dtype = float)
            for i in range(word_size):
                #cur_cooccurence[i][i] = cooccurence[word_idxes[i]][word_idxes[i]]
                for j in range(i + 1, word_size):
                    cur_cooccurence[i][j] = cooccurence[word_idxes[i]][word_idxes[j]]
                    cur_cooccurence[j][i] = cur_cooccurence[i][j]
            return cur_cooccurence

    def save_topics(self, chunk, topics_record, cf = -1):
        if cf == -1:
            save_dir = os.path.join(res_dir, self.exp_setting, "chunk_" + str(chunk))
        else:
            save_dir = os.path.join(res_dir, self.exp_setting, "kb_" + str(cf), "chunk_" + str(chunk))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        topic_nums = np.zeros(self.n_layers, dtype = int)
        topic_name_record_list = []
        for layer in range(self.n_layers):
            topic_nums[layer] = len(topics_record[layer])
            topics = []
            topic_name_record = {}
            for (i,topic_name) in enumerate(topics_record[layer]):
                topics.append(topics_record[layer][topic_name])
                topic_name_record[topic_name] = i
            topic_name_record_list.append(topic_name_record)
            utils.save_topics(topics, os.path.join(save_dir, "topics_{}.txt".format(layer)))
        for layer in range(self.n_layers - 1):
            relation = np.zeros(shape = (topic_nums[layer], topic_nums[layer + 1]), dtype = int)
            for (i, topic_name) in enumerate(topic_name_record_list[layer + 1]):
                topic_path = topic_name.split("_")
                topic_path.pop(len(topic_path) - 1)
                parent_topic_name = "_".join(topic_path)
                relation[topic_name_record_list[layer][parent_topic_name]][i] = 1
            utils.save_relation(relation, os.path.join(save_dir, "relation_{}.npy".format(layer)))
        return

    def eval_chunk(self, chunk):      
        res = self.eval_model.eval_chunk(chunk)
        return res

    def cal_cluster_prob(self, samples, sub_topic_centers):
        return cosine_similarity(samples, sub_topic_centers)

    def generate_guided_clusters(self, samples, word_idxes, topic_idx, sub_topic_softmax):
        unknown_samples = []
        unknown_word_idxes = []
        n_guided_nodes = len(sub_topic_softmax[0])
        guided_threshold = 1.0 - pow((1.0 - 1.0 / n_guided_nodes), self.beta)
        guided_samples_list = [[] for i in range(n_guided_nodes)]
        guided_word_idxes_list = [[] for i in range(n_guided_nodes)]
        guided_topic_idx_list = [topic_idx + "_g" + str(i) for i in range(n_guided_nodes)]
        #guided_guided_node_list = sub_topic_nodes
        max_idxes = np.argmax(sub_topic_softmax, axis = 1)
        for (i, sample) in enumerate(samples):
            if(sub_topic_softmax[i][max_idxes[i]] >= guided_threshold):
                guided_samples_list[max_idxes[i]].append(sample)
                guided_word_idxes_list[max_idxes[i]].append(word_idxes[i])
            else:
                unknown_samples.append(sample)
                unknown_word_idxes.append(word_idxes[i])
        return guided_samples_list, guided_word_idxes_list, guided_topic_idx_list, unknown_samples, unknown_word_idxes
        
    def match_guided_centers(self, cur_centers, guided_centers, guided_node_list):
        cur_guided_node_list = [None for i in range(len(cur_centers))]
        if len(guided_centers) > 0:
            guided_center_prob = self.cal_cluster_prob(cur_centers, guided_centers)
            guided_center_idxes = np.argmax(guided_center_prob, axis = 0)
            for i,idx in enumerate(guided_center_idxes):
                cur_guided_node_list[idx] = guided_node_list[i]
        return cur_guided_node_list

    def filter_clusters(self, samples_list, word_idxes_list, topic_idx_list, guided_node_list, centers, center_words):
        filter_idx_list = []
        for(i, samples) in enumerate(samples_list):
            if len(samples) < self.top_n:
                filter_idx_list.append(i)
        for(i, idx) in enumerate(filter_idx_list):
            samples_list.pop(idx - i)
            word_idxes_list.pop(idx - i)
            topic_idx_list.pop(idx - i)
            guided_node_list.pop(idx - i)
            centers.pop(idx - i)
            center_words.pop(idx - i)
        #assert len(samples_list) == len(word_idxes_list) 

        return samples_list, word_idxes_list, topic_idx_list, guided_node_list, centers, center_words

    def eval_coherence(self, topic_words, doc_word, word_dict):
        word_array = word_dict.doc2idx(topic_words)
        doc_size = len(doc_word)
        coherence = 0.0
        for top_n in self.top_ns:
            cur_coh = 0.0
            for n in range(top_n):
                flag_n = doc_word[:, word_array[n]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, top_n):
                    flag_l = doc_word[:, word_array[l]] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    if p_nl > 0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        cur_coh += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            cur_coh *= (2 / (top_n * top_n - top_n))
            coherence += cur_coh
        coherence /= len(self.top_ns)
        return coherence

    def extract_relevant_docs(self, doc_word, word_idxes):
        doc_size = len(doc_word)
        flag = np.zeros(doc_size, dtype = int)
        for word_idx in word_idxes:
            flag += (doc_word[:, word_idx] > 0)
        doc_idxes = []
        for idx in range(doc_size):
            if flag[idx] > 0:
                doc_idxes.append(idx)
        return doc_idxes

    def extract_relevant_words(self, rep_eval_model, word_idxes, dict):
        relevant_word_idxes = []
        for (i, idx) in enumerate(word_idxes):
            if rep_eval_model.get_term_score(dict[idx]) >= self.rep_threshold:
                relevant_word_idxes.append(i)
        return relevant_word_idxes

    def co_guided_embedding(self, samples, cooccurrence):
        coe = self.co_factor * cooccurrence + np.diag(np.ones(len(samples), dtype = float))
        deg = np.sum(coe, axis = 0)
        temp = np.diag(1.0 / deg)
        samples = np.array(samples)
        return temp @ coe @ samples

    def train_chunk(self, chunk:int, cf = -1):
        start_time = 0
        end_time = 0

        assert chunk < self.chunks
        chunk_dir = os.path.join(postprocessed_dir, self.dataset_name, self.filename_prefix + "_" + str(chunk))
        prepared_data_dir = os.path.join(prepared_dir, self.dataset_name, self.filename_prefix + "_" + str(chunk))
        start_time = time.perf_counter()
        chunk_corpus = utils.load_corpus(chunk_dir)
        end_time = time.perf_counter()
        print("Load Corpus Time {} s".format(end_time - start_time))

        start_time = time.perf_counter()
        self.embedding.load(os.path.join(prepared_data_dir, "_".join([self.embedding_option, str(self.embedding_dim)]) + ".model"))
        chunk_dict = corpora.Dictionary(chunk_corpus)
        #chunk_dict.filter_extremes(no_below = self.min_count)
        bad_ids = self.embedding.gather_bad_ids(chunk_dict)
        stop_words = [chunk_dict[bad_id] for bad_id in bad_ids]
        chunk_dict.filter_tokens(bad_ids = bad_ids)
        end_time = time.perf_counter()
        print("Load Embedding and Vocab Time {} s".format(end_time - start_time))

        self.vocab_size_list[chunk] = len(chunk_dict)

        start_time = time.perf_counter()
        doc_word = self.generate_doc_word(chunk_dir, chunk_dict, stop_words)
        cooccurrence = sp.load_npz(os.path.join(prepared_data_dir, "cooccurrence.npz")).toarray()#self.generate_cooccurrence(doc_word)
        end_time = time.perf_counter()
        print("Load Doc-Word and Co-occurrence {} s".format(end_time - start_time))
        
        if cf != -1:
            self.kb.reconstruct(self.exp_setting, cf)

        samples_list = [np.array([self.embedding.get_embedding(chunk_dict[i]) for i in range(len(chunk_dict))])]
        word_idxes_list = [[i for i in range(len(chunk_dict))]]
        topic_idx_list = ["0"]
        if self.with_kb:
            guided_node_list = [self.kb.get_root()]
        else:
            guided_node_list = [None]

        topics_record = [{} for i in range(self.n_layers)]

        for l in range(self.n_layers):
            next_samples_list = []
            next_word_idxes_list = []
            next_topic_idx_list = []
            next_guided_node_list = []
            topics_record[l] = {}
            pbar = tqdm(total = len(samples_list), desc = "Clustering on layer {}".format(l), leave = True)
            for (i,samples) in enumerate(samples_list):
                cur_guided_node = guided_node_list[i] 

                preference = None

                child_centers = []
                child_node_list = []
                if cur_guided_node is not None:
                    if self.kb.get_num_children(cur_guided_node) > 0:
                        child_center_words = self.kb.get_sub_topics(cur_guided_node)
                        child_centers = [self.embedding.get_embedding(sub_word) for sub_word in child_center_words]
                        child_node_list = self.kb.get_sub_nodes(cur_guided_node)
                        sub_topic_prob = self.cal_cluster_prob(samples, child_centers) 
                        preference = np.max(sub_topic_prob, axis = 1)
                        #preference = utils.normalize(preference)
                        
                if len(samples) > 0:
                    cur_cooccurrence = self.extract_cooccurrence(cooccurrence, word_idxes_list[i])
                    enhanced_samples = self.co_guided_embedding(samples, cur_cooccurrence)
                    affinity = self.generate_affinity(enhanced_samples)
                    if preference is not None:
                        preference *= np.min(affinity)
                    
                    pred_labels, center_indices = self.clustering.train(affinity, preference)
                    # Clustering doesn't converge
                    if -1 in pred_labels:
                        pbar.update(1)
                        continue
                    cur_centers = [samples[idx] for idx in center_indices]
                    cur_center_words = [chunk_dict[word_idxes_list[i][idx]] for idx in center_indices]
                    cur_samples_list, cur_word_idxes_list, cur_topic_idx_list = self.split_into_children(samples, pred_labels, word_idxes_list[i], topic_idx_list[i])
                    cur_guided_node_list = self.match_guided_centers(cur_centers, child_centers, child_node_list)

                cur_samples_list, cur_word_idxes_list, cur_topic_idx_list, cur_guided_node_list, cur_centers, cur_center_words = self.filter_clusters(cur_samples_list, cur_word_idxes_list, cur_topic_idx_list, cur_guided_node_list, cur_centers, cur_center_words)

                n_children = len(cur_samples_list)
                for child_idx in range(n_children):
                    topic_words, topic_words_idxes = self.extract_topic_words(cur_samples_list[child_idx], cur_centers[child_idx], cur_word_idxes_list[child_idx], chunk_dict)
                    topics_record[l][cur_topic_idx_list[child_idx]] = topic_words
                    if cur_guided_node is not None and cf == -1:
                        coherence = self.eval_coherence(topic_words, doc_word, chunk_dict)
                    #更新cur_guided_node_list[child_idx]的词
                        if cur_guided_node_list[child_idx] is not None and coherence >= self.kb_threshold:
                            self.kb.add_rm_count(cur_guided_node_list[child_idx], -1)
                            self.kb.update_node(cur_guided_node_list[child_idx], cur_center_words[child_idx])
                    #为cur_guided_node插入新的子节点（对应cur_guided_node_list[child_idx] = None的聚簇）
                        elif coherence >= self.kb_threshold:
                            new_node = self.kb.generate_new_node(cur_center_words[child_idx])
                            self.kb.insert_child_node(cur_guided_node, new_node)
                            cur_guided_node_list[child_idx] = new_node
                    #小于阈值时
                        elif cur_guided_node_list[child_idx] is not None and coherence < self.kb_threshold:
                            if self.kb.add_rm_count(cur_guided_node_list[child_idx], 1):
                                self.kb.remove_child_node(cur_guided_node, cur_guided_node_list[child_idx])
                    relevant_doc_idxes = self.extract_relevant_docs(doc_word, topic_words_idxes)
                    relevant_corpus = [chunk_corpus[idx] for idx in relevant_doc_idxes]
                    #self.embedding.fit_corpus(relevant_corpus)
                    cur_rep_eval_model = representativeness.Representativeness(relevant_corpus)
                    filter_word_idxes = self.extract_relevant_words(cur_rep_eval_model, cur_word_idxes_list[child_idx], chunk_dict)
                    cur_samples_list[child_idx], cur_word_idxes_list[child_idx] = self.filter_samples(cur_samples_list[child_idx], cur_word_idxes_list[child_idx], filter_word_idxes)
                cur_samples_list, cur_word_idxes_list, cur_topic_idx_list, cur_guided_node_list, cur_centers, cur_center_words = self.filter_clusters(cur_samples_list, cur_word_idxes_list, cur_topic_idx_list, cur_guided_node_list, cur_centers, cur_center_words)

                next_samples_list.extend(cur_samples_list)
                next_word_idxes_list.extend(cur_word_idxes_list)
                next_topic_idx_list.extend(cur_topic_idx_list)
                next_guided_node_list.extend(cur_guided_node_list)

                pbar.update(1)
            pbar.close()

            samples_list = next_samples_list
            word_idxes_list = next_word_idxes_list
            topic_idx_list = next_topic_idx_list
            guided_node_list = next_guided_node_list


        self.save_topics(chunk, topics_record, cf)
        if cf == -1:
            self.kb.save(self.exp_setting, chunk)

    def prepare_data(self):
        for chunk in range(self.chunks):
            chunk_dir = os.path.join(postprocessed_dir, self.dataset_name, self.filename_prefix + "_" + str(chunk))
            prepared_data_dir = os.path.join(prepared_dir, self.dataset_name, self.filename_prefix + "_" + str(chunk))
            print("Preparing Chunk {}".format(chunk))
            start_time = time.perf_counter()
            chunk_corpus = utils.load_corpus(chunk_dir)
            end_time = time.perf_counter()
            print("Load Corpus Time {} s".format(end_time - start_time))
            
            if not os.path.exists(prepared_data_dir):
                os.makedirs(prepared_data_dir)

            embedding_dir = os.path.join(prepared_data_dir, "_".join([self.embedding_option, str(self.embedding_dim)]) + ".model")
            start_time = time.perf_counter()
            if not os.path.exists(embedding_dir):
                if chunk == 0:
                    self.embedding.construct(chunk_corpus)
                else:
                    self.embedding.fit_corpus(chunk_corpus)
                self.embedding.save(embedding_dir)
            else:
                self.embedding.load(embedding_dir)
            chunk_dict = corpora.Dictionary(chunk_corpus)
            #chunk_dict.filter_extremes(no_below = self.min_count)
            bad_ids = self.embedding.gather_bad_ids(chunk_dict)
            stop_words = [chunk_dict[bad_id] for bad_id in bad_ids]
            chunk_dict.filter_tokens(bad_ids = bad_ids)
            end_time = time.perf_counter()
            print("Preparing Embedding and Vocab Time {} s".format(end_time - start_time))

            cooccurrence_dir = os.path.join(prepared_data_dir, "cooccurrence.npz")
            start_time = time.perf_counter()
            doc_word = self.generate_doc_word(chunk_dir, chunk_dict, stop_words)
            if not os.path.exists(cooccurrence_dir):
                cooccurrence = self.generate_cooccurrence(doc_word)
                cooccurrence = sp.csr_matrix(cooccurrence)
                sp.save_npz(cooccurrence_dir, cooccurrence, compressed = True)
            else:
                cooccurrence = sp.load_npz(cooccurrence_dir)
            end_time = time.perf_counter()
            print("Preparing Doc-Word and Co-occurrence {} s".format(end_time - start_time))

if __name__ == "__main__":
    exp_ini = sys.argv[1]
    model = LNCHTM(exp_ini)
    model.prepare_data()
    model.train()
    

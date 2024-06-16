from gensim.models.coherencemodel import CoherenceModel
import configparser as cp
import utils
import os
from gensim import corpora
import numpy as np
import sys
import json

postprocessed_dir = "../processed_dataset"
result_dir = "../result"
test_file_record = {"TaxoGen": "total", "NSEM-GMHTM": "test", "CluHTM": "total", "ECRTM": "test"}


class Topic_Evaluation(object):
    def __init__(self, exp_ini):
        self.exp_setting = exp_ini

        work_directory = os.path.dirname(os.path.abspath(__file__))
        exp_config = cp.ConfigParser()
        data_config = cp.ConfigParser()
        exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding = 'utf-8')
        data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding = 'utf-8')
        self.model_name = self.exp_setting.split("_")[0]
        self.coh_metrics = exp_config.get(exp_ini, "coh_metrics").split(',')
        self.top_ns = exp_config.get(exp_ini, "top_ns").split(",")
        self.top_ns = [int(val) for val in self.top_ns]
        dataset_name = exp_config.get(exp_ini, "dataset")
        self.is_hierarchical = False
        if exp_config.has_option(exp_ini, "n_layers"):
            self.n_layers = int(exp_config.get(exp_ini, "n_layers"))
            self.is_hierarchical = True
        self.is_lifelong = int(exp_config.get(exp_ini, "is_lifelong"))
        self.filename_prefix = data_config.get(dataset_name, "filename_prefix")
        self.chunk_num = int(data_config.get(dataset_name, "chunk_num"))
        self.corpus_dir = os.path.join(postprocessed_dir, dataset_name)

    def load_topics_flat(self, topic_file, corpus_file):
        self.flat_topics_list = utils.load_topics(topic_file)
        self.corpus = utils.load_corpus(corpus_file)
        self.word_dict = corpora.Dictionary(self.corpus)
        self.cal_doc_word()

    def load_topics_hierarchical(self, topic_files, relation_files, corpus_file):
        self.n_layers = len(topic_files)
        self.corpus = utils.load_corpus(corpus_file)
        self.word_dict = corpora.Dictionary(self.corpus)
        self.h_topics_list = [utils.load_topics(topic_file) for topic_file in topic_files]
        self.h_relation_list = [utils.load_relation(relation_file) for relation_file in relation_files]
        self.flat_topics_list = []
        for t in self.h_topics_list:
            self.flat_topics_list.extend(t)
        self.cal_doc_word()

    def eval_coherence(self, level = -1):
        #npmi
        if level == -1:
            origin_topics_list = self.flat_topics_list
        else:
            origin_topics_list = self.h_topics_list[level]
        topics_list = [self.word_dict.doc2idx(topic) for topic in origin_topics_list]
        doc_size = len(self.corpus)
        topic_size = len(topics_list)
        sum_coherence_score = 0.0
        for i in range(topic_size):
            word_array = topics_list[i]
            for top_n in self.top_ns:
                sum_score = 0.0
                for n in range(top_n):
                    if word_array[n] == -1:
                        continue
                    flag_n = self.doc_word[:, word_array[n]] > 0
                    p_n = np.sum(flag_n) / doc_size
                    for l in range(n + 1, top_n):
                        if word_array[l] == -1:
                            continue
                        flag_l = self.doc_word[:, word_array[l]] > 0
                        p_l = np.sum(flag_l)
                        p_nl = np.sum(flag_n * flag_l) 
                        if p_nl > 0:
                            p_l = p_l / doc_size
                            p_nl = p_nl / doc_size
                            sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                        else:
                            sum_score += 0
                sum_coherence_score += sum_score * (2 / (top_n * top_n - top_n))
        sum_coherence_score /= len(self.top_ns)
        sum_coherence_score /= topic_size
        
        return sum_coherence_score

    def eval_coherence_by_gensim(self, metric):
        window_size = max([len(doc) for doc in self.corpus])
        result = 0.0
        for top_n in self.top_ns:
            coh_model = CoherenceModel(topics = self.flat_topics_list, texts = self.corpus, dictionary = self.word_dict, coherence = metric, topn = top_n, window_size = window_size, processes = 40)
            result += coh_model.get_coherence()
        result /= len(self.top_ns)
        return result

    def eval_uniqueness(self):
        tu = 0.0
        for top_n in self.top_ns:
            count = np.zeros(len(self.word_dict))
            topics_list = [self.word_dict.doc2idx(topic)[: top_n] for topic in self.flat_topics_list]
            for topic_idx in topics_list:
                for word_idx in topic_idx:
                    count[word_idx] += 1
            for topic_idx in topics_list:
                tu_topic = 0.0
                for word_idx in topic_idx:
                    tu_topic += 1.0 / count[word_idx]
                tu_topic /= top_n
                tu += tu_topic
        tu /= len(self.top_ns)
        tu /= len(self.flat_topics_list)
        return tu

    def eval_diversity(self):
        td = 0.0
        for top_n in self.top_ns:
            
            topics_list = []
            for topic in self.flat_topics_list:
                topics_list.extend(topic[: top_n])
                print(topic[: top_n])
            print(len(set(topics_list)))
            td += len(set(topics_list)) / len(topics_list)
        td /= len(self.top_ns)
        return td
                
    def eval_hierarchical(self):
        avg_clnpmi = 0.0
        avg_or = 0.0
        doc_size = len(self.doc_word)
        for top_n in self.top_ns:
            pair_count = 0
            sum_clnpmi = 0.0
            sum_or = 0.0
            for (i, relation_mat) in enumerate(self.h_relation_list):
                topic_list_1 = [self.word_dict.doc2idx(topic)[: top_n] for topic in self.h_topics_list[i]]
                topic_list_2 = [self.word_dict.doc2idx(topic)[: top_n] for topic in self.h_topics_list[i + 1]]
                topic_size_1 = len(topic_list_1)
                topic_size_2 = len(topic_list_2)
                for idx1 in range(topic_size_1):
                    for idx2 in range(topic_size_2):
                        if relation_mat[idx1][idx2] == 1:
                            pair_count += 1
                            word_idx1 = topic_list_1[idx1]
                            word_idx2 = topic_list_2[idx2]
                            set1 = set(word_idx1)
                            set2 = set(word_idx2)
                            inter = set1.intersection(set2)
                            word_idx1 = list(set1.difference(inter))
                            word_idx2 = list(set2.difference(inter))

                            sum_or += len(inter) / top_n

                            cur_sum_clnpmi = 0.0
                            word_pair_count = 0
                            for n in range(len(word_idx1)):
                                if word_idx1[n] == -1:
                                    continue
                                flag_n = self.doc_word[:, word_idx1[n]] > 0
                                p_n = np.sum(flag_n) / doc_size
                                for l in range(len(word_idx2)):
                                    if word_idx2[l] == -1:
                                        continue
                                    flag_l = self.doc_word[:, word_idx2[l]] > 0
                                    p_l = np.sum(flag_l)
                                    p_nl = np.sum(flag_n * flag_l)
                                    if p_nl > 0:
                                        p_l = p_l / doc_size
                                        p_nl = p_nl / doc_size
                                        #p_nl += 1e-10
                                        cur_sum_clnpmi += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                                    word_pair_count += 1
                            if word_pair_count > 0:
                                cur_sum_clnpmi /= word_pair_count
                            else:
                                cur_sum_clnpmi = 0
                            sum_clnpmi += cur_sum_clnpmi
            sum_clnpmi /= pair_count
            sum_or /= pair_count
            avg_clnpmi += sum_clnpmi
            avg_or += sum_or
        avg_clnpmi /= len(self.top_ns)
        avg_or /= len(self.top_ns)
        return avg_clnpmi, avg_or

    def get_evaluation(self):
        #self.check()
        result = {}
        result["c_npmi"] = self.eval_coherence(level = -1)
        #result["c_npmi"] = self.eval_coherence_by_gensim(metric = "c_npmi")
        #result["c_v"] = self.eval_coherence_by_gensim(metric = "c_v")
        result["tu"] = self.eval_uniqueness()
        # #result["td"] = self.eval_diversity()
        result["tq"] = result["c_npmi"] * result["tu"]
        if self.is_hierarchical:
            # for i in range(self.n_layers):
            #     result["c_npmi_l{}".format(i)] = self.eval_coherence(level = i)
            result["clnpmi"], result["or"] = self.eval_hierarchical()
            for i in range(self.n_layers):
                result["n_topics_{}".format(i)] = len(self.h_topics_list[i])
        else:
            result["n_topics"] = len(self.flat_topics_list)
        return result
    
    def cal_doc_word(self):
        self.doc_word = np.zeros(shape = (len(self.corpus), len(self.word_dict)))
        for (i, doc) in enumerate(self.corpus):
            doc_idx = self.word_dict.doc2idx(doc)
            for word_idx in doc_idx:
                self.doc_word[i][word_idx] = 1
        return

    def get_info(self):
        info = {}
        if self.is_hierarchical:
            info["n_topics"] = {}
            for i in range(self.n_layers):
                info["n_topics"][i] = len(self.h_topics_list[i])
                #print("layer ", i, " num of topics: ", len(self.h_topics_list[i]))
        else:
            info["n_topics"] = len(self.flat_topics_list)
        return info

    def eval(self):
        if self.is_lifelong == 0:
            res_file = os.path.join(result_dir, self.exp_setting, "res.json")
            if os.path.exists(res_file):
                res = self.load_res(res_file)
            else:
                test_file = test_file_record[self.model_name]
                corpus_file = os.path.join(self.corpus_dir, test_file)
                #corpus_file = os.path.join(self.corpus_dir, "total")
                if self.is_hierarchical:
                    topic_files = [os.path.join(result_dir, self.exp_setting, "topics_{}.txt".format(i)) for i in range(self.n_layers)]
                    relation_files = [os.path.join(result_dir, self.exp_setting, "relation_{}.npy".format(i)) for i in range(self.n_layers - 1)]
                    self.load_topics_hierarchical(topic_files, relation_files, corpus_file)
                else:
                    topic_file = os.path.join(result_dir, self.exp_setting, "topics.txt")
                    self.load_topics_flat(topic_file, corpus_file)
                res = self.get_evaluation()
        else:
            res = self.eval_lifelong()
            # forgetting_res = self.eval_forgetting()
            # self.save_res(forgetting_res, os.path.join(result_dir, self.exp_setting, "lifelong.json"))
            # print(forgetting_res)
        self.save_res(res, os.path.join(result_dir, self.exp_setting, "res.json"))
        print(res)
        return res

    def eval_chunk(self, chunk, cf = -1):
        if cf == -1:
            root_dir = ""
        else:
            root_dir = "kb_" + str(cf)
        res_file = os.path.join(result_dir, self.exp_setting, root_dir, "chunk_{}".format(chunk), "res.json")
        if os.path.exists(res_file):
            res = self.load_res(res_file)
        else:
            corpus_file = os.path.join(self.corpus_dir, self.filename_prefix + "_{}".format(chunk))
            #corpus_file = os.path.join(self.corpus_dir, "total")
            if self.is_hierarchical:
                topic_files = [os.path.join(result_dir, self.exp_setting, root_dir, "chunk_{}".format(chunk), "topics_{}.txt".format(i)) for i in range(self.n_layers)]
                relation_files = [os.path.join(result_dir, self.exp_setting, root_dir, "chunk_{}".format(chunk), "relation_{}.npy".format(i)) for i in range(self.n_layers - 1)]
                self.load_topics_hierarchical(topic_files, relation_files, corpus_file)
            else:
                topic_file = os.path.join(result_dir, self.exp_setting, root_dir, "chunk_{}".format(chunk), "topics.txt")
                self.load_topics_flat(topic_file, corpus_file)
            res = self.get_evaluation()
            self.save_res(res, res_file)      
        return res

    def eval_lifelong(self):
        res = []
        for i in range(self.chunk_num):
            res.append(self.eval_chunk(i))
            print(i, res[i])
        avg_res = {}
        for index in res[0]:
            avg_res[index] = 0.0
            for i in range(self.chunk_num):
                avg_res[index] += res[i][index] / self.chunk_num
        return avg_res
    
    def eval_forgetting(self):
        metrics = ["c_npmi", "tu"]
        val_mat_record = {}
        for metric in metrics:
            val_mat_record[metric] = np.zeros(shape = (self.chunk_num, self.chunk_num))
        for kb_idx in range(self.chunk_num):
            for chunk in range(self.chunk_num):
                res = self.eval_chunk(chunk, kb_idx)
                for metric in metrics:
                    val_mat_record[metric][kb_idx][chunk] = res[metric]
        forgetting_res = {}
        for metric in metrics:
            forgetting_res[metric] = self.cal_forgetting(val_mat_record[metric])
        return forgetting_res
                            
    def cal_forgetting(self, val_mat):
        res = {"as": np.zeros(self.chunk_num), "fs": np.zeros(self.chunk_num), "ls": np.zeros(self.chunk_num)}
        for chunk in range(self.chunk_num):
            res["as"][chunk] = np.mean(val_mat[chunk])
            res["ls"][chunk] = np.mean([val_mat[i][i] for i in range(chunk + 1)])
        for chunk in range(1, self.chunk_num):
            prev_val_mat = val_mat[0: chunk, 0: chunk]
            max_vals = np.max(prev_val_mat, axis = 0)
            res["fs"][chunk] = np.mean(max_vals - val_mat[chunk, 0 : chunk])
        for metric in res.keys():
            res[metric] = res[metric].tolist()
        return res
  
    def check(self):
        invaild_words = []
        for topic in self.flat_topics_list:
            for word in topic:
                if word not in self.word_dict.token2id:
                    invaild_words.append(word)
        if len(invaild_words) > 0:
            print("Some topic words are not in the word_dict: ", invaild_words)

    def save_res(self, res, filename):
        with open(filename, "w") as f:
            json.dump(res, f)
            
    def load_res(self, filename):
        with open(filename, "r") as f:
            return json.load(f)
        
if __name__ == "__main__":
    exp_ini = sys.argv[1]#"ECRTM_20News_50"
    chunk = 0
    eval_model = Topic_Evaluation(exp_ini)
    res = eval_model.eval()

import os

kb_dir = "../knowledgebase"

class TopicTreeNode(object):
    def __init__(self, word):
        self.word = word
        self.children = []
        self.rm_count = 0
    def add_rm_count(self, val = 1):
        self.rm_count += val
        self.rm_count = max(self.rm_count, 0)
    def add_child(self, treenode):
        self.children.append(treenode)
    def remove_child(self, idx):
        self.children.pop(idx)
    def get_rm_count(self):
        return self.rm_count
    def get_children(self):
        return self.children.copy()
    def get_num_children(self):
        return len(self.children)
    def get_word(self):
        return self.word
    def set_word(self, new_word):
        self.word = new_word
    def show(self):
        print(self.word, len(self.children))

class KnowledgeBase(object):
    def __init__(self, rm_threshold):
        self.root = TopicTreeNode("ROOT")
        self.rm_threshold = rm_threshold
    def get_root(self):
        return self.root
    def get_rm_count(self, treenode):
        return treenode.get_rm_count()
    def get_num_children(self, treenode):
        return treenode.get_num_children()
    def get_sub_topics(self, treenode):
        topic_word_list = []
        children = treenode.get_children()
        for child in children:
            topic_word_list.append(child.get_word())
        return topic_word_list
    def get_sub_nodes(self, treenode):
        return treenode.get_children()
    def generate_new_node(self, word):
        return TopicTreeNode(word)
    def add_rm_count(self, treenode, val = 1):
        treenode.add_rm_count(val)
        return treenode.get_rm_count() >= self.rm_threshold
    def insert_child_node(self, parent, child):
        parent.add_child(child)
    def remove_child_node(self, parent, child):
        children = parent.get_children()
        child_idx = children.index(child)
        parent.remove_child(child_idx)
    def update_node(self, treenode, word):
        treenode.set_word(word)

    def level_traverse(self):
        level_node = [self.root]
        while(len(level_node) > 0):
            next_level_node = []
            cur_level_word = []
            for node in level_node:
                cur_level_word.append(node.get_word())
                next_level_node.extend(node.get_children())
            level_node = next_level_node
            print(",".join(cur_level_word))
        return

    def save(self, exp_ini, chunk):
        level_word_list = []
        parent_idxes_list = []
        level_node = [self.root]
        parent_idxes_list.append([-1])
        while(len(level_node) > 0):
            next_level_node = []
            cur_level_word = []
            cur_parent_idxes = []
            for (i,node) in enumerate(level_node):
                cur_level_word.append(node.get_word())
                next_level_node.extend(node.get_children())
                cur_parent_idxes.extend([i for n in range(node.get_num_children())])
            level_node = next_level_node
            level_word_list.append(cur_level_word)
            #排除最后一层无子节点
            if len(cur_parent_idxes) > 0:
                parent_idxes_list.append(cur_parent_idxes)
        save_dir = os.path.join(kb_dir, exp_ini)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "kb_" + str(chunk)), "w") as f:
            for l, (level_word, parent_idxes) in enumerate(zip(level_word_list, parent_idxes_list)):
                f.write(",".join(level_word) + "\n")
                f.write(",".join([str(idx) for idx in parent_idxes]) + "\n")
        return
        
    def reconstruct(self, exp_ini, chunk):
        level_node_list = []
        parent_idxes_list = []
        f = open(os.path.join(kb_dir, exp_ini, "kb_" + str(chunk)), "r")
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            level_word = lines[idx].strip("\n").split(",")
            level_node_list.append([TopicTreeNode(word) for word in level_word])
            parent_idxes = lines[idx + 1].split(",")
            parent_idxes_list.append([int(par_idx) for par_idx in parent_idxes])
            idx = idx + 2
        for l in range(1, len(level_node_list)):
            num_nodes = len(level_node_list[l])
            for node_idx in range(num_nodes):
                level_node_list[l - 1][parent_idxes_list[l][node_idx]].add_child(level_node_list[l][node_idx])
        self.root = level_node_list[0][0]
        return
        
        


'''
原始代码，构建依存图，和词典图，词典图
'''
from ltp import LTP
import tensorflow as tf
import numpy as np
ltp = LTP()
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27/', lang='zh')
# 截断128维度句子
class Parser:

    def __init__(self, sentence, max_seq_length, max_ngram_length, word2id, grams, d_parser='ltp' ):
        self.sentence = sentence
        self.max_seq_length = max_seq_length
        self.max_ngram_length = max_ngram_length 
        self.d_parser = d_parser
        self.word2id = word2id


        

    def execute(self):
        b = []
        if self.d_parser == 'ltp':
            b.append(self.sentence)
            seg, hidden = ltp.seg(b)
            dep = ltp.dep(hidden)
            word_tree = dep[0]
       
            a = seg[0]
        elif self.d_parser == 'st':
            
            word_tree = nlp.dependency_parse(self.sentence)
            a = nlp.word_tokenize(self.sentence)
            
        else:
            raise ValueError('no dependency parser,plase chack use_ltp or use_stanford!')
            
        word2char, j = {0: [0]}, 1
        for i, word in enumerate(a, start=1):
            word2char[i] = list(range(j, j + len(word)))
            j += len(word)
        # print(word2char)
        # # convert tree
        # char_tree = []
        # for arc in word_tree:
        #     dep, head, pos = arc
        #     dep_char, head_char = word2char[dep], word2char[head]
        #     for d in dep_char:
        #         for h in head_char:
        #             char_tree.append((d, h))
        # bulid matrix
        import numpy as np
        from itertools import product
        sent_len = sum((len(word) for word in a)) + 1
        # print(self.max_seq_length)
        char_tree_matrix = np.zeros((self.max_seq_length,self.max_seq_length))
        char_tree_matrix_bw = np.zeros((self.max_seq_length, self.max_seq_length))
        
        # print(char_tree_matrix)
        for arc in word_tree:
            if self.d_parser == 'ltp':
                dep, head, pos =  arc
            elif self.d_parser == 'st':
            
                pos, head, dep =  arc
            # print(dep,head,pos)
            dep_char, head_char = word2char[dep], word2char[head]
            # print(dep_char)
            # ind1, ind2 = zip(*product(dep_char, head_char))
            for d in dep_char:

                for h in head_char:
                    if d <= self.max_seq_length and h <= self.max_seq_length:
                        if h != 0:
                            # print(d - 1, h - 1)
                            char_tree_matrix[d+1 - 1, h +1- 1] = 1
                            char_tree_matrix_bw[h +1- 1, d +1- 1] = 1
        # for k in range(6):!!!!
 #             print(char_tree_matrix[k][0:7])
 #        print(word_tree)
        # np.savetxt("tree_matrix.csv", char_tree_matrix, delimiter=',')
        
        #到此语法树前向后向结束
	 
        # add dict graph从每个句子中找出n-gram ,保留在词典中的那些，然后将这些候选词和对应的字之间建立边
        word_graph = np.eye(len(self.sentence),k=-1) + np.eye(len(self.sentence),k=1) + np.eye(len(self.sentence),k=2) + np.eye(len(self.sentence),k=-2) #相邻的两个字之间有连接
        
        s = np.zeros((1,len(self.sentence))) #[ClS]将[CLS]对应的插在前面
        h = np.zeros((len(self.sentence)+1, 1)) #
        word_graph = np.concatenate((s, word_graph),0) #按列拼接
        word_graph = np.append(h, word_graph, -1)  # (len(self.sentence) +1 ) * (len(self.sentence) +1 ) #横向
        
        h = np.zeros(((len(self.sentence) +1 ), self.max_seq_length - (len(self.sentence) +1 )))
        word_graph = np.append(word_graph, h,  -1) 
        s = np.zeros((self.max_seq_length - (len(self.sentence) +1 ), self.max_seq_length))
        word_graph = np.concatenate(( word_graph, s),0)
        
       
        word_graph = word_graph.tolist()
        sen_word_id = [] #记录句子中所有字对应的候选词的id
        for i in range(len(self.sentence)):
            for j in range(self.max_ngram_length):
                if i + j > len(self.sentence):
                    break
                word = self.sentence[i:i+j+1]

                if word in self.word2id.keys(): #如果n-gram 在词典中，则记录id，以便对其初始化向量
                    # print(word)
                    sen_word_id.append(self.word2id[word])
                    print( word )
                    candidate = [0.0] * (len(word_graph)+1)
      
                    candidate[i+1] = 1.0
                    candidate[i+j+1] = 1.0
                    
                    for ii in range(len(word_graph)):
                        word_graph[ii].append(candidate[ii]) 
                    word_graph.append(candidate)
                          
        word_graph = np.array( word_graph) 
        # for k in range(0,8):
#                print(word_graph[k][128:143])
#.  
        # np.savetxt("word_graph.csv", word_graph, delimiter=',')
#         exit()
        char_tree_matrix = np.array(char_tree_matrix)
        char_tree_matrix_bw = np.array(char_tree_matrix_bw)
        # char_tree_matrix.astype(int)
        return char_tree_matrix, char_tree_matrix_bw, word_graph, sen_word_id

    # def execute_backward(self):
    #     b = []
    #     b.append(self.sentence)
    #     seg, hidden = ltp.seg(b)
    #     dep = ltp.dep(hidden)
    #     word_tree = dep[0]
    #     a = seg[0]
    #     word2char, j = {0: [0]}, 1
    #     for i, word in enumerate(a, start=1):
    #         word2char[i] = list(range(j, j + len(word)))
    #         j += len(word)
    #
    #     import numpy as np
    #     from itertools import product
    #     sent_len = sum((len(word) for word in a)) + 1
    #     char_tree_matrix = np.zeros((self.max_seq_length, self.max_seq_length))
    #     for arc in word_tree:
    #         # print(arc)
    #         dep, head, pos = arc
    #         dep_char, head_char = word2char[dep], word2char[head]
    #         # ind1, ind2 = zip(*product(dep_char, head_char))
    #         for d in dep_char:
    #             # print(d)
    #             for h in head_char:
    #                 # print(h)
    #                 if d <= self.max_seq_length and h <= self.max_seq_length:
    #                     if h != 0:
    #                         # print(h-1, d-1)
    #                         char_tree_matrix[h-1, d-1] = 1
                            
        # print(char_tree_matrix)
#         exit()
        # # 新加的
        # # 1.加单位矩阵
        # num_nodes = 128
        # identity = tf.eye(num_nodes)
        # identity = tf.expand_dims(identity, axis=0)
        # char_tree_matrix = identity + char_tree_matrix
        # char_tree_matrix.astype(int)
        # return char_tree_matrix

# a = '高腔是中国戏曲四大声腔之一。'
# b = Parser(a,6)
# print(b.execute())
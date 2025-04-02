'''
原始代码，构建依存图，和边界BIES图,在依存图中加了词，以及词语词之间的关系
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
       
            a = seg[0] #['你', '每天', '都', '要', '开心', '啊']
            # print(a)
        elif self.d_parser == 'st':
            
            word_tree = nlp.dependency_parse(self.sentence)
            a = nlp.word_tokenize(self.sentence) #分词结果['你', '每', '天', '都', '要', '开心', '啊']
            # print(a)
        else:
            raise ValueError('no dependency parser,plase chack use_ltp or use_stanford!')
            
        word2char, j = {0: [0]}, 1 #0对应的是root或者head
        for i, word in enumerate(a, start=1):
            word2char[i] = list(range(j, j + len(word)))
            j += len(word)
        # print(word2char)
 #        exit()
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
        dep_wrod2id = []
        for w in a:
            if w in self.word2id.keys():
                dep_wrod2id.append(self.word2id[w])
            else:
                dep_wrod2id.append(self.word2id['PAD'])  #分词结果的对应的id
                
        import numpy as np
        from itertools import product
        sent_len = sum((len(word) for word in a)) + 1 #加上一个root
        # print(self.max_seq_length)
        char_tree_matrix = np.zeros((self.max_seq_length + len(a),self.max_seq_length + len(a))) #在每个句子后面加上所有的分词结果长度 128+8
        char_tree_matrix_bw = np.zeros((self.max_seq_length + len(a), self.max_seq_length + len(a)))
        
        # print(char_tree_matrix)
        for arc in word_tree:
            # print(word_tree)
#             exit()
            if self.d_parser == 'ltp':
                dep, head, pos =  arc
            elif self.d_parser == 'st':
            
                pos, head, dep =  arc
            # if pos == 'ROOT' or pos == 'HED': #如果是头部就不进行操作
#                 continue
            dep_char, head_char = word2char[dep], word2char[head] #     取出每个词对应的位置
           
            for d in dep_char:
                char_tree_matrix[self.max_seq_length + dep - 1, d] = 1 #把每个词和对应的字相连,长度128，但其实位置是127
                char_tree_matrix_bw[d, self.max_seq_length + dep - 1] = 1
            for h in head_char:
                if h != 0:#如果是头部就不进行操作
                    char_tree_matrix[self.max_seq_length + head - 1, h] = 1
                    char_tree_matrix_bw[h, self.max_seq_length + head-1] = 1
            #把词和词之间的连起来
            if pos != 'ROOT' and pos != 'HED': #如果是头部就不进行操作
                char_tree_matrix[self.max_seq_length + dep - 1, self.max_seq_length + head - 1] = 1 
                char_tree_matrix_bw[self.max_seq_length + head - 1, self.max_seq_length + dep - 1] = 1
                
            for d in dep_char:
                
                for h in head_char:
                    if d <= self.max_seq_length and h <= self.max_seq_length:
                        if h != 0:#如果是头部就不进行操作
                            char_tree_matrix[d - 1 + 1, h - 1 + 1] = 1 #首先因为有一个root，所以每个字的位置都要-1，因为每个句子前面会加[CLS]因此所有的index要后移一位
                            char_tree_matrix_bw[h - 1 + 1, d - 1 + 1] = 1
        # for k in range(6):
 #             print(char_tree_matrix[k][0:7])
 #        print(word_tree)
        # np.savetxt("tree_matrix.csv", char_tree_matrix, delimiter=',')

        #到此语法树前向后向结束
        boundry2id = {'B':0,'I':1,'E':2,'S':3}
	 
        # add dict graph从每个句子中找出n-gram ,保留在词典中的那些，然后将这些字在候选词中的边界信息（BIES）和对应的字之间建立边
        word_graph = np.eye(len(self.sentence),k=-1) + np.eye(len(self.sentence),k=1) + np.eye(len(self.sentence),k=2) + np.eye(len(self.sentence),k=-2) #相邻的两个字之间有连接
        
        s = np.zeros((1,len(self.sentence))) #[ClS]将[CLS]对应的插在前面
        h = np.zeros((len(self.sentence)+1, 1)) #
        word_graph = np.concatenate((s, word_graph),0) #按列拼接
        word_graph = np.append(h, word_graph, -1)  # (len(self.sentence) +1 ) * (len(self.sentence) +1 ) #横向
        
        h = np.zeros(((len(self.sentence) +1 ), self.max_seq_length - (len(self.sentence) +1 )))
        word_graph = np.append(word_graph, h,  -1) 
        s = np.zeros((self.max_seq_length - (len(self.sentence) +1 ), self.max_seq_length))
        word_graph = np.concatenate(( word_graph, s),0)        
        #max_seq_length * max_seq_length
        
        #沿着矩阵的行拼接：
        bounary_graph = np.zeros((self.max_seq_length, len(boundry2id))) #128*4 用于边界信息
        # new_word_graph = np.append(word_graph, heng_m, -1) #128*132
        
        #沿着矩阵的列拼接
        # shu_m =np.zeros((len(boundry2id), max_seq_length + len(boundry2id))) # 4*132
        # new_word_graph = np.concatenate((new_word_graph,shu_m),0) #132*132
        
        #整个图矩阵是132*132，后4维对应着BIES边界信息
        
        # word_graph = word_graph.tolist()
        bounary_graph = bounary_graph.tolist()
        # print(len(self.sentence))
        # sen_word_id = [] #记录句子中所有字对应的边界
        for i in range(len(self.sentence)): #都以i为开始字
            for j in range(self.max_ngram_length):#以j为尾字
                if i + j > len(self.sentence):
                    break
                word = self.sentence[i:i+j+1]
                #B,I,E,S
                if word in self.word2id.keys(): #如果n-gram 在词典中，则记录id，以便对其初始化向量
                    # print(self.sentence[i])
                    if i == len(self.sentence) -1:
                        bounary_graph[i+1][boundry2id['S']] = 1 
                        continue
                    if j == 0:
                        #单字词
                        bounary_graph[i+1][boundry2id['S']] = 1 #最后一位是S，和S之间有边
                    else:
                        bounary_graph[1+1][boundry2id['B']] = 1
                        
                        for k in range(1,j):
                            #中间字z
                            bounary_graph[i + k + 1][boundry2id['I']] = 1
                #尾字
                        bounary_graph[i + j + 1 ][ boundry2id['E']] = 1
        
        
        bounary_graph = np.array(bounary_graph)  #128*4
        # print(bounary_graph[44:len(self.sentence)+2])
        new_word_graph = np.append(word_graph, bounary_graph, -1) #128*132
        # print(new_word_graph.shape)
        supply = np.zeros((len(boundry2id), len(boundry2id)))  #4*4
        # print(supply.shape)
        new_boundry = np.append(bounary_graph.T, supply, -1)  #bounary_graph转置加上4*4 = 4*132
        # print(new_boundry.shape)
        new_word_graph = np.concatenate((new_word_graph,new_boundry),0)  #132*132
        # print(new_word_graph.shape)
#         exit()
        # for k in range(0,8):
#                print(word_graph[k][128:143])
#.  
        # np.savetxt("word_graph.csv", word_graph, delimiter=',')
#         exit()
        sen_word_id = [self.word2id['B'], self.word2id['M'], self.word2id['E'], self.word2id['S']]
        char_tree_matrix = np.array(char_tree_matrix)
        char_tree_matrix_bw = np.array(char_tree_matrix_bw)
        # char_tree_matrix.astype(int)
        return char_tree_matrix, char_tree_matrix_bw, new_word_graph, sen_word_id, dep_wrod2id



# a = '高腔是中国戏曲四大声腔之一。'
# b = Parser(sentence='你每天都要开心啊', max_seq_length=10, max_ngram_length=30, word2id={}, grams={}, d_parser='ltp' )
# b.execute()
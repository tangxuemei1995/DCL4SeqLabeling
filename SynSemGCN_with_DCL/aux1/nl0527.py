'''
原始代码，构建依存图，和边界BIES图
'''
# from ltp import LTP
import tensorflow as tf
import numpy as np
# ltp = LTP()
from itertools import product
from nltk.tree import Tree
from scipy.sparse import *
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27/', lang='zh')
# 截断128维度句子

#成分分析：
# ADJP Adjective Phrase
# ADVP Adverbial Phrase
# CLP Classiﬁer Phrase
# DNP DEG Phrase
# DP Determiner Phrase
# DVP DEV phrase
# LCP Localizer Phrase
# LST List Marker
# NP Noun Phrase
# PP Prepositional Phrase
# QP Quantiﬁer Phrase
# VP Verb Phrase
def yasuo(graph):
    x = len(graph)
    y = len(graph[0])
    data,row,col = [],[],[]
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            data.append(graph[i][j])
            row.append(i)
            col.append(j)
    ys = csr_matrix((data,(row,col)),shape=(x,y))
    return ys
    
chunk_pos = {'ADJP':0, 'ADVP':1, 'CLP':2, 'DNP':3, 'DP':4, 'DVP':5, 'LCP':6, 'LST':7, 'NP':8, 'PP':9, 'QP':10, 'VP':11}
srl_pos = {'A0':0,'A1':1,'A2':2,'A3':3,'A4':4,'ADV':5,'BNF':6,'CND':7,'CRD':8,'DGR':9,'DIR':10,'DIS':11,'EXT':12,'FRQ':13,'LOC':14,'MNR':15,'PRP':16,'QTY':17,'TMP':18,'TPC':19,'PRD':20,'PSR':21,'PSE':22,'ROOT':23}
class Parser:
    def __init__(self, sentence='', max_seq_length=10, max_ngram_length=30, word2id={}, grams={}, d_parser='st' ):
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
            # print(len(self.sentence))
            a = seg[0] #['你', '每天', '都', '要', '开心', '啊']
            # print(srl)
 #            print(a)
 #            [[(1, [('A0', 0, 0), ('A1', 2, 5)]), (3, [('ARGM-TMP', 2, 2)]), (8, [('A0', 7, 7), ('A1', 9, 13)]), (13, [('A0', 9, 9), ('A0', 10, 11), ('ARGM-ADV', 12, 12)])]]
 #            ['我', '是', '昨天', '去', '的', '图书馆', '，', '我', '看见', '你', '和', '他们', '在', '说话', '。']
 #            exit()
        elif self.d_parser == 'st':
            
            word_tree = nlp.dependency_parse(self.sentence)
            a = nlp.word_tokenize(self.sentence) #分词结果['你', '每', '天', '都', '要', '开心', '啊']
        else:
            raise ValueError('no dependency parser,plase chack use_ltp or use_stanford!')
        #成分分析
        # (ROOT
        #   (CP
        #     (IP
        #       (NP (PN 你))
        #       (VP
        #         (DP (DT 每) (CLP (M 天)))
        #         (ADVP (AD 都))
        #         (VP (VV 要) (VP (VV 开心)))))
        #     (SP 啊)))
        #
        #     ['你'] NP
        #     ['每', '天', '都', '要', '开心'] VP
        #     ['每', '天'] DP
        #     ['天'] CLP
        #     ['都'] ADVP
        #     ['要', '开心'] VP
        #     ['开心'] VP
        # print(len(self.sentence))
        #生成语义角色图：
        b = []
        b.append(self.sentence)
        seg, hidden = ltp.seg(b)
        srl = ltp.srl(hidden,keep_empty=False)
        srl = srl[0]
        # print(srl)
#         print(seg)
        
        srl_tree = np.zeros((self.max_seq_length  + len(srl_pos),self.max_seq_length + len(srl_pos)))
        word2srl ={}
        for x in srl:
           word2srl[x[0]] = 'ROOT'
           for it in x[1]:
               # print(it)
               word2srl[it[1]] = it[0]
               word2srl[it[2]] = it[0]
        # print( word2srl)
        begin = 0
        a = seg[0]
        # print(len(self.sentence))
        for i in range(len(a)):
            # print('当前是第%d个词:%s'%(i,a[i]))
            if i not in word2srl.keys():
                begin += len(a[i])
                continue
            for j in range(len(a[i])):
                # print('%s是词%s的第%d个字'%(a[i][j],a[i], j))
                begin += 1
                # print('当前字"%s"的在句子中位置为%d'%(a[i][j],begin))
                key = word2srl[i]
                if key not in srl_pos.keys():
                    key = key.split('-')[1]
                if key not in srl_pos.keys():
                    raise ValueError('%s not in srl_pos!'%(key))
                
                srl_tree[self.max_seq_length + srl_pos[key], begin] = 1
                srl_tree[ begin, self.max_seq_length + srl_pos[key]] = 1
        srl_tree =yasuo(srl_tree)
        srl_word2id = list(range(0,24)) 
        

        #生成成分分析图
        word_seg = nlp.word_tokenize(self.sentence) #这里使用st的分词结果，和下面的成分分析相对应
        conts = nlp.parse(self.sentence)
        # print(len(self.sentence))
 #        for char in self.sentence:
 #            print(char)
 #        exit()
        coparse = Tree.fromstring(conts)
        word2pos = {}
        for s in coparse.subtrees(lambda t: t.label() in chunk_pos.keys()):#从外层开始剥离
                        leaves = s.leaves()
                        node = s.label()
                        for lef in leaves:
                            word2pos[lef] = node
      
        constuti_tree = np.zeros((self.max_seq_length + len(chunk_pos), self.max_seq_length + len(chunk_pos)))
        begin = 0
        for i in range(len(word_seg)):
            word = word_seg[i]
            if word not in word2pos.keys():
                begin += len(word)
                continue
            # print(word)
            for j in range(len(word)):
                begin += 1
                if j != 0:#将每个词之间连接起来
                   constuti_tree[begin - 1, begin] = 1
                   constuti_tree[begin - 1, begin] = 1
                constuti_tree[self.max_seq_length + chunk_pos[word2pos[word]], begin ] = 1
                constuti_tree[begin , self.max_seq_length + chunk_pos[word2pos[word]]] = 1
                
        
            
        constuti_tree = yasuo(constuti_tree)
        #依存图
        word2char, j = {0: [0]}, 1 #0对应的是root或者head
        for i, word in enumerate(a, start=1):
            word2char[i] = list(range(j, j + len(word)))
            j += len(word)

        conts_wrod2id = list(range(0,12)) #成分分析结果的对应的id，每个句子对应的成分标记是一致的
                

        sent_len = sum((len(word) for word in a)) + 1
        # print(len(self.sentence))
 #        exit()
        char_tree_matrix = np.zeros((self.max_seq_length,self.max_seq_length))
        char_tree_matrix_bw = np.zeros((self.max_seq_length, self.max_seq_length))
        # print(word2char)
  #       print(char_tree_matrix)
        for arc in word_tree:
            if self.d_parser == 'ltp':
                dep, head, pos =  arc
            elif self.d_parser == 'st':
            
                pos, head, dep =  arc
            # print(dep,head,pos)
            dep_char, head_char = word2char[dep], word2char[head]
            
            
            for d in dep_char:

                for h in head_char:
                    if d <= self.max_seq_length and h <= self.max_seq_length:
                        if h != 0:
                            # print(d - 1, h - 1)
                            char_tree_matrix[d - 1 +1, h - 1 + 1] = 1 #因为每个句子前面会加[CLS]因此所有的index要后移一位
                            char_tree_matrix_bw[h - 1 + 1, d - 1 + 1] = 1
        char_tree_matrix = yasuo(char_tree_matrix)
        char_tree_matrix_bw = yasuo(char_tree_matrix_bw)
        # for k in range(6):
 #             print(char_tree_matrix[k][0:7])
 #        print(word_tree)
        # np.savetxt("tree_matrix.csv", char_tree_matrix, delimiter=',')

        #到此语法树前向后向结束
        max_word_size = 160
        word_attention = np.zeros((self.max_seq_length,max_word_size),dtype=int) #
        
        # s = np.zeros((1,len(self.sentence))) #[ClS]将[CLS]对应的插在前面
        # h = np.zeros((len(self.sentence)+1, 1)) #
        # word_attention = np.concatenate((s, word_attention),0) #按列拼接
        # word_attention = np.append(h, word_attention, -1)  # (len(self.sentence) +1 ) * (len(self.sentence) +1 ) #横向
        #
        # h = np.zeros(((len(self.sentence) +1 ), self.max_seq_length - (len(self.sentence) +1 )))
        # word_attention = np.append(word_attention, h,  -1)
        # s = np.zeros((self.max_seq_length - (len(self.sentence) +1 ), self.max_seq_length))
        # word_attention = np.concatenate(( word_attention, s),0)
        
        
        word_attention = word_attention.tolist()
        sen_word_id = [] #记录句子中所有字对应的候选词的id
        position = 0
        for i in range(len(self.sentence)):
            for j in range(self.max_ngram_length):
                if i + j > len(self.sentence):
                    break
                word = self.sentence[i:i+j+1]

                if word in self.word2id.keys(): #如果n-gram 在词典中，则记录id，以便对其初始化向量
                    position += 1 #有一个ngram或者word了，将
                    if  position > max_word_size - 1:
                        continue
                    else:
                        sen_word_id.append(self.word2id[word])
                        for index in range(i+1, i+j+1+1):   #将ngram里面每个字和ngram对应上
                            word_attention[index][position-1] = 1
                            word_attention[index][position-1] = 1

        word_attention = yasuo(word_attention)

        return char_tree_matrix, char_tree_matrix_bw, word_attention, sen_word_id, constuti_tree, conts_wrod2id, srl_tree, srl_word2id


if __name__=="__main__":
# #     # a = '纽西兰的盛暑，却是中国农历年期间，在奥克兰东区有许多来自亚洲的移民定居在此，既然纽西兰人可以在大热天过圣诞节，我们这些亚洲移民，又何尝不能过东方新年呢？于是，亚洲联合协会（UAA_UnitedAsianAssociation）与来自台湾的华夏协会主办了一场《庆祝东方新年》，没想到，吸引上万的人潮前来看表演，连当时的'
    a='我是昨天去的图书馆。'
#     word2id = {'pad':0,'我':1, "昨天":3, '图书馆':2}
    b = Parser(a, max_seq_length=15, max_ngram_length=10,word2id=word2id, grams={}, d_parser='ltp' )
    b.execute()
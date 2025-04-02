'''
原始代码，构建依存图，和边界BIES图
'''
from ltp import LTP
import tensorflow as tf
import numpy as np
ltp = LTP()
from itertools import product
from nltk.tree import Tree
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
        # exit()
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
        sen_word_id = [0,1,2,3]
        char_tree_matrix = np.array(char_tree_matrix)
        char_tree_matrix_bw = np.array(char_tree_matrix_bw)
        # char_tree_matrix.astype(int)
        return char_tree_matrix, char_tree_matrix_bw, new_word_graph, sen_word_id, constuti_tree, conts_wrod2id, srl_tree, srl_word2id


if __name__=="__main__":
    from scipy.sparse import coo_matrix
    # 建立稀疏矩阵
    data = [1,2,3,4]
    row = [3,6,8,2]
    col = [0,7,4,9]
    c = coo_matrix((data,(row,col)),shape=(10,10)) #构建10*10的稀疏矩阵，其中不为0的值和位置在第一个参数
    print(c)
    d = c.todense()
    print(d)
    e = coo_matrix(d) #将一个0值很多的矩阵转为稀疏矩阵
    print(e)
    import numpy as np
    aa = np.array(d)
    print(aa)
    # save
    np.save('test_save_1.npy', aa) #保存一个数组
    a_ = np.load('test_save_1.npy')
    print(type(a_))
#     # a = '纽西兰的盛暑，却是中国农历年期间，在奥克兰东区有许多来自亚洲的移民定居在此，既然纽西兰人可以在大热天过圣诞节，我们这些亚洲移民，又何尝不能过东方新年呢？于是，亚洲联合协会（UAA_UnitedAsianAssociation）与来自台湾的华夏协会主办了一场《庆祝东方新年》，没想到，吸引上万的人潮前来看表演，连当时的'
#     a='我是昨天去的图书馆，我看见你和他们在说话。'
#     b = Parser(a, max_seq_length=30, max_ngram_length=30, word2id={}, grams={}, d_parser='ltp' )
#     b.execute()
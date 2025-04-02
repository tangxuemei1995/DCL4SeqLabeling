import re
import numpy as np
import scipy.sparse as sp



def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

            if character in ['，', '。', '？', '！', '：', '；', '（', '）', '、'] and len(sentence) > 64:
                sentence_list.append(sentence)
                label_list.append(labels)
                sentence = []
                labels = []

    return sentence_list, label_list
    
def create_ngram_list(train_path, eval_path, test_path, ngram_num=5):
    '''
    创建n-gram字典，返回ngram和频次
    
    '''
    train_sentences, _ = read_tsv(train_path)
    eval_sentences, _  = read_tsv(eval_path)
    test_sentences, _  = read_tsv(test_path)
    
    all_sentences = train_sentences +  eval_sentences + test_sentences

    new_all_sentences = []


    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)
                
    ngram_dict = {}
    for input_list in new_all_sentences:
        for num in range(1, ngram_num+1) :
            if len(input_list) <= num:
                if input_list not in ngram_dict.keys():
                     ngram_dict[input_list] = 1
            else:
                for tmp in zip(*[input_list[i:] for i in range(num)]):
                    tmp = "".join(tmp)
                    if tmp not in ngram_dict.keys():
                        ngram_dict[tmp] = 1
                    else:
                        ngram_dict[tmp] += 1
    return ngram_dict


def av(train_path, eval_path, test_path, min_freq, av_threshold=5):
    '''提取训练集和验证集，测试集的高频n-gram，数据格式就是训练conll格式'''

    train_sentences, _ = read_tsv(train_path)
    eval_sentences, _ = read_tsv(eval_path)
    test_sentences, _ = read_tsv(test_path)
    
    all_sentences = train_sentences +  eval_sentences + test_sentences

    n_gram_dict = {}
    new_all_sentences = []

    ngram2av = {}

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, av_threshold+1):
                if i + n > len(sentence):
                    break
                left_index = i - 1
                right_index = i + n
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                    ngram2av[n_gram] = {'l': {}, 'r': {}}
                else:
                    n_gram_dict[n_gram] += 1
                if left_index >= 0:
                    ngram2av[n_gram]['l'][sentence[left_index]] = 1
                if right_index < len(sentence):
                    ngram2av[n_gram]['r'][sentence[right_index]] = 1
    remaining_ngram = {}
    for ngram, av_dict in ngram2av.items():
        avl = len(av_dict['l'])
        avr = len(av_dict['r'])
        av = min(avl, avr)
        if av >= av_threshold and n_gram_dict[ngram] >= min_freq:
            remaining_ngram[ngram] = n_gram_dict[ngram]

    return remaining_ngram



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    return adj_normalized
  
    
def create_graph_from_sentence_and_word_vectors(sentence, max_seq_length, max_ngram_length, word2id, grams, use_ltp, use_stanford):
    '''在主代码中直接引用本函数作为构建依存语法图的函数'''
    from .nl0527 import Parser

    if not isinstance(sentence, str):
        raise TypeError("String must be an argument")

    # sentence = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1_\2', sentence)
    if use_ltp:
        d_parser = 'ltp'
    elif use_stanford:
        d_parser = 'st'
    else:
        raise ValueError('no dependency parser,plase chack use_ltp or use_stanford!')

    parser = Parser(sentence, max_seq_length, max_ngram_length, word2id, grams, d_parser)

    A_fw, A_bw, word_graph, sen_word_id, constuti_tree, syn_word_id, srl_graph,srl_wrod_id = parser.execute()
    # A_bw = parser.execute_backward()

    return A_fw, word_graph, sen_word_id ,A_bw, constuti_tree, syn_word_id, srl_graph,srl_wrod_id

tags = ['O','B','S','I','E','[CLS]','[SEP]']
# tags = ["B-noun", "I-noun", "B-verb", "I-verb", "B-adjective", "I-adjective", "B-numeral", "I-numeral", "B-classifier", "I-classifier", "B-pronoun", "I-pronoun", "B-preposition", "I-preposition", "B-multiword-expression", "I-multiword-expression", "B-time-word", "I-time-word", "B-noun-of-locality", "I-noun-of-locality", "O", "[CLS]", "[SEP]"]
classes = ["X", "B-剧种", "I-剧种", "B-剧目", "I-剧目", "B-乐器", "I-乐器", "B-地点", "I-地点", "B-唱腔曲牌", "I-唱腔曲牌", "B-脚色行当", "I-脚色行当", "O", "[CLS]", "[SEP]"]


def build_vocab(path,path1):
    '''
    传入地址，返回词表，key为词，value为频次
    '''
    vocab = {}

    f = open(path,'r',encoding='utf-8')
    f2 = open(path1,'r',encoding='utf-8')
    all = f.read().split('\n\n') +f2.read().split('\n\n')
    for sen_pos in all:
        lines = sen_pos.strip().split('\n')
        sentence, pos = [], []
        for line in lines:
            if line == '\n' or line == '':
                continue
            line = line.strip().split('\t')
            sentence.append(line[0])
            pos.append(line[1])
        word = ''
        for i in range(len(sentence)):
            if pos[i].startswith('B'):
                if word != '':
                    if word not in  vocab.keys():
                         vocab[word] = 1
                    else:
                         vocab[word] += 1
                    word = ''
                         
                word += sentence[i]
                
                
            elif  pos[i].startswith('I'):
                word += sentence[i]
                
            elif pos[i].startswith('E'):
                    
                word += sentence[i] 
                if word not in  vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1
                word = ''
            elif pos[i].startswith('S'):
                if word != '':
                    if word not in  vocab.keys():
                         vocab[word] = 1
                    else:
                         vocab[word] += 1
                    word = ''
                    
                word = sentence[i] 
                if word not in  vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1
                word = ''
    return vocab           
    
def get_sentences(path,max_seq_length):
     f = open(path,'r',encoding='utf-8')
     text = f.read().strip().split('\n\n')
     sen_list = []

     for x in text:
         sen = []
         x = x.split('\n')
         for xx in x:
             sen.append(xx.strip().split('\t')[0])
         sen_list.append(sen[0:max_seq_length-2])
     return sen_list
     

def get_all_sentences(filename):
    file = open(filename, encoding='utf-8')
    sentences = []
    items = []
    for line in file.readlines():
        elements = line.split()
        # print(elements)
        if len(elements) == 0:
            if items != []:
                sentences.append(items)
                items = []
                continue
        word = elements[0]
        
        entity = elements[1]
        tag = 'O'
        items.append((word, tag, entity))
    sentences.append(items)
    return sentences


def decide_entity(string, prior_entity):
    if string == '*)':
        return prior_entity, ''
    if string == '*':
        return prior_entity, prior_entity
    entity = ''
    for item in classes:
        if string.find(item) != -1:
            entity = item
    prior_entity = ''
    if string.find(')') == -1:
        prior_entity = entity
    return entity, prior_entity


def get_clean_word_vector(word):

    from spacy.lang.zh import Chinese
    parser = Chinese()
    default_vector = parser('entity')[0].vector
    # print(default_vector)
    # exit()
    parsed = parser(word)
    try:
        vector = parsed[0].vector
        if vector_is_empty(vector):
            vector = default_vector
    except:
        vector = default_vector
    return np.array(vector, dtype=np.float64)


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False


def get_class_vector(class_name):
    vector = [0.] * (len(classes) + 1)
    index = len(classes)
    try:
        index = classes.index(class_name)
    except:
        pass
    vector[index] = 1.
    return vector


def get_tagging_vector(tag):
    vector = [0.] * (len(tags) + 1)
    index = len(tags)
    try:
        index = tags.index(tag)
    except:
        pass
    vector[index] = 1.
    return vector


def get_data_from_sentences(sentences):
    all_data = []
    for sentence in sentences:
        word_data = []
        class_data = []
        tag_data = []
        words = []
        for word, tag, entity in sentence:
            # print(word,tag,entity)
            words.append(word)
            
            # word_vector = get_clean_word_vector(word)
            # word_data.append(word_vector)

            # tag_vector = get_tagging_vector(tag)
#             tag_data.append(tag_vector)
#
#             class_vector = get_class_vector(entity)
#             class_data.append(class_vector)
#    
        # all_data.append((words, word_data, tag_data, class_data))
        all_data.append((words))

    return all_data


def create_full_sentence(words):
    import re
    sentence = ''.join(words)
    return sentence


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sequence = item[0]
        length = len(sequence)
        try:
            size_to_data_dict[length].append(item)
        except:
            size_to_data_dict[length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets
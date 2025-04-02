import os
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import aux1
import scipy.sparse as sp
import numpy as np




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
        
def add_identity(adj):
    # adj = adj.tolist()
#     adj = np.array(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
         
    return adj
        
        
        
        
def build_graph_new(mode, data, graph_max_length_dep, graph_max_length_word, graph_max_length_syn, word2id, index, args):
    '''mode = 'test'/'dev'/'train' data : 如果图不存在，构建图，并将图保存'''
    # os.makedirs(os.path.join('sample_data',args.dataset_name,'graph'))
    # f1 = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_fw.npy'),'w',encoding='utf-8')
    # f2 = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_bw.npy'),'w',encoding='utf-8')

    # f4 = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_wg.npy'),'w',encoding='utf-8')
    # f5 = open(os.path.join('sample_data',args.dataset_name,'graph',mode +  '_syn_word_id.txt'),'a+',encoding='utf-8')
    # f6 = open(os.path.join('sample_data',args.dataset_name,'graph',mode +  '_syn_graph.npy'),'w',encoding='utf-8')
    # f7 = open(os.path.join('sample_data',args.dataset_name,'graph',mode +  '_srl_word_id.txt'),'a+',encoding='utf-8')
    # f8 = open(os.path.join('sample_data',args.dataset_name,'graph',mode +  '_srl_graph.npy'),'w',encoding='utf-8')
    
    # forward, backward, word_attention, candidate_id, syn_graphs, syn_word_ids, srl_graphs, srl_word_ids = [], [], [], [], [], [], [], []
    '''如果每次使用的数据和词典都是一样的话，可以将图存储然后再读入，每次都构建图会很费时间'''
    
    # datapath = os.path.join('sample_data', args.dataset_name)
    # if args.use_weight:
#         grams = aux1.create_ngram_list(datapath +'/train.txt' , datapath +'/dev.txt', datapath +'/test.txt', ngram_num=7)
#     else:
    grams = []
    # count = 0
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw'))
        
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'bfw') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'bfw'))
        
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'wgrh') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'wgrh'))
        
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'syn') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'syn'))
        
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'srl') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'srl'))
    if not os.path.exists(os.path.join('sample_data',args.dataset_name,'graph',mode,'wordid') ):
        os.makedirs(os.path.join('sample_data',args.dataset_name,'graph',mode,'wordid'))
    # for j in range():
    item = data
    sentence = item.text_a.replace(' ','')
         
    A_fw, word_graph, sen_word_id, A_bw ,syn_graph, syn_word_id, srl_graph, srl_word_id = aux1.create_graph_from_sentence_and_word_vectors(sentence, args.max_seq_length, args.max_ngram_length, word2id, grams, args.use_ltp, args.use_st)
        
         # word_graph_ = word_graph.toarray()
  #        syn_graph_ = syn_graph.toarray()
  #        A_fw_ = A_fw.toarray()
  #        A_bw_ = A_bw.toarray()
  #        srl_graph_ = srl_graph.toarray()
         # word_graph = add_identity(word_graph)
    # A_fw = add_identity(A_fw)
 #    A_bw = add_identity(A_bw)
 #    syn_graph = add_identity(syn_graph)
 #    srl_graph =  add_identity(srl_graph)
         
         # word_attention.append(word_graph)
#          forward.append(A_fw)
#          backward.append(A_bw)
#          candidate_id.append(sen_word_id)
#          syn_graphs.append(syn_graph)
#          syn_word_ids.append(syn_word_id)
#          srl_graphs.append(srl_graph)
#          srl_word_ids.append(srl_word_id)
#
    A_fw = coo_matrix(A_fw)
    word_graph = coo_matrix(word_graph)
    A_bw = coo_matrix(A_bw)
    syn_graph = coo_matrix(syn_graph)
    srl_graph = coo_matrix(srl_graph)
         
    A_fw = add_identity(A_fw)
    A_bw = add_identity(A_bw)
    syn_graph = add_identity(syn_graph)
    srl_graph =  add_identity(srl_graph)

         
    np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') + '/' + str(index)+'.npy', A_fw)
         # a_ = np.load(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') + '/' + str(count)+'.npy', allow_pickle=True)
    np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'bfw') + '/' + str(index)+'.npy', A_bw)
    np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'wgrh') + '/' + str(index)+'.npy',word_graph)
    np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'syn') + '/' + str(index)+'.npy', syn_graph)
    np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'srl') + '/' + str(index)+'.npy', srl_graph)
    count += 1
         
         # print(sen_word_id)
#          exit()
    x = ''
    f3 = open(os.path.join('sample_data',args.dataset_name,'graph', 'wordid') + '/' + str(index)+ '.txt','w',encoding='utf-8')
    
    for i in sen_word_id:
        x +=str(i) + ' '
    f3.write(x+'\n')
         
    # x = ''
 #    for i in syn_word_id:
 #        x +=str(i) + ' '
 #    f5.write(x+'\n')
 #
 #    x = ''
 #    for i in srl_word_id:
 #        x +=str(i) + ' '
 #    f7.write(x+'\n')
 #
         # np.save(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') + '/' + str(count)+'.npy', A_fw)
         
    return A_fw, A_bw, word_graph, syn_graph, srl_graph, sen_word_id, syn_word_id, srl_word_id
         

def read_graph(mode, index, args):
    '''图已经存在，读取即可'''
    # print(mode +' graph is existing!')

    #
    candidate_ids_file = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_word_id.txt'),'r',encoding='utf-8')
    candidate_ids = candidate_ids_file.read().strip()
    
    # syn_ids_file = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_syn_word_id.txt'),'r',encoding='utf-8')
 #    syn_ids = syn_ids_file.read().strip()
 #
 #    srl_ids_file = open(os.path.join('sample_data',args.dataset_name,'graph',mode + '_srl_word_id.txt'),'r',encoding='utf-8')
 #    srl_ids = srl_ids_file.read().strip()
        # a_ = np.load(os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') + '/' + str(count)+'.npy', allow_pickle=True)
    syn_file = os.path.join('sample_data',args.dataset_name,'graph',mode,'syn') + '/' 
    srl_file = os.path.join('sample_data',args.dataset_name,'graph',mode,'srl') + '/' 
    wg_file = os.path.join('sample_data',args.dataset_name,'graph',mode,'wgrh') + '/'
    fw_file = os.path.join('sample_data',args.dataset_name,'graph',mode,'afw') + '/'
    bw_file = os.path.join('sample_data',args.dataset_name,'graph',mode,'bfw') + '/'
    

    candidate_ids = candidate_ids.split('\n')
    # syn_ids = syn_ids.split('\n')
#srl_ids = srl_ids.split('\n')
    # print(len(srl_ids))
#     print(len(syn_ids))
#     print(len(candidate_ids))
#     exit()
    forward, backward, word_attention, syngraphs, srlgraphs, candidate_id, syn_id, srl_id = [], [], [], [], [], [], [], []
    # for i in range(start_index, end_index):
    A_fw = np.load(fw_file + str(index) + '.npy', allow_pickle=True)
        
    A_fw = np.mat(A_fw)
    A_fw = np.array(A_fw)
        
        
    A_bw = np.load(bw_file + str(index) + '.npy', allow_pickle=True)
        
    A_bw = np.mat(A_bw)
    A_bw = np.array(A_bw)
        
        
        
    srl = np.load(srl_file + str(index) + '.npy', allow_pickle=True)
        
    srl = np.mat(srl)
    srl = np.array(srl)
        
        
    syn = np.load(syn_file + str(index) + '.npy', allow_pickle=True)
        
    syn = np.mat(syn)
    syn = np.array(syn)
      
        
    wg = np.load(wg_file + str(index) + '.npy', allow_pickle=True)
    wg = np.mat(wg)
    wg = np.array(wg)
        
        
        
        # wg = add_identity(wg)

        
    A_fw = csr_matrix(A_fw.all())
    A_bw = csr_matrix(A_bw.all())
    srl = csr_matrix(srl.all())
    syn = csr_matrix(syn.all())
    wg = csr_matrix(wg.all())
    A_fw = add_identity(A_fw)
    A_bw = add_identity(A_bw)
    syn = add_identity(syn)
    srl =  add_identity(srl)

    #仅适用于当前ud1
    # if srgs.dataset_name == 'ud1':
    
    sen_word_id = []
        # for x in candidate_ids[index].strip().split():
#         # sen_word_id = []
#             sen_word_id.append(int(x))
    # else:
        # print(len(sen_word_id))
        # sen_word_id = pad_sen_id(sen_word_id, graph_max_length_word)
        # candidate_id.append(sen_word_id)
   
    # for x in syn_ids[index]:
#         sen_word_id = []
#         x = x.strip().split()
#         for i in x:
#             sen_word_id.append(int(i))
#
#         syn_id.append(sen_word_id)
    syn_id = [i for i in range(0,12)]
    
    srl_id = [i for i in range(0,24)]
    # print(len(forward))
#     exit()
    return A_fw, A_bw, wg, syn, srl, sen_word_id, syn_id, srl_id
    
    
    
    

from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from sklearn.metrics import confusion_matrix 
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from smseg_helper import get_word2id, get_gram2id, get_dicts, get_dcits_from_voc
from smseg_eval import eval_sentence, cws_evaluate_word_PRF, cws_evaluate_OOV
from model_curriculum import WMSeg
import datetime
import evaluation
import os
# import metrics
from seg_eval import eval_sentence, pos_evaluate_word_PRF, cws_evaluate_OOV, cws_evaluate_word_PRF
from spl import TrainingScheduler, calculate_spl_weights

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def class_metrics(true_cls,pre_cls):
    # pre_list = []
    # for x in pre_cls:
    #     pre_list += x  
    ma = confusion_matrix(true_cls,pre_cls)
    co = 0
    for i in range(len(true_cls)):
        if true_cls[i] == pre_cls[i]:
            co += 1
    print('混淆阵\n:',ma)
    return co/len(true_cls)

def write_dict(name,word2id,output_model_dir):
    if not os.path.exists(output_model_dir):
        os.mkdir(output_model_dir)
    f = open(output_model_dir+'/' + name +'_voc.txt', 'w',encoding= 'utf-8')
    for key in word2id.keys():
        f.write(key + '\t' + str(word2id[key]) + '\n') 
    
    
def train(args):

    if args.use_bert and args.use_zen and args.use_lstm and args.use_trans:
        raise ValueError('We cannot use both BERT, ZEN, LSTM, TRANSFORMER')

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # if args.local_rank == -1 or args.no_cuda:
   #      device = torch.device("cpu")
   #      # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
   #      n_gpu = 0
   #      # n_gpu = torch.cuda.device_count()
   #  else:
   #      torch.cuda.set_device(args.local_rank) #local rank 进程GPU编号
   
    device = torch.device("cuda:0")
    n_gpu = 1
 

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.dataset_name is None:
        raise Warning('dataset name(equal to model name) is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./output', args.dataset_name + '_' + args.model_set)

    word2id = get_word2id(args.train_data_path) #the train set word -> dict，key:word value:id
    logger.info('# of word in train: %d: ' % len(word2id))
    
    # dataset_map = {'MSR':0,'SGW':1,'ZGW':2,'JGW':3}
    # id2dataset = {}
    # for key in dataset_map.keys():
#         id2dataset[dataset_map[key]] = key
#
    
    # if args.use_dict:
#         '''
#      use dict，only use frequency > 1 word
#         '''
#         path0 = './sample_data/msr_vocab.txt'
#         path1 = './sample_data/shanggu_vocab.txt'
#         path2 = './sample_data/jingu_vocab.txt'
#path3 = './sample_data/zhonggu_vocab.txt'
        # dicts2id, big_word2id = get_dcits_from_voc(dataset_map,path0,path1,path2,path3)
        
        # OOV_dicts2id,_ = get_dicts(args.train_data_path, args.eval_data_path,dataset_map) #from train and dev extract dicts
        # logger.info('# there are : %d  dictionaries' % len(dicts2id))
 #        for key in dicts2id.keys():
 #            logger.info(key + '#  have ' + str(len(dicts2id[key]))+ ' words' )
 #            write_dict(key,dicts2id[key],output_model_dir)
 #        # exit()
 #    else:
 #        dicts2id = None
    
    # print(dicts2id['MSR'])
    # exit()
        
    
        
    # if args.use_memory:
#         '''
#         使用n-gram,只使用频次大于阈值的ngram
#         '''
#         if args.ngram_num_threshold <= 1:
#             raise Warning('The threshold of n-gram frequency is set to %d. '
#                           'No n-grams will be filtered out by frequency. '
#                           'We only filter out n-grams whose frequency is lower than that threshold!'
#                           % args.ngram_num_threshold)
#
#         gram2id = get_gram2id(args.train_data_path, args.eval_data_path,
#                               args.ngram_num_threshold, args.ngram_flag, args.av_threshold) #使用AV提取训练集和验证集/测试集ngram 然后后见ngram2id
#         logger.info('# of n-gram in memory: %d' % len(gram2id))
#     else:
#         gram2id = None

     #数据集对应的标签
    # label_list = []
    # begin = ['B','S','I','E']
    if args.dataset_name == 'zz':
        label_list = ['PAD','S_w', 'S_ns', 'S_d', 'S_v', 'S_n', 'S_p', 'B_ns', 'E_ns', 'S_nr', 'S_u', 'S_y', 'S_r', 'S_a', 'B_nr', 'E_nr', 'S_c', 'B_n', 'E_n', 'B_v', 'E_v', 'S_m', 'S_j', 'M_nr', 'S_f', 'B_t', 'E_t', 'M_n', 'B_a', 'E_a', 'B_d', 'E_d', 'B_r', 'E_r', 'M_t', 'B_m', 'E_m', 'S_q', 'B_c', 'E_c', 'S_sv', 'S_t', 'B_p', 'E_p', 'M_ns', 'B_y', 'E_y', 'M_m', 'B_w', 'E_w', 'M_r', 'M_v', 'B_nsr', 'E_nsr', 'S_wv', 'B_f', 'E_f', 'S_b', 'M_c', 'M_a', 'B_rs', 'E_rs', 'S_mr', 'B_s', 'E_s', 'B_rr', 'M_rr', 'E_rr', 'S_s', 'S_yv', 'B_nn', 'E_nn', 'B_mr', 'E_mr', 'B_rn', 'E_rn', 'B_u', 'E_u', 'S_rs', 'M_y', 'M_f']
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'ud1':
        label_list = ['PAD']
        for line in open('./sample_data/ud1/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'pku':
        label_list = ['PAD']
        for line in open('./sample_data/pku/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'ud2':
        label_list = ['PAD']
        for line in open('./sample_data/ud2/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'zx':
        label_list = ['PAD']
        for line in open('./sample_data/zx/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'ctb5':
        label_list = ['PAD']
        for line in open('./sample_data/ctb5/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'ctb9':
        label_list = ['PAD']
        for line in open('./sample_data/ctb9/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    elif  args.dataset_name == 'ctb6':
        label_list = ['PAD']
        for line in open('./sample_data/ctb6/labels.txt'):
            label_list.append(line.strip())
        label_list += ['[CLS]','[SEP]']
    else:
        if args.dataset_name == 'sg_gerneral':
            pos = ['A', 'ADV', 'DC','DF' ,'DJ', 'ASP', 'C', 'M', 'FW', 'NA','NA4','NA5','NI','NB1','NB2','NB3','NB4','NB5','NH','NG','P','T','I','VI','VT','VP','U']
        elif args.dataset_name == 'sg':
            # pos = ['A','DA','DB','DC','DD','DF','DG','DH','DJ','DL','DN','DV','ASP','C','S','FW','NF','NA1',
 #                'NA2','NA3','NA4','NA5','NI','NB1','NB2','NB3','NB4','NB5','NB5','NH','NG','P','POST','I','T','VA','VH1','VI',
 #            'VC1','VC2','VD','VE','VG','VF','VH2','VJ','VK','VM','VP','U','O']
 #            # pos = ['A', 'ADV', 'DC','DF' ,'DJ', 'ASP', 'C', 'M', 'FW', 'NA','NA4','NA5','NI','NB1','NB2','NB3','NB4','NB5','NH','NG','P','T','I','VI','VT','VP','U']
            label_list =['PAD','O', 'S_VK', 'B_NA2', 'E_NA2', 'S_VC1', 'S_NA5',
            'S_T', 'S_VP', 'B_NA3', 'E_NA3', 'S_NH', 'S_NA2', 'S_NI', 'S_VG',
             'S_P', 'S_NB2', 'S_C', 'S_NA1', 'B_VG', 'E_VG', 'S_VA', 'S_NA4',
             'B_NB2', 'E_NB2', 'B_NA1', 'E_NA1', 'S_VH1', 'B_NB1', 'E_NB1',
             'S_VH2', 'S_VF', 'B_NI', 'E_NI', 'B_NA4', 'E_NA4', 'S_NB1', 'B_VH1',
             'E_VH1', 'S_VC2', 'B_VA', 'E_VA', 'S_DC', 'B_C', 'E_C', 'S_VE',
             'S_NA3', 'S_S', 'S_NF', 'S_NG', 'S_DD', 'S_DB', 'B_NH', 'E_NH',
              'I_NA1', 'B_VI', 'E_VI', 'S_DA', 'S_NB3', 'I_VI', 'B_NA5',
               'E_NA5', 'S_DV', 'B_VF', 'E_VF', 'S_VM', 'B_NG', 'E_NG',
               'B_DJ', 'E_DJ', 'S_VD', 'B_DC', 'E_DC', 'B_VP', 'E_VP',
                'I_NB1', 'S_NB4', 'S_VJ', 'B_NB3', 'E_NB3', 'S_DL',
                'S_DF', 'S_DH', 'S_DN', 'B_U', 'I_U', 'E_U', 'B_VK',
                'E_VK', 'B_VC1', 'E_VC1', 'S_A', 'B_NB4', 'I_NB4',
                'E_NB4', 'S_U', 'B_DD', 'E_DD', 'I_NI', 'B_VH2', 'E_VH2',
                'I_NG', 'S_NB5', 'B_DN', 'E_DN', 'B_VM', 'E_VM', 'I_NB2',
                'B_S', 'E_S', 'B_I', 'E_I', 'B_NB5', 'E_NB5', 'B_VE', 'I_VE',
                'E_VE', 'I_NB5', 'I_NA5', 'B_A', 'E_A', 'I_VH1', 'I_NA3',
                'B_DH', 'E_DH', 'I_NB3', 'B_T', 'E_T', 'B_DF', 'E_DF', 'I_C',
                'B_VC2', 'E_VC2', 'S_I', 'I_DJ', 'I_NH', 'I_NA2', 'B_DV', 'E_DV',
                'B_VJ', 'E_VJ', 'B_DA', 'E_DA', 'S_DG', 'B_DB', 'E_DB', 'S_DJ',
                 'I_VG', 'B_DG', 'E_DG', 'I_VC1', 'I_I', 'I_VP', 'B_NF', 'E_NF',
                 'B_P', 'E_P', 'I_VK', 'B_DL', 'E_DL', 'I_NA4', 'I_VA', 'S_H1',
                 'I_VM', 'B_VD', 'E_VD', 'S_O', 'S_SHI', 'I_DA', 'I_DH', 'I_DB',
                  'I_S', 'I_VH2', 'I_P', 'B_PROP', 'I_PROP', 'E_PROP', 'I_DG',
                  'S_VI', 'S_ON', 'I_DV', 'I_VJ', 'I_VF', 'I_DF', 'I_A', 'S_T，',
                  'S_FW', 'I_VC2', 'I_T', 'I_DD', 'B_ATTR', 'I_ATTR', 'E_ATTR',
                  'I_DL', 'S_PROP', 'I_DN', 'B_OTHERS,', 'I_OTHERS,', 'E_OTHERS,', 'I_DC']
            label_list += ['[CLS]','[SEP]']
    # label_list = ["O", "B", "I", "E", "S", "[CLS]", "[SEP]"] #tag only segmentation, no pos
    label_map = {label: i for i, label in enumerate(label_list, 0)}
    # print(label_map )
#     exit()
    id2label = {}
    for key in label_map.keys():
        id2label[label_map[key]] = key  
    
    # print(len(label_list),len(label_map),len(id2label))
#     exit()
    #读取部首
    # radical2unicode, radical2id = get_radical_dic('./sample_data/radical.txt')
    
    
    voc2id = {'PAD': 0} #用于构建词表attention的词典
    index= 1
    if os.path.isfile('sample_data/' + args.dataset_name + '/' + args.voc):
        for line in open('sample_data/' + args.dataset_name + '/' + args.voc):
            word = line.strip().split('\t')[0]
            if word not in voc2id.keys():
                voc2id[word] = index
                index += 1
    logger.info('# 用于构建词点attention的词有: %d个' % len(voc2id))
    
    
    
    if args.old_model:
        print("read old model!")
        seg_model_checkpoint = torch.load(args.old_model)
        seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)
    else:
        hpara = WMSeg.init_hyper_parameters(args)
        # print(hpara)
 #        exit()
        seg_model = WMSeg(word2id, label_map, voc2id, hpara, args)

    train_examples = seg_model.load_data(args.train_data_path, label_list, args)

    eval_examples = seg_model.load_data(args.eval_data_path, label_list, args)
    
    num_labels = seg_model.num_labels
    
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    
    
    

    
    
    total_params = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    # if args.local_rank != -1:
#         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        seg_model.half()
        
    seg_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    param_optimizer = list(seg_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_oov = -1
    
    history = {'epoch': [], 'class':[]}
    # for key in dataset_map.keys():
    history['p' ] = []
    history['f' ] = []
    history['r' ] = []
    history['oov' ] = []
        
    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        scheduler_type='rootp' #['const', 'exp', 'linear', 'rootp', 'geom']
        alpha=0.3  #nitial threshold (default 0.1)'
        max_thresh=1.0 #maximum of threshold (default 2.0)
        grow_steps = args.grow_steps ##number of epoch to grow to max_thresh (default 30)
        p=2 #p for scheduler root-p (default 2)
        eta=1.1 #alpha *= eta in each step for scheduler exp (default 1.1)
        spl_type = 'linear'  #'hard', 'linear', 'log', 'mix2', 'logistic', 'poly', 'welsch', 'cauchy', 'huber', 'l1l2'
        mix2_gamma=1 #gamma in mixture2 (default 1.0)
        poly_t = 3 #t in polynomial (default 3)'

        cl_scheduler = TrainingScheduler(scheduler_type, alpha, max_thresh, grow_steps, p, eta)
        thresh = alpha
        id2train = {}
        
        last_count = 0
        if args.use_data_level:
            if args.use_bayesian:
                difficult_mode = 'bayesian'
            elif args.use_top_k_LC:
                difficult_mode = 'LC'
            elif args.use_nor_log_P:
                difficult_mode = 'log_p'
            else:
                #使用数据层排序，但是不使用模型层排序
                difficult_mode = 'bayesian'
            rank_index = []
            for line in open('./sample_data/' + args.dataset_name + '/train_rank' + difficult_mode + '.txt'):
                if line != '\n':
                    rank_index.append(int(line.strip()))
            for x in range(len(rank_index)): #将训练集按照原来data-level 的难度排列，train_samples_rank.txt 中记录的是每个sample的index
                id2train[rank_index[x]] = train_examples[rank_index[x]]
            # print(len(id2train))
#             exit()
        else:
            rank_index = [i for i in range(len(train_examples))]#这里是原始的难度排序，但是在下面更新时是剩下的训练数据按照难度的 index排序
            for i in range(len(train_examples)):
                id2train[rank_index[i]] = train_examples[i]
        
        # rank_index = [i for i in range(len(id2train))] #这里是原始的难度排序，但是在下面更新时是剩下的训练数据按照难度的 index排序
        # train_index = [i for i in range(len(id2train))]
        last_train_examples = []
        last_train_index = []
        last_rank_index = []
        update = 0
        train_index = rank_index
        # last_f = 0
        last_f = 0 #每一次增加新的数据之后，都要将f值置为0，因为是看当前这些数据是否收敛，不能跟上一次的数据对比
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # np.random.shuffle(train_examples)
            # thresh = cl_scheduler.get_next_ratio() #控制有多少数据进入训练
            
            if args.use_curri and last_f == 0:
              if rank_index != last_rank_index :
                # last_f = 0 #每一次增加新的数据之后，都要将f值置为0，因为是看当前这些数据是否收敛，不能跟上一次的数据对比
                #当使用课程学习时，要重新组织训练数据
                train_examples = []#当前使用哪些训练数据
                train_index = [] #当前训练数据在原始数据集中的index
                print(thresh)
                count = thresh * len(id2train) - last_count# #当前要使用count条数据，但是在上一轮的基础上增加，last_count记录上一轮使用了多少数据
                train_index += last_train_index #之所以要记录每条数据的index，是因为后面生成图的时候，每条句子对应的图是可以区分的
                print('上一次训练数据index：', len(train_index))

                train_examples += last_train_examples#加上原来的训练数据
                print('上一次训练数据：', len(train_examples ))
                
                print('本次需要加入的数据：', count)
#                 exit()
                for x in range(0, int(count)+1):
                    if x < len(rank_index):
                        if rank_index[x] < len(id2train):
                            train_examples.append(id2train[rank_index[x]])
                            train_index.append(rank_index[x]) #rank_index 中已经是排序过的index,下面会更新
                print('当前训练数据index：', len(train_index))
                print('当前训练数据：', len(train_examples ))
                print('当前阈值：', thresh)
                last_train_index = train_index
                last_train_examples = train_examples
                last_count = len(last_train_examples)
                last_rank_index = rank_index
            #按照阈值取相应的训练数据
            #剩下的训练数据,此轮需要难度排序的数据
                need_rank_examples = {}
                for key in id2train.keys():
                    if key not in train_index:
                        need_rank_examples[key] = id2train[key]
                print('需要排序的数据：',len(need_rank_examples))
                # print(thresh)
#                 exit()
            seg_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                seg_model.train()
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                
                batch_index = train_index[start_index: min(start_index +
                                                             args.train_batch_size, len(train_examples))]
                # print(batch_index)
#                 exit()
                # if len(batch_examples) == 0:
#                     continue
                
                train_features = convert_examples_to_features(batch_examples, batch_index, args, 'train',) #按照一个batch处理，就要记住这个batch 之内是哪些对应的数据,根据start_index 和len(batch_examples)来确定
                # exit()
                input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids = feature2input(device, train_features)
                

                max_loss, loss, seq, _ = seg_model(input_ids=input_ids, token_type_ids=segment_ids, 
                                       attention_mask=input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                        syn_ids=syn_ids, srl_ids=srl_ids, device=device)
                loss =  torch.mean(loss)
                # print(seq)
#                 exit()
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.to(device)
                    loss.backward(retain_graph=True)
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            seg_model.to(device)

                
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                seg_model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true, y_cls_true = [],[]
                y_pred, y_cls_pre = [],[]
                # label_map = {label: i for i, label in enumerate(label_list, 1)}
                label_map = {i: label for i, label in enumerate(label_list, 0)}
                print('begin dev!')
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]
                    index = [i for i in range(start_index, start_index+len(eval_batch_examples))]
                    
                    eval_features = convert_examples_to_features(eval_batch_examples, index , args, 'test')

                    input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids = feature2input(device, eval_features)

                    with torch.no_grad():
                       max_lossi1, total_loss1, tag_seq, logits  = seg_model(input_ids=input_ids, token_type_ids=segment_ids, 
                                       attention_mask=input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                        syn_ids=syn_ids, srl_ids=srl_ids, device=device)
                                    # seg_model(input_ids, segment_ids, input_mask, label_ids, label_cls_id, valid_ids, l_mask, word_ids,
                                                                                       # matching_matrix, word_mask, ngram_ids, ngram_positions,device)

                    # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
                    # logits = logits.detach().cpu().numpy()
                    # print(tag_seq)
                    logits = tag_seq.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    # print(logits)
#                     exit()
                    for i, label in enumerate(label_ids):
                        # print(label_ids)
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1: #这里说明已经是‘SEP’了，因此前面的标签里面，一定要将"SEP"放在标签的最后一位
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])

                # print(y_pred)
                sentence_all = []
                for example in eval_examples:
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    sentence_all.append(sen)
        

                if not os.path.exists(output_model_dir):
                    os.mkdir(output_model_dir)
                fr = open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8')

                y_t, y_p = [], []
                for i in range(len(y_true)):
                    for j in range(len(y_true[i])):
                            y_t.append(y_true[i][j].replace('M-', 'I-'))
                            y_p.append(y_pred[i][j].replace('M-', 'I-'))
                # print(y_t,y_p)
                (p, r, f), (pp, pr, pf) = pos_evaluate_word_PRF(y_p, y_t)
                oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id) #这里其实穿的word2id 参数不对，因为，这里是训练集和测试集所有词的都在，在这里评估的也是测试集

                fr.write('\nEPoch:\t' + str(epoch + 1) + '\nP:\t' + str(p) + '\nR:\t' + str(r) + '\nF:\t' + str(f))
                fr.write('\nEPoch:\t' + str(epoch + 1) + '\npP:\t' + str(pp) + '\npR:\t' + str(pr) + '\npF:\t' + str(pf))
                fr.write('\noov:\t' + str(oov) + '\n')

                logger.info('OOV: %f' % oov)
                history['epoch'].append(epoch)
                history['p'].append(pp)
                history['r'].append(pr)
                history['f'].append(pf)
                history['oov'].append(oov)
                # history['class'].append(class_)
                logger.info("=======entity level========")
                logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f ", epoch + 1,  pp, pr, pf, oov)
                logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f ", epoch + 1,  p, r, f, oov) 
                
    
                
                
                logger.info("=======entity level========")
                # the evaluation method of NER
                # report = classification_report(y_true, y_pred, digits=4)

                # if args.model_name is not None:
                #     if not os.path.exists(output_model_dir):
                #         os.mkdir(output_model_dir)
                #
                #     with open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8') as fr:
                #         fr.write('\nEPoch:\t' + str(epoch + 1) + '\nbest_P:\t' + str(p) +  '\nbest_R:\t' + str(r) + '\nbest_F:\t' + str(f))
                #         fr.write('\nbest_oov:\t' + str(oov) + '\n' + '\nclass:\t' + str(class_) + '\n')
                    #
                    # output_eval_file = os.path.join(args.model_name, "eval_results.txt")
                    #
                    # if os.path.exists(output_eval_file):
                    #     with open(output_eval_file, "a") as writer:
                    #         logger.info("***** Eval results *****")
                    #         logger.info("=======token level========")
                    #         logger.info("\n%s", report)
                    #         logger.info("=======token level========")
                    #         writer.write(report)

                if pf > best_f:
                    best_epoch = epoch + 1
                    best_p = pp
                    best_r = pr
                    best_f = pf
                    best_oov = oov
                    num_of_no_improvement = 0

                    if args.dataset_name:
                        with open(os.path.join(output_model_dir, 'CWS_result.txt'), "w") as writer:
                            for i in range(len(y_pred)):
                                sentence = eval_examples[i].text_a
                                seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
                                # logger.info("true: %s", seg_true_str)
                                # logger.info("pred: %s", seg_pred_str)
                                writer.write('True: %s\n' % seg_true_str)
                                writer.write('Pred: %s\n\n' % seg_pred_str)

                        best_eval_model_path = os.path.join(output_model_dir, 'model.pt')

                        if n_gpu > 1:
                            torch.save({
                                'spec': seg_model.module.spec,
                                'state_dict': seg_model.module.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                        else:
                            torch.save({
                                'spec': seg_model.spec,
                                'state_dict': seg_model.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                else:
                    num_of_no_improvement += 1
                # if (pf- last_f) < 0.01:#当前的数据的F值增加不多
                
                if args.use_curri and epoch % args.u == 0 and thresh != 1:
                    # arg.u用于控制在课程学习的每个更新数据之后，要迭代几次，通常是等待本次数据收敛，但在很多论文中通常是迭代一次

                    
                  update += 1 #更新阈值的次数
                  
                  thresh = cl_scheduler.get_next_ratio(update) #不在每个epoch更新下一次的阈值
                  
                  print('上一轮的f:', last_f)
                  print('当前轮的f:', pf)
                  if args.use_curri:
                    last_f = 0
                    '''对剩余部分的训练数据进行排序'''
                    rank_examples = []
                    init_rank_index = []
                    p_var = [] #存放最后算出来的用于排序的难度值
                    for key in need_rank_examples.keys():
                        init_rank_index.append(key)
                        # print(need_rank_examples[key])
                        rank_examples.append(need_rank_examples[key])
                    
                    seg_model.eval()
                    for start_index in range(0, len(rank_examples), args.eval_batch_size):
                        eval_batch_examples = rank_examples[start_index: min(start_index + args.eval_batch_size,
                                                                             len(rank_examples))]
                        init_index = init_rank_index[start_index: min(start_index + args.eval_batch_size,
                                                                             len(rank_examples))]
                                                                         
                        rank_features = convert_examples_to_features(eval_batch_examples, init_index , args, 'train')

                        input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids = feature2input(device, rank_features)
                        with torch.no_grad():
                            if args.use_bayesian:
                                difficult_mode = 'bayesian'
                                max_loss, sample_loss, probs = [], [], []
                                for i in range(args.dropout_times):
                                    '''测试多次，只在排序时使用'''
                                    maxloss, total_loss, _, prob = seg_model(input_ids=input_ids,
                                                                             token_type_ids=segment_ids,
                                                                             attention_mask=input_mask,
                                                                             labels=label_ids,
                                                                             valid_ids=valid_ids,
                                                                             attention_mask_label=l_mask,
                                                                             fw=fw, bw=bw, syn=syn, srl=srl, wg=wg,
                                                                             c_ids=c_ids,
                                                                             syn_ids=syn_ids, srl_ids=srl_ids,
                                                                             device=device, rank=True)
                                    max_loss.append(maxloss)
                                    sample_loss.append(total_loss)

                                    probs.append(prob)
                                    # print(prob.shape)
                                input_mask = torch.unsqueeze(l_mask, dim=2)
                                # print(l_mask.shape)
                                mask = input_mask.repeat(1, 1, len(label_list))
                                p_set = [torch.mul(mask, prob) for prob in probs]
                                # p_st[0]  batch_size, seq_lenth, label_number
                                total_max_loss = max_loss[0]
                                # print(max_loss)

                                total_max_loss = sum(max_loss)
                                aver_max_loss = torch.div(total_max_loss, args.dropout_times)  # token 最大损失的平均值

                                # 开始计算每个样本的期望
                                v0 = torch.div(sum([torch.pow(prob, 2) for prob in p_set]), args.dropout_times)
                                # v1 = torch.div(sum([torch.pow (loss for loss in sample_loss]),args.dropout_times**2)
                                v1 = torch.div(torch.pow(sum([prob for prob in p_set]), 2), args.dropout_times ** 2)
                                v = v0 - v1  # 求的每个token的变化量再每个标签上的变化量
                                v = torch.sum(v, -1)
                                # print(v)
                                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist()  # 每个句子的真实长度
                                # print(lengths)
                                # 求的每个token的变化量
                                # 将句子中的每个token的变化量做平均，得到每个句子的变化量
                                # 同时找出每个句子中token最大的变化量
                                # v 中要去掉所有的[cls],[sep],pad
                                var = v.to('cpu').numpy()
                                # 找到句子中的最大值，并且对

                                for i in range(len(lengths)):
                                    # print(lengths[i])
                                    # exit()
                                    # print(i)
                                    true_var = var[i][1:lengths[i][0] - 1]
                                    # print(true_var)
                                    max = np.max(true_var)
                                    aver = np.mean(true_var)
                                    p_var.append(max + aver)
                                # print(p_var)
                                # exit()
                            elif args.use_top_k_LC:
                                difficult_mode = 'LC'
                                maxloss, total_loss, _, logits = seg_model(input_ids=input_ids,
                                                                           token_type_ids=segment_ids,
                                                                           attention_mask=input_mask, labels=label_ids,
                                                                           valid_ids=valid_ids,
                                                                           attention_mask_label=l_mask,
                                                                           fw=fw, bw=bw, syn=syn, srl=srl, wg=wg,
                                                                           c_ids=c_ids,
                                                                           syn_ids=syn_ids, srl_ids=srl_ids,
                                                                           device=device, rank=True)
                                # prob中是每个样本每个token对应的概率   batch_size * sequende_length*label_number
                                prob = F.softmax(logits, -1)
                                # print(prob.size())
                                prob, index = torch.max(prob, 2)  # 找到每个token对应的最大概率标签的概率

                                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist() 

                                one_p = torch.ones_like(prob)
                                uncertainty = one_p - prob
                                uncertainty = uncertainty.to('cpu').numpy()
                                # print(type(uncertainty))
                                # exit()
                                n = 5
                                for i in range(len(lengths)):
                                    # print(uncertainty[i])
                                    # print(lengths[i])
                                    unc = uncertainty[i][1:lengths[i] - 1]  # 找到句子中真实token对应的所有概率
                                    import heapq
                                    if lengths[i] < n:
                                        # max_n_uncer = heapq.nlargest(n, p)
                                        mean_top_n = sum(unc) / n
                                    else:
                                        max_n_uncer = heapq.nlargest(n, unc)  # 找到最大的5个值
                                        # print(max_n_uncer)
                                        mean_top_n = sum(max_n_uncer) / n
                                    p_var.append(mean_top_n)

                            elif args.use_nor_log_P:
                                difficult_mode = 'log_p'
                                maxloss, total_loss, _, logits = seg_model(input_ids=input_ids,
                                                                           token_type_ids=segment_ids,
                                                                           attention_mask=input_mask, labels=label_ids,
                                                                           valid_ids=valid_ids,
                                                                           attention_mask_label=l_mask,
                                                                           fw=fw, bw=bw, syn=syn, srl=srl, wg=wg,
                                                                           c_ids=c_ids,
                                                                           syn_ids=syn_ids, srl_ids=srl_ids,
                                                                           device=device, rank=True)
                                prob = F.softmax(logits, -1)
                                # print(prob.size())
                                prob, index = torch.max(prob, 2)  # 找到每个token对应的最大概率标签的概率

                                log_p = torch.log(prob)

                                log_p = log_p.to('cpu').numpy().tolist()
                                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist()  # 每个句子的真实长度
                                # print(lengths)
#                                 exit()

                                for i in range(len(lengths)):
                                    p = log_p[i][1:lengths[i] - 1]
                                    mean_log_p = sum(p) / lengths[i]
                                    p_var.append(0 - mean_log_p)
                            else:
                                maxloss, total_loss, _, logits = seg_model(input_ids=input_ids,
                                                                           token_type_ids=segment_ids,
                                                                           attention_mask=input_mask, labels=label_ids,
                                                                           valid_ids=valid_ids,
                                                                           attention_mask_label=l_mask,
                                                                           fw=fw, bw=bw, syn=syn, srl=srl, wg=wg,
                                                                           c_ids=c_ids,
                                                                           syn_ids=syn_ids, srl_ids=srl_ids,
                                                                           device=device, rank=True)
                                #数据依然按照原始的顺序，因此将所有的数据从小到大给赋一个值
                                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist()  # 每个句子的真实长度
                                
                                for i in range(len(lengths)):
                                    if  p_var == []:
                                        p_var.append(0)
                                    else:
                                        p_var.append(p_var[-1]+1)
                    # print(p_var)
#                     exit()
                    index2var = {}
                    for i in range(len(p_var)):
                        index2var[init_rank_index[i]] = p_var[i]
                    index2var = sorted(index2var.items(), key=lambda x: x[1])  #按照var值进行排序 [(2453, 0.01), (1312, 0.02), (1341, 0.04), (1351, 0.12)]
                    rank_index =[t[0] for t in index2var] #得到难度排序的index,从易到难，var越小，越简单
                else:
                    print('当前数据的模型未收敛，上一轮的f:', last_f)
                    print('当前数据的模型未收敛，当前轮的f:', pf)
                    last_f = pf #模型还未收敛
                    print('当前数据的模型未收敛，上一轮的f:', last_f)
                    print('当前数据的模型未收敛，当前轮的f:', pf)

            # if num_of_no_improvement >= patient: #当前模型收敛了，可以加数据

               
                
                # logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
#                 break
            
        
        logger.info("\n=======best f entity level========")
        logger.info("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f\n", best_epoch, best_p, best_r, best_f, best_oov)
        logger.info("\n=======best f entity level========")
        
        with open(os.path.join(output_model_dir, 'dev_metrics.tsv'), 'a', encoding='utf8') as fw:
            fw.write('\nEPoch:\t' + str(best_epoch) + '\nbest_P:\t' + str(best_p) +  '\nbest_R:\t' + str(best_r) + '\nbest_F:\t' + str(best_f))
            fw.write('\nbest_oov:\t' + str(best_oov) + '\n')

        if os.path.exists(output_model_dir):
            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')
                
def rank_sample(args):
    n_gpu = 1

    device = torch.device("cuda:0")
    
        # checkpoint = torch.load(path)
     #    model.load_state_dict(checkpoint['model'])
     #    optimizer.load_state_dict(checkpoint['optimizer'])

    seg_model_checkpoint = torch.load(args.eval_model)
        # print(seg_model_checkpoint['state_dict'].keys())
    #     exit()
    seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)
        # label_map = seg_model.label_map
    
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id
    voc2id = seg_model.voc2id
    id2labels = {}
    label_map = {v: k for k, v in seg_model.labelmap.items()}
    label_list = []
    for k in label_map.keys():
        id2labels[label_map[k]] = k
        label_list.append(k)
    if args.fp16:
        seg_model.half()
    eval_examples = seg_model.load_data(args.train_data_path, label_list, args)  #对训练数据排序  
    seg_model.to(device)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true, y_cls_true= [], []
    y_pred, y_cls_pre = [], []
    
    rank_index = [i for i in range(len(eval_examples))]
    p_var = [] #存放最后算出来的用于排序的难度值
    
    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        index = [i for i in range(start_index, start_index+len(eval_batch_examples))]
                                                             
        eval_features = convert_examples_to_features(eval_batch_examples, index, args, 'train')

        input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids = feature2input(device, eval_features)
        with torch.no_grad():
            if args.use_bayesian:
                difficult_mode = 'bayesian'
                max_loss, sample_loss, probs = [], [],[]
                for i in range(args.dropout_times):

                    '''测试多次，只在排序时使用'''
                    maxloss, total_loss, _ , prob = seg_model(input_ids=input_ids, token_type_ids=segment_ids,
                                           attention_mask=input_mask, labels=label_ids,
                                           valid_ids=valid_ids, attention_mask_label=l_mask,
                                           fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                            syn_ids=syn_ids, srl_ids=srl_ids, device=device,rank=True)
                    max_loss.append(maxloss)
                    sample_loss.append(total_loss)

                    probs.append(prob)
                                    # print(prob.shape)
                input_mask = torch.unsqueeze(l_mask,dim=2)
                               # print(l_mask.shape)
                mask = input_mask.repeat(1, 1, len(label_list))
                p_set = [torch.mul(mask,prob) for prob in probs ]
                               #p_st[0]  batch_size, seq_lenth, label_number
                total_max_loss = max_loss[0]
                               # print(max_loss)

                total_max_loss= sum(max_loss)
                aver_max_loss = torch.div(total_max_loss, args.dropout_times)#token 最大损失的平均值

                #开始计算每个样本的期望
                v0 = torch.div(sum([torch.pow(prob,2) for prob in p_set] ), args.dropout_times)
                               # v1 = torch.div(sum([torch.pow (loss for loss in sample_loss]),args.dropout_times**2)
                v1 = torch.div(torch.pow(sum([prob for prob in p_set]),2), args.dropout_times**2)
                v = v0 - v1 #求的每个token的变化量再每个标签上的变化量
                v = torch.sum(v,-1)
                               # print(v)
                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist() #每个句子的真实长度
                               # print(lengths)
                               #求的每个token的变化量
                               #将句子中的每个token的变化量做平均，得到每个句子的变化量
                               #同时找出每个句子中token最大的变化量
                               #v 中要去掉所有的[cls],[sep],pad
                var = v.to('cpu').numpy()
                               #找到句子中的最大值，并且对

                for i in range(len(lengths)):
                    # print(lengths[i])
                    # exit()
                                   # print(i)
                    true_var = var[i][1:lengths[i][0]-1]
                                   # print(true_var)
                    max = np.max(true_var)
                    aver = np.mean(true_var)
                    p_var.append(max+aver)
                # print(p_var)
                # exit()
            elif args.use_top_k_LC:
                difficult_mode = 'LC'
                maxloss, total_loss, _, logits = seg_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                         attention_mask=input_mask, labels=label_ids,
                                                         valid_ids=valid_ids, attention_mask_label=l_mask,
                                                         fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                                         syn_ids=syn_ids, srl_ids=srl_ids, device=device, rank=True)
                #prob中是每个样本每个token对应的概率   batch_size * sequende_length*label_number
                prob = F.softmax(logits, -1)
                # print(prob.size())
                prob, index = torch.max(prob,2)#找到每个token对应的最大概率标签的概率

                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist()  # 每个句子的真实长度

                one_p = torch.ones_like(prob)
                uncertainty = one_p - prob
                uncertainty = uncertainty.to('cpu').numpy()
                # print(type(uncertainty))
                # exit()
                n = 5
                for i in range(len(lengths)):
                    # print(uncertainty[i])
                    # print(lengths[i])
                    unc = uncertainty[i][1:lengths[i]-1] #找到句子中真实token对应的所有概率
                    import heapq
                    if lengths[i] < n:
                        # max_n_uncer = heapq.nlargest(n, p)
                        mean_top_n = sum(unc)/n
                    else:
                        max_n_uncer = heapq.nlargest(n, unc) #找到最大的5个值
                        # print(max_n_uncer )
                        mean_top_n = sum(max_n_uncer)/n
                    p_var.append(mean_top_n)

            elif args.use_nor_log_P:
                difficult_mode = 'log_p'
                maxloss, total_loss, _, logits = seg_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                         attention_mask=input_mask, labels=label_ids,
                                                         valid_ids=valid_ids, attention_mask_label=l_mask,
                                                         fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                                         syn_ids=syn_ids, srl_ids=srl_ids, device=device, rank=True)
                prob = F.softmax(logits, -1)
                # print(prob.size())
                prob, index = torch.max(prob, 2)  # 找到每个token对应的最大概率标签的概率

                log_p = torch.log(prob)
                # print(log_p)
                log_p = log_p.to('cpu').numpy().tolist()
                lengths = torch.sum(input_mask, 1).to('cpu').numpy().tolist()  # 每个句子的真实长度

                for i in range(len(lengths)):

                    p = log_p[i][1:lengths[i]-1]
                    mean_log_p = sum(p)/lengths[i]
                    p_var.append(0-mean_log_p)
                # print(p_var)
                # exit()
    
    index2var = {}
    for i in range(len(p_var)):
        index2var[rank_index[i]] = p_var[i]
    index2var = sorted(index2var.items(), key=lambda x: x[1])  #按照var值进行排序 [(2453, 0.01), (1312, 0.02), (1341, 0.04), (1351, 0.12)]
    rank_index =[ t[0]  for t  in index2var] #得到难度排序的index,从易到难，var越小，越简单
    f1 = open('./sample_data/' + args.dataset_name + '/train_rank'+ difficult_mode +'.txt', 'w', encoding='utf-8')
    for x in rank_index:
        f1.write(str(x) + '\n')



def test(args):


    n_gpu = 1

    device = torch.device("cuda:0")
    
    # checkpoint = torch.load(path)
 #    model.load_state_dict(checkpoint['model'])
 #    optimizer.load_state_dict(checkpoint['optimizer'])

    seg_model_checkpoint = torch.load(args.eval_model)
    # print(seg_model_checkpoint['state_dict'].keys())
#     exit()
    seg_model = WMSeg.from_spec(seg_model_checkpoint['spec'], seg_model_checkpoint['state_dict'], args)
    # label_map = seg_model.label_map
    
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id
    voc2id = seg_model.voc2id
    # dataset_map = seg_model.dataset_map
    
    # id2dataset, id2labels = {}, {}
    
    # for key in dataset_map.keys():
#         id2dataset[dataset_map[key]] = key
#  
    
    # big_word2id = seg_model.big_word2id
    # dicts2id = seg_model.dict2id
    # dataset_map = seg_model.dataset_map
    # OOV_dicts2id,_ = get_dicts(args.train_data_path, args.eval_data_path,dataset_map) #from train and dev extract dicts
    
    id2labels = {}
    label_map = {v: k for k, v in seg_model.labelmap.items()}
    label_list = []
    for k in label_map.keys():
        id2labels[label_map[k]] = k
        label_list.append(k)
    if args.fp16:
        seg_model.half()
    eval_examples = seg_model.load_data(args.test_data_path, label_list, args)   
    seg_model.to(device)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        seg_model = DDP(seg_model)
    elif n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true, y_cls_true= [], []
    y_pred, y_cls_pre = [], []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        index = [i for i in range(start_index, start_index+len(eval_batch_examples))]
                                                             
        eval_features = convert_examples_to_features(eval_batch_examples, index , args, 'test')

        input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids = feature2input(device, eval_features)

        with torch.no_grad():
            _, _, tag_seq,_ = seg_model(input_ids=input_ids, token_type_ids=segment_ids, 
                                       attention_mask=input_mask, labels=label_ids,
                                       valid_ids=valid_ids, attention_mask_label=l_mask,
                                       fw=fw, bw=bw, syn=syn, srl=srl, wg=wg, c_ids=c_ids,
                                        syn_ids=syn_ids, srl_ids=srl_ids, device=device)

        # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
        # logits = logits.detach().cpu().numpy()
        # print()
        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        # labels_cls = label_cls_ids.to('cpu').numpy().tolist()
        # print(logits)
#         exit()
        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1: #这里说明已经是‘SEP’了，因此前面的标签里面，一定要将"SEP"放在标签的最后一位
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    # print(y_pred)
 #    exit()
    sentence_all = []
    for example in eval_examples:
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        sentence_all.append(sen)
        

    ave_f, ave_p, ave_r, ave_oov = 0, 0, 0, 0 
        
    fr = open(os.path.join(args.eval_model[0:-9], 'test_metrics.tsv'),'a+', encoding='utf8')
    
    y_t, y_p = [], []
    for i in range(len(y_true)):
        # print(len(y_true[i]),len(y_pred[i]))
        for j in range(len(y_true[i])):
                y_t.append(y_true[i][j].replace('M-','I-'))
                y_p.append(y_pred[i][j].replace('M-','I-'))
    (p,r,f), (pp, pr, pf) = pos_evaluate_word_PRF(y_p, y_t)
    oov = cws_evaluate_OOV(y_pred, y_true, sentence_all, word2id)

    ave_f += pf
    ave_p += pp
    ave_r += pr
    ave_oov += oov
    
    print('p\t', pp)
    print('r\t', pr)
    print('f\t', pf)
    print('oov\t', oov)
    print('p\t', p)
        
    print('r\t', r)
    print('f\t', f)
    fr.write('\t' + '\nP:\t' + str(pp) +  '\nR:\t' + str(pr) + '\nF:\t' + str(pf))
    fr.write('\t' + '\nP:\t' + str(p) +  '\nR:\t' + str(r) + '\nF:\t' + str(f))
        
    fr.write('\noov:\t' + str(oov) + '\n'  + '\n\n')

    
    with open(os.path.join(args.eval_model[0:-9], 'test_result.txt'), "w") as writer:
        for i in range(len(y_pred)):
            sentence = eval_examples[i].text_a
            seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
            # logger.info("true: %s", seg_true_str)
            # logger.info("pred: %s", seg_pred_str)
            writer.write('\tTrue:\t' + seg_true_str + '\n')
            writer.write('\tPred:\t' + seg_pred_str + '\n')
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_rank",
                        action='store_true',
                        help="Whether to rank training samples.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,

                        help="The output path of segmented file")
    parser.add_argument("--dataset_name",
                        default=None,
                        type=str,
                        help="datset name of current data")
    parser.add_argument("--old_model",
                        default=None,
                        type=str,
                        help="training based on lod model")
    parser.add_argument("--use_lstm",
                        action='store_true',
                        help="Whether to use bi-lstm.")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
			   
    parser.add_argument("--use_trans",
                        action='store_true',
                        help="Whether to use transformer.")
			   
    parser.add_argument("--bert_model", default='bert-base-chinese', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
                        
    parser.add_argument('--dropout_times',
                        type=int,
                        default=3,
                        help="Number of dropout times when rank training data")

    parser.add_argument('--u',
                        type=int,
                        default=2,
                        help="Number of iteration for curriculum learning")

    parser.add_argument('--grow_steps',
                        type=int,
                        default=5,
                        help="Number of dropout times when rank training data")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--ngram_num_threshold', type=int, default=0, help="The threshold of n-gram frequency")
    parser.add_argument('--av_threshold', type=int, default=5, help="av threshold")
    parser.add_argument('--max_ngram_length', type=int, default=5,
                        help="The maximum length of n-grams to be considered.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument('--model_set', type=str, default=None, help="")
    parser.add_argument("--use_memory", action='store_true',help="Whether to run training.")
    parser.add_argument("--use_dict", action='store_true', help="use dictionary or not")
    parser.add_argument("--switch", type=str, default='', help="use classifier is hard switch or soft switch")
    parser.add_argument("--classifier", action='store_true', help="use classifier for different era dataset or not")
    parser.add_argument('--decoder', type=str, default='softmax',
                        help="the decoder to be used, should be one of softmax and crf.")
    parser.add_argument('--ngram_flag', type=str, default='av, ngram tool', help="")
    parser.add_argument('--attention_mode', type=str, default='add', help="use concat or add")
    parser.add_argument("--use_radical", action='store_true', help="Whether to use radical information.")
    parser.add_argument("--use_gcn", action='store_true', help="Whether to use gcn")
    parser.add_argument("--use_attention", action='store_true', help="Whether to use lexcicon attention")
    parser.add_argument("--use_gate", action='store_true', help="Whether to use gate in GCN")
    parser.add_argument("--use_curri", action='store_true', help="Whether to use curriculum learning")
    parser.add_argument("--use_data_level", action='store_true', help="Whether to use data-level curriculum learning")
    
    parser.add_argument("--use_ltp", action='store_true', help="Whether to use ltp")
    parser.add_argument("--use_st", action='store_true', help="Whether to use stanford corenlp")
    parser.add_argument("--voc", type=str, default='voc_train_dev_ngrams_jieba.txt', help="vocabulary type:voc_train_test.txt or voc_train_jieba.txt")
    parser.add_argument("--use_bayesian", action='store_true', help="Whether to use bayesian as difficulty")
    parser.add_argument("--use_top_k_LC", action='store_true', help="Whether to use top_k uncertainty as difficulty")
    parser.add_argument("--use_nor_log_P", action='store_true', help="Whether to use normalize log probability as difficulty")

    parser.add_argument('--save_top',
                        type=int,
                        default=1,
                        help="")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_rank:
        rank_sample(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()

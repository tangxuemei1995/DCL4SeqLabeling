from __future__ import absolute_import, division, print_function

import os

import math, copy

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter

import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import tokenization
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.crf import CRF
import transformer
from torch.autograd import Variable
from gcn import GraphConvolution
# from torch_GCN import GCN
from graph import read_graph, build_graph_new
DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'max_ngram_length': 5,
    'use_bert': False,
    'use_lstm': False,
    'use_trans': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_memory': False,
    'decoder': 'crf',
    'use_radical':False,
    'use_gcn':False
}





class WMSeg(nn.Module):

    def __init__(self, word2id, labelmap, voc2id, hpara, args):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec.pop('args')
        
        self.word2id = word2id
        self.labelmap = labelmap
        self.train_batch_size = args.train_batch_size
        self.hpara = hpara
        self.num_labels = len(self.labelmap) 
        # self.num_labels_cls = len(self.dataset_map)
        self.max_seq_length = self.hpara['max_seq_length']
        self.bert_tokenizer = None
        self.bert = None
        self.voc2id = voc2id
        self.istraining = args.do_train
        self.use_gate = args.use_gate
        self.use_gcn = args.use_gcn
        self.dropout_MC = nn.Dropout(0.8)
#         self.dropout_MC_1 = nn.Dropout(0.7)
#         self.dropout_MC_2 = nn.Dropout(0.6)
#         self.dropout_MC_3 = nn.Dropout(0.5)
        
        if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                                                    do_lower_case=self.hpara['do_lower_case'])
                # print(args.bert_model)
                self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                # print(self.bert)
                self.hpara['bert_tokenizer'] = self.bert_tokenizer
                self.hpara['config'] = self.bert.config
        else:
                self.bert_tokenizer = self.hpara['bert_tokenizer']
                self.bert = BertModel(self.hpara['config'])
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        self.feat_dim = hidden_size
        self.output_dim = hidden_size
        
        if self.use_gate:
            self.weight_fw = Parameter(torch.FloatTensor(self.feat_dim,self.output_dim))
            self.weight_bw = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.weight_syn = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.weight_srl = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.bias_fw = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_bw = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_syn = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_srl = Parameter(torch.FloatTensor(self.output_dim))
            
            self.weight_fw_ = Parameter(torch.FloatTensor(self.feat_dim,self.output_dim))
            self.weight_bw_ = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.weight_syn_ = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.weight_srl_ = Parameter(torch.FloatTensor(self.feat_dim ,self.output_dim))
            self.bias_fw_ = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_bw_ = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_syn_ = Parameter(torch.FloatTensor(self.output_dim))
            self.bias_srl_ = Parameter(torch.FloatTensor(self.output_dim))
            self.weight_fw.data.uniform_(0, 0.1) #pytorch 中Tensor.uniform_代替numpy.random.uniform, 将tensor用从均匀分布中抽样得到的值填充。
            self.bias_fw.data.uniform_(-0.1, 0.1)
            self.weight_bw.data.uniform_(0, 0.1)
            self.bias_bw.data.uniform_(-0.1, 0.1)
            self.weight_syn.data.uniform_(0, 0.1)
            self.bias_syn.data.uniform_(-0.1, 0.1)
            self.weight_srl.data.uniform_(0, 0.1)
            self.bias_srl.data.uniform_(-0.1, 0.1)
            
            self.weight_fw_.data.uniform_(0, 0.1)
            self.bias_fw_.data.uniform_(-0.1, 0.1)
            self.weight_bw_.data.uniform_(0, 0.1)
            self.bias_bw_.data.uniform_(-0.1, 0.1)
            self.weight_syn_.data.uniform_(0, 0.1)
            self.bias_syn_.data.uniform_(-0.1, 0.1)
            self.weight_srl_.data.uniform_(0, 0.1)
            self.bias_srl_.data.uniform_(-0.1, 0.1)

            # stdv = 1. / math.sqrt(self.weight.size(1))
            
        # if self.hpara['use_bert']:

            

       
        if self.use_gcn:
            self.gcn1 = GraphConvolution(self.feat_dim, self.feat_dim)
            self.gcn2 = GraphConvolution(self.feat_dim, self.feat_dim)
            self.embedding_syn = nn.Embedding(13,  self.feat_dim)
        
            self.embedding_srl = nn.Embedding(25,  self.feat_dim)

        self.classifier = nn.Linear(hidden_size * 2, self.num_labels, bias=False)
        if self.hpara['decoder'] == 'crf':
            self.crf = CRF(tagset_size=self.num_labels - 3, gpu=False)
        else:
            self.crf = None
             
        if args.do_train:
            self.spec['hpara'] = self.hpara
            # istraining = True

    def gate(self, input, g_weight, g_bias):
        logsig = nn.LogSigmoid() 
        x = torch.matmul(input, g_weight) + g_bias
        # print(x.shape)
        gate = logsig(x) #标量门
        output = input * gate
        return output
        
    def softmax(self, output, attention_mask, labels):
        logits = self.classifier(output)
        # logits= F.softmax(logits, -1)
        # logits = soft(logits)
        loss_fct = CrossEntropyLoss(ignore_index=0,reduction='none')
        total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        total_loss = total_loss.reshape(output.size(0),output.size(1))
                # print('total_loss',total_loss.size())# reshape成batch * sequence_length 16*97
        valid = attention_mask.sum(1).float()
               
        max_lossi, idx = torch.max(total_loss,1)
        total_loss = total_loss.sum(1) #将每句话中的每个字的损失加起来，这会使得句子长的损失大，句子短的损失小
    
        total_loss = torch.div(total_loss,valid)
        # print(total_loss)
#         exit()
        # print(max_lossi)
#         exit()
        tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        
        return max_lossi, total_loss, tag_seq
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                valid_ids=None, attention_mask_label=None, fw=None, 
                 bw=None, syn=None, srl=None, wg=None, c_ids=None, syn_ids=None,
                 srl_ids=None, device=None):
                
        if self.bert is not None:
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                       output_all_encoded_layers=False)
        else:
            raise ValueError()

        #完成bert 编码
        
        if self.istraining:
            sequence_output = self.dropout(sequence_output)

        #dep graph
        if self.use_gcn:
            forward = self.gcn1(sequence_output, fw)
            backward = F.relu(self.gcn1(sequence_output, bw))
            syn_embed = self.embedding_syn(syn_ids)
            syn_embedding = torch.cat([sequence_output, syn_embed],-2)
            syn_output = self.gcn1(syn_embedding, syn)
            srl_embed = self.embedding_srl(srl_ids)
            srl_embedding = torch.cat([sequence_output, srl_embed],-2)
            srl_output = self.gcn1(srl_embedding, srl)
        
            if self.use_gate:
            
                forward = self.gate(forward, self.weight_fw, self.bias_fw)
                backward = self.gate(backward, self.weight_bw, self.bias_bw)
                syn_output = self.gate(syn_output, self.weight_syn, self.bias_syn)
                srl_output = self.gate(srl_output, self.weight_srl, self.bias_srl)
            
        #整合第一层的embedding
            common = forward + backward + syn_output[:,:sequence_output.size(1),:] + srl_output[:,:sequence_output.size(1),:] 
            fisrt_layer_output = torch.cat([common, syn_output[:,sequence_output.size(1):,:], srl_output[:,sequence_output.size(1):,:]], 1) 
            fisrt_layer_output = F.relu(fisrt_layer_output)
            forward_1 = fisrt_layer_output[:,:sequence_output.size(1),:]
            backward_1 = fisrt_layer_output[:,:sequence_output.size(1),:]
            syn_1 = fisrt_layer_output[:,:syn_output.size(1),:]
            srl_1 = torch.cat([fisrt_layer_output[:,:sequence_output.size(1),:], fisrt_layer_output[:, syn_output.size(1):,:]],1)
        
        # print(forward_1.shape)
#         print(backward_1.shape)
#         print(srl_1.shape)
#         print(syn_1.shape)
#         exit()
        #secon layer
            forward = self.gcn2(forward_1, fw)
            backward = self.gcn2(backward_1 , bw)
            syn_output = self.gcn2(syn_1, syn)
            srl_output = self.gcn2(srl_1, srl)
        
            if self.use_gate:
                forward = self.gate(forward, self.weight_fw_, self.bias_fw_)
                backward = self.gate(backward, self.weight_bw_, self.bias_bw_)
                syn_output = self.gate(syn_output, self.weight_syn_, self.bias_syn_)
                srl_output = self.gate(srl_output, self.weight_srl_, self.bias_srl_)
            
            common = forward + backward + syn_output[:,:sequence_output.size(1),:] + srl_output[:,:sequence_output.size(1),:] 
            second_layer_output = torch.cat([common, syn_output[:,sequence_output.size(1):,:], srl_output[:,sequence_output.size(1):,:]], 1) 
            second_layer_output = F.relu(second_layer_output)
        
            gcn_final = second_layer_output[:,:sequence_output.size(1),:]
        
    
        
            sequence_output = torch.cat([gcn_final, sequence_output],-1)
        

        
        output1 = self.dropout_MC(sequence_output)
        # output2 = self.dropout_MC(sequence_output)
#         output3 = self.dropout_MC(sequence_output)
#         output4 = self.dropout_MC(sequence_output)
        
        

        # if self.crf is not None:
#             # crf = CRF(tagset_size=number_of_labels+1, gpu=True)
#             total_loss = self.crf.neg_log_likelihood_loss(logits, attention_mask,labels)
#             scores, tag_seq = self.crf._viterbi_decode(logits, attention_mask)
#             # Only keep active parts of the loss
        # else:
        max_lossi, total_loss, tag_seq = self.softmax(output1, attention_mask, labels)
        # max_lossi2, total_loss2, tag_seq = self.softmax(output2, attention_mask, labels)
#         max_lossi3, total_loss3, tag_seq = self.softmax(output3, attention_mask, labels)
#max_lossi4, total_loss4, tag_seq = self.softmax(output4, attention_mask, labels)
        # print(max_lossi1)
        # print(max_lossi2)
#         print(max_lossi3)
#         print(max_lossi4)
        
#             exit()
        # index_ = [0] * self.train_batch_size
        #max_lossi batch_size*1 每个案例中的token loss最大值
        # total_loss batch_size*1 #token loss的平均值
        # maxloss_ = max_lossi1 + max_lossi2 + max_lossi3 + max_lossi4
#maxloss_aver  = torch.div( maxloss_, 4)
        # print(maxloss_aver)
#         exit()
        # print(total_loss)
        # exit()
        return max_lossi, total_loss, tag_seq, max_lossi









    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['max_ngram_length'] = args.max_ngram_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_lstm'] = args.use_lstm
        hyper_parameters['use_trans'] = args.use_trans
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        hyper_parameters['decoder'] = args.decoder
        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model, args):
        spec = spec.copy()
        res = cls(args=args, **spec)
        res.load_state_dict(model)
        return res

    def load_data(self, data_path, label_list, args, do_predict=False):
        f = open(data_path, 'r', encoding='utf-8')
        text = f.read().strip().split('\n\n')
        lines = []
        count = 0
        for item in text:
            count += 1
            
            words = []
            labels = []
            # poses = []
            for x in item.split('\n'):
                if len(x.split('\t')) != 2:
                    print(x)
                    continue
                else:
                    word, label = x.split('\t')[0],  x.split('\t')[1]
                    words.append(word)
                    # if label not in label_list:
#                         print(label)
                    labels.append(label)
                    
                    # pos = pos2id['O']
 #                    poses.append(pos)
            if len(words) > 0:
                # s = ' '.join([str(pos) for pos in poses[:FLAGS.max_seq_length-2]])
                l = ' '.join([label for label in labels[:args.max_seq_length-2] if len(label) > 0])
                w = ' '.join([word for word in words[:args.max_seq_length-2] if len(word) > 0])
                lines.append([w, l ])
                
               
        examples = []
        for (i, line) in enumerate(lines):
            guid = '{}'.format(i)
            text = line[0]
            label = line[1]
            
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples
        

    def convert_examples_to_features(self, examples, start_index, end_index, args, mode='train' ):
        
        
        if args.use_gcn or args.use_attention: #使用gcn和词表attention
            # graph_file = os.path.join('sample_data', args.dataset_name, 'graph', mode, 'syn', '0.npy')
            forward, backward, word_attention, candidate_id, syn_graphs, syn_word_ids, srl_graphs, srl_word_ids = [], [], [], [], [], [], [], []
            
            graph_max_length_dep = args.max_seq_length
            graph_max_length_word = args.max_seq_length
            graph_max_length_syn = args.max_seq_length + 12
            graph_max_length_syn = args.max_seq_length + 24
            # forward, backward, word_attention, candidate_id, syngraphs, syn_word_ids, srlgraphs, srl_word_ids = [],
            # print(index)
#             exit()
        
            index = [x for x in range(start_index, end_index)]
            if len(index) != len(examples):
                print('训练数据条数不一致！')
            for i in range(len(examples)):
                # print(x)
                x = index[i]
                graph_file = os.path.join('sample_data', args.dataset_name, 'graph', mode, 'srl', str(x) + '.npy')
                # print(graph_file)
 #                exit()
                if not os.path.isfile(graph_file):
                    
                    A_fw, A_bw, wg, syn, srl, sen_word_id, syn_id, srl_id = build_graph_new(mode, examples[i], graph_max_length_dep, graph_max_length_word, graph_max_length_syn, self.voc2id, x, args)
                else:
                    A_fw, A_bw, wg, syn, srl, sen_word_id, syn_id, srl_id = read_graph(mode, x, args) 
                    # print('1')
                word_attention.append(wg)
                forward.append(A_fw)
                backward.append(A_bw)
                candidate_id.append(sen_word_id)
                syn_graphs.append(syn)
                syn_word_ids.append(syn_id)
                srl_graphs.append(srl)
                srl_word_ids.append(srl_id)
        else:
             forward = [[0]*self.max_seq_length]*self.max_seq_length] * len(examples)
             backward = [[0]*self.max_seq_length]*self.max_seq_length] * len(examples)
             
             word_attention = [[0]*self.max_seq_length]*self.max_seq_length] * len(examples)
             syn_graphs = [[0]*self.max_seq_length]*self.max_seq_length] * len(examples)
             candidate_id = [[0]*self.max_seq_length] * len(examples)
             syn_word_ids = [[0]*self.max_seq_length] * len(examples)
             srl_graphs [[0]*self.max_seq_length]*self.max_seq_length] * len(examples)
             srl_word_ids = [[0]*self.max_seq_length] * len(examples)
        # max_seq_length = min(int(max([len(e.text_a.split(' ')) for e in examples]) * 1.1 + 2), self.max_seq_length)
        max_seq_length = self.max_seq_length
        # exit()


        features = []

        tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer
        # print(len(examples),len(forward),len(srl_graphs))
#         exit()
        for ex_index, (example, fw, bw, wg, syn_g, srl_g, c_id, syn_id, srl_id) in enumerate(zip(examples, forward, backward, word_attention, syn_graphs, srl_graphs, candidate_id, syn_word_ids,srl_word_ids )):
            fw = fw.toarray()
            bw = bw.toarray()
            wg = wg.toarray()
        
            syn_g = syn_g.toarray()
            srl_g = srl_g.toarray()
            
            textlist = example.text_a.split(' ')
            labellist = example.label.split(' ')
            # print(labellist)
            # exit()
            # label_cls_id = example.label_cls
            tokens = []
            labels = []
            valid = []  ####
            label_mask = []
            if len(textlist) != len(labellist):
                print(textlist, labellist)
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):  # 每个字的第一个字符valid为1，其他的为0，因为tokenize之后会变成多个字符段，特别是英语
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []
            # radical_ids = [] #部首
            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)  # 在字符有效列表前面增加一个1
            label_mask.insert(0, 1)  # 在标签有效列表之前增加1
            label_ids.append(self.labelmap["[CLS]"])
            # radical_ids.append(self.radical2id['<NULL>'])
            # print(self.labelmap)
            # exit()
#             labels =
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    # print(labels)
                    label_ids.append(self.labelmap[labellist[i]])
                    
                # if token==  '[UNK]' :
#                     radical_ids.append(self.radical2id['<NULL>'])
#                 elif is_chinese_char(ord(token)):
#                     # print(token)
#                     radical_ids.append(self.radical2id[get_radical_idx(token,self.radical2unicode)])
#                     # print(radical_ids)
# #
#                 else:
#                     radical_ids.append(self.radical2id['<NULL>'])
            # print(radical_ids)
#             exit()
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])
            # radical_ids.append(self.radical2id['<NULL>'])
         
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            # if step == 105:
   #              print(label_ids)
   #              print(radical_ids)
   #              print(input_ids)
   #              print(label_mask)
   #              print(ntokens)
   #              print(labels)
   #              print(len(input_ids),len(radical_ids),len(label_ids),len(label_mask),len(ntokens),len(labels))
   #
                
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
                # radical_ids.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length
            # 到此所有句子相关的输入完成
            
            # print(label_ids)
#             # print(radical_ids)
#             print(input_ids)
#             print(label_mask)
#             print(ntokens)
#     # print(textlist)
# #
#             print(labels)
#             exit()
            while len(c_id) < max_seq_length:
                c_id.append(0)
            
            c_id = c_id[0:max_seq_length]
                # label_mask.append(0)
   #          print(len(c_id))
            # print(syn_g.shape)
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              fw=fw, #正向
                              bw=bw,
                              wg=wg,
                              syn_g=syn_g,
                              srl_g=srl_g,
                              c_id=c_id,
                              syn_id=syn_id,
                              srl_id=srl_id
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        # all_radical_ids = torch.tensor([f.radical_ids for f in feature], dtype=torch.long)
# >        all_label_cls_id = torch.tensor([f.label_cls_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_mask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        all_fw = torch.tensor([f.fw[0] for f in feature], dtype=torch.float)
        all_bw = torch.tensor([f.bw[0] for f in feature], dtype=torch.float)
        all_wg = torch.tensor([f.wg[0] for f in feature], dtype=torch.long)
        all_syn = torch.tensor([f.syn_g[0] for f in feature], dtype=torch.float)
        all_srl = torch.tensor([f.srl_g[0] for f in feature], dtype=torch.float)
        all_c_ids = torch.tensor([f.c_id for f in feature], dtype=torch.long)
        all_syn_ids = torch.tensor([f.syn_id[0] for f in feature], dtype=torch.long)
        all_srl_ids = torch.tensor([f.srl_id for f in feature], dtype=torch.long)
        # for f in feature:
#             print(f.valid_ids)
#             exit()
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        # label_cls_id = all_label_cls_id.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_mask_ids.to(device)
        fw = all_fw.to(device)
        bw = all_bw.to(device)
        syn = all_syn.to(device)
        srl = all_srl.to(device)
        wg = all_wg.to(device)
        c_ids = all_c_ids.to(device)
        syn_ids = all_syn_ids.to(device)
        srl_ids = all_srl_ids.to(device)
        # print(bw.shape)
# #         print(srl_ids.shape)
#         exit()
        return input_ids, input_mask, l_mask, label_ids, segment_ids, valid_ids, fw, bw, syn, srl, wg, c_ids, syn_ids, srl_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, label_cls=None, word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, 
                       valid_ids=None, label_mask=None, 
                       fw=None, bw=None,
                       wg=None, syn_g=None, 
                       srl_g=None, c_id=None,
                       syn_id=None, srl_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.fw=fw, #正向
        self.bw=bw,
        self.wg=wg,
        self.syn_g=syn_g,
        self.srl_g=srl_g,
        self.c_id=c_id,
        self.syn_id=syn_id,
        self.srl_id=srl_id
    


# def get_radical_dic(path='radical.txt'):
#     rad2uni = {} #key 是unicode，value是部首
#     rad2id = {'<NULL>' : 0} #key 是部首的unicode，value是
#     count = 1
#     for line
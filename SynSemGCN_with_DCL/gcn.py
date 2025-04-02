import math
 
import torch
 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
 
import numpy as np 

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
 
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
 
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def forward(self, input, adj):
         # support = lambda x: torch.mm(input,self.weight)
         support = []
         for t in range(input.size(0)):
             # print(input[t])
             support.append(torch.mm(input[t],self.weight)) #[seq_length,out_features] 
        
         support = torch.cat([torch.unsqueeze(x,0) for x in support],0) #x= [seq_length,out_features] ->[1, seq_length,out_features]->[batch_size, seq_length,out_features]
         
         # print(support.shape)
         output = [] 
         for j in range(support.size(0)):
 
             output.append(torch.spmm(adj[j],support[j])) #batch_size* out_features
         
         output = torch.cat([torch.unsqueeze(x,0) for x in output],0)
         if self.bias is not None:
             return output+self.bias
         else:
             return output
 
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


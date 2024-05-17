import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


'''
Transformer Implementation 

Source : https://github.com/IpsumDominum/Pytorch-Simple-Transformer
'''

class  SelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, mask = False):
        super(SelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_in, key_in, value_in):
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        key_transposed = torch.transpose(key, 1, 2)

        # caculate attention weights
        attention_weights = torch.matmul(query, key_transposed) #(n_query, n_key)
        attention_weights = attention_weights / math.sqrt(self.d_k)

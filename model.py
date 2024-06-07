import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy
from base_architecture import *
from attention import *

def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core Encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "PASS the input(and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) # trainable parameter
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        x = [10,20,30]
        self.a_2 = [1,1,1]
        self.b_2 = [0,0,0]
        mean = 20 , std = 8.16
        
        output = [-1.22, 0.00, 1.22]
        """

        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        output = self.a_2 * (x - mean) / (std + self.eps) + self.b_2

        return output

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as oppsed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        x = x + self.dropout(sublayer(self.norm(x)))
        return x


class EncoderLayer(nn.Module):
    "Encoder is made upt of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)

        return x
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m , m, src_mask))
        
        return self.sublayer[2](x, self.feed_forward)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"

    def __init__(self, d_model, d_ff, dropout = 0.1) :
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 드롭아웃 레이어를 정의합니다.
        
        # Compute the positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)  # 최대 길이와 모델 차원을 기준으로 0으로 초기화된 행렬
        position = torch.arange(0, max_len).unsqueeze(1)  # 0부터 max_len까지의 정수 시퀀스를 만듭니다. (max_len, 1) 형태의 텐서
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  
        # 위치 인코딩의 주기를 조절하기 위한 스케일링 텀을 계산합니다.
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 대해 사인 함수를 적용
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 대해 코사인 함수를 적용
        pe = pe.unsqueeze(0)  # 배치 차원을 추가합니다. (1, max_len, d_model)
        self.register_buffer("pe", pe)  # 학습하지 않는 상태로 위치 인코딩을 등록

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)  # 입력 x에 위치 인코딩 덧셈
        return self.dropout(x)  # 드롭아웃을 적용한 결과를 반환합니다.
    
def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff =2048, h=8, dropout=0.1
):
    "Helper : Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
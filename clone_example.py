import torch
import torch.nn as nn
import copy

# 모듈 정의

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

def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# SimpleModule 인스턴스 생성
module = SublayerConnection(10, 0.1)
print(module)

# clones 함수로 동일한 모듈 5개 생성
cloned_modules = clones(module, 5)

# cloned_modules는 이제 5개의 동일한 SimpleModule 인스턴스를 포함합니다
print(cloned_modules)

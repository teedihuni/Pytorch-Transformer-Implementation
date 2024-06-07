from model import *
import pandas as pd

def PE_example():
    # PositionalEncoding 레이어 초기화
    d_model = 512
    dropout = 0.1
    pos_encoder = PositionalEncoding(d_model, dropout)

    # 예시 입력 (배치 크기: 2, 시퀀스 길이: 10, 임베딩 차원: 512)
    input_tensor = torch.randn(2, 10, d_model)

    # 위치 인코딩을 더한 출력
    output_tensor = pos_encoder(input_tensor)
    print(output_tensor.shape)  # torch.Size([2, 10, 512])

def Embedding_example():
    # 임베딩 레이어 초기화
    vocab_size = 10000  # 어휘 크기
    d_model = 512  # 임베딩 차원
    embedding_layer = Embeddings(d_model, vocab_size)

    # 예시 입력 (배치 크기: 2, 시퀀스 길이: 5)
    input_indices = torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    # 임베딩 벡터 출력
    output_embeddings = embedding_layer(input_indices)
    print(output_embeddings.shape)  # torch.Size([2, 5, 512])


def FFN_example():
    x = torch.randn(64, 50, 512)  # (배치 크기, 시퀀스 길이, d_model)
    d_model = 512
    d_ff = 2048
    dropout = 0.1

    # 모델 초기화
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 예시 입력 처리
    output = ffn(x)
    print(output.shape)  # torch.Size([64, 50, 512])


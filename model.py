import torch 
import torch.nn as nn
import math 

class InputEmbedding(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        return self.embedding(X)* math.sqrt(self.d_model)   #from the paper sec 3.4 Embeddings and softmax (In the emnbedding layer, we multiply those weights by sqrt(dmodel))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model =  d_model
        self.seq_len =  seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        #Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe=pe.unsqueeze(0)  #(1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x= x + (self.pe[: :x.shape[1], :]).requires_grad_(False)
        return self.dropout(X)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float= 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameters(torch.ones(features))
        self.bias = nn.Parameters(torch.zeros(features))

    def forward(self, x):
        #X: (batch,seq_len, hidden_size)
        # Keeep the dimensions for broadcasting
        mean = x.mean(dim = -1, keepdim = True) #(batch, seq_len,1)
        #keep the dimensions for broadcasting
        std = x.std(dim = -1, keepdim = True)  # (batch, seq_len, 1)
        #eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x-mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
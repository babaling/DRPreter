import torch
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
from typing import Optional, Any, Union, Callable
from torch.nn import functional as F
from torch import Tensor

class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=.1, max_len=1024):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
    position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

    positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
    positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

    # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
    positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

    # register pe to buffer and require no grads
    self.register_buffer('pe', positional_encoding)

  def forward(self, x):
    # x: [seq_len, batch, d_model]
    # we can add positional encoding to x directly, and ignore other dimension
    x = x + self.pe[:x.size(0), ...]

    return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
  '''
  Padding, because of unequal in source_len and target_len.

  parameters:
  seq_q: [batch, seq_len]
  seq_k: [batch, seq_len]

  return:
  mask: [batch, len_q, len_k]

  '''

  batch, len_q = seq_q.size()
  batch, len_k = seq_k.size()
  # we define index of PAD is 0, if tensor equals (zero) PAD tokens
  pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch, 1, len_k]

  return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]

class FeedForwardNetwork(nn.Module):
  '''
  Using nn.Conv1d replace nn.Linear to implements FFN.
  '''
  def __init__(self, d_model, d_ff, p_drop):
    super(FeedForwardNetwork, self).__init__()
    self.ff1 = nn.Linear(d_model, d_ff)
    self.ff2 = nn.Linear(d_ff, d_model)
    # self.ff1 = nn.Conv1d(d_model, d_ff, 1)
    # self.ff2 = nn.Conv1d(d_ff, d_model, 1)
    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(p=p_drop)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x):
    # x: [batch, seq_len, d_model]
    residual = x
    # x = x.transpose(1, 2) # [batch, d_model, seq_len]
    x = self.ff1(x)
    x = self.relu(x)
    x = self.ff2(x)
    # x = x.transpose(1, 2) # [batch, seq_len, d_model]

    return self.layer_norm(residual + x)

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    '''
    Q: [batch, n_heads, len_q, d_k]
    K: [batch, n_heads, len_k, d_k]
    V: [batch, n_heads, len_v, d_v]
    attn_mask: [batch, n_heads, seq_len, seq_len]
    '''
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # [batch, n_heads, len_q, len_k]
    scores.masked_fill_(attn_mask, -1e9)

    attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]
    prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]
    return prob, attn

class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, d_k, d_v, n_heads):
    super(MultiHeadAttention, self).__init__()
    # do not use more instance to implement multihead attention
    # it can be complete in one matrix
    self.n_heads = n_heads
    self.d_k = d_k
    self.d_v = d_v

    # we can't use bias because there is no bias term in formular
    self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
    self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
    self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
    self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, input_Q, input_K, input_V, attn_mask):
    '''
    To make sure multihead attention can be used both in encoder and decoder,
    we use Q, K, V respectively.
    input_Q: [batch, len_q, d_model]
    input_K: [batch, len_k, d_model]
    input_V: [batch, len_v, d_model]
    '''
    residual, batch = input_Q, input_Q.size(0)

    # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view -->
    # [batch, len_q, n_heads, d_k,] -- transpose --> [batch, n_heads, len_q, d_k]

    Q = self.W_Q(input_Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
    K = self.W_K(input_K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
    V = self.W_V(input_V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

    attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]

    # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
    prob, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

    prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
    prob = prob.view(batch, -1, self.n_heads * self.d_v).contiguous() # [batch, len_q, n_heads * d_v]

    output = self.fc(prob) # [batch, len_q, d_model]

    return self.layer_norm(residual + output), attn

class EncoderLayer(nn.Module):

  def __init__(self, d_model, d_k, d_v, n_heads, d_ff, p_drop):
    super(EncoderLayer, self).__init__()
    self.encoder_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
    self.ffn = FeedForwardNetwork(d_model, d_ff, p_drop)

  def forward(self, encoder_input, encoder_pad_mask):
    '''
    encoder_input: [batch, source_len, d_model]
    encoder_pad_mask: [batch, n_heads, source_len, source_len]

    encoder_output: [batch, source_len, d_model]
    attn: [batch, n_heads, source_len, source_len]
    '''
    encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
    encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

    return encoder_output, attn

class Encoder(nn.Module):

  def __init__(self, d_model, d_k, d_v, nhead, num_encoder_layers, dim_feedforward, dropout, activation, layer_norm_eps):
    super(Encoder, self).__init__()
    # self.source_embedding = nn.Embedding(source_vocab_size, d_model)
    # self.positional_embedding = PositionalEncoding(d_model)
    self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, nhead,
                                              dim_feedforward, dropout) for layer in range(num_encoder_layers)])

  def forward(self, encoder_input):
    # encoder_input: [batch, source_len]
    # encoder_output = self.source_embedding(encoder_input) # [batch, source_len, d_model]
    # encoder_output = self.positional_embedding(encoder_output.transpose(0, 1)).transpose(0, 1) # [batch, source_len, d_model]

    encoder_self_attn_mask = get_attn_pad_mask(encoder_input[:,:,0], encoder_input[:,:,0]) # [batch, source_len, source_len]
    encoder_self_attns = list()
    encoder_output = encoder_input
    for layer in self.layers:
      # encoder_output: [batch, source_len, d_model]
      # encoder_self_attn: [batch, n_heads, source_len, source_len]
      encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
      encoder_self_attns.append(encoder_self_attn)

    return encoder_output, encoder_self_attns

class Transformer(nn.Module):

  def __init__(self, d_model: int = 512, d_k: int = 64, d_v: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
               dim_feedforward: int = 2048, dropout: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-5) -> None:
    super(Transformer, self).__init__()

    self.encoder = Encoder(d_model, d_k, d_v, nhead, num_encoder_layers, dim_feedforward, dropout, activation, layer_norm_eps)
    # self.projection = nn.Linear(d_model, target_vocab_size, bias=False)

  def forward(self, encoder_input):
    '''
    encoder_input: [batch, source_len]
    decoder_input: [batch, target_len]
    '''
    # encoder_output: [batch, source_len, d_model]
    # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
    encoder_output, encoder_attns = self.encoder(encoder_input)
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
    # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
    # decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input, encoder_output)
    # decoder_logits = self.projection(decoder_output) # [batch, target_len, target_vocab_size]

    # decoder_logits: [batch * target_len, target_vocab_size]
    return encoder_output, encoder_attns


# if __name__ == '__main__':

#       model = Transformer()
#       x = torch.rand((16, 34, 512))
#       model(x)
#       print(x.shape)
#       print(x)
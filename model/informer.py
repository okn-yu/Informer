import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mask import TriangularCasualMask, ProbMask
from model.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.decoder import Decoder, DecoderLayer
from model.attention import FullAttention, ProbAttention, AttentionLayer
from model.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len, factor=5, d_model=512, n_heads=8, e_layers=3,
                 d_layers=2, d_ff=512, dropout=0.0, attention_type='prob', embded='fixed', freeq='h', activation='gelu',
                 output_attention=False, distil=True, device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attention_type = attention_type
        self.output_attention = output_attention

        # 1.Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed_type='fixed', freq='h', dropout=0.0)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed_type='fixed', freq='h', dropout=0.0)

        # 2.Attention
        attention = ProbAttention if attention_type == 'prob' else FullAttention

        # 3-1. encoder_layer
        encoder_layer = EncoderLayer(
            AttentionLayer(attention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                           d_model, d_ff, dropout=dropout, activation=activation))

        # 3-2. attention_layers
        attention_layers = [encoder_layer for e in range(e_layers)]

        # 3-3.conv_layer
        conv_layers = [ConvLayer(d_model) for l in range(e_layers - 1)]

        # 3-4.norm_layer
        norm_layer = torch.nn.LayerNorm(d_model)

        # 3.Encoder
        self.encoder = Encoder(attention_layers, conv_layers, norm_layer)


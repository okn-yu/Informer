import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mask import TriangularCasualMask, ProbMask
from model.encoder import Encoder, AttentionConvLayer, ConvLayer, EncoderStack
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

        #############
        # 3.Encoder #
        #############

        # 3-1. AttentionConvLayers
        # 3-2. ConvLayers
        # 3-3. NormLayers

        attention_conv_layer = AttentionConvLayer(
            AttentionLayer(attention(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        attention_conv_layers = [attention_conv_layer for e in range(e_layers)]
        conv_layers = [ConvLayer(d_model) for l in range(e_layers - 1)]
        norm_layer = torch.nn.LayerNorm(d_model)
        self.encoder = Encoder(attention_conv_layers, conv_layers, norm_layer)

        #############
        # 4.Decoder #
        #############

        # 4-1. DecoderLayers

        decoder_layer = DecoderLayer(
            AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
            AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        decoder_layers = [decoder_layer for l in range(d_layers)]
        norm_layer = torch.nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layers, norm_layer)

        self.projection = nn.Linear(d_model, c_out, bias=True)

        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_out = self.projection(dec_out)

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]




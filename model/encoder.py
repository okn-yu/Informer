import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.down_conv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2,
                                   padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attention_mask=None):
        _x, attention = self.attention(x, x, x, attention_mask=attention_mask)
        x = x + self.dropout(_x)

        x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attention


class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers=None, norm_layers=None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layers

    def forward(self, x, attention_mask=None):
        attentions = []
        if self.conv_layers is not None:
            for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                x, attention = attention_layer(x, attention_mask=attention_mask)
                x = conv_layer(x)
                attentions.append(attention)

            x, attention = self.attention_layers[-1](x)
            attentions.append(attention)

        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x, attention_mask=attention_mask)
                attentions.append(attention)

        if self.norm is not None:
            x = self.norm(x)

        return x, attentions


class EncoderStack(nn.Module):
    def __init__(self, encoders):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, attention_mask=None):
        # x.shape = [B, L, D]
        inp_len = x.shape[1]
        x_stack = []
        attentions = []

        for encoder in self.encoders:
            if encoder is None:
                inp_len = inp_len // 2
                continue

            x, attention = encoder(x[:, -inp_len:, :])
            x_stack.append(x);
            attentions.append(attention)
            inp_len = inp_len // 2

        x_stack = torch.cat(x_stack, -2)

        return x_stack, attentions

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.transformer import Transformer
from models.vificlip import returnCLIP


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


# 主力model
class QGVL(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, config, class_names):
        super(QGVL, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        self.weight2 = torch.linspace(0, 1, 101, requires_grad=False).cuda()
        print(self.weight)
        print(self.weight2)
        self.w1 = nn.Parameter((torch.ones(1) * 0.5).cuda().requires_grad_())
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.regressor2 = nn.Linear(hidden_dim, 101)
        self.CLIP = returnCLIP(config, class_names=class_names, )

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        text_fea = self.CLIP(None, None)
        fea_grade = text_fea[-4:]
        fea_score = text_fea[:-4]
        fea_score1 = fea_score[:25]
        fea_score2 = fea_score[25:50]
        fea_score3 = fea_score[50:75]
        fea_score4 = fea_score[75:]
        fea_grade = fea_grade.unsqueeze(0).repeat(b, 1, 1)
        fea_score1 = fea_score1.unsqueeze(0).repeat(b, 1, 1)
        fea_score2 = fea_score2.unsqueeze(0).repeat(b, 1, 1)
        fea_score3 = fea_score3.unsqueeze(0).repeat(b, 1, 1)
        fea_score4 = fea_score4.unsqueeze(0).repeat(b, 1, 1)

        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(fea_grade, encode_x)

        fea_score1 = self.transformer.decoder(fea_score1, q1[:, 0:1, :])
        fea_score2 = self.transformer.decoder(fea_score2, q1[:, 1:2, :])
        fea_score3 = self.transformer.decoder(fea_score3, q1[:, 2:3, :])
        fea_score4 = self.transformer.decoder(fea_score4, q1[:, 3:, :])
        q2 = torch.cat([fea_score1,fea_score2,fea_score3,fea_score4],dim=1)

        clip_grade, clip_score = self.CLIP(q1, q2)

        s = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        s2 = self.regressor2(q2)  # (b, n, n)
        s2 = torch.diagonal(s2, dim1=-2, dim2=-1)  # (b, n)

        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        norm_s2 = torch.sigmoid(s2)
        norm_s2 = norm_s2 / torch.sum(norm_s2, dim=1, keepdim=True)
        out2 = torch.sum(self.weight2.unsqueeze(0).repeat(b, 1) * norm_s2, dim=1)

        out = (self.w1 * out) + ((1. - self.w1) * out2)
        return {'output': out, 'embed': q1, 'embed2': q2, 'clip_grade': clip_grade, 'clip_score': clip_score}

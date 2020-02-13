import torch
import torch.nn as nn
import sys
import os
from SubLayers import *
from Modules import *


import logging
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda:0' if use_cuda else 'cpu')

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation=='tanh':
            layers.append(nn.Tanh())
        elif activation=='softmax':
            layers.append(nn.Softmax(dim=-1))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask!=None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output) #40X60X16
        if non_pad_mask != None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class TemporalEncoder(nn.Module):
    #Model 1: Temporal Information encoding model
    def __init__(self, embedding_dim, seq_len, n_head, d_k, d_v, d_inner, num_layers=3, dropout=0.5):
        super(TemporalEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.num_layers=num_layers
        n_position=seq_len+1
        self.positional_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embedding_dim, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(embedding_dim, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)])
        self.spatial_embedding=nn.Linear(3, embedding_dim)
        self.seq_len=seq_len

    def forward(self, singleInput, return_attns=False):
        batch=singleInput.shape[1] # 60X40X3
        # logger.info('one point temporal input size = {}'.format(singleInput.size()))
        singleInput=singleInput.type(torch.cuda.FloatTensor)
        singleInput=singleInput.permute(1,0,2) # 40X60X3
        singleInputPos=torch.tensor([range(1,self.seq_len+1) for i in range(batch)])
        singleInputPos=singleInputPos.type(torch.cuda.LongTensor)
        singleOutput=self.spatial_embedding(singleInput)+self.positional_embedding(singleInputPos)
        singleInputPos=singleInputPos.cpu()
        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            singleOutput, enc_slf_attn = enc_layer(singleOutput)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return singleOutput, enc_slf_attn_list
        return singleOutput, #40X60X16

class SpatialEncoder(nn.Module):
    def __init__(self, embedding_dim, seq_len, n_head, d_k, d_v, d_inner, num_layers, dropout, num_markers):
        super(SpatialEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.num_layers=num_layers
        self.num_markers=num_markers
        self.temporalEncoder=TemporalEncoder(embedding_dim=embedding_dim, seq_len=seq_len, n_head=n_head,
                                             d_k=d_k, d_v=d_v, d_inner=d_inner, num_layers=num_layers, dropout=dropout)
        self.num_layers=num_layers
        self.positional_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(num_markers+1, embedding_dim, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(embedding_dim, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)])
        self.mlp1=make_mlp([seq_len*embedding_dim, 512, 128, embedding_dim],activation='tanh')
        self.mlp2 = make_mlp([num_markers * embedding_dim, 256, embedding_dim], activation='tanh')

    def forward(self, BodyInput, return_attns=False):
        # num_markers=BodyInput.size(-1)
        batch=BodyInput.shape[0]
        bodyOutput=[]
        bodyTemporalAttention=[]
        for i in range(self.num_markers):
            singleInput=BodyInput[:,:,i,:]
            singleInput=singleInput.permute(1,0,2)
            output,=self.temporalEncoder(singleInput) # 40X60X16
            output=self.mlp1(output.view(batch,-1))  #40X16
            bodyOutput.append(output)

        bodyData=torch.stack(bodyOutput,dim=2) # 40X16X37

        bodyInputPos=torch.tensor([range(1,self.num_markers+1) for i in range(batch)])
        bodyInputPos = bodyInputPos.type(torch.cuda.LongTensor)

        bodyOutput=bodyData.permute(0,2,1)+self.positional_embedding(bodyInputPos)
        bodyInputPos=bodyInputPos.cpu()

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            bodyOutput, enc_slf_attn = enc_layer(bodyOutput) # 40X37X16
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return bodyOutput, enc_slf_attn_list
        bodyOutput=self.mlp2(bodyOutput.view(batch,-1))  # 40X16

        return bodyOutput,

class GroupEncoder(nn.Module):
    def __init__(self, embedding_dim, seq_len, n_head, d_k, d_v, d_inner, num_layers, dropout, num_markers):
        super(GroupEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        # self.h_dim=h_dim
        self.num_layers=num_layers
        self.joinEncoder=SpatialEncoder(embedding_dim=embedding_dim, seq_len=seq_len, n_head=n_head,
                                             d_k=d_k, d_v=d_v, d_inner=d_inner, num_layers=num_layers, dropout=dropout, num_markers=num_markers)
        self.groupEncoder = SpatialEncoder(embedding_dim=embedding_dim, seq_len=seq_len, n_head=n_head,
                                             d_k=d_k, d_v=d_v, d_inner=d_inner, num_layers=num_layers,
                                          dropout=dropout, num_markers=num_markers)
        self.mlp1=make_mlp([4*embedding_dim, 4],activation='tanh')
        self.mlp2=make_mlp([4,4],activation='softmax')
        self.mlp3=make_mlp([4*embedding_dim,2], activation='softmax')

    def forward(self, GroupInput):
        # groupSize=GroupInput.size(-1)
        batch=GroupInput.shape[0] # GroupInput num_dataXseq_lenX4X37X3
        # logger.info('one group input size = {}'.format(GroupInput.size()))
        groupSize=3
        joinInput=GroupInput[:,:,0,:,:]
        TAtt=[]
        BAtt=[]
        # logger.info('one joining agent size = {}'.format(joinInput.size()))
        groupInputs=GroupInput[:,:,1:4,:,:]
        joinBodyOutput,=self.joinEncoder(joinInput) # 40X16
        # TAtt.append(joinTAtt)
        # BAtt.append(joinBAtt)
        allBodyOutputs=[]
        allBodyOutputs.append(joinBodyOutput)
        for i in range(groupSize):
            groupInput=GroupInput[:,:,i+1,:,:]
            groupBodyOutput,=self.groupEncoder(groupInput)
            # TAtt.append(groupTAtt)
            # BAtt.append(groupBAtt)
            allBodyOutputs.append(groupBodyOutput)
        allGroupData=torch.stack(allBodyOutputs,dim=2) # 40X16X4
        groupAttention=self.mlp1(allGroupData.reshape(batch,-1))
        groupAttention=self.mlp2(groupAttention) # 40X4
        # groupAttention=groupAttention.unsqueeze(dim=1) # 40X1X4
        groupAttentionOutput=torch.mul(allGroupData,groupAttention.unsqueeze(dim=1)) # 40X16X4
        groupAttentionOutput=groupAttentionOutput.reshape(batch,-1) # 40X64
        # logger.info('one group1 size = {}'.format(groupAttentionOutput.size()))
        groupAttentionOutput=self.mlp3(groupAttentionOutput) # 40X2
        # TAtt=torch.stack(TAtt,dim=1)
        # BAtt = torch.stack(BAtt, dim=1)
        return TAtt, BAtt, groupAttention, groupAttentionOutput



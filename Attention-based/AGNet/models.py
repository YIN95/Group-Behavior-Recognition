import torch
import torch.nn as nn
import sys
import os
# print(torch.cuda.current_device())
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

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

class TemporalEncoder(nn.Module):
    #Model 1: Temporal Information encoding model
    def __init__(self, embedding_dim, h_dim, num_layers=3, dropout=0.5):
        super(TemporalEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.h_dim=h_dim
        self.num_layers=num_layers
        self.encoder=nn.LSTM(embedding_dim,h_dim,num_layers,dropout=dropout)
        self.spatial_embedding=nn.Linear(3, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )


    def forward(self, singleInput):
        batch=singleInput.shape[1]
        # logger.info('one point temporal input size = {}'.format(singleInput.size()))
        singleInput=singleInput.type(torch.cuda.FloatTensor)
        singleInput_embedding=self.spatial_embedding(singleInput.reshape(-1,3))
        singleInput_embedding=singleInput_embedding.reshape(-1,batch,self.embedding_dim)
        state_tuple=self.init_hidden(batch)
        output, state=self.encoder(singleInput_embedding, state_tuple)
        final_h=state[0]
        return output, final_h

class SpatialEncoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, num_layers, dropout, num_markers):
        super(SpatialEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.h_dim=h_dim
        self.num_layers=num_layers
        self.num_markers=num_markers
        self.temporalEncoder=TemporalEncoder(embedding_dim=embedding_dim, h_dim=h_dim, num_layers=num_layers, dropout=dropout)
        self.con1d=nn.Conv1d(h_dim,out_channels=1,kernel_size=1,stride=1)
        self.softmax=nn.Softmax(dim=-1)
        self.mlp1=make_mlp([num_markers*h_dim, num_markers],activation='tanh')
        self.mlp2=make_mlp([num_markers,num_markers],activation='softmax')
        self.mlp3=make_mlp([num_markers,2], activation='softmax')
        # self.mlp3 = make_mlp([num_markers, 2], activation='softmax') # using CrossEntropyLoss already contains softmax

    def forward(self, BodyInput):
        # num_markers=BodyInput.size(-1)
        batch=BodyInput.shape[0]
        bodyOutput=[]
        bodyTemporalAttention=[]
        for i in range(self.num_markers):
            singleInput=BodyInput[:,:,i,:]
            singleInput=singleInput.permute(1,0,2)
            output, final_h=self.temporalEncoder(singleInput)
            output=output.permute(1,2,0) # 40X16X60
            # logger.info('one point temporal output size = {}'.format(output.size()))
            temporalAttention=self.con1d(output) # 40X1X60
            # logger.info('one point temporal output1 size = {}'.format(temporalAttention.size()))
            temporalAttention=self.softmax(temporalAttention) # 40X1X60
            # logger.info('one point temporal output2 size = {}'.format(temporalAttention.size()))
            bodyTemporalAttention.append(temporalAttention)
            temporalOutput=torch.mul(temporalAttention,output) # 40X16X60
            # logger.info('one point temporal output3 size = {}'.format(temporalOutput.size()))
            temporalOutput=torch.sum(temporalOutput,dim=2) # 40X16
            # logger.info('one point temporal output4 size = {}'.format(temporalOutput.size()))
            bodyOutput.append(temporalOutput)

        bodyData=torch.stack(bodyOutput,dim=2) # 40X16X37
        bodyAttention=self.mlp1(bodyData.reshape(batch, -1))
        bodyAttention=self.mlp2(bodyAttention) # 40X37
        # bodyAttention=bodyAttention.unsqueeze(dim=1)
        # logger.info('bodyAttention size = {}'.format(bodyAttention.size()))
        bodyAttentionOutput=torch.mul(bodyData,bodyAttention.unsqueeze(dim=1)) # 40X16X37 * 40X1X37
        bodyAttentionOutput=torch.sum(bodyAttentionOutput,dim=2) # 40X16
        # bodyAttentionOutput=bodyAttentionOutput.view(batch,-1)
        # predicts=self.mlp3(bodyAttentionOutput)
        bodyTemporalAttention=torch.cat(bodyTemporalAttention,dim=1)
        return bodyTemporalAttention, bodyAttention, bodyAttentionOutput

class GroupEncoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, num_layers, dropout, num_markers):
        super(GroupEncoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.h_dim=h_dim
        self.num_layers=num_layers
        self.joinEncoder=SpatialEncoder(embedding_dim=embedding_dim, h_dim=h_dim, num_layers=num_layers, dropout=dropout, num_markers=num_markers)
        self.groupEncoder = SpatialEncoder(embedding_dim=embedding_dim, h_dim=h_dim, num_layers=num_layers,
                                          dropout=dropout, num_markers=num_markers)
        self.mlp1=make_mlp([4*h_dim, 4],activation='tanh')
        self.mlp2=make_mlp([4,4],activation='softmax')
        self.mlp3=make_mlp([4*h_dim,2], activation='softmax')

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
        joinTAtt, joinBAtt,joinBodyOutput=self.joinEncoder(joinInput) # 40X16
        TAtt.append(joinTAtt)
        BAtt.append(joinBAtt)
        allBodyOutputs=[]
        allBodyOutputs.append(joinBodyOutput)
        for i in range(groupSize):
            groupInput=GroupInput[:,:,i+1,:,:]
            groupTAtt,groupBAtt,groupBodyOutput=self.groupEncoder(groupInput)
            TAtt.append(groupTAtt)
            BAtt.append(groupBAtt)
            allBodyOutputs.append(groupBodyOutput)
        allGroupData=torch.stack(allBodyOutputs,dim=2) # 40X16X4
        groupAttention=self.mlp1(allGroupData.reshape(batch,-1))
        groupAttention=self.mlp2(groupAttention) # 40X4
        # groupAttention=groupAttention.unsqueeze(dim=1) # 40X1X4
        groupAttentionOutput=torch.mul(allGroupData,groupAttention.unsqueeze(dim=1)) # 40X16X4
        groupAttentionOutput=groupAttentionOutput.reshape(batch,-1) # 40X64
        # logger.info('one group1 size = {}'.format(groupAttentionOutput.size()))
        groupAttentionOutput=self.mlp3(groupAttentionOutput) # 40X2
        TAtt=torch.stack(TAtt,dim=1)
        BAtt = torch.stack(BAtt, dim=1)
        return TAtt, BAtt, groupAttention, groupAttentionOutput



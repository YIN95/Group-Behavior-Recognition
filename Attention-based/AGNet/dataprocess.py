import logging
import os
import os.path
import math
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import six

logger = logging.getLogger(__name__)


def read_file(_path, delim=','):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        csvreader=csv.reader(f,delimiter=delim)
        for row in csvreader:
            row = row[0].split(',')
            row = [float(irow) for irow in row]
            data.append(row)
    return np.asarray(data)

def RepresentsInt(s):
    return re.match(r"[-+]?\d+$", s) is not None

def SlWindow(df, seq_len, slidingwindow_len, overlap=0.75):
    # Returen nslid*seqlen* all
    frame_len = df.shape[0]
    if slidingwindow_len>=frame_len:
        slidingwindow_len=frame_len

    skip=int(slidingwindow_len/seq_len)
    # seq_len = 100
    # overlap = 0.75
    move = slidingwindow_len - int(slidingwindow_len * overlap)
    left = 0
    nslide = 0
    win = []
    while left <= frame_len - slidingwindow_len:
        d = df.iloc[left:left + slidingwindow_len:skip]
        # print(len(d.values))
        win.append(torch.tensor(d.values))
        nslide += 1
        left += move
    win = torch.stack(win) # num_windows X seq_len X (37*3)
    # print(win.size())
    return win


def SplitData(loaddata, info, seq_len, slidingwindow_len, num_groupmem=4):
    loaddata.columns = info

    # order of bone marker names
    # uniqueinfo=info.unique()
    # print(uniqueinfo)

    index=[]
    data=[]

    for i in range(num_groupmem):
        index.append([ind for ind, x in enumerate(loaddata.columns) if x[0:2] == 'p' + str(i + 1)])

    for i in range(num_groupmem):
        player_data=loaddata.iloc[:, index[i]]
        sld = SlWindow(player_data, seq_len, slidingwindow_len)
        player_data_xyz=[sld[:, :, m::3] for m in range(3)]
        player_data_xyz=torch.stack(player_data_xyz, dim=3) # num_windows X seq_len X 37 X 3
        data.append(player_data_xyz)
    data=torch.stack(data) # 4 X num_windows X seq_len X 37 X 3
    data=data.permute(1,2,0,3,4)
    # print(data.size())
    return data # num_windows X seq_len X 4 X 37 X 3

def RemoveHead(read_name):
    csvFile = open(read_name)
    readerObj = csv.reader(csvFile)
    csvRows = []
    # if read_name[-5:] == '1.csv': n = 2
    # else: n = 7
    n = 2
    for row in readerObj:
        if readerObj.line_num <= n:
            continue
        csvRows.append(row)
    csvFile.close()
    return csvRows


def read_label(path, delim=','):
    label_data=[]
    with open(path, 'r', encoding="utf8", errors='ignore') as f:
        csvreader=csv.reader(f,delimiter=delim)
        for row in csvreader:
            label_data.append(row)
        f.close()
    return label_data


def ExtractBoneMarker(df):
    df.columns = df.iloc[0]
    boneMarker = df['Bone Marker']
    data = boneMarker.apply(pd.to_numeric, errors='coerce')
    infoBonMarker = boneMarker.iloc[1]
    data_boneMarker = data.iloc[5:].reset_index(drop=True) #remove title
    data_iboneMarker= data_boneMarker.interpolate(method='linear',limit_direction='both')

    # TODO fill in missing data
    # data_iboneMarker.fillna(0, inplace=True)
    if data_iboneMarker.isnull().any().any():
        # print(data_iboneMarker.isnull().any().sum())
        idnull = list(set((np.where(np.asanyarray(np.isnan(data_iboneMarker)))[1])))
        bM = np.array(infoBonMarker.iloc[idnull])
        bM = bM.flatten().tolist()
        missbM = list(set(bM))

        # change left right get index missing data
        for ind, n in enumerate(missbM):
            # player No.
            nply = n[:2]
            if n[3] == 'R':
                missbM[ind] = n[:3] + 'L' + n[4:]
            elif n[3] == 'L':
                missbM[ind] = n[:3] + 'R' + n[4:]
            elif n[3:8] == 'Waist':
                if n[9] == 'R':
                    missbM[ind] = n[:8] + 'L' + n[9:]
                if n[9] == 'L':
                    missbM[ind] = n[:8] + 'R' + n[9:]
        indmD = []
        for i in missbM:
            a = np.where(infoBonMarker == i)[0]
            indmD.extend(list(a))

        # get index chest
        for index, n in enumerate(missbM):
            missbM[index] = n[:3] + 'Chest'
        indChest = []
        for i in missbM:
            a = np.where(infoBonMarker == i)[0]
            indChest.extend(list(a))

        data_chest = data_iboneMarker.iloc[:, indChest]
        data_mD = data_iboneMarker.iloc[:, indmD]
        data_iboneMarker.iloc[:, idnull] = data_chest * 2 - data_mD

    if data_iboneMarker.isnull().any().any(): print('!!Nan in this data!!')
    return data_iboneMarker, infoBonMarker


def seq_collate(data):
    (loaddata,label) = zip(*data)
    out =[loaddata,label]
    return tuple(out)


class CongreG8Dataset(Dataset):
    """Dataloder for the CongreG8 datasets"""
    def __init__(
        self, data_dir, seq_len, slidingwindow_len, delim='\t'
    ):
        super(CongreG8Dataset, self).__init__()

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.delim = delim
        seq_list=[]
        labeldf=pd.DataFrame()
        label_group_list=[]
        label_group_index=[]

        all_files = os.listdir(self.data_dir)
        file_list=[]
        for file in all_files:
            if file.endswith(".xlsx"):
                label_path = os.path.join(self.data_dir, str(file))
                labeldf = pd.read_excel(label_path)
                # test=label_group_list = labeldf.iloc[:, 0]
                for ind, item in labeldf.iloc[:, 0].iteritems():
                    #TODO add something here to read t1, t2, .., i.e. robot data

                    if isinstance(item, six.string_types):
                        if RepresentsInt(item):
                            label_group_list.append(int(item))
                    else:
                        label_group_list.append(int(item))
                label_group_eindex = [len(label_group_list)-label_group_list[::-1].index(i)-1 for i in range(1, 11)]
                label_group_sindex = [label_group_list.index(i) for i in range(1, 11)]
            if os.path.isdir(os.path.join(self.data_dir, file)):
                if file[0]=='.':
                    pass
                else:
                    f_name=str(file)
                    if f_name[0:5] == 'Group':
                        filename = f_name
                        file_list.append(filename)


        all_files = [os.path.join(self.data_dir, _path) for _path in file_list]

        total_data=[]
        total_label=[]
        # label_start=0
        for read_path in all_files:
            last_letter = read_path[-1]
            if last_letter == '0':
                group_id = 10
            else:
                group_id = int(last_letter)

            trail_list=labeldf.iloc[label_group_sindex[group_id - 1]:label_group_eindex[group_id - 1]+1, 2].reset_index()

            for f in range(1,43):  # Round
                # read_name = read_path + '\\' + str(f) + '.csv'
                read_name = os.path.join(read_path,str(f)+'.csv')
                if os.path.exists(read_name):
                    # print('Import data from ' + read_name)
                    logger.info('reading {}'.format(read_name))
                    data = RemoveHead(read_name)
                    df = pd.DataFrame(data)
                    data_boneM, info_boneM = ExtractBoneMarker(df)
                    # print(data_boneM)
                    # print(info_boneM)
                    data=SplitData(data_boneM,info_boneM, seq_len, slidingwindow_len) # num_windows X seq_len X 4 X 37 X 3
                    try:
                        label=int(trail_list.iloc[f-1]['label'])
                        cat_label = torch.zeros(data.shape[0], 2)
                        cat_label[range(data.shape[0]), label] = 1.0
                        total_label.append(cat_label)
                    except:
                        logger.info('Label and actual data does not match')

                    if f <= 10:
                        # p4 is the joining player
                        joinPlayer=data[:,:,3,:,:]
                        joinPlayer=joinPlayer.unsqueeze(dim=2)
                        restPlayers=data[:,:,0:3,:,:]
                        data=torch.cat([joinPlayer,restPlayers],dim=2)
                    elif f <= 20:
                        # p3
                        joinPlayer=data[:,:,2,:,:]
                        joinPlayer=joinPlayer.unsqueeze(dim=2)
                        index=[0,1,3]
                        restPlayers=data[:,:,index,:,:]
                        data=torch.cat([joinPlayer,restPlayers],dim=2)
                        # pass
                    elif f <= 30:
                        # p2
                        joinPlayer=data[:,:,1,:,:]
                        joinPlayer=joinPlayer.unsqueeze(dim=2)
                        index=[0,2,3]
                        restPlayers=data[:,:,index,:,:]
                        data=torch.cat([joinPlayer,restPlayers],dim=2)
                        # pass
                    else:
                        # p1
                        pass
                    # print(data.size())
                    total_data.append(data)
                    seq_list.append(data.shape[0])
        total_data=torch.cat(total_data,dim=0)
        total_label=torch.cat(total_label,dim=0)
        self.num_seq = total_data.shape[0]
        self.data=total_data
        self.label=total_label

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        # start, end = self.seq_start_end[index]
        out=[self.data[index], self.label[index]]
        return out
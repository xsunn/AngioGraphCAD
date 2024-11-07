from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
import torch_geometric.transforms as T

import os
import numpy as np 
import os.path as op
import torch 
import logging
import pandas as pd
import random
from itertools import permutations

# load all data for training and testing 
allDataPath ="/home/sun/data/patientLevel/allData/"
allDataList = os.listdir(allDataPath)
allDataList.sort()
labelCSV=pd.read_csv("/home/sun/project/FAME2/patientLeveLabel.csv")

#load training or testing patients 
def load_train_split(save_split_path: str,train_fname):
    train_patient_ids = []
    with open(op.join(save_split_path, train_fname), "r") as fp:
        for i in fp.readlines():
            train_patient_ids.append(i.strip())
    return train_patient_ids


def expandList(graphList,lesionLabel,target_length):

    #according to the lesionlabel sort two list
    sorted_indices = [index for index, value in sorted(enumerate(lesionLabel), key=lambda x: x[1],reverse=True)]
    s_lesionLabel=[lesionLabel[i] for i in sorted_indices]
    s_graphList = [graphList[i] for i in sorted_indices]

    #expand the graphlist to a fixed length
    if len(s_graphList)<target_length:
        s_graphList = s_graphList * ((target_length + len(s_graphList) - 1) // len(s_graphList))

    # crop graphlist to a fixed length
    if len(s_graphList)>=target_length:
        s_graphList = s_graphList[:target_length]
    
    return s_graphList,s_lesionLabel


def getTrainPath_Patient(trainFoldPath,train_fname):
    viewNum =9
    trainList=[]
    patientList = load_train_split(trainFoldPath,train_fname)
    for patient in patientList:

        if any(char.isalpha() for char in patient):
            cleanName= patient[:2]+patient[-5:]
        else:
            cleanName = patient

        label = labelCSV[labelCSV['lname'] == cleanName]['Label']
        label = torch.tensor(label.values[0]).float()
        ptDataList=[]
        lesionLabel=[]
        for ptfile in allDataList:
            if patient in ptfile:

                fullPTPath = os.path.join(os.path.join("/home/sun/data/patientLevel/allData/",ptfile))
                pt = torch.load(fullPTPath)

                ptDataList.append(pt)
                lesionLabel.append(pt.y.item())

        init_len=len(ptDataList)
        # featureUse=[torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)]
        featureList=[float(0)]*viewNum

        # if 0<len(ptDataList)<5:
        ptDataList,_=expandList(ptDataList,lesionLabel,target_length=viewNum)

        # mask: 1 having this view, 0 padding views
        # maskMatirx = torch.zeros(viewNum,32)

        for i in range(min(init_len,viewNum)):
            featureList[i]=float(1)
        #     maskMatirx[i,:]=1
        # maskMatirx = maskMatirx.reshape(1,-1)[0]

        case={"view0":ptDataList[0],"view1":ptDataList[1],"view2":ptDataList[2],"view3":ptDataList[3],"view4":ptDataList[4],
              "view5":ptDataList[5],"view6":ptDataList[6],"view7":ptDataList[7],"view8":ptDataList[8],
                  "label":label,"mask":featureList,"len":init_len}
        
        trainList.append(case)
    return trainList




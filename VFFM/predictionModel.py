import torch
import sys
sys.path.append('')
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 
from VFFM.maskeAttention import MAM, SinconPos
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#MASKED ATTENTION LAYERS 
class MAL (nn.Module):
        # def __init__(self,embedding_dim=1024, dim_feedforward=512, nhead=4, num_layers=8, drop_rate= 0.3,lesionNum=5,device='cuda') -> None:
        def __init__(self,MAM,num_layers) -> None:

            super().__init__()
            self.layers = _get_clones(MAM,num_layers)

            # self.sinconPos = SinconPos(max_len=lesionNum,embedding_dim=embedding_dim,device=device)
        def forward(self, x, mask):
                    # position embedding 
            # x_with_pos= self.sinconPos(x) # b， lesion num，feat_dim
            output = x
            for layer in self.layers:
                output,atten_prob= layer(output,mask)
            
            return output,atten_prob

class mergModule(nn.Module):
    def __init__(self, encoder, cfg, lesionNum=6, embedding_dim=512,dim_feedforward=256, nhead=4,num_layers=4,qkv_bias=True,drop_rate=0.2,device='cuda'):
        super(mergModule,self).__init__()
        self.lesionNum = lesionNum
        self.device = device
        self.enc=encoder
        self.fusion_method = cfg.fusion_method
        self.sinconPos = SinconPos(max_len=lesionNum,embedding_dim=embedding_dim,device=device)
        self.dim_feedforward=dim_feedforward
        self.emb_dim = embedding_dim
        # self.fusion = fusionModel 
        self.fusion = MAM(feat_dim=embedding_dim,att_dim=dim_feedforward, qkv_bias=True, nhead=nhead,lesionNum=lesionNum,drop_rate=0.2,device=device)

        self.fusionNetwork = MAL(self.fusion,num_layers=num_layers)

        out_dim = cfg.nb_classes

        self.clic = nn.Sequential(nn.Linear(3*lesionNum,32),
                                  nn.BatchNorm1d(32),nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
                                    nn.Linear(32,32),)
        
        #feezing backbone or not 
        for p in encoder.parameters():
            p.requires_grad = True
        

        self.preHeader= nn.Sequential(     

            nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            
            nn.Linear(self.emb_dim,128),
            nn.BatchNorm1d(128), 
            nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(128, 64),nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(64, 32),

            nn.BatchNorm1d(32), 
            nn.Linear(32,32),
            nn.BatchNorm1d(32), 
            nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(32, 16),nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(cfg.gnn_dropout),
            nn.Linear(16, 2),
        )
        self.norma = nn.BatchNorm1d(32)

    def forward(self, batchData):
        # print("training model datalist",type(batchData),len(batchData))
        length = batchData['len']
        views_0=batchData['view0']
        views_1=batchData['view1']
        views_2=batchData['view2']
        views_3=batchData['view3']
        views_4=batchData['view4']
        views_5=batchData['view5']
        views_6=batchData['view6']
        views_7=batchData['view7']
        views_8=batchData['view8']

        view0_label = batchData['view0'].y.unsqueeze(1).to(torch.long).to(self.device)
        view1_label = batchData['view1'].y.unsqueeze(1).to(torch.long).to(self.device)
        view2_label = batchData['view2'].y.unsqueeze(1).to(torch.long).to(self.device)
        view3_label = batchData['view3'].y.unsqueeze(1).to(torch.long).to(self.device)
        view4_label = batchData['view4'].y.unsqueeze(1).to(torch.long).to(self.device)
        view5_label = batchData['view5'].y.unsqueeze(1).to(torch.long).to(self.device)
        view6_label = batchData['view6'].y.unsqueeze(1).to(torch.long).to(self.device)
        view7_label = batchData['view7'].y.unsqueeze(1).to(torch.long).to(self.device)
        view8_label = batchData['view8'].y.unsqueeze(1).to(torch.long).to(self.device)

        lesion_gt = torch.concat([view0_label, view1_label,view2_label,view3_label,view4_label,view5_label,view6_label,view7_label,view8_label],dim=1)


        feat0,view_pre0=self.enc(views_0.to(self.device))
        feat1,view_pre1=self.enc(views_1.to(self.device))
        feat2,view_pre2=self.enc(views_2.to(self.device))
        feat3,view_pre3=self.enc(views_3.to(self.device))
        feat4,view_pre4=self.enc(views_4.to(self.device))
        feat5,view_pre5=self.enc(views_5.to(self.device))
        feat6,view_pre6=self.enc(views_6.to(self.device))
        feat7,view_pre7=self.enc(views_7.to(self.device))
        feat8,view_pre8=self.enc(views_8.to(self.device))
        lesion_Pre =torch.concat([view_pre0, view_pre1,view_pre2,view_pre3,view_pre4,view_pre5,view_pre6,view_pre7,view_pre8],dim=1)
        # print("feat size", feat0.size())

        mask = batchData['mask']
        padding_mask = [list(row) for row in zip(*mask)]
        padding_mask = np.array(padding_mask)
        padding_mask = torch.tensor(padding_mask).to(self.device) # batch, lesion Num
        padding_mask = padding_mask[:,:self.lesionNum]

        feat0 = feat0.unsqueeze(1)
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(1)
        feat3 = feat3.unsqueeze(1)
        feat4 = feat4.unsqueeze(1)
        feat5 = feat5.unsqueeze(1)
        feat6 = feat6.unsqueeze(1)
        feat7 = feat7.unsqueeze(1)
        feat8 = feat8.unsqueeze(1)
        # print("feat size", feat0.size())
        allfeat = torch.concat([feat0,feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8],dim=1)
        data = allfeat[:,:self.lesionNum]
        lesion_gt=lesion_gt[:,:self.lesionNum]
        lesion_feats= data*padding_mask.unsqueeze(2) 


        if self.fusion_method =="mean":
            data = data*padding_mask.unsqueeze(2) 
            feat_sum = torch.sum(data,dim=1)
            weighted_Feat = feat_sum/(torch.sum(padding_mask,dim=1).unsqueeze(1))

        if self.fusion_method =="sum":
            data = data*padding_mask.unsqueeze(2) 
            weighted_Feat = torch.sum(data,dim=1)

        if self.fusion_method =="max":
            weighted_Feat,_ = torch.max(data,dim=1)

        if self.fusion_method =="min":
            weighted_Feat,_ = torch.min(data,dim=1)

        if self.fusion_method=="atten":
            org_data = data
            # print("data size",data.size())
            data = self.sinconPos(data)

            fusedFeat,atten_prob=self.fusionNetwork(data,padding_mask) # batch, lesion No., dimension

            # #filter out the padding views
            fusedFeat = data *(padding_mask.unsqueeze(2)) 
            
            # # compute the weights 
            weight_Matrix = fusedFeat
            weight_Matrix = torch.where(weight_Matrix != 0, torch.exp(weight_Matrix), weight_Matrix) # b, lesion num, dim

            # # normalize weight matrix
            normal_weight = F.normalize(weight_Matrix,p=2,dim=1)

            # weighted sum features 
            weighted_Feat = torch.multiply(normal_weight, org_data)# batch, lesion Num, dim 
            weighted_Feat = torch.sum(weighted_Feat,dim=1) #batch, dim

        patient_out = self.preHeader(weighted_Feat) 

        return patient_out, lesion_feats, lesion_gt, weighted_Feat
        

    

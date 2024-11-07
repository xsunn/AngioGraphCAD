import os 
import torch 
import torch.nn as nn 
import random

# define patient level triplet loss 

class patientTripletLoss(nn.Module):
    def __init__(self,margin=1, p=2, eps=1e-7) -> None:
        super().__init__()

        self.loss = nn.TripletMarginLoss(margin=margin,p=p,eps=eps)
        self.normal = nn.BatchNorm1d(32)
    def forward(self, feats, labels):
        b,D= feats.size()
        device= feats.device

        all_anchor = torch.rand(1,D).to(device)
        all_positive = torch.rand(1,D).to(device)
        all_negtive = torch.rand(1,D).to(device)

        b,D= feats.size()
        for i in range(b):
            anchor_label = labels[i]
            anchor_feat = feats[i:i+1,:]

            positive_indices = torch.where(labels==anchor_label)
            negtive_indices = torch.where(labels==1-anchor_label)
            positive_indices = positive_indices[0]
            negtive_indices = negtive_indices[0]
            # print(negtive_indices, len(negtive_indices))
            # print("neg",anchor_label,negtive_indices, len(negtive_indices))

            if len(positive_indices)>1:
                random_index = torch.randint(0, len(positive_indices), (1,), device='cuda:0')
                random_vale = positive_indices[random_index]
                positive_feat = feats[random_vale:random_vale+1]

            if len(positive_indices)==0:
                positive_feat = anchor_feat

            if len(negtive_indices)>1:
                random_index = torch.randint(0, len(negtive_indices), (1,), device='cuda:0')
                random_vale = negtive_indices[random_index]
                negtive_feat = feats[random_vale:random_vale+1]

            if len(negtive_indices)==0:
                negtive_feat = anchor_feat            

            all_positive = torch.cat((all_positive,positive_feat),dim=0)
            all_negtive = torch.cat((all_negtive,negtive_feat),dim=0)
            # loss = self.loss(anchor_feat,positive_feat,negtive_feat) + patient_loss
            # print(loss)

        all_positive=all_positive[1:,]
        all_negtive =all_negtive[1:]
        patient_loss = self.loss(feats,all_positive,all_negtive)
        return patient_loss
    



class LesionTriplet(nn.Module):
    def __init__(self,tem) -> None:
        super().__init__()

        # self.normal = nn.BatchNorm1d(32)
        self.t = tem

    def forward(self, feats, labels):
        b,leison_num, lesion_dim = feats.size()

        reshape_feats = feats.view(b*leison_num, lesion_dim)

        reshape_label = labels.reshape(b*leison_num)
        #filter out the padding features
        non_zero_rows = torch.any(reshape_feats!=0,dim=1)

        feats_non_zero = reshape_feats[non_zero_rows]  #[86,32]
        label_non_zero = reshape_label[non_zero_rows] #[86]

        # label size = batch, 9 

        # divide features into negative and positive features according to its label 
        one_rows = (label_non_zero == 1).nonzero(as_tuple=True)[0] 
        positive_len = one_rows.size()
        positive_feat = feats_non_zero[one_rows]
        
        zero_rows = (label_non_zero == 0).nonzero(as_tuple=True)[0] 
        negative_len = zero_rows.size()
        negative_feat = feats_non_zero[zero_rows]

        # anchor = feats_non_zero[0:1]
        # result = torch.matmul(anchor,feats_non_zero.t())
        loss = 0 
        for i in range(feats_non_zero.size()[0]):
            anchor = feats_non_zero[i:i+1]
            anchor_label = label_non_zero[i]

            deno = torch.matmul(anchor,feats_non_zero.t())/self.t
            deno_exp = torch.exp(deno)
            deno_exp_sum=torch.sum(deno_exp)

            if anchor_label == 0:
                positive_paire = negative_feat
            else:
                positive_paire = positive_feat

            p_num = positive_paire.size()[1]

            numerator = torch.matmul(anchor,positive_paire.t())/self.t
            numerator_exp = torch.exp(numerator)

            numerator_exp_nor = numerator_exp/deno_exp_sum
            numerator_exp_nor_log = torch.log(numerator_exp_nor)
            numerator_exp_nor_log_sum=torch.sum(numerator_exp_nor_log)/(-p_num)

            loss+=numerator_exp_nor_log_sum

        # return loss/(feats_non_zero.size()[0])
        return loss/b

        
        # all_anchor = torch.rand(1,leison_num).to(device)
        # all_positive = torch.rand(1,leison_num).to(device)
        # all_negtive = torch.rand(1,leison_num).to(device)

        # b,D= feats.size()
        # for i in range(b):
        #     anchor_label = labels[i]
        #     anchor_feat = feats[i:i+1,:]

        #     positive_indices = torch.where(labels==anchor_label)
        #     negtive_indices = torch.where(labels==1-anchor_label)
        #     positive_indices = positive_indices[0]
        #     negtive_indices = negtive_indices[0]



        #     if len(positive_indices)>1:
        #         random_index = torch.randint(0, len(positive_indices), (1,), device='cuda:0')
        #         random_vale = positive_indices[random_index]
        #         positive_feat = feats[random_vale:random_vale+1]

        #     if len(positive_indices)==0:
        #         positive_feat = anchor_feat

        #     if len(negtive_indices)>1:
        #         random_index = torch.randint(0, len(negtive_indices), (1,), device='cuda:0')
        #         random_vale = negtive_indices[random_index]
        #         negtive_feat = feats[random_vale:random_vale+1]

        #     if len(negtive_indices)==0:
        #         negtive_feat = anchor_feat            

        #     all_positive = torch.cat((all_positive,positive_feat),dim=0)
        #     all_negtive = torch.cat((all_negtive,negtive_feat),dim=0)
            # loss = self.loss(anchor_feat,positive_feat,negtive_feat) + patient_loss
            # print(loss)



        # return 0











# define lesion level contrastive loss 

# class lesionContraLoss (nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

# define patient level contrastive loss  
        
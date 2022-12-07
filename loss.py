# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F
from dataset_qua_semihard import Normalize
from lib import Distance, Cosine_Distance, Normalize_01,SC_Distance

##计算mutual局部特征距离
def Mutual_Distance(input_1, input_2,neighbor_k=3):
    input2_norm = torch.norm(input_2, p=2, dim=1, keepdim=True)   
    input2 = input_2/input2_norm
    
    B, C, L = input_1.size()
    if torch.cuda.is_available():
        output = torch.zeros(len(input_1), len(input_2)).cuda()
    for b in range(B):
        input1 = input_1[b]
        input1 = torch.transpose(input1, 0, 1)
        input1_norm = torch.norm(input1, p=2, dim=1, keepdim=True)   
        input1 = input1/input1_norm
        input1 = input1.unsqueeze(0)

        innerproduct_matrix = torch.matmul(input1, input2)
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)     
        row = -1*torch.sum(topk_value, dim=(1,2))
        #row=torch.div(row,(input1.size(1)*neighbor_k))
        output[b, :] = row 

    return output

# compute all similarities between all network traces.
def build_mutual_similarities(embs,neighbor_k=3):

    size=len(embs)
    all_sims = torch.zeros([size,size]).to('cuda')
    
    input2 = embs.type(torch.float32)
    input2 = F.normalize(input2, dim=1)
    for r in range(len(embs)):
        #要用GPU来算快很多
        input1 = embs[r].type(torch.float32).to('cuda')
        input1 = torch.transpose(input1, 0, 1)
        input1 = F.normalize(input1,dim=1)
        
        innerproduct_matrix = torch.matmul(input1,input2)
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)
        row=torch.sum(topk_value,dim=(1,2))
        row=row.div(input1.size(0)*neighbor_k)
        all_sims[[r],:]=row

    return all_sims

# compute all similarities between all network traces.
def build_mutual_similarities_Older(embs,neighbor_k=3):

    size=len(embs)
    all_sims = torch.zeros([size,size]).to('cuda')
    
    input2 = embs.type(torch.float32)
    input2 = torch.transpose(input2, 1, 2)
    input2 = F.normalize(input2, dim=1)
    for r in range(len(embs)):
        #要用GPU来算快很多
        input1 = embs[r].type(torch.float32).to('cuda')
        input1 = F.normalize(input1,dim=1)
        
        innerproduct_matrix = torch.matmul(input1,input2)
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 2)
        row=torch.sum(topk_value,dim=(1,2))
        row=row.div(len(input1)*neighbor_k)
        all_sims[[r],:]=row

    return all_sims


    
    
    
def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    #added
    normed_feature = F.normalize(normed_feature)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]






# class QuadrupletLoss(torch.nn.Module):
#     """
#     QuadrupletLoss
#     """
#     def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
#         super(QuadrupletLoss, self).__init__()
#         self.margin1=margin1
#         self.margin2=margin2
#         self.dist=dist

#     def forward(self, anchor, positive, negative1, negative2):
        
#         if self.dist=="euclidean":
#             squarred_distance_pos=(anchor-positive).pow(2).sum(1)
#             squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
#             squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
#         else:
#             squarred_distance_pos=Distance(anchor,positive)
#             squarred_distance_neg=Distance(anchor,negative1)
#             squarred_distance_neg2=Distance(negative1,negative2)

#         loss =F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg)\
#                 +F.relu(self.margin2+squarred_distance_pos-squarred_distance_neg2)

#         return loss.mean()

# #sample-class-distance
# class QuadrupletLoss1(torch.nn.Module):
#     """
#     QuadrupletLoss
#     """
#     def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
#         super(QuadrupletLoss1, self).__init__()
#         self.margin1=margin1
#         self.margin2=margin2
#         self.dist=dist

#     def forward(self, anchor, positive, negative1, negative2, all_embs, id_to_classid, classid_to_id, traces_a, traces_p, traces_n1, traces_n2):
        
#         if self.dist=="euclidean":
#             squarred_distance_pos=(anchor-positive).pow(2).sum(1)
#             squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
#             squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
#         else:
#             squarred_distance_pos=SC_Distance(anchor, positive, id_to_classid, classid_to_id, traces_a, all_embs)
#             squarred_distance_neg=SC_Distance(anchor, negative1, id_to_classid, classid_to_id, traces_a, all_embs)
#             squarred_distance_neg2=SC_Distance(negative1, negative2, id_to_classid, classid_to_id, traces_n1, all_embs)

#         loss =F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg)\
#                 +F.relu(self.margin2+squarred_distance_pos-squarred_distance_neg2)

#         return loss.mean()



# class QuadrupletLoss2(torch.nn.Module):
#     """
#     QuadrupletLoss
#     """
#     def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
#         super(QuadrupletLoss2, self).__init__()
#         self.margin1=margin1
#         self.margin2=margin2
#         self.dist=dist

#     def forward(self, anchor, positive, negative1, negative2):
        
#         if self.dist=="euclidean":
#             squarred_distance_pos=(anchor-positive).pow(2).sum(1)
#             squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
#             squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
#         else:
#             squarred_distance_pos=Distance(anchor,positive)
#             squarred_distance_neg=Distance(anchor,negative1)
#             squarred_distance_neg2=Distance(anchor,negative2)
#             squarred_distance_neg3=Distance(negative1,negative2)

#         loss =F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg)\
#                 +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg2)\
#                 +F.relu(self.margin2+squarred_distance_pos-squarred_distance_neg3)

#         return loss.mean()


# class QuadrupletLoss3(torch.nn.Module):
#     """
#     QuadrupletLoss
#     """
#     def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
#         super(QuadrupletLoss3, self).__init__()
#         self.margin1=margin1
#         self.margin2=margin2
#         self.dist=dist

#     def forward(self, anchor, positive, negative1, negative2):
        
#         if self.dist=="euclidean":
#             squarred_distance_pos=(anchor-positive).pow(2).sum(1)
#             squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
#             squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
#         else:
#             squarred_distance_pos=Distance(anchor,positive)
#             squarred_distance_neg=Distance(anchor,negative1)
#             squarred_distance_neg2=Distance(anchor,negative2)
#             squarred_distance_neg3=Distance(positive,negative1)
#             squarred_distance_neg4=Distance(positive,negative2)
#             squarred_distance_neg5=Distance(negative1,negative2)

#         loss =F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg)\
#                 +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg2)\
#                 +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg3)\
#                 +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg4)\
#                 +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg5)

#         return loss.mean()


class MQuadrupletLoss(torch.nn.Module):
    """
    QuadrupletLoss
    """
    def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
        super(MQuadrupletLoss, self).__init__()
        self.margin1=margin1
        self.margin2=margin2
        self.dist=dist

    def forward(self, anchor, positive, negative1, negative2):
        
        if self.dist=="euclidean":
            squarred_distance_pos=(anchor-positive).pow(2).sum(1)
            squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
            squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
            squarred_distance_neg3=(positive-negative1).pow(2).sum(1)
            squarred_distance_neg4=(positive-negative2).pow(2).sum(1)
            squarred_distance_neg5=(negative1-negative2).pow(2).sum(1)
        else:
            squarred_distance_pos=Distance(anchor,positive)
            squarred_distance_neg=Distance(anchor,negative1)
            squarred_distance_neg2=Distance(anchor,negative2)
            squarred_distance_neg3=Distance(positive,negative1)
            squarred_distance_neg4=Distance(positive,negative2)
            squarred_distance_neg5=Distance(negative1,negative2)

        loss =F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg)\
                +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg2)\
                +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg3)\
                +F.relu(self.margin1+squarred_distance_pos-squarred_distance_neg4)\
                +F.relu(self.margin2+squarred_distance_pos-squarred_distance_neg5)

        return loss.mean()

def calculate_adaptive_margin(traces_a, traces_p, traces_n1, traces_n2, w1=1,w2=0.5):
    dis_aps=Distance(traces_a, traces_p)
    avg_dis_aps=torch.sum(dis_aps)/len(traces_a)

    dis_an1s=Distance(traces_a, traces_n1)
    dis_an2s=Distance(traces_a, traces_n2)
    avg_dis_ans=torch.sum(dis_an1s+dis_an2s)/(2*len(traces_a))

    m1=w1*(avg_dis_ans-avg_dis_aps)
    m2=w2*(avg_dis_ans-avg_dis_aps)

    return m1,m2

class AdaptiveQuadrupletLoss(torch.nn.Module):
    """
    QuadrupletLoss
    """
    def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
        super(AdaptiveQuadrupletLoss, self).__init__()
        self.margin1=margin1
        self.margin2=margin2
        self.dist=dist

    def forward(self, anchor, positive, negative1, negative2):
        
        if self.dist=="euclidean":
            squarred_distance_pos=(anchor-positive).pow(2).sum(1)
            squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
            squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
        else:
            squarred_distance_pos=Distance(anchor,positive)
            squarred_distance_neg=Distance(anchor,negative1)
            squarred_distance_neg2=Distance(negative1,negative2)

        m1,m2=calculate_adaptive_margin(anchor, positive, negative1, negative2, w1=1, w2=0.5)

        loss =F.relu(m1+squarred_distance_pos-squarred_distance_neg)\
                +F.relu(m2+squarred_distance_pos-squarred_distance_neg2)

        return loss.mean()

class AdaptiveQuadrupletLoss2(torch.nn.Module):
    """
    QuadrupletLoss
    """
    def __init__(self, margin1=2.0, margin2=1.0, dist="euclidean") -> None:
        super(AdaptiveQuadrupletLoss2, self).__init__()
        self.margin1=margin1
        self.margin2=margin2
        self.dist=dist

    def forward(self, anchor, positive, negative1, negative2, m1=2, m2=1):
        
        distances=torch.tensor([]).cuda()
        if self.dist=="euclidean":
            squarred_distance_pos=(anchor-positive).pow(2).sum(1)
            squarred_distance_neg=(anchor-negative1).pow(2).sum(1)
            squarred_distance_neg2=(negative1-negative2).pow(2).sum(1)
        else:
            squarred_distance_pos=Distance(anchor,positive)
            squarred_distance_neg=Distance(anchor,negative1)
            squarred_distance_neg2=Distance(negative1,negative2)

            distances=torch.cat((squarred_distance_pos.view(1,-1), squarred_distance_neg.view(1,-1), squarred_distance_neg2.view(1,-1)),0)
            #distances=Normalize_01(distances)


        loss =F.relu(m1+distances[0]-distances[1])\
                +F.relu(m2+distances[0]-distances[2])

        return loss.mean()




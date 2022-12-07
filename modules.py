import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler

from collections import OrderedDict
from numpy.lib.twodim_base import diag




output_classes=95
seq_length=5000
struct_seq_length=5000





def conv3(in_planes, out_planes, stride=1,padding=1,dilation=1,groups=1):
    """3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False,groups=groups,dilation=dilation)

class BasicBlock_dilated(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dilations=(1, 2)):
        super(BasicBlock_dilated, self).__init__()
        ###O=floor(I+2P-d(k-1))/S+1 => P=ceil((O-1)*S+d(k-1)-O)/2
        padding_size=math.floor((dilations[0]*(3-1)+1)/2)
        # self.conv1 = nn.Conv1d(
        #     in_planes, planes, kernel_size=3, stride=stride, padding=padding_size,dilation=dilations[0], bias=False)
        self.conv1=conv3(in_planes,planes, stride=stride,padding=padding_size,dilation=dilations[0])
        self.bn1 = nn.BatchNorm1d(planes,eps=1e-5)
        
        padding_size=math.floor((dilations[1]*(3-1)+1)/2)
        # self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
        #                        stride=1, padding=padding_size,dilation=dilations[1], bias=False)
        self.conv2=conv3(planes,planes, padding=padding_size,dilation=dilations[1],groups=32)
        self.bn2 = nn.BatchNorm1d(planes,eps=1e-5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
                
                nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.expansion*planes,eps=1e-5)

                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=3, stride=stride, padding=1,bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_dilated_SE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dilations=(1, 2)):
        super(BasicBlock_dilated_SE, self).__init__()
        ###O=floor(I+2P-d(k-1))/S+1 => P=ceil((O-1)*S+d(k-1)-O)/2
        padding_size=math.floor((dilations[0]*(3-1)+1)/2)
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=padding_size,dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(planes,eps=1e-5)
        
        padding_size=math.floor((dilations[1]*(3-1)+1)/2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=padding_size,dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(planes,eps=1e-5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
                
                nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.expansion*planes,eps=1e-5)

                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=3, stride=stride, padding=1,bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)
        #out = F.relu(out)
        return out


        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dilations=(1, 2)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes,eps=1e-5)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes,eps=1e-5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
                ##MODIFICATION2-resnet-D
                nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
                # nn.Conv1d(in_planes, self.expansion*planes,
                #           kernel_size=3, stride=stride, padding=1,bias=False),
                # nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock_dilated(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dilations=(1, 2)):
        super(PreActBlock_dilated, self).__init__()
        padding_size=math.floor((dilations[0]*(3-1)+1)/2)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=padding_size, dilation=dilations[0],bias=False)

        padding_size=math.floor((dilations[1]*(3-1)+1)/2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=padding_size, dilation=dilations[1],bias=False)


        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dilations=(1, 2)):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool1d(kernel_size=3,stride=2,padding=1),
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.expansion*planes,eps=1e-5)
            )

        # SE layers
        self.fc1 = nn.Conv1d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, padding_size=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.padding_size=padding_size

        # self.conv1 = nn.Conv1d(1, 64, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        ###MODIFICATION-1:resnet_C
        self.conv1_1 = nn.Conv1d(1, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_3 = nn.Conv1d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64,eps=1e-5)
        self.relu=nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        dilations=(1, 2)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,dilations))
            self.in_planes = planes * block.expansion
            dilations=(4, 8)
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv1_1(x)
        out=self.conv1_2(out)
        out=self.conv1_3(out)
        out = self.relu(self.bn1(out))
        #print(out.size())
        ###added
        out = F.max_pool1d(out,kernel_size=3, stride=2,padding=self.padding_size)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.mean(dim=-1)
        #out = self.avgpool(out)
        #out=F.adaptive_avg_pool1d(out, 1)
        # out = F.avg_pool1d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
    
class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, padding_size=1):
        super(ResNet2, self).__init__()
        self.in_planes = 64
        self.padding_size=padding_size

        # self.conv1 = nn.Conv1d(1, 64, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        ###MODIFICATION-1:resnet_C
        self.conv1_1 = nn.Conv1d(1, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_3 = nn.Conv1d(64, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64,eps=1e-5)
        self.relu=nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        dilations=(1, 2)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,dilations))
            self.in_planes = planes * block.expansion
            dilations=(4, 8)
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv1_1(x)
        out=self.conv1_2(out)
        out=self.conv1_3(out)
        out = self.relu(self.bn1(out))
        #print(out.size())
        ###added
        out = F.max_pool1d(out,kernel_size=3, stride=2,padding=self.padding_size)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        ##由于cuda内存不足，额外增加的层
        #out = F.max_pool1d(out,kernel_size=8, stride=8)
        out = out.mean(dim=-1)
        #out = self.avgpool(out)
        #out=F.adaptive_avg_pool1d(out, 1)
        # out = F.avg_pool1d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


#========================== Define NResNet18 ==========================#

class NResNet18(nn.Module):
	def __init__(self, num_classes=5, neighbor_k=3):
		super(NResNet18, self).__init__()

		self.features = ResNet(BasicBlock, [2, 2, 2, 2],padding_size=1)
		
		self.imgtoclass = InstoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes



	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			support_set_sam = self.features(input2[i])
			B, C, l = support_set_sam.size()
			support_set_sam = support_set_sam.permute(1, 0, 2)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			S.append(support_set_sam)

		x = self.imgtoclass(q, S) # get Batch*num_classes

		return x


#========================== Define an image-to-class layer ==========================#
class InstoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3):
		super(InstoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k


	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_cosinesimilarity(self, input1, input2):
		B, C, l = input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)   
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
				support_set_sam = support_set_sam/support_set_sam_norm

				# cosine similarity between a query sample and a support category
				innerproduct_matrix = query_sam@support_set_sam

				# choose the top-k nearest neighbors
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
				inner_sim[0, j] = torch.sum(topk_value)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)    

		return Similarity_list 


	def forward(self, x1, x2):

		Similarity_list = self.cal_cosinesimilarity(x1, x2)

		return Similarity_list


class InstoClass_CosMetric(nn.Module):
	def __init__(self):
		super(InstoClass_CosMetric, self).__init__()


	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_similarity(self, input1, input2):
		B, _= input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i].unsqueeze(0)

			if torch.cuda.is_available():
				sim_mat = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				sims=[]
				for k in range(len(support_set_sam)):
					support_sam=support_set_sam[k].unsqueeze(0)
                    # similarity between a query sample and a support category
					sim = query_sam.mm(support_sam.t())
					sims.append(sim)
                
				sim_mat[0, j] = max(sims)

			Similarity_list.append(sim_mat)

		Similarity_list = torch.cat(Similarity_list, 0)    

		return Similarity_list 


	def forward(self, x1, x2):

		Similarity_list = self.cal_similarity(x1, x2)

		return Similarity_list
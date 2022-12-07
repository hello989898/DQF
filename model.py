import os
from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ResNet, ResNet2, BasicBlock_dilated, BasicBlock,InstoClass_Metric, InstoClass_CosMetric
from lib import load_model_statedict,Normalize_01


##########for EightLayerDF training to evaluate local features
class Sub_EightLayerDF(nn.Module):
	def __init__(self):
		super(Sub_EightLayerDF, self).__init__()
		self.feat_dim=256
		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
	def forward(self, input1):
 
		q = self.features(input1)
  
		return q

##########for EightLayerDF test
class Sub_EightLayerDF_Test(nn.Module):
	def __init__(self, neighbor_k=3):
		super(Sub_EightLayerDF_Test, self).__init__()

		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
		self.simlayer= InstoClass_Metric(neighbor_k=neighbor_k)
		

	def forward(self, input1, input2):
 
		q = self.features(input1)

    	# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			s = self.features(input2[i])
			B, C, l = s.size()
			s = s.permute(1, 0, 2)
			s = s.contiguous().view(C, -1)
			S.append(s)

		sim=self.simlayer(q, S)
  
		return sim

##########for EightLayerDF training to evaluate local features
class Sub_EightLayerDF_Test3(nn.Module):
	def __init__(self):
		super(Sub_EightLayerDF_Test3, self).__init__()

		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
	def forward(self, input1):
		
		#for test_opw
		q = self.features(input1)
		q = F.normalize(q,dim=1)
		q = q.view(len(q),-1)



  
		return q


##########for EightLayerDF2 training
class Sub_EightLayerDF2(nn.Module):
	def __init__(self):
		super(Sub_EightLayerDF2, self).__init__()

		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
		self.proj=nn.Linear(256,256)
		

	def forward(self, input1):
 
		q = self.features(input1)
		q = q.mean(dim=-1)
		q = self.proj(q)
		q = F.normalize(q, p=2, dim=-1)
  
		return q

##########for EightLayerDF test2
class Sub_EightLayerDF_Test2(nn.Module):
	def __init__(self):
		super(Sub_EightLayerDF_Test2, self).__init__()

		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
		self.simlayer= InstoClass_CosMetric()
		

	def forward(self, input1, input2):
 
		q = self.features(input1)
		q = q.mean(dim=-1)
		q = self.proj(q)
		q = F.normalize(q, p=2, dim=-1)

    	# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			s = self.features(input2[i])
			s = s.mean(dim=-1)
			s = self.proj(s)
			s = F.normalize(s, p=2, dim=-1)
   		
			S.append(s)

		sim=self.simlayer(q, S)
  
		return sim

##for test nodlffs
class Sub_EightLayerDF3(nn.Module):
	def __init__(self):
		super(Sub_EightLayerDF3, self).__init__()
		self.feat_dim=256
		self.features = nn.Sequential( 
			# block1                             # 1*5000
			nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [4999]),
			nn.ELU(alpha=1.0), 
			nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [5000]),
			nn.ELU(alpha=1.0), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block2                             # 32*1250
			nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [1249]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [1250]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block3                             # 64*312
			nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [311]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [312]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1),

			# block4                             # 128*78
			nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
			nn.LayerNorm(normalized_shape = [77]),
			# 激活函数变换为leakyrelu
			nn.LeakyReLU(), 
			nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
			nn.LayerNorm(normalized_shape = [78]),
			nn.LeakyReLU(), 
			nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
			nn.Dropout(0.1)							# 256*19
			
		)
  
	def forward(self, input1):
 
		q = self.features(input1)
		q = q.mean(-1)
  
		return q


#########for NResNet18 training
class Sub_NResNet18(nn.Module):
	def __init__(self):
		super(Sub_NResNet18, self).__init__()
		self.features = ResNet(BasicBlock_dilated, [2, 2, 2, 2], padding_size=1)


	def forward(self, input1):
		# extract features of input1--query image
		q = self.features(input1)
  
		return q



#########for NResNet18 test
class Sub_NResNet18_Test(nn.Module):
	def __init__(self,neighbor_k=3):
		super(Sub_NResNet18_Test, self).__init__()
		self.features = ResNet(BasicBlock_dilated, [2, 2, 2, 2], padding_size=1)
		self.simlayer= InstoClass_Metric(neighbor_k=neighbor_k)
		
		
	def forward(self, input1, input2):
		# extract features of input1--query image
		q = self.features(input1) #(1,1,5000)->(1,512)

  		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			s = self.features(input2[i])
			B, C, l = s.size()
			s = s.permute(1, 0, 2)
			s = s.contiguous().view(C, -1)
			S.append(s)

		sim=self.simlayer(q, S)
  
		return sim



#########for NResNet18-2 training
class Sub_NResNet18_2(nn.Module):
	def __init__(self):
		super(Sub_NResNet18_2, self).__init__()
		self.features = ResNet2(BasicBlock_dilated, [2, 2, 2, 2], padding_size=1)
		self.proj = nn.Linear(512, 512)
		
		
	def forward(self, input1):
		# extract features of input1--query image
		q = self.features(input1)
		q = self.proj(q)
		q = F.normalize(q, p=2, dim=-1)
  
		return q

#########for NResNet18-2 test
class Sub_NResNet18_Test2(nn.Module):
	def __init__(self):
		super(Sub_NResNet18_Test2, self).__init__()
		self.features = ResNet2(BasicBlock_dilated, [2, 2, 2, 2], padding_size=1)
		self.proj = nn.Linear(512, 512)
		self.simlayer= InstoClass_CosMetric()
		
		
	def forward(self, input1, input2):
		# extract features of input1--query image
		q = self.features(input1) #(1,1,5000)->(1,512)
		q = self.proj(q)
		q = F.normalize(q, p=2, dim=-1)
  
  		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			s = self.features(input2[i])
			s = self.proj(s)
			s = F.normalize(s, p=2, dim=-1)
   
			S.append(s)

		sim=self.simlayer(q, S)
  
		return sim
        
        

### Triplet model
class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        return E1, E2, E3


###For EightLayerDF
def get_model():
    # Model
    embeddingNet = Sub_EightLayerDF()
    model = TripletNet(embeddingNet)
    return model


###For NResNet18
def get_nresnet18_model():
    # Model
    embeddingNet = Sub_NResNet18()
    model = TripletNet(embeddingNet)

    return model


### Quadruplet model
class Quadruplet(nn.Module):
    def __init__(self, embeddingNet):
        super(Quadruplet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3, i4):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        E4 = self.embeddingNet(i4)
        return E1, E2, E3, E4

###For EightLayerDF
def get_qua_model():
    # Model
    embeddingNet = Sub_EightLayerDF()
    model = Quadruplet(embeddingNet)
    return model

###For EightLayerDF
def get_qua_model3():
    # Model
    embeddingNet = Sub_EightLayerDF3()
    model = Quadruplet(embeddingNet)
    return model

###For NResNet18
def get_qua_nresnet18_model():
    # Model
    embeddingNet = Sub_NResNet18()
    model = Quadruplet(embeddingNet)

    return model

### Quintuplet model
class Quintuplet(nn.Module):
    def __init__(self, embeddingNet):
        super(Quintuplet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3, i4, i5):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        E4 = self.embeddingNet(i4)
        E5 = self.embeddingNet(i5)
        return E1, E2, E3, E4, E5

###For EightLayerDF
def get_quint_model():
    # Model
    embeddingNet = Sub_EightLayerDF()
    model = Quintuplet(embeddingNet)
    return model

###For NResNet18
def get_quint_nresnet18_model():
    # Model
    embeddingNet = Sub_NResNet18()
    model = Quintuplet(embeddingNet)

    return model

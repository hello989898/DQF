from sklearn import neighbors
import torch
import os
from bisect import bisect_right
import numpy as np
import sys
import math
import torch.nn.functional as F

from torch.nn.functional import normalize
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve

def Normalize_01(data):
    m=torch.mean(data)
    mx=torch.amax(data)
    mn=torch.amin(data)
    data=torch.add(data,-mn)
    data=torch.divide(data,(mx-mn))
    return data

def RowNormalize_01(data):
    m=torch.mean(data,dim=1)
    mx=torch.amax(data,dim=1)
    mn=torch.amin(data,dim=1)
    data=torch.add(data,-mn)
    data=torch.divide(data,(mx-mn))
    return data


# def sim(a,b):
#     M,N=len(a),len(b)
#     C=256
#     L=19
#     a=a.view(M,C,L)
#     b=b.view(N,C,L)
#     sim=[]
#     a=a.T
#     neighbor_k=3
#     for i in range(len(a)):
#         for j in range(len(b)):
#             sim.append(np.multiply(a[i],b[j]))
#     sim=np.topk(sim, neighbor_k, 0)
#     sim=np.sum(sim)
#     return sim

# def my_KNN_Distance(a,b):
#     M,N=len(a),len(b)
#     a,b=np.array(a),np.array(b)
#     cost=sys.maxsize*np.ones((M,N))

#     for i in range(1,M):
#         ai=a[i]
#         for j in range(1,N):
#             bj=b[j]
#             cost[i,j]=sim(ai,bj)

#     return cost[-1,-1]

def my_KNN_Distance2(a,b):
    M,N=len(a),len(b)
    a,b=np.array(a),np.array(b)
    C=256
    L=19
    a=a.reshape((C,L))
    b=b.reshape((C,L))
    
    a=a.T
    sumsim=[]
    sim=[]
    neighbor_k=3
    for i in range(1,L):
        ai=a[i]
        ai=ai.reshape((1,-1))
        for j in range(1,L):
            bj=b[:,j]
            bj=bj.reshape((-1,1))
            sim.append((ai@bj)[0][0])
        sim=torch.from_numpy(np.array(sim))
        sim,_=torch.topk(sim, neighbor_k, 0)
        sim=sim.numpy()
        sim=np.sum(sim)
        sumsim.append(sim)
        sim=[]
    sumsim=np.sum(sumsim)
    return 1/sumsim
            

def my_KNN_Distance(a,b):
    M,N=len(a),len(b)
    a,b=np.array(a),np.array(b)
    C=256
    L=19
    a=a.reshape((C,L))
    b=b.reshape((C,L))
    neighbor_k=1
    
    a=a.T
    sim=a@b
    sim=torch.tensor(sim)
    sumsim,_=torch.topk(sim,neighbor_k,1)
    sumsim=torch.sum(sumsim)
    sumsim=sumsim.item()
    return 1/sumsim

def my_KNN_Distance3(a,b):
    M,N=len(a),len(b)
    a,b=np.array(a),np.array(b)
    C=256
    L=19
    a=a.reshape((C,L))
    a=torch.from_numpy(a)
    a=F.normalize(a)
    b=b.reshape((C,L))
    b=torch.from_numpy(b)
    b=F.normalize(b)
    neighbor_k=3
    
    a=a.T
    sim=a@b
    sumsim,_=torch.topk(sim,1,1)
    sumsim,_=torch.topk(sumsim,neighbor_k,0)
    sumsim=torch.sum(sumsim)
    sumsim=sumsim.item()
    return 1/sumsim



##计算mutual局部特征距离
def Mutual_Distance(input_1,input_2,neighbor_k=3):
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

##计算pair-wise局部特征距离
def Distance(input_1,input_2,neighbor_k=3):

    B, C, L = input_1.size()
    if torch.cuda.is_available():
        output = torch.zeros(1, len(input_1)).cuda()
    for b in range(B):
        input1 = input_1[b]
        input1 = torch.transpose(input1, 0, 1)
        input1_norm = torch.norm(input1, 2, 1, True)   
        input1 = input1/input1_norm

        input2 = input_2[b]
        input2_norm = torch.norm(input2, 2, 0, True)   
        input2 = input2/input2_norm

        innerproduct_matrix = input1@input2
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)     
        output[0, b] = -1*torch.sum(topk_value)
    
    output=output.view(-1)

    return output

##计算sample-to-class局部特征距离
def SC_Distance(input_1,input_2,id_to_classid, classid_to_id, traces_1, all_embs, neighbor_k=3):

    B, C, L = input_2.size()
    if torch.cuda.is_available():
        output = torch.zeros(1, len(input_2)).cuda()
    for b in range(B):
        input2 = input_2[b]
        input2 = torch.transpose(input2, 0, 1)
        input2_norm = torch.norm(input2, 2, 1, True)   
        input2 = input2/input2_norm

        clsid=id_to_classid[traces_1[b]]
        ids=classid_to_id[clsid]
        embs=all_embs[ids]

        input1=torch.tensor(embs.squeeze()).float().cuda()
        input1=input1.view(C,-1)
        input1_norm = torch.norm(input1, 2, 0, True)
        input1 = input1/input1_norm
        
        innerproduct_matrix = input2@input1
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)     
        output[0, b] = -1*torch.sum(topk_value)
        torch.cuda.empty_cache()
    
    output=output.view(-1)

    return output

def filter_tpr_bdr(tpr_thred,bdr_thred,tpr_list,bdr_list):
    L=len(tpr_list)
    M=len(bdr_list)
    new_tpr_list=[]
    new_bdr_list=[]
    assert L==M
    for i in range(L):
        if tpr_list[i]>=tpr_thred and bdr_list[i]>=bdr_thred:
            new_tpr_list.append(round(tpr_list[i],4))
            new_bdr_list.append(round(bdr_list[i],4))
    return new_tpr_list,new_bdr_list


def count_ow_4(logits, label, opt = None):
    # logits_new = F.softmax(logits, dim=1)
    logits_new = logits
    label_new = np.array(label)

    n_way_pos_y = np.where(label_new != 100)
    n_way_pos_n = np.where(label_new == 100)
    # print(type(n_way_pos_y), n_way_pos_y)
    two_label = label_new.copy()
    two_label[n_way_pos_y] = 1
    two_label[n_way_pos_n] = 0
    
    
    predict_label = np.argmax(logits_new, axis=1)
    #scores = torch.max(logits_new, axis=1)
    scores = np.max(logits_new, axis=1)
    #sorted_index = torch.argsort(scores)
    sorted_index = np.argsort(scores)
    scores = scores[sorted_index]
    label_new = label_new[sorted_index]
    predict_label= predict_label[sorted_index]
    two_label = two_label[sorted_index]
    label_new2 = label_new.copy()
    scores_new2 = scores.copy()
    label_new2 = label_new2[::-1]
    scores_new2 = scores_new2[::-1]
    predict_label2= predict_label.copy()
    predict_label2 = predict_label2[::-1]
    tprm = []
    fpr2, tpr2, thresholds = roc_curve(two_label, scores, pos_label=1, sample_weight=None, drop_intermediate=False)
    tmp_tprm = 0
    for i,score in enumerate(thresholds):
        if i == 0:
            tprm.append(0.0)
        else:
            if i >=len(label_new2):
                tprm.append(tmp_tprm)
                break
            if label_new2[i] != 100 and predict_label2[i] == label_new2[i]:
                tmp_tprm +=1
            tprm.append(tmp_tprm)

    p_all = len(n_way_pos_y[0])
    tprm = np.array(tprm)
    tpr_all = tprm / p_all
    two_class_auc = auc(fpr2, tpr2)
    multi_class_auc = auc(fpr2, tpr_all)

    two_tpr_list = tpr2
    multi_class_tpr_list = tpr_all
    fpr_list = fpr2

    fp_number=fpr2*len(n_way_pos_n[0])
    tp2_number=tpr2*len(n_way_pos_y[0])
    tpm_number=tprm
    append=np.ones((len(tp2_number)))*math.exp(-100)
    bdr2=np.divide(tp2_number,tp2_number+fp_number+append)
    bdrm=np.divide(tpm_number,tpm_number+fp_number+append)

    two_bdr_list=bdr2
    multi_bdr_list=bdrm
    
    return two_bdr_list,multi_bdr_list,two_tpr_list, multi_class_tpr_list, fpr_list, two_class_auc, multi_class_auc, logits_new,label_new

def log_ow(model_name, predictions, **parameters):
    results={}
    print('%s model:' % model_name)
    # predictions_cw=predictions[:1500]
    # predictions_opw=predictions[1500:]
    max_predictions=torch.amax(predictions,1)
    max_predictions_opw=torch.amax(predictions,1)
    maxvalue=torch.max(max_predictions)
    minvalue=torch.min(max_predictions)
    # maxvalue_opw=torch.max(max_predictions_opw)
    # minvalue_opw=torch.min(max_predictions_opw)
    stepvalue=(maxvalue-minvalue)/21
    fpr_list=[]
    two_tpr_list=[]
    multi_tpr_list=[]
    for conf_thresh in np.arange(minvalue, maxvalue, stepvalue):
        two_class_tpr, multi_class_tpr, two_class_bdr, multi_class_bdr, fpr = find_accuracy(
            predictions, conf_thresh, **parameters)
        print('\t conf: %f' % conf_thresh)
        print('\t \t two-class TPR: %s' % two_class_tpr)
        print('\t \t multi-class TPR: %s' % multi_class_tpr)
        print('\t \t two-class BDR: %s' % two_class_bdr)
        print('\t \t multi-class BDR: %s' % multi_class_bdr)
        print('\t \t FPR: %s' % fpr)
        fpr_list.append(fpr)
        two_tpr_list.append(two_class_tpr)
        multi_tpr_list.append(multi_class_tpr)

        prefix = '%s_%f' % (model_name, conf_thresh)
        results['%s_two_TPR' % prefix] = two_class_tpr
        results['%s_multi_TPR' % prefix] = multi_class_tpr
        results['%s_two_BDR' % prefix] = two_class_bdr
        results['%s_multi_BDR' % prefix] = multi_class_bdr
        results['%s_FPR' % prefix] = fpr

    two_class_auc = auc(fpr_list, two_tpr_list)
    multi_class_auc = auc(fpr_list, multi_tpr_list)
    return results, two_class_tpr, multi_class_tpr, two_class_bdr, multi_class_bdr, fpr

def find_accuracy(model_predictions, conf_thresh, actual_labels=None,
                  num_mon_sites=None, num_mon_inst_test=None,
                  num_unmon_sites_test=None, num_unmon_sites=None):
    """Compute TPR and FPR based on softmax output predictions."""

    # Changes predictions according to confidence threshold
    thresh_model_labels = np.zeros(len(model_predictions))
    for inst_num, softmax in enumerate(model_predictions):
        predicted_class = np.argmax(softmax)
        if predicted_class==num_mon_sites:
            thresh_model_labels[inst_num] = num_mon_sites
        elif softmax[predicted_class] < conf_thresh:
            thresh_model_labels[inst_num] = num_mon_sites
        else:
            thresh_model_labels[inst_num] = predicted_class

    # Computes TPR and FPR
    two_class_true_pos = 0  # Mon correctly classified as any mon site
    multi_class_true_pos = 0  # Mon correctly classified as specific mon site
    false_pos = 0  # Unmon incorrectly classified as mon site

    for inst_num, inst_label in enumerate(actual_labels):
        if inst_label == num_mon_sites:  # Supposed to be unmon site
            if thresh_model_labels[inst_num] < num_mon_sites:
                false_pos += 1
        else:  # Supposed to be mon site
            if thresh_model_labels[inst_num] < num_mon_sites:
                two_class_true_pos += 1
            if thresh_model_labels[inst_num] == inst_label:
                multi_class_true_pos += 1

    two_class_tpr = two_class_true_pos / \
                    (num_mon_sites * num_mon_inst_test) 
    multi_class_tpr = multi_class_true_pos / \
                      (num_mon_sites * num_mon_inst_test) 

    if num_unmon_sites == 0:  # closed-world
        fpr = 0
    else:
        fpr = false_pos / num_unmon_sites_test

    two_class_wrong_pos=0
    two_class_bdr=two_class_true_pos/(two_class_true_pos+false_pos)

    multi_class_wrong_pos=num_mon_sites*num_mon_inst_test-multi_class_true_pos
    multi_class_bdr=multi_class_true_pos/(multi_class_true_pos+false_pos)


    return two_class_tpr, multi_class_tpr, two_class_bdr, multi_class_bdr, fpr

##计算pair-wise局部特征距离
def Cosine_Distance(input_1,input_2,neighbor_k=3):

    B, C, L = input_1.size()
    if torch.cuda.is_available():
        output = torch.zeros(1, len(input_1)).cuda()
    for b in range(B):
        input1 = input_1[b]
        input1 = torch.transpose(input1, 0, 1)
        input1_norm = torch.norm(input1, 2, 1, True)   
        input1 = input1/input1_norm

        input2 = input_2[b]
        input2_norm = torch.norm(input2, 2, 0, True)   
        input2 = input2/input2_norm

        innerproduct_matrix = input1@input2
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1)   
        output[0, b] = -1*torch.sum(topk_value)
    output=torch.div(output,input1.size(0)*neighbor_k)
    output=output.view(-1)

    return output

##从文件加载模型
def load_model_statedict(model, model_path, appendflag=False):
    if os.path.isfile(model_path):
        print("=> Loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        if appendflag:
            for k in list(checkpoint['state_dict'].keys()):
                # 删去预训练模型中line开头的所有结构
                # if k.startswith('proj1.weight') or k.startswith('proj1.bias'):
                #     del checkpoint['state_dict'][k]
                checkpoint['state_dict'][k]="embeddingNet."+checkpoint['state_dict'][k]
    
        model.load_state_dict({k.replace('module.','').replace('embeddingNet.',''):v for k,v in checkpoint['state_dict'].items()})
        print("=> Loaded checkpoint '{}'".format(model_path))
    else:
        print("=> No checkpoint found at '{}'".format(model_path))
        
    return model

##从模型加载模型
def copy_model_statedict(src_model, des_model, appendflag=False):
    ##temp write
    model_to_save = {
                'state_dict': src_model.state_dict(),
            }
    file_name = os.path.join("/test/data", "temp_model.pth")
    save_checkpoint(model_to_save, file_name)
    ##restore
    des_model=load_model_statedict(des_model, "/test/data/temp_model.pth")
    return des_model

##保存模型
def save_checkpoint(state, file_name):
    torch.save(state, file_name)


def build_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 1.0
        if "backbone" in key:
            lr_mul = 0.1
        params += [{"params": [value], "lr_mul": lr_mul}]
    optimizer = getattr(torch.optim, 'SGD')(params,lr=0.01,weight_decay=0.0005)
    return optimizer


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(
                self.milestones,
                self.last_epoch
            )
            for base_lr in self.base_lrs
        ]
        
        
def build_lr_scheduler(optimizer):
    return WarmupMultiStepLR(
        optimizer,
        [1000, 2000, 3000],
        0.1,
        warmup_factor=0.01,
        warmup_iters=200,
        warmup_method='linear',
    )



##平均准确率
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


##计算准确率
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			#correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			#top3计算有误？
			correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res
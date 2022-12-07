import os, torch, random, pickle
import numpy as np
import torch.nn.functional as F


def build_pos_pairs_for_id(classid, classid_to_ids): # classid --> e.g. 0
    traces = classid_to_ids[classid]
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range, classid_to_ids):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classid_to_ids)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

def data_processing(data_path):
        
        # This is tuned hyper-parameters
        print (data_path)

        dirs = sorted(os.listdir(data_path))
        # Each given folder name (URL of each class), we assign class id
        # e.g. {'adp.com' : 23, ...}
        name_to_classid = {d:i for i,d in enumerate(dirs)}
        classid_to_name = {v:k for k,v in name_to_classid.items()}

        num_classes= len(name_to_classid)
        print ("number of classes: "+str(num_classes))


        trace_paths = {c:[directory + "/" + img for img in sorted(os.listdir(data_path + directory))]
                for directory,c in name_to_classid.items()}

        all_traces_path = []
        for trace_list in trace_paths.values():
            all_traces_path += trace_list

        # map to integers
        # just map each path to sequence of ID (from 1 to len(all_trace_path)
        path_to_id = {v: k for k, v in enumerate(all_traces_path)}
        id_to_path = {v: k for k, v in path_to_id.items()}


        # build mapping between traces and class
        classid_to_ids = {k: [path_to_id[path] for path in v] for k, v in trace_paths.items()}
        id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}
        all_classes=[c for c, traces in classid_to_ids.items() for v in traces]
        all_classes=np.array(all_classes)

        # open trace
        all_traces = []
        for path in id_to_path.values():
            each_path = data_path + path
            with open(each_path, 'rb') as handle:
                each_trace = pickle.load(handle,encoding="bytes")
            all_traces += [each_trace]

        all_traces = np.vstack((all_traces))
        all_traces = all_traces[:, np.newaxis, :]
        print ("Load traces with ",all_traces.shape)
        print ("Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB')
        Xa_train, Xp_train = build_positive_pairs(range(0, num_classes), classid_to_ids)
        all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
        
        return Xa_train, Xp_train, all_traces, all_classes, num_classes, all_traces_train_idx, id_to_classid

def data_processing_opw(data_path):
        
        # This is tuned hyper-parameters
        print (data_path)

        dirs = sorted(os.listdir(data_path))
        # Each given folder name (URL of each class), we assign class id
        # e.g. {'adp.com' : 23, ...}
        name_to_classid = {d:i for i,d in enumerate(dirs)}
        classid_to_name = {v:k for k,v in name_to_classid.items()}

        num_classes= len(name_to_classid)
        print ("number of classes: "+str(num_classes))


        trace_paths = {c:[directory + "/" + img for img in sorted(os.listdir(data_path + directory))]
                for directory,c in name_to_classid.items()}

        all_traces_path = []
        for trace_list in trace_paths.values():
            all_traces_path += trace_list

        # map to integers
        # just map each path to sequence of ID (from 1 to len(all_trace_path)
        path_to_id = {v: k for k, v in enumerate(all_traces_path)}
        id_to_path = {v: k for k, v in path_to_id.items()}


        # build mapping between traces and class
        classid_to_ids = {k: [path_to_id[path] for path in v] for k, v in trace_paths.items()}
        id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}
        all_classes=[c for c, traces in classid_to_ids.items() for v in traces]
        all_classes=np.array(all_classes)

        # open trace
        all_traces = []
        for path in id_to_path.values():
            each_path = data_path + path
            with open(each_path, 'rb') as handle:
                each_trace = pickle.load(handle,encoding="bytes")
            all_traces += [each_trace]

        all_traces = np.vstack((all_traces))
        all_traces = all_traces[:, np.newaxis, :]
        print ("Load traces with ",all_traces.shape)
        print ("Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB')

        
        return all_traces, all_classes, num_classes, id_to_classid



def Normalize(data):
    m=np.mean(data)
    mx=np.amax(data)
    mn=np.amin(data)
    data=np.add(data,-m)
    data=np.divide(data,(mx-mn))
    data=np.add(data,1)
    return data

# #78.74(5epoch),81.18(10epoch)19v-mean-I1I2
# def build_similarities_testing(conv, all_instances,neighbor_k):

#     all_sims = np.zeros((len(all_instances), len(all_instances)))
    
#     #使用GPU计算
#     embs=[]
#     for b in range(len(all_instances)):
#         temp=conv(torch.Tensor(all_instances[b]).view(-1,1,5000).to('cuda'))
#         temp=temp.detach().cpu().numpy().tolist()
#         embs.append(temp)
#     embs=np.array(embs)
     
#     input2 = torch.from_numpy(embs.squeeze(1)).type(torch.float32).to('cuda') #(B*256*19)
#     ##19个向量
#     input2 = F.normalize(input2,dim=1) #(B*256*19)
#     for r in range(len(embs)):
#         ##要用GPU来算快很多
#         input1 = torch.from_numpy(embs[r]).type(torch.float32).to('cuda') #(1*256*19)
#         ##19个向量
#         input1 = torch.transpose(input1, 1, 2)#(1*19*256)
#         input1 = F.normalize(input1, dim=2) #(1*19*256)

#         innerproduct_matrix = torch.matmul(input1,input2) #(B*19*19)
#         ##在input1维取top3
#         topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1) #(B*3*19)
#         row=torch.sum(topk_value,dim=(1,2)).cpu().numpy()
#         row=row.reshape(1,-1)
#         all_sims[[r],:]=np.divide(row,(input1.size(1)*neighbor_k))

#     return all_sims

# ###79.36%(5epo)--19V-Mean-I2I1
# def build_similarities_testing(conv, all_instances,neighbor_k):

#     all_sims = np.zeros((len(all_instances), len(all_instances)))
    
#     #使用GPU计算
#     embs=[]
#     for b in range(len(all_instances)):
#         temp=conv(torch.Tensor(all_instances[b]).view(-1,1,5000).to('cuda'))
#         temp=temp.detach().cpu().numpy().tolist()
#         embs.append(temp)
#     embs=np.array(embs)
     
#     input2 = torch.from_numpy(embs.squeeze(1)).type(torch.float32).to('cuda') #(B*256*19)
#     input2 = torch.transpose(input2, 1, 2) #(B*19*256)
#     ##19个向量
#     input2 = F.normalize(input2,dim=2) #(B*19*256)
#     for r in range(len(embs)):
#         ##要用GPU来算快很多
#         input1 = torch.from_numpy(embs[r]).type(torch.float32).to('cuda') #(1*256*19)
#         ##19个向量
#         input1 = F.normalize(input1,dim=1) #(1*256*19)

#         innerproduct_matrix = torch.matmul(input2,input1) #(B*19*19)
#         ##在input2维取top3
#         topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1) #(B*3*19)
#         row=torch.sum(topk_value,dim=(1,2)).cpu().numpy()
#         row=row.reshape(1,-1)
#         all_sims[[r],:]=np.divide(row,(input1.size(2)*neighbor_k))

#     return all_sims


# def build_similarities_NO1_256V(conv, all_instances,neighbor_k):

#     all_sims = np.zeros((len(all_instances), len(all_instances)))
#     with torch.no_grad():
#         #使用GPU计算
#         embs=torch.tensor([]).float().cuda() #(NB*256*19)
#         for b in range(len(all_instances)):
#             temp=conv(torch.Tensor(all_instances[b]).view(-1,1,5000).to('cuda'))
#             embs=torch.cat((embs, temp), 0)
        
#         embs = torch.transpose(embs, 1, 2) #(NB*19*256)
#         ##256个向量
#         embs = F.normalize(embs,dim=1) #(NB*19*256)

#         for r in range(len(embs)):
#             ##要用GPU来算快很多
#             input1 = embs[r] #(256*19)
#             ##256个向量
#             input1 = F.normalize(input1,dim=1) #(19*256)
#             input1 = torch.transpose(input1, 0, 1)#(256*19)
#             input1=input1.unsqueeze(0)#(1*256*19)
#             ##store sim of one query 
#             prod_all=torch.tensor([]).float().cuda()
#             ##divide into batches
#             batches=100
#             batchsize=len(embs)//batches
#             for batch in range(batches):
#                 input2=embs[batch*batchsize:(batch+1)*batchsize]#(B*19*256)

#                 innerproduct_matrix = torch.matmul(input1,input2) #(B*256*256)
#                 prod_all=torch.cat((prod_all, innerproduct_matrix), 0)
#                 #release gpu memmory
#                 torch.cuda.empty_cache()
            
#             ##在input2维取top3
#             topk_value, topk_index = torch.topk(prod_all, neighbor_k, 2) #(B*256*3)
#             row=torch.sum(topk_value,dim=(1,2)).cpu().numpy()
#             ##top3的平均余弦相似度
#             all_sims[[r],:]=np.divide(row,(input1.size(1)*neighbor_k))
#             #release gpu memmory
#             torch.cuda.empty_cache()

#     return all_sims

# ##83.56 --19V-Norm-I2I1
# def build_similarities_ok(conv, all_instances,neighbor_k):

#     all_sims = np.zeros((len(all_instances), len(all_instances)))
    
#     #使用GPU计算
#     embs=[]
#     for b in range(len(all_instances)):
#         temp=conv(torch.Tensor(all_instances[b]).view(-1,1,5000).to('cuda'))
#         temp=temp.detach().cpu().numpy().tolist()
#         embs.append(temp)
#     embs=np.array(embs)
     
#     input2 = torch.from_numpy(embs.squeeze(1)).type(torch.float32).to('cuda') #(B*256*19)
#     input2 = torch.transpose(input2, 1, 2) #(B*19*256)
#     ##19个向量
#     input2 = F.normalize(input2,dim=2) #(B*19*256)
#     for r in range(len(embs)):
#         ##要用GPU来算快很多
#         input1 = torch.from_numpy(embs[r]).type(torch.float32).to('cuda') #(1*256*19)
#         ##19个向量
#         input1 = F.normalize(input1,dim=1) #(1*256*19)

#         innerproduct_matrix = torch.matmul(input2,input1) #(B*19*19)
#         ##在input2维取top3
#         topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1) #(B*3*19)
#         row=torch.sum(topk_value,dim=(1,2)).cpu().numpy()
#         row=row.reshape(1,-1)
#         all_sims[[r],:]=row

#     all_sims=Normalize(all_sims)

#     return all_sims

##83.61----19V-Norm-I1I2
def build_similarities(conv, all_instances,neighbor_k):

    all_sims = np.zeros((len(all_instances), len(all_instances)))
    
    #使用GPU计算
    embs=[]
    for b in range(len(all_instances)):
        temp=conv(torch.Tensor(all_instances[b]).view(-1,1,5000).to('cuda'))
        temp=temp.detach().cpu().numpy().tolist()
        embs.append(temp)
    embs=np.array(embs)
     
    input2 = torch.from_numpy(embs.squeeze(1)).type(torch.float32).to('cuda') #(B*256*19)
    ##19个向量
    input2 = F.normalize(input2,dim=1) #(B*256*19)
    for r in range(len(embs)):
        ##要用GPU来算快很多
        input1 = torch.from_numpy(embs[r]).type(torch.float32).to('cuda') #(1*256*19)
        ##19个向量
        input1 = torch.transpose(input1, 1, 2)#(1*19*256)
        input1 = F.normalize(input1, dim=2) #(1*19*256)

        innerproduct_matrix = torch.matmul(input1,input2) #(B*19*19)
        ##在input1维取top3
        topk_value, topk_index = torch.topk(innerproduct_matrix, neighbor_k, 1) #(B*3*19)
        row=torch.sum(topk_value,dim=(1,2)).cpu().numpy()
        row=row.reshape(1,-1)
        all_sims[[r],:]=row

    all_sims=Normalize(all_sims)

    return all_sims


def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, id_to_classid, alpha_value, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg

###for training
class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, all_traces, neg_traces_idx, id_to_classid, batch_size, conv, alpha_value=0.1):
        
        self.alpha_value=alpha_value
        self.batch_size = batch_size
        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.id_to_classid = id_to_classid
        if conv:
            self.similarities = build_similarities(conv, self.traces,3)
        else:
            self.similarities = None

    def __getitem__(self,index):
        
        self.cur_train_index=index
        if self.cur_train_index >= self.num_samples:
            self.cur_train_index = 0


        # fill one batch
        traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
        traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
        traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx, self.id_to_classid, self.alpha_value)

        return self.traces[traces_a], self.traces[traces_p], self.traces[traces_n], np.zeros(shape=(traces_a.shape[0]))
    
    def __len__(self):
        return self.num_samples
    
def get_train_dataset(Xa_train, Xp_train, all_traces, all_traces_train_idx, id_to_classid, batch_size, conv, alpha):
    
    return SemiHardTripletGenerator(Xa_train, Xp_train, all_traces, all_traces_train_idx, id_to_classid, batch_size, conv, alpha)



#####test data
class BatchTestDataGenerator():
    def __init__(self, Test_Data_PATH, Test_Data_PATH_OPW, num_shot, num_query=15, num_way=100):
        
        
        _, _, all_traces, all_classes, num_classes, _, _=data_processing(Test_Data_PATH)
        all_traces_opw, all_classes_opw, num_classes_opw, id_to_classid_opw=data_processing_opw(Test_Data_PATH_OPW)
        
        self.num_shot=num_shot
        self.num_query=num_query
        self.num_way=num_way
        
        self.all_traces = all_traces
        self.all_classes = all_classes
        self.num_classes = num_classes
        self.num_instances = all_traces.shape[0]//num_classes

        self.all_traces_opw = all_traces_opw
        self.all_classes_opw = all_classes_opw
        self.num_classes_opw = num_classes_opw
        
        
        self.batch_size = num_way*(num_shot+num_query)
        self.batch_num_insts = num_shot+num_query
        self.batch_num_classes = num_way
        #初始排序类别
        perm_classes = np.random.permutation(num_classes)
        self.perm_classes = perm_classes
        
        self.cur_index = 0
        
    def generate_batch_for_onetask_in_conceptdrift(self, batch_labels):
        batch_spt_labels=[]
        batch_spt_traces=[]
        batch_query_labels=[]
        batch_query_traces=[]
        #redefine label 
        num_spt=20
        num_query=self.num_instances-20
        new_label=-1
        for label in batch_labels:
            new_label += 1
            perm_instances_id = np.random.permutation(num_spt)+label*(self.num_instances)
            label_spt_ids=perm_instances_id[:self.num_shot]
            #label_spt_classes= self.all_classes[label_spt_ids]
            label_spt_traces=self.all_traces[label_spt_ids]
            #label_spt_traces=np.expand_dims(label_spt_traces,axis=1)
            batch_spt_labels.append(new_label)
            batch_spt_traces.append(label_spt_traces.tolist())
            
            perm_instances_id = np.random.permutation(num_query)+label*(self.num_instances)+num_spt
            label_query_ids=perm_instances_id[:self.num_query]
            label_query_classes= self.all_classes[label_query_ids]
            label_query_traces=self.all_traces[label_query_ids]
            batch_query_labels.extend([new_label]*len(label_query_classes))
            batch_query_traces.extend(label_query_traces.tolist())
            
        return np.array(batch_spt_labels), np.array(batch_spt_traces), np.array(batch_query_labels), np.array(batch_query_traces)
        

    def generate_batch_for_onetask(self, batch_labels):
        batch_spt_labels=[]
        batch_spt_traces=[]
        batch_query_labels=[]
        batch_query_traces=[]
        #redefine label 
        new_label=-1
        for label in batch_labels:
            new_label += 1
            perm_instances_id = np.random.permutation(self.num_instances)+label*(self.num_instances)
            label_spt_ids=perm_instances_id[:self.num_shot]
            #label_spt_classes= self.all_classes[label_spt_ids]
            label_spt_traces=self.all_traces[label_spt_ids]
            #label_spt_traces=np.expand_dims(label_spt_traces,axis=1)
            batch_spt_labels.append(new_label)
            batch_spt_traces.append(label_spt_traces.tolist())
            
            label_query_ids=perm_instances_id[self.num_shot:self.num_shot+self.num_query]
            label_query_classes= self.all_classes[label_query_ids]
            label_query_traces=self.all_traces[label_query_ids]
            batch_query_labels.extend([new_label]*len(label_query_classes))
            batch_query_traces.extend(label_query_traces.tolist())
            
        return np.array(batch_spt_labels), np.array(batch_spt_traces), np.array(batch_query_labels), np.array(batch_query_traces)

    def generate_batch_for_onetask_opw(self, batch_labels):
        batch_spt_labels=[]
        batch_spt_traces=[]
        batch_query_labels=[]
        batch_query_traces=[]
        #redefine label 
        new_label=-1
        for label in batch_labels:
            new_label += 1
            perm_instances_id = np.random.permutation(self.num_instances)+label*(self.num_instances)
            label_spt_ids=perm_instances_id[:self.num_shot]
            #label_spt_classes= self.all_classes[label_spt_ids]
            label_spt_traces=self.all_traces[label_spt_ids]
            #label_spt_traces=np.expand_dims(label_spt_traces,axis=1)
            batch_spt_labels.append(new_label)
            batch_spt_traces.append(label_spt_traces.tolist())
            
            label_query_ids=perm_instances_id[self.num_shot:self.num_shot+self.num_query]
            label_query_classes= self.all_classes[label_query_ids]
            label_query_traces=self.all_traces[label_query_ids]
            batch_query_labels.extend([new_label]*len(label_query_classes))
            batch_query_traces.extend(label_query_traces.tolist())

        ##add opw
        new_label+=1
        perm_instances_id_opw=np.random.permutation(len(self.all_traces_opw))
        label_spt_ids_opw=perm_instances_id[:self.num_shot]
        label_spt_traces_opw=self.all_traces[label_spt_ids_opw]
        batch_spt_labels.append(new_label)
        batch_spt_traces.append(label_spt_traces_opw.tolist())



        #num_query_opw=self.num_way*self.num_query
        label_query_ids_opw=perm_instances_id_opw[self.num_shot:]
        label_query_classes_opw= self.all_classes_opw[label_query_ids_opw]
        label_query_traces_opw=self.all_traces_opw[label_query_ids_opw]
        batch_query_labels.extend([new_label]*len(label_query_classes_opw))
        batch_query_traces.extend(label_query_traces_opw.tolist())
            
        return np.array(batch_spt_labels), np.array(batch_spt_traces), np.array(batch_query_labels), np.array(batch_query_traces)
            
            
        
    def __getitem__(self,index):
        
        self.cur_index=index%self.num_classes
        if self.cur_index==0 and index!=0:
            #重新排序类别
            perm_classes = np.random.permutation(self.num_classes)
            self.perm_classes = perm_classes

        if self.cur_index+self.batch_num_classes>self.num_classes:
            self.cur_index=0
            #重新排序类别
            perm_classes = np.random.permutation(self.num_classes)
            self.perm_classes = perm_classes

        # fill one batch for one task
        batch_classes=self.perm_classes[self.cur_index:self.cur_index + self.batch_num_classes]
        # for cw
        #batch_spt_classes, batch_spt_traces, batch_query_classes, batch_query_traces=self.generate_batch_for_onetask(batch_classes)
        # for opw
        batch_spt_classes, batch_spt_traces, batch_query_classes, batch_query_traces=self.generate_batch_for_onetask_opw(batch_classes)
        # for concept drift experiments
        #batch_spt_classes, batch_spt_traces, batch_query_classes, batch_query_traces=self.generate_batch_for_onetask_in_conceptdrift(batch_classes)

        return batch_spt_classes, batch_spt_traces, batch_query_classes, batch_query_traces
    
    def __len__(self):
        return self.num_classes

def get_test_dataset(test_path, test_path_opw, num_shot, num_query, num_way):
    return BatchTestDataGenerator(test_path, test_path_opw, num_shot, num_query, num_way)


    

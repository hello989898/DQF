import os, argparse, time, torch, pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import scipy
from scipy.stats import t
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve

from torch.autograd import Variable

from model import Sub_NResNet18, Sub_NResNet18_2, Sub_EightLayerDF, Sub_EightLayerDF2
from model import Sub_NResNet18_Test, Sub_NResNet18_Test2, Sub_EightLayerDF_Test, Sub_EightLayerDF_Test2
from dataset_opw import get_train_dataset, data_processing, get_test_dataset
from lib import build_lr_scheduler, save_checkpoint, load_model_statedict, AverageMeter, accuracy, Distance, log_ow, Normalize_01, count_ow_4,filter_tpr_bdr


torch.manual_seed(1)
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def test(data, model):
    model.eval()
    maxiter = 15
    top1 = AverageMeter()
    accuracies=[]
    tprs_2 = []
    tprs_m = []
    bdrs_2 = []
    bdrs_m = []
    aucs_2 = []
    aucs_m = []
    mean_fpr = np.linspace(0,1,100)
    for task_idx in range(maxiter):
        
        spt_labels, spt_datas, query_labels, query_datas= data[task_idx * args.num_way]  # (N), (N, C)
        spt_labels = torch.from_numpy(spt_labels).float().cuda()
        spt_datas = torch.from_numpy(spt_datas).float().cuda()
        query_labels = torch.from_numpy(query_labels).float().cuda()
        query_datas = torch.from_numpy(query_datas).float().cuda()
        
        input_var2 = []
        for i in range(len(spt_datas)):
            temp_support = spt_datas[i]
            input_var2.append(temp_support)
            
        batches=100
        divide=len(query_datas)//batches
        prediction=torch.tensor([])
        for batch in range(batches):
			
            if batch==0:
                input_var1=query_datas[:(batch+1)*divide]
            else:
                input_var1=query_datas[batch*divide:(batch+1)*divide]

            with torch.no_grad():
                prediction_batch = model(input_var1, input_var2)  # (B, C)
                prediction=torch.cat((prediction, prediction_batch.cpu()), 0)
                torch.cuda.empty_cache()
        
        print('\n............Current task............'+str(task_idx+1))
        # parameters = {'actual_labels': query_labels.cpu(),
        #         'num_mon_sites': args.num_way,
        #         'num_mon_inst_test': args.num_query,
        #         'num_unmon_sites_test': args.num_way*args.num_query,
        #         'num_unmon_sites': 1}
        prediction=F.softmax(prediction,dim=1)
        #_, btpr, mtpr, bbdr, mbdr, fpr=log_ow("QF",prediction,**parameters)
        
        
        two_bdr_list,multi_bdr_list,two_tpr_list, multi_class_tpr_list, fpr_list, two_class_auc, multi_class_auc, logits_new,label_new=count_ow_4(prediction.numpy(),query_labels.cpu().numpy())
        #top1.update(btpr, query_datas.size(0))
        interp_tpr2 = np.interp(mean_fpr, fpr_list, two_tpr_list)
        interp_tpr2[0] = 0.0
        interp_tprm = np.interp(mean_fpr, fpr_list, multi_class_tpr_list)
        interp_tprm[0] = 0.0
        interp_bdr2 = np.interp(mean_fpr, fpr_list, two_bdr_list)
        interp_bdr2[0] = 0.0
        interp_bdrm = np.interp(mean_fpr, fpr_list, multi_bdr_list)
        interp_bdrm[0] = 0.0
        tprs_2.append(interp_tpr2)
        tprs_m.append(interp_tprm)
        bdrs_2.append(interp_bdr2)
        bdrs_m.append(interp_bdrm)
        aucs_2.append(two_class_auc)
        aucs_m.append(multi_class_auc)


        if (task_idx % 5 == 0) and (task_idx != 0):
            current_time = time.time()
            time_array = time.localtime(current_time) #时间戳转时间数组
            time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
            print('\n............Current time............'+time_str)
            mean_tpr2 = np.mean(tprs_2,axis=0)
            mean_tpr2[-1] = 1.0
            mean_auc2 = auc(mean_fpr, mean_tpr2)
            std_auc2 = np.std(aucs_2,axis=0)
            mean_tprm = np.mean(tprs_m,axis=0)
            mean_tprm[-1] = 1.0
            mean_aucm = auc(mean_fpr, mean_tprm)
            std_aucm = np.std(aucs_m,axis=0)

            mean_bdr2 = np.mean(bdrs_2,axis=0)
            mean_bdr2[-1] = 1.0
            mean_bdrm = np.mean(bdrs_m,axis=0)
            mean_bdrm[-1] = 1.0
            dataset= "400k"
            classifier= "QF"
            # print('tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_tpr2)))
            # print('tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_tprm)))
            # print('bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_bdr2)))
            # print('bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_bdrm)))
            # print('fpr_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_fpr)))
            print('auc2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, mean_auc2))
            print('auc2_{}_{}_shot_{}_std={}'.format(classifier, args.num_shot, dataset, std_auc2))
            print('aucm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, mean_aucm))
            print('aucm_{}_{}_shot_{}_std={}'.format(classifier, args.num_shot, dataset, std_aucm))

            ##tuned for precision
            tpr2_thred=0.1
            bdr2_thred=0.1
            tprm_thred=0.1
            bdrm_thred=0.1
            new_tpr2_list,new_bdr2_list=filter_tpr_bdr(tpr2_thred,bdr2_thred,mean_tpr2,mean_bdr2)
            new_tprm_list,new_bdrm_list=filter_tpr_bdr(tprm_thred,bdrm_thred,mean_tprm,mean_bdrm)
            print('filter_tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tpr2_list)))
            print('filter_bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdr2_list)))
            print('filter_tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tprm_list)))
            print('filter_bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdrm_list)))
    
    current_time = time.time()
    time_array = time.localtime(current_time) #时间戳转时间数组
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
    print('\n............Outputing: Current time............'+time_str)
    mean_tpr2 = np.mean(tprs_2,axis=0)
    mean_tpr2[-1] = 1.0
    mean_auc2 = auc(mean_fpr, mean_tpr2)
    std_auc2 = np.std(aucs_2,axis=0)
    mean_tprm = np.mean(tprs_m,axis=0)
    mean_tprm[-1] = 1.0
    mean_aucm = auc(mean_fpr, mean_tprm)
    std_aucm = np.std(aucs_m,axis=0)

    mean_bdr2 = np.mean(bdrs_2,axis=0)
    mean_bdr2[-1] = 1.0
    mean_bdrm = np.mean(bdrs_m,axis=0)
    mean_bdrm[-1] = 1.0
    dataset= "400k"
    classifier= "QF"
    # print('tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_tpr2)))
    # print('tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_tprm)))
    # print('bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_bdr2)))
    # print('bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_bdrm)))
    # print('fpr_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(mean_fpr)))
    print('auc2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, mean_auc2))
    print('auc2_{}_{}_shot_{}_std={}'.format(classifier, args.num_shot, dataset, std_auc2))
    print('aucm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, mean_aucm))
    print('aucm_{}_{}_shot_{}_std={}'.format(classifier, args.num_shot, dataset, std_aucm))

    ##tuned for precision
    tpr2_thred=0.1
    bdr2_thred=0.1
    tprm_thred=0.1
    bdrm_thred=0.1
    new_tpr2_list,new_bdr2_list=filter_tpr_bdr(tpr2_thred,bdr2_thred,mean_tpr2,mean_bdr2)
    new_tprm_list,new_bdrm_list=filter_tpr_bdr(tprm_thred,bdrm_thred,mean_tprm,mean_bdrm)
    print('filter_tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tpr2_list)))
    print('filter_bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdr2_list)))
    print('filter_tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tprm_list)))
    print('filter_bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdrm_list)))

    # filename="QF_OPW_"+dataset+"_"+str(args.num_shot)+"shot.txt"
    # out_file="/test/data/paper3/exp5/QF/"+filename
    # with open(out_file,"w") as fo:
    #     fo.writelines('filter_tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tpr2_list)))
    #     fo.writelines('filter_bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdr2_list)))
    #     fo.writelines('filter_tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tprm_list)))
    #     fo.writelines('filter_bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdrm_list)))

    ##tuned for recall
    # tpr2_thred=0.7
    # bdr2_thred=0.7
    # tprm_thred=0.7
    # bdrm_thred=0.7
    # new_tpr2_list,new_bdr2_list=filter_tpr_bdr(tpr2_thred,bdr2_thred,mean_tpr2,mean_bdr2)
    # new_tprm_list,new_bdrm_list=filter_tpr_bdr(tprm_thred,bdrm_thred,mean_tprm,mean_bdrm)
    # print('filter_tpr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tpr2_list)))
    # print('filter_bdr2_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdr2_list)))
    # print('filter_tprm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_tprm_list)))
    # print('filter_bdrm_{}_{}_shot_{}={}'.format(classifier, args.num_shot, dataset, list(new_bdrm_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch metric learning for WF')
    #basic info
    parser.add_argument('--exp_name', type=str, default='exp1', help='name of experiment')
    parser.add_argument('--dataset_name', type=str, default='tor_775w_25tr', help='dataset name (default:awf775)')
    parser.add_argument('--trainingdata_path', type=str, default='/test/data/paper3/exp1/FL/extracted_AWF775/', help='path for training data')
    parser.add_argument('--valdata_path', type=str, default='E:\\数据\\workspace\\TF\\exp0_144w_40tr_val\\', help='path for validation data')
    parser.add_argument('--testdata_path', type=str, default='/test/data/paper3/exp5/QF/extracted_AWF100/', help='path for test data')
    #parser.add_argument('--testdata_path', type=str, default='/test/data/paper3/exp7/QF/wang_wtf_100w_90tr/', help='path for test data')
    parser.add_argument('--testdata_path_opw', type=str, default='/test/data/paper3/exp5/QF/tf_tor_test_opw_399989w_1tr/', help='path for test data')
    
    #data info
    parser.add_argument('--training_batchsize', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--training_batchclasses', type=int, default=16, help='the number of input batch classes for training (default: 16)')
    parser.add_argument('--num_shot', type=int, default=1, help='number of training samples (default: 1)')
    parser.add_argument('--num_query', type=int, default=70, help='number of test samples (default: 15)')
    parser.add_argument('--num_way', type=int, default=100, help='size of problem (default: 100)')
    #model training
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 10)')
    #parser.add_argument('--ckp', type=str, default="/test/data/TF/results/exp1/qqqdnf_awf75_200_semihardpn_epoch30_qualoss4_FIX_selfdist_checkpoint_1.pth-67.61", help='path to load checkpoint')
    parser.add_argument('--ckp', type=str, default="/test/data/TF/results/exp1/qqqdnf_awf775_25_semihardpn_epoch30_qualoss4_FIX_selfdist_checkpoint_20.pth-87.15", help='path to load checkpoint')
    parser.add_argument('--ckp_freq', type=int, default=1, help='checkpoint frequency (default: 1)')
    parser.add_argument('--model_path', type=str, default='E:\\数据\\workspace\\hollidaytakeout\\results\\tf\\', help='directory to store model')
    #optimizer info
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    #loss info
    parser.add_argument('--margin', type=float, default=1.0, help='margin for triplet loss (default: 1.0)')
    #log info
    parser.add_argument('--train_log_step', type=int, default=100, help='number of iterations after which to log the loss')
    parser.add_argument('--test_log_step', type=int, default=10, help='number of iterations after which to statistic and log the results')
    
    global args, device
    args = parser.parse_args()

    
     # Build Model
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        device='cuda'
        cudnn.benchmark = True
    else:
        device='cpu'
    
    model = Sub_EightLayerDF_Test()
    model = model.to(device)


    
    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp)
            for k in list(checkpoint['state_dict'].keys()):
                # 删去预训练模型中line开头的所有结构
                if k.startswith('proj.weight') or k.startswith('proj.bias'):
                    del checkpoint['state_dict'][k]
     
            model.load_state_dict({k.replace('module.','').replace('embeddingNet.',''):v for k,v in checkpoint['state_dict'].items()})
            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))
            
    # tmp = filter(lambda x: x.requires_grad, model.parameters())
    # temp=list(model.parameters())
    # for _,param in enumerate(model.named_parameters()):
    #     print(param[0])
    #     print(param[1])
    #     print('----------------')


    start_time = time.time()
    time_array = time.localtime(start_time) #时间戳转时间数组
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
    print('\n............Start predicting............'+time_str)

    # batch sampler
    test_data_loader = get_test_dataset(args.testdata_path, args.testdata_path_opw, num_shot=args.num_shot, num_query=args.num_query, num_way=args.num_way)
    # test
    test(test_data_loader, model)


            


    
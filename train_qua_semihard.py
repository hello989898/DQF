import os, argparse, time, torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

from model import Sub_NResNet18, Sub_NResNet18_2, Sub_EightLayerDF, Sub_EightLayerDF2, get_model, get_qua_model,get_qua_nresnet18_model
from model import Sub_NResNet18_Test, Sub_NResNet18_Test2, Sub_EightLayerDF_Test, Sub_EightLayerDF_Test2
#from dataset_qua_random import get_train_dataset, data_processing ##83.85-30epoch 原版本
from dataset_qua_semihard import get_train_dataset, data_processing ##新版本
from lib import build_lr_scheduler, save_checkpoint, load_model_statedict, Distance
from loss import MQuadrupletLoss


torch.manual_seed(1)
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='2'


def train(data, model, criterion, optimizer, epoch):
    total_loss = 0
    model.train()
    model.embeddingNet.train()
    # if epoch>2:
    #     model.eval()
    #     model.embeddingNet.eval()
        
    for batch_idx in range(len(data)//args.training_batchsize-1):
        img_triplet=data[batch_idx*args.training_batchsize]
        anchor_img, pos_img, neg_img1, neg_img2 = img_triplet
        anchor_img, pos_img, neg_img1, neg_img2 = torch.Tensor(anchor_img).to(device).view(args.training_batchsize,1,5000), torch.Tensor(pos_img).to(device).view(args.training_batchsize,1,5000), torch.Tensor(neg_img1).to(device).view(args.training_batchsize,1,5000), torch.Tensor(neg_img2).to(device).view(args.training_batchsize,1,5000)
        anchor_img, pos_img, neg_img1, neg_img2 = Variable(anchor_img), Variable(pos_img), Variable(neg_img1), Variable(neg_img2)
        E1, E2, E3, E4 = model(anchor_img, pos_img, neg_img1, neg_img2)
        # dist_E1_E2 = Distance(E1,E2,3)
        # dist_E1_E3 = Distance(E1,E3,3)
        # dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        # dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        # target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        # target = target.to(device)
        # target = Variable(target)
        loss = criterion(E1,E2,E3,E4)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_step = args.train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            current_time = time.time()
            time_array = time.localtime(current_time) #时间戳转时间数组
            time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
            print('\n............Current time............'+time_str)
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data)//args.training_batchsize, total_loss / log_step))
            total_loss = 0

        #for debugging
        # if batch_idx==1:
        #     break
    print("****************"+str(batch_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch metric learning for WF')
    #basic info
    parser.add_argument('--exp_name', type=str, default='exp1', help='name of experiment')
    parser.add_argument('--dataset_name', type=str, default='extracted_AWF775', help='dataset name (default:awf775)')
    #parser.add_argument('--trainingdata_path', type=str, default='/test/data/paper3/exp1/FL/extracted_AWF775/', help='path for training data')
    parser.add_argument('--trainingdata_path', type=str, default='/test/data/paper3/exp3/df_wtf_train_75w_200tr/', help='path for training data')
    parser.add_argument('--valdata_path', type=str, default='E:\\数据\\workspace\\TF\\exp0_144w_40tr_val\\', help='path for validation data')
    parser.add_argument('--testdata_path', type=str, default='/test/data/paper3/exp1/FL/extracted_AWF100/', help='path for test data')
    #data info
    parser.add_argument('--training_batchsize', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--num_shot', type=int, default=1, help='number of training samples (default: 1)')
    parser.add_argument('--num_query', type=int, default=15, help='number of test samples (default: 15)')
    parser.add_argument('--num_way', type=int, default=100, help='size of problem (default: 100)')
    #model training
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
    parser.add_argument('--ckp', type=str, default=None, help='path to load checkpoint')
    #parser.add_argument('--ckp', type=str, default="/test/data/TF/results/exp1/qqqdnf_awf775_25_semihardpn_epoch30_qualoss_selfdist_checkpoint_30.pth", help='path to load checkpoint')
    parser.add_argument('--ckp_freq', type=int, default=1, help='checkpoint frequency (default: 5)')
    parser.add_argument('--model_path', type=str, default='/test/data/TF/results/', help='directory to store model')
    #optimizer info
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    #loss info
    parser.add_argument('--margin', type=float, default=1, help='margin for triplet loss (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha value for triplet loss (default: 0.1)')
    #log info
    parser.add_argument('--train_log_step', type=int, default=500, help='number of iterations after which to log the loss')
    parser.add_argument('--test_log_step', type=int, default=600, help='number of iterations after which to statistic and log the results')
    
    global args, device
    args = parser.parse_args()

    exp_dir = os.path.join(args.model_path, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    
     # Build Model
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        device='cuda'
        cudnn.benchmark = True
        #cudnn.deterministic = True
    else:
        device='cpu'
    
    model = get_qua_model()
    model = model.to(device)
    
    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            load_model_statedict(model.embeddingNet, args.ckp)
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))

    # tmp = filter(lambda x: x.requires_grad, model.parameters())
    # temp=list(model.embeddingNet.parameters())
    # for _,param in enumerate(model.embeddingNet.named_parameters()):
    #     print(param[0])
    #     print(param[1])
    #     print('----------------')
            

    criterion=MQuadrupletLoss(margin1=0.3, margin2=0.15, dist="SD")
    #criterion=torch.nn.TripletMarginLoss(margin=0.3,p=2)
    optimizer = optim.Adam(model.embeddingNet.parameters(), lr=args.lr)

    start_time = time.time()
    time_array = time.localtime(start_time) #时间戳转时间数组
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
    print('\n............Start training............'+time_str)
    
    
    Xa_train, Xp_train, all_traces, all_classes, _, all_traces_train_idx, id_to_classid=data_processing(args.trainingdata_path)
    # Init data loaders
    #train_data_loader= get_train_dataset(Xa_train, Xp_train, all_classes, all_traces, all_traces_train_idx, id_to_classid, args.training_batchsize, None, args.alpha)
    train_data_loader= get_train_dataset(Xa_train, Xp_train, all_traces, all_traces_train_idx, id_to_classid, args.training_batchsize, None, args.alpha)
    for epoch in range(1, args.epochs + 1):
        # train
        train(train_data_loader, model, criterion, optimizer, epoch)

        #test
        #test(test_data_loader, model, criterion)
        # Save model
        if epoch % args.ckp_freq == 0 or epoch==args.epochs:
            model_to_save = {
            "epoch_index": epoch,
            'best_prec1': 0,
            'optimizer' : optimizer.state_dict(),
            'state_dict': model.embeddingNet.state_dict(),
            }
            file_name = os.path.join(exp_dir, "qqqdnf_awf75_200_semihardpn_epoch30_qualoss4_FIX_selfdist_checkpoint_" + str(epoch) + ".pth")
            save_checkpoint(model_to_save, file_name)
        
        current_time = time.time()
        time_array = time.localtime(current_time) #时间戳转时间数组
        time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_array) # 时间数组转时间字符串
        print('\n............Current time............'+time_str)
        ##shuffle
        perm=np.random.permutation(len(Xa_train))
        Xa_train=Xa_train[perm]
        Xp_train=Xp_train[perm]

        if epoch==args.epochs:
            break
        
        #train_data_loader= get_train_dataset(Xa_train, Xp_train, all_classes, all_traces, all_traces_train_idx, id_to_classid, args.training_batchsize, model.embeddingNet, args.alpha)
        train_data_loader= get_train_dataset(Xa_train, Xp_train, all_traces, all_traces_train_idx, id_to_classid, args.training_batchsize, model.embeddingNet, args.alpha)


            


    
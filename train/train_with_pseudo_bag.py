import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from nystrom_attention import Nystromformer
import abmil as abmil
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from utils import get_cam_1d

def group_data(data_list, category_list):
    groups = defaultdict(list)
    for data, category in zip(data_list, category_list):
        groups[category].append(data)
    return groups

def get_bag_feats(csv_file_df, args):
    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    pos = feats.iloc[:,int(args.feats_size):]
    feats = feats.iloc[:,:int(args.feats_size)]
    feats = feats.to_numpy()
    pos = pos.to_numpy()
    pos = np.squeeze(pos)
    pos = np.array([np.fromstring(s.strip("[] "), dtype=int, sep=" ") for s in pos])
    
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats, pos

'''
def get_bag_feats(csv_file_df, args):

    feats_csv_path = csv_file_df.iloc[0]
    
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    
    #pos = feats.iloc[:,int(args.feats_size):]
    feats = feats.iloc[:,:int(args.feats_size)]
    #print(feats)
    feats = feats.to_numpy()
    
    label = np.zeros(args.num_classes)
    
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats
'''

def train(train_df, milnet, attention, classifier, criterion, optimizer_t1, optimizer_t2, args):
    torch.autograd.set_detect_anomaly(True)
    b_loss = args.b_loss
    m_loss = 1-b_loss
    milnet.train()
    attention.train()
    classifier.train()
    
    distill = args.distill_type
    instance_per_group = len(train_df) // args.num_k
    
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        label, feats, pos_arr = get_bag_feats(train_df.iloc[i], args)
        
        scaler = MinMaxScaler()
        nor_pos_arr = scaler.fit_transform(pos_arr)
        
        max_x = np.amax(pos_arr, 0)[0]
        max_y = np.amax(pos_arr, 0)[1]
        
        #feats = dropout_patches(feats, args.dropout_patch)
        encoded_vectors =  np.concatenate((feats, 16*nor_pos_arr), axis=1)
        
        kmeans = KMeans(n_clusters=args.num_k, n_init=20)
        kmeans.fit(encoded_vectors)
        labels = kmeans.labels_
        result = group_data(feats, labels)
        
        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []
        bag_label = Tensor(np.array([label])).clone()
        for category, data_group in result.items():
            
            slide_sub_labels.append(Tensor(np.array([label])).clone())
            bag_feats = Variable(Tensor(np.array([data_group])))
            bag_feats = bag_feats.view(-1, args.feats_size)
            
            A = attention(bag_feats)
                                
            A = A.view(-1, 1)
            attFeats = bag_feats * A
            attFeat_tensor = torch.mean(attFeats, dim=0).unsqueeze(0)
            sub_predict = classifier(attFeat_tensor) 
            
            
            slide_sub_preds.append(sub_predict)
            
            patch_pred_logits = get_cam_1d(classifier, attFeat_tensor.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

            if distill == 'MaxMinS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                d_inst_feat = bag_feats.index_select(dim=0, index=topk_idx)
                slide_pseudo_feat.append(d_inst_feat.detach())
            elif distill == 'MaxS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx = topk_idx_max
                d_inst_feat = bag_feats.index_select(dim=0, index=topk_idx)
                slide_pseudo_feat.append(d_inst_feat.detach())
            elif distill == 'AFS':
                slide_pseudo_feat.append(attFeat_tensor.detach())
                            
        slide_sub_preds_tensor = torch.cat(slide_sub_preds, dim=0)
        slide_sub_labels_tensor = torch.cat(slide_sub_labels, dim=0) 
        #print(slide_sub_preds_tensor)
        loss_t1 = criterion(slide_sub_preds_tensor, slide_sub_labels_tensor).mean()
        optimizer_t1.zero_grad()
        loss_t1.backward()        
        optimizer_t1.step()
        
        slide_pseudo_feat_tensor = torch.cat(slide_pseudo_feat, dim=0)

        ins_prediction, bag_prediction, _, _ = milnet(slide_pseudo_feat_tensor)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss_t2 = b_loss*bag_loss + m_loss*max_loss
        optimizer_t2.zero_grad()
        loss_t2.backward()
        optimizer_t2.step()
        
        total_loss = total_loss + loss_t2.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss_t2.item()))
        #print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
        
        
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_df, milnet, attention, classifier, criterion, optimizer_t1, optimizer_t2, args):
    b_loss = args.b_loss
    m_loss = 1-b_loss

    milnet.eval()
    attention.eval()
    classifier.eval()
    
    distill = args.distill_type
    instance_per_group = len(test_df) // args.num_k
    
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):            
            label, feats, pos_arr = get_bag_feats(test_df.iloc[i], args)
        
            scaler = MinMaxScaler()
            nor_pos_arr = scaler.fit_transform(pos_arr)

            max_x = np.amax(pos_arr, 0)[0]
            max_y = np.amax(pos_arr, 0)[1]

            #feats = dropout_patches(feats, args.dropout_patch)
            encoded_vectors =  np.concatenate((feats, 16*nor_pos_arr), axis=1)

            kmeans = KMeans(n_clusters=args.num_k, n_init=20)
            kmeans.fit(encoded_vectors)
            labels = kmeans.labels_
            result = group_data(feats, labels)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []
            bag_label = Tensor(np.array([label])).clone()
        
            for category, data_group in result.items():
                slide_sub_labels.append(Tensor(np.array([label])).clone())
                bag_feats = Variable(Tensor(np.array([data_group])))
                bag_feats = bag_feats.view(-1, args.feats_size)
                
                A = attention(bag_feats)
                
                A = A.view(-1, 1)
                attFeats = bag_feats * A
                attFeat_tensor = torch.mean(attFeats, dim=0).unsqueeze(0)
                sub_predict = classifier(attFeat_tensor) 
            
                print(sub_predict)
                slide_sub_preds.append(sub_predict)
                
                patch_pred_logits = get_cam_1d(classifier, attFeat_tensor.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                if distill == 'MaxMinS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx_min = sort_idx[-instance_per_group:].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    d_inst_feat = bag_feats.index_select(dim=0, index=topk_idx)
                    slide_pseudo_feat.append(d_inst_feat.detach())
                elif distill == 'MaxS':
                    topk_idx_max = sort_idx[:instance_per_group].long()
                    topk_idx = topk_idx_max
                    d_inst_feat = bag_feats.index_select(dim=0, index=topk_idx)
                    slide_pseudo_feat.append(d_inst_feat.detach())
                elif distill == 'AFS':
                    slide_pseudo_feat.append(attFeat_tensor.detach())
                
            slide_sub_preds_tensor = torch.cat(slide_sub_preds, dim=0)
            slide_sub_labels_tensor = torch.cat(slide_sub_labels, dim=0) 
            loss_t1 = criterion(slide_sub_preds_tensor, slide_sub_labels_tensor).mean()

            slide_pseudo_feat_tensor = torch.cat(slide_pseudo_feat, dim=0)
              
            ins_prediction, bag_prediction, _, _ = milnet(slide_pseudo_feat_tensor)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss_t2 = b_loss*bag_loss + m_loss*max_loss

            total_loss = total_loss + loss_t2.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f\n' % (i, len(test_df), loss_t2.item()))
            #print('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(m_loss*torch.sigmoid(max_prediction)+b_loss*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: 
                test_predictions.extend([(m_loss*torch.sigmoid(max_prediction)+b_loss*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])

            print((m_loss*torch.sigmoid(max_prediction)+b_loss*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy())

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.1 , type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--store_dir', type=str, default=None) 
    parser.add_argument('--sigmoid', type=str, default=False)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--b_loss', type=float, default=0.9)
    parser.add_argument('--with_pos', type=str, default=0)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--num_block', default=2, type=int)
    parser.add_argument('--num_landmark', default=256, type=int)
    parser.add_argument('--num_k', default=5, type=int)
    parser.add_argument('--distill_type', default='MaxS', type=str) ## MaxMinS, MaxS, AFS

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    b_loss = args.b_loss
    m_loss = 1-b_loss
    print('lr: ', args.lr )
    print('b_loss: ', b_loss)
    print('m_loss: ', m_loss)
    
    '''
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    '''
    if args.with_pos == '1':
        import dsmil_efficient_vit_with_pos as mil
        print('with pos')
    else:
        import dsmil_efficient_vit as mil
        print('without pos')

    import nystrom_attention

    print(nystrom_attention.__path__)
    efficient_transformer = Nystromformer(dim = args.feats_size,
                                          depth = args.num_block,
                                          heads = args.num_head,
                                          num_landmarks = args.num_landmark)
    
    
    attention = abmil.GatedAttention(args.feats_size).cuda()
    classifier = abmil.Classifier_1fc(args.feats_size, args.num_classes, args.dropout_patch).cuda()
    
    
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.ViT_1d( num_classes = args.num_classes,
                            dim = args.feats_size,
                            transformer = efficient_transformer).cuda()
    
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    criterion = nn.BCEWithLogitsLoss()
    
    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    
    optimizer_t1 = torch.optim.Adam(trainable_parameters, lr=args.lr,  weight_decay=args.weight_decay)
    optimizer_t2 = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler_t1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_t1, [100])
    scheduler_t2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t2, args.num_epochs, 0.000005)
    
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        
    bags_path = pd.read_csv(bags_csv)
    train_path = bags_path.iloc[0:int(len(bags_path)*(1-args.split)), :]
    test_path = bags_path.iloc[int(len(bags_path)*(1-args.split)):, :]
    best_score = 0
    save_path = os.path.join(args.store_dir, datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, attention, classifier, criterion, optimizer_t1, optimizer_t2, args) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, attention, classifier, criterion, optimizer_t1, optimizer_t2, args)
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        scheduler_t1.step()
        scheduler_t2.step()
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            milnet_save_name = os.path.join(save_path, 'milnet_' + args.prefix + '_' + str(run+1)+ '_' + str(epoch) +'.pth')
            cls_save_name = os.path.join(save_path, 'classifier_' + args.prefix + '_' + str(run+1)+ '_' + str(epoch) +'.pth')
            attn_save_name = os.path.join(save_path, 'attn_' + args.prefix + '_' + str(run+1)+ '_' + str(epoch) +'.pth')
            torch.save(milnet.state_dict(), milnet_save_name)
            torch.save(attention.state_dict(), attn_save_name)
            torch.save(classifier.state_dict(), cls_save_name)
            if args.dataset=='TCGA-lung':
                print('Best model saved at: ' + milnet_save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + milnet_save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            

if __name__ == '__main__':
    b_loss = None
    m_loss = None

    main()

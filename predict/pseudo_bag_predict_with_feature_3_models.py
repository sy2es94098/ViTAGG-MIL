import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
import copy
import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
from nystrom.nystrom_attention import Nystromformer
import json
import datetime
from sklearn.utils import shuffle
from torch.autograd import Variable
import pickle

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

import abmil as abmil
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from utils import get_cam_1d

b = 1
m = 1-b

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

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
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
    return aucs, thresholds, thresholds_optimal, fpr, tpr

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, test_df, milnet, classifier, attention):
    b_loss = args.b_loss
    m_loss = 1-b_loss
    jsfile = open(args.model + '_' + datetime.date.today().strftime("%m%d%Y") + '_' + args.postfix + '.json','w')
    
    score = {}
    milnet.eval()
    attention.eval()
    classifier.eval()
    
    distill = args.distill_type
    instance_per_group = len(test_df) // args.num_k
    
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    csvs = shuffle(test_df).reset_index(drop=True)
    #Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):

            print(test_df.iloc[i].iloc[0])
            label, feats, pos_arr = get_bag_feats(test_df.iloc[i], args)

            scaler = MinMaxScaler()
            nor_pos_arr = scaler.fit_transform(pos_arr)

            max_x = np.amax(pos_arr, 0)[0]
            max_y = np.amax(pos_arr, 0)[1]

            #feats = dropout_patches(feats, args.dropout_patch)
            encoded_vectors =  np.concatenate((feats, 16*nor_pos_arr), axis=1)

            kmeans = KMeans(n_clusters=args.num_k, n_init=10, random_state=87)
            kmeans.fit(encoded_vectors)
            labels = kmeans.labels_
            result = group_data(feats, labels)

            bag_label = Tensor(np.array([label])).clone()
            slide_pseudo_feat = []
            for category, data_group in result.items():
                print(len(data_group))
                bag_feats = Variable(Tensor(np.array([data_group])))
                bag_feats = bag_feats.view(-1, args.feats_size)

                A = attention(bag_feats)
                
                A = A.view(-1, 1)
                attFeats = bag_feats * A
                attFeat_tensor = torch.mean(attFeats, dim=0).unsqueeze(0)
                sub_predict = classifier(attFeat_tensor) 
            
                print(sub_predict)
                
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
                    
                    
            slide_pseudo_feat_tensor = torch.cat(slide_pseudo_feat, dim=0)

            ins_prediction, bag_prediction, _, _ = milnet(slide_pseudo_feat_tensor)
            max_prediction, _ = torch.max(ins_prediction, 0) 

            del ins_prediction
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            max_prediction = torch.sigmoid(max_prediction).squeeze().cpu().numpy()


            bag_prediction = m_loss*max_prediction+b_loss*bag_prediction
            print(bag_prediction)
            test_labels.extend([label])
            test_predictions.extend([bag_prediction])



    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal, fpr, tpr = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    bag_score = 0
    for i in range(0, len(test_predictions)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_predictions)  
    
    print('acc: ' + str(avg_score))
    print('auc: ' + str(auc_value))
    print('threshold: ' + str(thresholds_optimal))

    with open('tcga_' + args.model + '_'  + args.postfix + '.pickle','wb') as p:
        pickle.dump([fpr,tpr], p)

    json.dump(score, jsfile)
    jsfile.close()
    
    return avg_score, auc_value, thresholds_optimal  
            
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_luad', type=float, default=0.7371)
    parser.add_argument('--thres_lusc', type=float, default=0.2752)
    parser.add_argument('--aggr_weight', type=str, default=None)
    parser.add_argument('--attn_weight', type=str, default=None)
    parser.add_argument('--cls_weight', type=str, default=None)
    parser.add_argument('--store_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--patch', type=str, default='test_patches/patches/')
    parser.add_argument('--postfix', type=str, default='0')
    parser.add_argument('--b_loss', type=float, default=0.9)
    parser.add_argument('--dataset', type=str, default='TCGA')
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--num_block', default=2, type=int)
    parser.add_argument('--num_landmark', default=256, type=int)
    parser.add_argument('--with_pos', type=str, default=0)
    parser.add_argument('--num_k', default=5, type=int)
    parser.add_argument('--distill_type', default='MaxS', type=str) ## MaxMinS, MaxS, AFS

    args = parser.parse_args()
    
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.model == 'vit':
        if args.with_pos == '1':
            import dsmil_efficient_vit_with_pos as mil
            print('with pos')
        else:
            import dsmil_efficient_vit as mil
            print('without pos')

        from nystrom_attention import Nystromformer

        efficient_transformer = Nystromformer(dim = args.feats_size,
                                          depth = args.num_block,
                                          heads = args.num_head,
                                          num_landmarks = args.num_landmark)

        b_classifier = mil.ViT_1d( num_classes = args.num_classes,
                            dim = args.feats_size,
                            transformer = efficient_transformer).cuda()   
    
    elif args.model == 'origin':
        import dsmil as mil
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
        
    
    attention = abmil.GatedAttention(args.feats_size).cuda()
    classifier = abmil.Classifier_1fc(args.feats_size, args.num_classes).cuda()
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    #i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    
    state_dict_weights = torch.load(args.cls_weight)
    
    new_state_dict = OrderedDict()

    state_dict_init = classifier.state_dict()
    print('-------------CLS--------------')
        
    for k,v in state_dict_weights.items():
        print(k)

    print('-----------------------------')
    for k,v in state_dict_init.items():
        print(k)
        
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict, strict=True)
    
    state_dict_weights = torch.load(args.attn_weight)
    
    new_state_dict = OrderedDict()

    state_dict_init = attention.state_dict()
    print('-------------ATTN-------------')
        
    for k,v in state_dict_weights.items():
        print(k)

    print('-----------------------------')
    for k,v in state_dict_init.items():
        print(k)
        
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    attention.load_state_dict(new_state_dict, strict=True)
    
    state_dict_weights = torch.load(args.aggr_weight)

    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)
    
    bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')        
    bags_path = pd.read_csv(bags_csv)
    test_path = bags_path.iloc[0:,:]
    
    os.makedirs(os.path.join(args.store_dir, 'output'), exist_ok=True)
    test(args, test_path, milnet, classifier, attention)

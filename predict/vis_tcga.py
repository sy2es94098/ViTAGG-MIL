import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

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

b = 1
m = 1-b

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

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

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def test(args, bags_list, milnet):
    jsfile = open('tcga_' + args.model + '_' + datetime.date.today().strftime("%m%d%Y") + '_' + args.post_fix + '.json','w')
    score = {}
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        print(bags_list[i])
        feats_list = []
        pos_list = []
        classes_list = []

        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            try:
                bag_prediction, A, idx = milnet.b_classifier(bag_feats, ins_classes)
                print(bag_prediction)
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
                print(bag_prediction)
                max_prediction, _ = torch.max(ins_classes, 0)
                max_prediction = torch.sigmoid(max_prediction).squeeze().cpu().numpy()
                print(max_prediction)
                bag_prediction = b*bag_prediction+m*max_prediction
                '''
                if bag_prediction < args.thres_luad:
                    print(bags_list[i] + ' is detected as LUAD')
                else:
                    print(bags_list[i] + ' is detected as LUSC')

                '''
                '''
                if bag_prediction[0] >= args.thres_luad and bag_prediction[1] < args.thres_lusc:
                    print(bags_list[i] + ' is detected as: LUAD')
                    
                elif bag_prediction[1] >= args.thres_lusc and bag_prediction[0] < args.thres_luad:
                    print(bags_list[i] + ' is detected as: LUSC')
                elif bag_prediction[0] < args.thres_luad and bag_prediction[1] < args.thres_lusc:
                    print(bags_list[i] + ' is detected as: benign')
                else:
                    print(bags_list[i] + ' is detected as: both LUAD and LUSC')
                '''
                print(bag_prediction)

                file_name = bags_list[i]
                bag_prediction = float(bag_prediction)
                #bag_prediction = list(bag_prediction)
                #bag_prediction = [float(i) for i in bag_prediction]

                score[file_name[file_name.rfind('/')+1:]] = bag_prediction
                print(score)
            except Exception as e:
                print(e)

    json.dump(score, jsfile)
    jsfile.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_luad', type=float, default=0.7371)
    parser.add_argument('--thres_lusc', type=float, default=0.2752)
    parser.add_argument('--aggr_weight', type=str, default=None)
    parser.add_argument('--store_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--patch', type=str, default='test_patches/patches/')
    parser.add_argument('--post_fix', type=str, default='0')


    args = parser.parse_args()
    
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.model == 'vit':
        import dsmil_efficient_vit as mil
        from nystrom.nystrom_attention import Nystromformer

        efficient_transformer = Nystromformer(dim = args.feats_size,
                                          depth = 2,
                                          heads = 2,
                                          num_landmarks = 256)

        b_classifier = mil.ViT_1d( num_classes = args.num_classes,
                            dim = args.feats_size,
                            transformer = efficient_transformer).cuda()   
    
    elif args.model == 'origin':
        import dsmil as mil
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()

    
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()

    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    state_dict_weights = torch.load(os.path.join('test', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
    state_dict_weights = torch.load(args.aggr_weight)
    for k,v in state_dict_weights.items():
        print(k)

    print('-----------------------------')
    for k,v in milnet.state_dict().items():
        print(k)

    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)
    
    bags_list = glob.glob(os.path.join(args.patch, '*'))
    os.makedirs(os.path.join(args.store_dir, 'output'), exist_ok=True)
    test(args, bags_list, milnet)

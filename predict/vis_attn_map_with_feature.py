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
from sklearn.utils import shuffle
from torch.autograd import Variable


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

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    #print(args.feats_size)
    pos = feats.iloc[:,int(args.feats_size):]
    feats = feats.iloc[:,:int(args.feats_size)]
    feats = feats.to_numpy()
    pos = pos.to_numpy()
    pos = np.squeeze(pos)
    pos = np.array([np.fromstring(s.strip("[] "), dtype=int, sep=" ") for s in pos])
    
    #print(feats)
    #print(pos)
    #print(np.shape(feats))
    #print(np.shape(pos))
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats, pos

def test(args, test_df, milnet):
    b = args.b_loss
    m = 1-b
    
    print(b)
    print(m)

    jsfile = open(args.model + '_' + datetime.date.today().strftime("%m%d%Y") + '_' + args.postfix + '.json','w')
    score = {}
    milnet.eval()
    

    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    csvs = shuffle(test_df).reset_index(drop=True)
    
    for i in range(len(test_df)):
        #print(test_df.iloc[i])
        try:
            feats_list = []
            pos_arr = []
            classes_list = []

            file_name = os.path.basename(test_df.iloc[i].iloc[0])
            file_name = os.path.splitext(file_name)
            #print(file_name)
            idx_path = os.path.join(args.bg_dir,file_name[0],'bg')
            files = os.listdir(idx_path)
            x_arr = []
            y_arr = []
            for f in files:
                x_arr.append(int(f.split('.')[0].split('_')[0]))
                y_arr.append(int(f.split('.')[0].split('_')[1]))

            x_arr.sort(reverse=True)
            y_arr.sort(reverse=True)

            bg_x = int(x_arr[0])
            bg_y = int(y_arr[0])
            del x_arr
            del y_arr
            del files

            #bg_x = 336
            #bg_y = 209
            #print(bg_x, bg_y)

            with torch.no_grad():
                classes_list, feats_arr, pos_arr = get_bag_feats(test_df.iloc[i], args)
                #print(np.shape(classes_list))
                #print(np.shape(feats_arr))
                bag_feats = Variable(Tensor(np.array([feats_arr])))
                bag_feats = bag_feats.view(-1, args.feats_size)
                ins_prediction, bag_prediction, A, idx = milnet(bag_feats)
                #print(np.shape(ins_prediction))
                #print(np.shape(bag_prediction))
                #print(np.shape(A))

                if args.model == 'vit':
                    A = torch.squeeze(A, dim=0)
                    A = torch.mean(A,dim=0)
                    data_num = len(feats_arr)
                    A = A[-data_num-1,-data_num:]
                    #A = A[-data_num+idx,-data_num:]
                    A = A.view(-1,1)
                    #print(A)

                max_prediction, _ = torch.max(ins_prediction, 0)
                max_prediction = torch.sigmoid(max_prediction).squeeze().cpu().numpy()
                bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()

                #print(max_prediction)
                #print(bag_prediction)

                bag_prediction = b*bag_prediction+m*max_prediction

                color = [0, 0, 0]

                if bag_prediction >= args.thres_tumor:
                    print(file_name[0] + ' is detected as malignant')
                    color = [1, 0, 0]
                    attentions = A
                else:
                    attentions = A
                    print(file_name[0] + ' is detected as benign')
                #print(np.amax(pos_arr, 0))
                #print(bg_x)
                #print(bg_y)
                #print(pos_arr)
                max_x = max(bg_x,np.amax(pos_arr, 0)[0])
                max_y = max(bg_y,np.amax(pos_arr, 0)[1])

                print(bag_prediction)

                del A
                #max_x = np.amax(pos_arr, 0)[0]
                #max_y = np.amax(pos_arr, 0)[1]
                color_map = np.zeros((max_x+1, max_y+1, 3))
                #color_map = np.zeros((219, 494, 3))

                #print(np.amax(pos_arr, 0),  np.amax(pos_arr, 0))
                attentions = attentions.cpu().numpy()
                attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
                #print(attentions)
                for k, pos in enumerate(pos_arr):
                    tile_color = np.asarray(color) * attentions[k]
                    color_map[pos[0], pos[1]] = tile_color
                slide_name = file_name[0].split(os.sep)[-1]
                color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
                io.imsave(os.path.join(args.store_dir, 'output', slide_name+'.png'), img_as_ubyte(color_map))   
                score[file_name[0]] = float(bag_prediction)
                del color_map
                #print(score)
        except:    
            print(file_name)
            


    json.dump(score, jsfile)
    jsfile.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.1964)
    parser.add_argument('--aggr_weight', type=str, default=None)
    parser.add_argument('--store_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--patch', type=str, default='test_patches/patches/')
    parser.add_argument('--postfix', type=str, default='0')
    parser.add_argument('--b_loss', type=float, default=0.9)   
    parser.add_argument('--dataset', type=str, default='C16_ms')
    parser.add_argument('--bg_dir', type=str, default='C16_ms')
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--num_block', default=2, type=int)
    parser.add_argument('--num_landmark', default=256, type=int)

    args = parser.parse_args()
    
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.model == 'vit':
        import dsmil_efficient_vit as mil
        from nystrom.nystrom_attention import Nystromformer

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

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    #i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()

    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    '''
    state_dict_weights = torch.load(os.path.join('test-c16-vit', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    
    for k,v in state_dict_weights.items():
        print(k)

    print('-----------------------------')
    for k,v in state_dict_init.items():
        print(k)
        
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
    '''
    state_dict_weights = torch.load(args.aggr_weight)
    '''
    for k,v in state_dict_weights.items():
        print(k)

    print('-----------------------------')
    for k,v in milnet.state_dict().items():
        print(k)
    '''
    
    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)
    
    bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')        
    bags_path = pd.read_csv(bags_csv)
    test_path = bags_path.iloc[0:,:]
    
    os.makedirs(os.path.join(args.store_dir, 'output'), exist_ok=True)
    test(args, test_path, milnet)

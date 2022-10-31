import glob
import os
import random
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import xml.etree.ElementTree as ET
from typing import Dict, Any
import albumentations as A
import collections
import transforms as T
from tool.utils import *
from engine import train_one_epoch, evaluate
from metric import *

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

"""
voc data
"""
class load_voc():
    def __init__(self, root_dir, set_name='VOC2007_test'):
        self.root_dir = root_dir
        self.set_name = set_name
        
        self.classes = [
                        "aeroplane",
                        "bicycle",
                        "bird",
                        "boat",
                        "bottle",
                        "bus",
                        "car",
                        "cat",
                        "chair",
                        "cow",
                        "diningtable",
                        "dog",
                        "horse",
                        "motorbike",
                        "person",
                        "pottedplant",
                        "sheep",
                        "sofa",
                        "train",
                        "tvmonitor"
                    ]
        
        # preparing a data list
        #dataset = 'VOC2007' # or VOC2007_test, VOC2012, VOC2007+2012
        #path = '/nasdata2/khj/objectdetection/segmentation/VOCdevkit/'
        if self.set_name in ['VOC2007', 'VOC2007_test', 'VOC2012']:

            img_base_path = os.path.join(self.root_dir,self.set_name,'JPEGImages')

            self.img_files = glob.glob(os.path.join(img_base_path,'*.jpg'))
            self.label_files = [ i.replace('JPEGImages','Annotations').replace('jpg','xml') for i in self.img_files ]

        elif self.set_name == 'VOC2007+2012':

            img_base_path_2007, img_base_path_2012 = os.path.join(self.root_dir,'VOC2007','JPEGImages'), os.path.join(self.root_dir,'VOC2012','JPEGImages')

            img_files_2007, img_files_2012 = glob.glob(os.path.join(img_base_path_2007,'*.jpg')), glob.glob(os.path.join(img_base_path_2012,'*.jpg'))
            label_files_2007 = [ i.replace('JPEGImages','Annotations').replace('jpg','xml') for i in img_files_2007 ]
            label_files_2012 = [ i.replace('JPEGImages','Annotations').replace('jpg','xml') for i in img_files_2012 ]

            self.img_files, self.label_files = img_files_2007 + img_files_2012, label_files_2007 + label_files_2012

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml 파일을 dictionary로 반환
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def get_annotations(self,annot_file):
        target = self.parse_voc_xml(ET.parse(annot_file).getroot())
        annotations = np.zeros((0, 5))
        
        for t in target['annotation']['object']:
            annotation = np.zeros((1, 5))
            annotation[0, 1:] = np.array( [ t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'] ] )
            annotation[0, 0] = self.classes.index(t['name'])
            annotations = np.append(annotations, annotation, axis=0)
        
        return annotations

    def get_dataset(self):
        
        print('We found {} files... Read images/labels...'.format(len(self.img_files)))
        cnt = 0
        since = time.time()
        img_all, label_all = [], []
        for idx in range(len(self.img_files)):
            if cnt % 1000 == 0 and cnt != 0: 
                time_elapsed = time.time() - since
                print('\t Current idx {}... complete in {:.0f}m {:.0f}s'.format(cnt, time_elapsed // 60, time_elapsed % 60))
            img_all.append(np.array(Image.open(self.img_files[idx]).convert('RGB')))
            label_all.append(self.get_annotations(self.label_files[idx]))
            cnt += 1
            
        return img_all, label_all

"""
    Parcing params
"""
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None, help="Please give a pretrained-weight path.")
parser.add_argument('--prefix', default='voc_tmp', help="Please give an output prefix.")
parser.add_argument('--GPU_NUM', type=int, default=0, help="Please give a GPU_NUM.")
parser.add_argument('--save_dir', default=None, help="Please give a save_dir.")
parser.add_argument('--img_size', type=int, default=608, help="Please give a image size.")
parser.add_argument('--batch_size', type=int, default=8, help="Please give a batch_size.")
parser.add_argument('--nbepoch', type=int, default=20, help="Please give a # of epochs.")
parser.add_argument('--lr', type=float, default=0.001, help="Please give a learning rate.")
parser.add_argument('--momentum', type=float, default=0.9, help="Please give a momentum.")
parser.add_argument('--wd', type=float, default=5e-4, help="Please give a weight decay coefficient.")
parser.add_argument('--stepsize', nargs='+', type=int, default=[8,16], help="Please give a stepsize for lr scheduler.")
parser.add_argument('--gamma', type=float, default=0.1,help="Please give a gamma for lr scheduler.")
parser.add_argument('--print_freq', type=int, default=5,help="Please give a print_freq.")
args = parser.parse_args()

"""
Dataset 
"""
def get_transformA(mode='train',image_size=608,format='pascal_voc'):
    if mode == 'train':
        t = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,border_mode=cv2.BORDER_CONSTANT),  
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
        ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.5)
        )
    elif mode == 'test':
        t =  A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,border_mode=cv2.BORDER_CONSTANT),  
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
        ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'])
        )
        
    return t

class VOCdataset(torch.utils.data.Dataset):
    def __init__(self, img_set, label_set, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = img_set
        self.labs = label_set

    def __getitem__(self, idx):
        # load image
        img = copy.copy(self.imgs[idx])
        lab = copy.copy(self.labs[idx])
        
        # get bounding box coordinates for each mask
        num_objs = np.shape(lab)[0]
        boxes, labels = [], []
        for i in range(num_objs):
            xmin = lab[i][1]
            xmax = lab[i][3]
            ymin = lab[i][2]
            ymax = lab[i][4]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(lab[i][0] + 1) # assign '0' to the bg class
        
        if self.transforms is not None:
            augmentations = self.transforms(image=np.array(img), bboxes=np.vstack(boxes), class_labels=np.vstack(labels))
            img = augmentations['image']
            boxes = np.array(augmentations['bboxes'])
            labels = np.array(augmentations['class_labels'])
            
        nl = len(boxes)
        boxes_out = torch.zeros((nl, 4)).type(torch.FloatTensor)
        labels_out = torch.zeros((nl)).type(torch.LongTensor)
        if nl:
            boxes_out = torch.from_numpy(boxes).type(torch.FloatTensor)
            labels_out = torch.from_numpy(labels.reshape(-1)).type(torch.LongTensor)

        image_id = torch.tensor([idx])
        area = (boxes_out[:, 3] - boxes_out[:, 1]) * (boxes_out[:, 2] - boxes_out[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_out
        target["labels"] =  labels_out
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
                               
        return img, target

    def __len__(self):
        return len(self.imgs)

"""
    Main
"""
def main(args):
    
    device = gpu_setup(True, args.GPU_NUM)
    
    print("Data loading")
    voc_loader_train = load_voc(root_dir='/nasdata2/khj/objectdetection/segmentation/VOCdevkit/', set_name='VOC2007+2012')
    voc_loader_test = load_voc(root_dir='/nasdata2/khj/objectdetection/segmentation/VOCdevkit/', set_name='VOC2007_test')

    imgs_train, labs_train = voc_loader_train.get_dataset()
    imgs_test, labs_test = voc_loader_test.get_dataset()
    print('After loading, there are {}/{} samples for the train/test respectively.'.format(len(imgs_train),len(imgs_test)))
    
    # use our dataset and defined transformations
    dataset_train = VOCdataset(imgs_train, labs_train, get_transformA('train',args.img_size))
    dataset_test = VOCdataset(imgs_test, labs_test, get_transformA('test',args.img_size))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,collate_fn=collate_fn)
    
    # Model creating
    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # (optional) load the pre-trained weights
    if args.pretrained is not None:
        pretrained_dict = torch.load(args.pretrained,map_location=torch.device('cpu')) 
        mod_dicts = dict((key.replace('module.',''), value) for (key, value) in pretrained_dict.items())
        model.backbone.load_state_dict(mod_dicts)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21) # 20 classes + bg
    
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    
    # Training
    print('Start training')
    start_time = time.time()
    for epoch in range(args.nbepoch):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))  
    
    #
    print('Saving')
    if args.save_dir:
        save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args},
            os.path.join(args.save_dir, '{}_model_{}.pth'.format(args.prefix,epoch)))
     
            
main(args)

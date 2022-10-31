import os
import numpy as np
import torch
import copy
from PIL import Image
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import albumentations as A
import albumentations.pytorch 

class PBBMDataset_yolo(torch.utils.data.Dataset):
    def __init__(self, img_set, label_set, transforms,train):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = img_set
        self.labs = label_set
        self.train = train

    def __getitem__(self, idx):
        if not self.train:
            return self._get_val_item(idx)
        # load image
        img = copy.copy(self.imgs[idx])
        lab = copy.copy(self.labs[idx])
        
        # augmentations
        if self.transforms is not None:
            augmentations = self.transforms(image=img, bboxes=lab[:, 1:], class_labels=lab[:, 0])
            img = augmentations['image']
            boxes = np.array(augmentations['bboxes'])
            labels = np.array(augmentations['class_labels'])
            
        
        nl = len(labels)
        labels_out = np.zeros((nl, 5))
        if nl:
            labels_out[:, :4] = torch.from_numpy(boxes)
            labels_out[:, 4] = torch.from_numpy(labels)
        out_bboxes1 = np.zeros([60, 5])
        out_bboxes1[:min(labels_out.shape[0], 60)] = labels_out[:min(labels_out.shape[0], 60)]
        
        # Convert 
        img = np.ascontiguousarray(img) * 255
                               
        #return img.astype(np.uint8), labels_out
        return img.astype(np.uint8), out_bboxes1
    
    def _get_val_item(self, idx):
        # load image
        img = copy.copy(self.imgs[idx])
        lab = copy.copy(self.labs[idx])
        
        # augmentations
        if self.transforms is not None:
            augmentations = self.transforms(image=img, bboxes=lab[:, 1:], class_labels=lab[:, 0])
            img = augmentations['image']
            boxes = np.array(augmentations['bboxes'])
            labels = np.array(augmentations['class_labels'])
            
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # x1,y1,x2,y2 --> x1, y1, w, h 
        boxes[...,:2] = boxes[...,:2] + boxes[...,2:]/2 # x1, y1, w, h --> xc,yc,w,h
        num_objs = np.shape(boxes)[0]
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels.flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        
        img = np.ascontiguousarray(img) * 255
        img = img.astype(np.uint8)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
    

class PBBMDataset(torch.utils.data.Dataset):
    def __init__(self, img_set, label_set, transforms, is_bg_in=False):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = img_set
        self.labs = label_set
        self.is_bg_in = is_bg_in

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
            if self.is_bg_in:
                labels.append(lab[i][0])
            else:
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
    
def collate_yolo(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes

def get_transformA(mode='train',image_size=608,format='pascal_voc'):
    if mode == 'train':
        t = A.Compose([
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        # 수평 뒤집기
        A.HorizontalFlip(p=0.5),
        # blur
        A.GaussianBlur(sigma_limit=(.1,2.), p=0.5),
        # grayscale
        A.ToGray(p=0.2),
        # normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
        ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.5)
        )
    elif mode == 'test':
        t =  A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
        ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'])
        )
        
    return t
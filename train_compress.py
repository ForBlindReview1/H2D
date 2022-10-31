import glob
import json
import os
import random
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
from tool.utils import *
from data import *
from engine import train_one_epoch, evaluate
from metric import *

'''
Parcing arguments
'''

parser = argparse.ArgumentParser('argument for training')

parser.add_argument('--root_dir', type=str, help='path to dataset')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
parser.add_argument('--save_freq', type=int, default=15, help='save frequency')
parser.add_argument('--save_fig', type=bool, default=True, help='save figure')
parser.add_argument('--img_size', type=int, default=608, help='image size')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--samp', type=int, default=0, help='sampling number')

# optimization
parser.add_argument('--learning_rate', type=float, default=0.9, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

# Compress
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden embedding dimension for the predict layers')
parser.add_argument('--feat_dim', type=int, default=12544, help='Feature dimension for the Roi align output (e.g. 7*7*256 for the FPN)')
parser.add_argument('--numk', type=int, default=64, help='The number of anchor points')
parser.add_argument('--temp_t', type=float, default=0.001)
parser.add_argument('--temp_s', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.99)

# GPU setting
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

# paths
parser.add_argument('--checkpoint_path', default='./output_compress/', type=str,
                    help='where to save checkpoints. ')

opt = parser.parse_args()
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp','gif'] 
img_dir, lab_dir = opt.root_dir + '/images/', opt.root_dir + '/labels/'


'''
Read caches
'''
f = []  # image files
for p in path if isinstance(img_dir, list) else [img_dir]:
    p = Path(p)
    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
label_files = img2label_paths(img_files)  # labels

print('Cached images / labels : {}, and {}'.format(len(img_files),len(label_files)))

'''
Load data
'''
img_all, label_all = [], []
nins = 0
for sample in range(len(img_files)):
    # read
    img, labels = read_data(img_files,label_files,xywhn=False,subj_index=sample,img_size=opt.img_size)
    if labels.ndim == 1: 
        labels = labels.reshape(1,-1)
    if np.shape(labels)[-1] == 0: 
        continue
    nins += np.shape(labels)[0]

    # stack
    img_all.append(img)
    label_all.append(labels)
print('From all images, {} instances were detected.'.format(nins))

'''
Define the transformation
'''

def get_transformA(image_size=608,format='pascal_voc'):
    t = A.Compose([
    A.RandomResizedCrop (608, 608, scale=(0.2, 1.), p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    # grayscale
    A.ToGray(p=0.2),
    # blur
    A.GaussianBlur(sigma_limit=(.1,2.), p=0.5),    
    # 수평 뒤집기
    A.HorizontalFlip(p=0.5),
    # normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255)
    ],
        bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.5,label_fields=['class_labels'])
    )
    
    return t

'''
Dataloader
'''
class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, img_set, label_set, samp=0, transforms=None):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.numk = numk
        self.imgs = img_set
        self.labs = label_set
        self.samp = samp
        self.tuples = self.generate_tuple_queues(self.labs,self.samp)
        self.nins = len(self.tuples)
        
    def generate_tuple_queues(self,labels,samp):
        
        memory_querys = []
        idx = 0
        for lab in labels:
            nins = np.shape(lab)[0]
            for ins in range(nins):
                memory_querys.append([idx,ins])
            idx += 1
        
        '''
        random sampling
        '''
        if samp > 0:
            memory_querys = random.sample(memory_querys, samp) 
            print('After random sampling, {} instances were survived.'.format(len(memory_querys)))
            
        return memory_querys
      
    def set_data_from_tuples(self,tup):

        img = self.imgs[tup[0]]
        lab = self.labs[tup[0]][:,1:]
        bboxes = xywhn2xyxy(lab,np.shape(img)[0],np.shape(img)[1])
        bbox = bboxes[tup[1],:]

        # transform for the query tuple
        if self.transforms is not None:
            while True:
                augmentations = self.transforms(image=img, bboxes=[list(bbox)], class_labels=[1])
                img_tmp = augmentations['image']
                bbox_tmp = np.array(augmentations['bboxes'])
                if len(bbox_tmp) > 0: # if the instance was excluded from the image (e.g. near the boundary): retry the transform
                    img = img_tmp
                    bbox = bbox_tmp
                    break

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
            
        return img, bbox       

    def __getitem__(self, idx):
        
        # get a key tuple 
        k_img, k_bbox = self.set_data_from_tuples(self.tuples[idx])
        q_img, q_bbox = self.set_data_from_tuples(self.tuples[idx])
        
        return k_img, q_img, k_bbox, q_bbox
        
    def __len__(self):
        return len(self.tuples)

'''
model
'''
class Compress(torch.nn.Module):
    def __init__(self,key_model,query_model,
                 hidden_dim=1024,feat_dim=12544, m=0.99,T_k=0.01,T_q=0.1,K=1024):
        super(Compress,self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.m = m
        self.K = K
        self.T_k = T_k
        self.T_q = T_q
        self.key_model = key_model
        self.query_model = query_model

        # prediction layer for the query tuples (for student)
        self.predict = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim,self.hidden_dim,bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_dim,self.hidden_dim,bias=True),
        )
        
        for param_k in self.key_model.parameters():
            param_k.requires_grad = False
            
        # setup queue
        self.register_buffer('queue', torch.randn(self.K, self.hidden_dim))
        # normalize the queue
        self.queue = F.normalize(self.queue, dim=1)
        print(self.queue.shape)

        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
       
    def _batch_roi_align(self, multiscale_batch_output,batch_bboxex,k0=4,W=608,H=608,out_dim=7):

        w,h = batch_bboxex[:,0,2] - batch_bboxex[:,0,0], batch_bboxex[:,0,3] - batch_bboxex[:,0,1]
        val = torch.round(torch.log2(torch.sqrt( (w*h) / (W*H) )) + k0) 
        val[val<=2] = 2

        pooled=[]
        for batch in range(batch_bboxex.size(0)):
            lev = str(int(val[batch]-2))
            bbox = batch_bboxex[batch,0,:].unsqueeze(0)
            output = multiscale_batch_output[lev]
            pooled.append(torchvision.ops.roi_align(output,[bbox.float()],out_dim,output.size(-1)/W,2))

        return torch.vstack(pooled) # batch,channel,out_dim,out_dim

    def forward(self, k_img, q_img, k_bbox, q_bbox):
        
        nb = k_img.size(0) # batchsize
        
        # query features in the student model
        out_q = self.query_model(q_img)
        aligned_q = self._batch_roi_align(out_q,q_bbox).view(nb,-1)
        q = self.predict(aligned_q)
        q = F.normalize(q, dim=1) # batch,channel
        
        # compute key features in the teacher model
        with torch.no_grad():
            
            # shuffle keys
            shuffle_ids, reverse_ids = get_shuffle_ids(k_img.shape[0])
            k_img = k_img[shuffle_ids]
            k_bbox = k_bbox[shuffle_ids]
            
            out_k = self.key_model(k_img)
            aligned_k = self._batch_roi_align(out_k,k_bbox).view(nb,-1)
        
        k = self.predict(aligned_k)
        k = F.normalize(k, dim=1) # batch,channel
    
        # undo shuffle
        k = k[reverse_ids]
        
        # similarities
        queue = self.queue.clone().detach()
        sim_q = torch.mm(q, queue.t())
        sim_k = torch.mm(k, queue.t())
        
        # temperature
        sim_q /= self.T_q
        sim_k /= self.T_k
        
        # convert to prob dist
        sim_q = F.log_softmax(sim_q,dim=1)
        sim_k = F.softmax(sim_k,dim=1)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return sim_q, sim_k

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds

'''
training
'''    
def train_epoch(epoch, loader, model, criterion, optimizer, opt):
    
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    
    end = time.time()
    for idx, (k_img, q_img, k_bbox, q_bbox) in enumerate(loader):
        data_time.update(time.time() - end)

        k_img = k_img.cuda(non_blocking=True)
        q_img = q_img.cuda(non_blocking=True)
        k_bbox = k_bbox.cuda(non_blocking=True)
        q_bbox = q_bbox.cuda(non_blocking=True)
        
        sim_q, sim_k = model(k_img, q_img, k_bbox, q_bbox)
        loss = criterion(sim_q,sim_k)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), k_img.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(loader), batch_time=batch_time,
                   data_time=data_time,loss=loss_meter))
            sys.stdout.flush()
            
    return loss_meter.avg    
    
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
    main
"""
def main(opt):
    
    torch.manual_seed(opt.seed)
    # cuda setting
    gpu_setup(True,opt.gpu)
    
    # training dataloader
    dataset = SSLDataset(img_all, label_all, opt.samp,get_transformA())
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
            pin_memory=True, drop_last=True)
    
    # model initialization
    big_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # for the faster-RCNN model
    key_model = big_model.backbone # the backbone with fpn
    query_model = big_model.backbone
    
    model = Compress(key_model,query_model,
             hidden_dim = opt.hidden_dim, feat_dim = opt.feat_dim, # in the original faster-RCNN, ROIalign calculate the feature map with size of (batch,256,7,7)
             m=opt.momentum,T_k=opt.temp_t,T_q=opt.temp_s,K=opt.numk)
    model = model.cuda()
    
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=opt.learning_rate,
                                momentum=opt.sgd_momentum,
                                weight_decay=opt.weight_decay)
    
    loss_epochs = []
    for epoch in range(opt.epochs):
        print("==> training...")
        time1 = time.time()
        
        loss_epoch = train_epoch(epoch, loader, model, criterion, optimizer,opt)
        loss_epochs.append(loss_epoch)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        if epoch != 0 and epoch % opt.save_freq == 0 or epoch == opt.epochs-1:
            print('==> Saving...')
            save_file = os.path.join(opt.checkpoint_path, 'ckpt_epoch_{epoch}_tempT_{T_t}_tempS_{T_s}_lr_{lr}_k_{k}_samp{samp}.pth'.format(epoch=epoch,T_t=opt.temp_t,T_s=opt.temp_s,
                                                                                                                                           lr=opt.learning_rate,k=opt.numk,
                                                                                                                                          samp=opt.samp))
            torch.save(model.query_model.state_dict(), save_file)
            torch.cuda.empty_cache()
    if opt.save_fig:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        ax.plot(np.arange(epoch+1),np.array(loss_epochs))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('KLdiv loss')
        save_path = os.path.join(opt.checkpoint_path, 'loss_epoch_{epoch}_tempT_{T_t}_tempS_{T_s}_lr_{lr}_k_{k}_samp{samp}.png'.format(epoch=epoch,T_t=opt.temp_t,T_s=opt.temp_s,
                                                                                                                                       lr=opt.learning_rate,k=opt.numk,
                                                                                                                                          samp=opt.samp))
        fig.savefig(save_path, dpi=250)
        plt.close()    
    
main(opt)   
    
    
    
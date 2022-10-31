import glob
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

import transforms as T
from tool.utils import *
from data import *
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
    data util
"""
# preparing a data list
def loadcaches(path,idx):
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)
        f += glob.glob(str(p / '**' / '*.*'), recursive=True)
    img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    label_files = img2label_paths(img_files)  # labels
    img_all, label_all = [], []
    for sample in range(len(img_files)):
        if sample in idx: # for only training samples
            # read
            img, labels = read_data(img_files,label_files,subj_index=sample)
            # stack
            img_all.append(img)
            label_all.append(labels)
    return img_all, label_all

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# get evaluation metrics (precision, recall, AP@.5, AP@.5:.95)
def get_coco_metrics(predn,labelm,num_classes,CLASS_NAMES,save_dir='.',plotpr=False,plotconf=False):
    iouv = torch.linspace(0.5, 0.95, 10, device=torch.device("cpu")) # IoU vector (.5:.95)
    confusion_matrix = ConfusionMatrix(nc=num_classes)
    names = {k: v for k, v in enumerate(CLASS_NAMES)}
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    correct = process_batch(predn, labelm, iouv)
    confusion_matrix.process_batch(predn, labelm)

    # p,r,f1 : precision, recall, and f1score; size of (nc,)
    # ap : average precision per classes; size of (nc,10)
    # ap_class : class indices
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(correct.cpu().numpy(),predn[:, 4].cpu().numpy(),predn[:, 5].cpu().numpy(),labelm[:, 0].cpu().numpy(), 
                                                  plot=plotpr, save_dir=save_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5 (size: ncx1), AP@0.5:0.95 (size: ncx1))
    nt = np.bincount(labelm[:, 0].cpu().numpy().astype(np.int64), minlength=num_classes) # number of targets (instances) per class

    if plotconf: confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    return ap50,ap,p,r, nt

def print_evals(ap50,ap,p,r,nt,CLASS_NAMES,spacing=35):
    print('%-*s | P           | R           | mAP@.5           | mAP@.5:.95' %(spacing,'Class (#ins)'))
    print('%-*s | ------------|-------------|------------------|-----------' %(spacing,'-'*spacing))
    cnt = 0
    for c in range(len(CLASS_NAMES)):
        print('%-*s | %.4f      | %.4f      | %.4f           | %.4f' % (spacing,CLASS_NAMES[c]+' (' + str(int(nt[c])) + ')',p[c], r[c], ap50[c], ap[c]))
    print('%-*s | ------------|-------------|------------------|-----------' %(spacing,'-'*spacing))
    print('%-*s | %.4f      | %.4f      | %.4f           | %.4f' % (spacing,'All'+' (' + str(int(nt.sum())) + ')',p.mean(), r.mean(), ap50.mean(), ap.mean()))
    print('%-*s | ------------|-------------|------------------|-----------' %(spacing,'-'*spacing))
"""
    Parcing params
"""
parser = argparse.ArgumentParser()
parser.add_argument('--config', help="Please give a config.json file.")
parser.add_argument('--path', help="Please give a data path.")
parser.add_argument('--pretrained', help="Please give a pretrained-weight path.")
parser.add_argument('--prefix', help="Please give an output prefix.")
parser.add_argument('--test_index', help="Please give an index of test dataset.")
parser.add_argument('--train_index', help="Please give an index of train dataset.")
parser.add_argument('--model', help="Please give a model type.")
parser.add_argument('--GPU_NUM', help="Please give a GPU_NUM.")
parser.add_argument('--save_dir', help="Please give a save_dir.")
parser.add_argument('--img_size', help="Please give a image size.")
parser.add_argument('--plotpr', help="Please give an whether plot pr curve or not.")
parser.add_argument('--plotconf', help="Please give an whether plot confusion matrix or not.")
parser.add_argument('--K', help="Please give a fold number.")
parser.add_argument('--batch_size', help="Please give a batch_size.")
parser.add_argument('--nbepoch', help="Please give a # of epochs.")
parser.add_argument('--lr', help="Please give a learning rate.")
parser.add_argument('--momentum', help="Please give a momentum.")
parser.add_argument('--wd', help="Please give a weight decay coefficient.")
parser.add_argument('--stepsize', help="Please give a stepsize for lr scheduler.")
parser.add_argument('--gamma', help="Please give a gamma for lr scheduler.")
parser.add_argument('--print_freq', help="Please give a print_freq.")
parser.add_argument('--iou_thres', help="Please give an iou_thres for nms.")
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)
    
if args.path is not None:
    path = args.path
else:
    path = config['path']
    
if args.save_dir is not None:
    save_dir = args.save_dir
else:
    save_dir = config['save_dir']
if not os.path.exists(save_dir): os.mkdir(save_dir)
    
if args.pretrained is not None:
    pretrained = args.pretrained
else:
    pretrained = config['pretrained'] 
    
if args.prefix is not None:
    train_prefix = args.prefix
else:
    train_prefix = config['prefix'] 

if args.test_index is not None:
    test_fname = args.test_index
else:
    test_fname = config['test_index'] 
    
if args.train_index is not None:
    train_fname = args.train_index
else:
    train_fname = config['train_index'] 

if args.model is not None:
    model_type = args.model
else:
    model_type = config['model'] 

# device
if args.GPU_NUM is not None:
    config['gpu']['id'] = int(args.GPU_NUM)
    config['gpu']['use'] = True
device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
# parameters
params = config['params']
if args.img_size is not None:
    params['img_size'] = int(args.img_size)
if args.K is not None:
    params['K'] = int(args.K)
if args.plotpr is not None:
    params['plotpr'] = bool(args.plotpr)
if args.plotconf is not None:
    params['plotconf'] = bool(args.plotconf)

# network parameters
net_params = config['net_params']
if args.batch_size is not None:
    net_params['batch_size'] = int(args.batch_size)
if args.nbepoch is not None:
    net_params['nbepoch'] = int(args.nbepoch)
if args.lr is not None:
    net_params['lr'] = float(args.lr)
if args.momentum is not None:
    net_params['momentum'] = float(args.momentum)
if args.wd is not None:
    net_params['wd'] = float(args.wd)
if args.stepsize is not None:
    net_params['stepsize'] = int(args.stepsize)
if args.gamma is not None:
    net_params['gamma'] = float(args.gamma)
if args.print_freq is not None:
    net_params['print_freq'] = int(args.print_freq)
if args.iou_thres is not None:
    net_params['iou_thres'] = float(args.iou_thres)
    
#CLASS_NAMES = ['band neutrophil', 'segmented neutrophil', 'basophil', 'eosinophil', 'monocyte', 'lymphocyte', 'myelocyte', 'metamyelocyte', 'polychromatic normoblast', 'orthochromatic normoblast', 'splenic marginal zone lymphoma', 'sezary syndrome', 'reactive lymphocyte', 'prolymphocytic leukemia', 'mantle cell lymphoma', 'large granular lymphocytic leukemia', 'hairy cell leukemia', 'follicular lymphoma', 'chronic lymphocytic leukemia', 'peripheral T cell lymphoma', 'burkitts lymphoma', 'diffuse large cell lymphoma', 'waldenstrom macroglobulinemia']
CLASS_NAMES = ['__background__','band neutrophil', 'basophil', 'chronic lymphocytic leukemia', 'eosinophil', 'follicular lymphoma', 'hairy cell leukemia', 'large granular lymphocytic leukemia', 'lymphocyte', 'mantle cell lymphoma', 'monocyte', 'prolymphocytic leukemia', 'reactive lymphocyte', 'segmented neutrophil', 'sezary syndrome', 'splenic marginal zone lymphoma']
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'] 
img_size = params['img_size']







"""
    Training pipeline
"""
def train_test_pipeline(model,model_type,fold,data_loader_train,data_loader_test,device,params,net_params,save_dir,train_prefix):
    # move model to the right device
    model.to(device)

    # construct an optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=net_params['lr'],
                                momentum=net_params['momentum'], weight_decay=net_params['wd'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=net_params['stepsize'],
                                                   gamma=net_params['gamma'])

    for epoch in range(net_params['nbepoch']):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=net_params['print_freq'],
                       header=f"Fold [{fold}]-> Epoch: [{epoch}]")
        # update the learning rate
        lr_scheduler.step()
        # print info
        if (epoch + 1) % net_params['print_freq'] == 0:
            # evaluate on the test dataset (just for a reporting)
            evaluate(model, data_loader_test, device=device)

"""
    Main
"""
def main(model,model_type,device,params,net_params,save_dir,train_prefix,path,test_fname,num_classes,CLASS_NAMES,img_size):
    
    train_indices = np.load(train_fname,allow_pickle=True)
    test_indices = np.load(test_fname,allow_pickle=True)
    fold_stats = []
    
    for fold in range(params['K']): # for each fold..
        # read data - train
        train_idx_one = train_indices[fold]
        imgs_train, labs_train = loadcaches(path,train_idx_one)

        # read data - test
        test_idx_one = test_indices[fold]
        imgs_test, labs_test = loadcaches(path,test_idx_one)

        # use our dataset and defined transformations
        dataset_train = PBBMDataset(imgs_train, labs_train, get_transformA('train',img_size))
        dataset_test = PBBMDataset(imgs_test, labs_test, get_transformA('test',img_size))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=net_params['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False,collate_fn=collate_fn)
        
        # after each fold, stack the outputs
        train_test_pipeline(model,model_type,fold,
                                            data_loader,data_loader_test,
                                            device,params,net_params,save_dir,train_prefix)
        print('')
        print('After training, evaluation metric for this fold:')
        print('')
        model.eval()
        fold_eval = evaluate(model, data_loader_test, device=device)
        fold_stats.append(fold_eval.stats)
        

    all_stats = np.stack(fold_stats,axis=0) # fold,12
    # calculate the mean and std across the folds without undesirable case (e.g. the fold which has a value -1 for AP small)
    mask = (all_stats < 0).astype(np.float) # masking the value -1
    ma_stats = np.ma.masked_array(all_stats,mask) # masked array
    mean_stats, std_stats = ma_stats.mean(axis=0).data, ma_stats.std(axis=0).data
    
    print('Mean, and Std for the cross validation:')
    print('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f, %.3f' %(mean_stats[0],std_stats[0]))
    print('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %.3f, %.3f' %(mean_stats[1],std_stats[1]))
    print('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %.3f, %.3f' %(mean_stats[2],std_stats[2]))
    print('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f, %.3f' %(mean_stats[3],std_stats[3]))
    print('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f, %.3f' %(mean_stats[4],std_stats[4]))
    print('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f, %.3f' %(mean_stats[5],std_stats[5]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %.3f, %.3f' %(mean_stats[6],std_stats[6]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %.3f, %.3f' %(mean_stats[7],std_stats[7]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f, %.3f' %(mean_stats[8],std_stats[8]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f, %.3f' %(mean_stats[9],std_stats[9]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f, %.3f' %(mean_stats[10],std_stats[10]))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f, %.3f' %(mean_stats[11],std_stats[11]))
        
'''
        if fold == 0: 
            predn_all = predn
            labelm_all = labelm
        else:
            predn_all = torch.cat((predn_all,predn),dim=0)
            labelm_all = torch.cat((labelm_all,labelm),dim=0)
            
    # plot the pr curve & confusion matrix
    ap50,ap,p,r,nt = get_coco_metrics(predn_all,labelm_all,num_classes,CLASS_NAMES,
                               save_dir=save_dir,
                               plotpr=params['plotpr'],plotconf=params['plotconf'])
    
    # summary
    print_evals(ap50,ap,p,r,nt,CLASS_NAMES)
'''            
            
"""
    Model setting
"""
# num_classes which is user-defined
num_classes = len(CLASS_NAMES) + 1 # classes + background
# load a model pre-trained on COCO
if model_type == 'fasterRCNN':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # (optional) load the pre-trained weights
    if pretrained is not None:
        pretrained_dict = torch.load(pretrained,map_location=torch.device('cpu')) 
        model.backbone.load_state_dict(pretrained_dict)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
elif model_type == 'retinaNet':
    num_anchors = 9
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    in_features = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head = RetinaNetClassificationHead(in_features,num_anchors,num_classes)
    
elif model_type == 'SSD':
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    in_features = [i.in_channels for i in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_features,num_anchors,num_classes)
        
#
main(model,model_type,device,
     params,net_params,
     save_dir,train_prefix,path,test_fname,
     num_classes,CLASS_NAMES,img_size)
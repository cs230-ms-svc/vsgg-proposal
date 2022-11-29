import torch
import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np
import copy
import cv2
import os

#%%
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList

MODEL="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

class detectron():
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
        self.predictor = DefaultPredictor(self.cfg)

    def predict_for_frame(self, im_data):
        outputs = self.predictor(im_data)
        return outputs

#%%

class detector(nn.Module):
    def __init__(self, is_train, object_classes, use_SUPPLY, mode='sgdet'):
        super(detector, self).__init__()
        self.is_train = is_train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        counter = 0
        counter_image = 0

# %%
def run_partial():
    im1 = cv2.imread('/home/cs230/code/action-genome-data/frames/0A8ZT.mp4/000385.png')
    im2 = cv2.imread('/home/cs230/code/action-genome-data/frames/0A8ZT.mp4/000374.png')
    im3 = cv2.imread('/home/cs230/code/action-genome-data/frames/0A8ZT.mp4/000353.png')
    
    im1 = cv2.resize(im1, (512, 384))
    im2 = cv2.resize(im2, (512, 384))
    im3 = cv2.resize(im3, (512, 384))
    
    im1 = np.rollaxis(im1, 2)
    im2 = np.rollaxis(im2, 2)
    im3 = np.rollaxis(im3, 2)
    imgs_np = np.stack((im1, im2, im3))
    imgs_tensor = torch.from_numpy(imgs_np)
    imgs_tensor_gpu = imgs_tensor.to("cuda", dtype=torch.float32)

    imgs_list = ImageList(imgs_tensor, [(384, 512), (384,512), (384, 512)])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)

    model = build_model(cfg)
    model.eval()
    
    features = model.backbone(imgs_tensor_gpu)
    proposals, _ = model.proposal_generator(imgs_list, features)
    instances, _ = model.roi_heads(imgs_list, features, proposals)
    mask_features = [features[f] for f in model.roi_heads.in_features]
    mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
    return features, proposals, instances, mask_features



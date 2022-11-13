#%%
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

#%%
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#%%
# TESTFILE='/home/cs230/code/action-genome-data/frames/001YG.mp4/000089.png'

# im = cv2.imread(TESTFILE)
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# cv2.imwrite('boxes_first.jpg', out.get_image()[:, :, ::-1])

# print(outputs["instances"].pred_masks.shape)

#%%
# TESTFILE='/home/cs230/code/action-genome-data/frames/001YG.mp4/000093.png'

# im = cv2.imread(TESTFILE)
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# cv2.imwrite('boxes_second.jpg', out.get_image()[:, :, ::-1])

# print(outputs["instances"].pred_masks.shape)

#%%
# FRAME_1_ID = 337
# FRAME_2_ID = 382
# FRAME_3_ID = 436


# FRAME1=f'/home/cs230/code/action-genome-data/frames/001YG.mp4/000{FRAME_1_ID}.png'
# FRAME2=f'/home/cs230/code/action-genome-data/frames/001YG.mp4/000{FRAME_2_ID}.png'
# FRAME3=f'/home/cs230/code/action-genome-data/frames/001YG.mp4/000{FRAME_3_ID}.png'

def define_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def predict_for_frame(cfg, predictor, filename):
    im = cv2.imread(filename)
    outputs = predictor(im)
    vis = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    vis_out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    return outputs, vis, vis_out

cfg, predictor = define_config()

def plot_instance_seg(filename, vis_out):
    cv2.imwrite(filename, vis_out.get_image()[:, :, ::-1])

# out_1, vis_1, vis_1_out = predict_for_frame(cfg, predictor, FRAME1)
# out_2, vis_2, vis_2_out = predict_for_frame(cfg, predictor, FRAME2)
# out_3, vis_3, vis_3_out = predict_for_frame(cfg, predictor, FRAME3)
# plot_instance_seg(f'seg_{FRAME_1_ID}.jpg', vis_1_out)
# plot_instance_seg(f'seg_{FRAME_2_ID}.jpg', vis_2_out)
# plot_instance_seg(f'seg_{FRAME_3_ID}.jpg', vis_3_out)

#%%
import os

SRC_BASE='/home/cs230/code/action-genome-data/frames'
TESTDIR='0A8ZT.mp4'
SRCDIR=os.path.join(SRC_BASE, TESTDIR)

DST_BASE = os.getcwd()
DSTDIR=os.path.join(DST_BASE, TESTDIR)

os.makedirs(DSTDIR, exist_ok=True)

for root, dirs, files in os.walk(SRCDIR):
    for file in files:
        if file.endswith("png"):
            full_filename = os.path.join(root, file)
            test_name = os.path.splitext(file)[0]
            out, vis, vis_out = predict_for_frame(cfg, predictor, full_filename)
            full_dst_filename = os.path.join(DSTDIR, f'seg_{test_name}.jpg')
            plot_instance_seg(full_dst_filename, vis_out)

#%%
# Best frames from previous were 374, 385, 387, 389
# Let's try 385 and 387 first

frames = [374, 385]
file_names = [os.path.join(SRCDIR, f'000{x}.png') for x in frames]
processed = [(predict_for_frame(cfg, predictor, x)) for x in file_names]



# %%
out_374 = processed[0][0]
out_385 = processed[1][0]
class_0 = out_374["instances"].pred_masks.to("cpu").numpy()[0]
class_59 = out_374["instances"].pred_masks.to("cpu").numpy()[1]
class_75 = out_374["instances"].pred_masks.to("cpu").numpy()[2]
cv2.imwrite("class_0.jpg", class_0.astype(np.uint8)*255)

#%%
print(torch.min(out_374["instances"].pred_classes))

min_class_label = torch.min(out_374["instances"].pred_classes)
max_class_label = torch.max(out_374["instances"].pred_classes)

result_374 = np.zeros(class_0.shape)
pred_classes = out_374["instances"].pred_classes.to("cpu").numpy()
pred_masks = out_374["instances"].pred_masks.to("cpu").numpy()

index = 0
for pred_class in pred_classes:
    if pred_class == 0:
        pred_class = max_class_label.to("cpu").numpy() + 1
    pred_mask = pred_masks[index]
    result_374[pred_mask] = pred_class
    index += 1



#%%

print(torch.min(out_385["instances"].pred_classes))

min_class_label = torch.min(out_385["instances"].pred_classes)
max_class_label = torch.max(out_385["instances"].pred_classes)

result_385 = np.zeros(class_0.shape)
pred_classes = out_385["instances"].pred_classes.to("cpu").numpy()
pred_masks = out_385["instances"].pred_masks.to("cpu").numpy()

index = 0
for pred_class in pred_classes:
    if pred_class == 0:
        pred_class = max_class_label.to("cpu").numpy() + 1
    pred_mask = pred_masks[index]
    result_385[pred_mask] = pred_class
    index += 1

#%%
diff_image = result_385 - result_374

intersection = diff_image != 0
intersection = intersection * result_385

#%%

pseudocolor_374 = result_374.astype(np.uint8)
# pseudocolor_374 = cv2.applyColorMap(result_374.astype(np.uint8), 
#                                     colormap=cv2.COLORMAP_BONE)
cv2.imwrite("classes_only_374.jpg", pseudocolor_374)

pseudocolor_385 = result_385.astype(np.uint8)
# cv2.applyColorMap(result_385.astype(np.uint8),
#                                     colormap=cv2.COLORMAP_BONE)
cv2.imwrite("classes_only_385.jpg", pseudocolor_385)

pseudocolor_intersection = intersection.astype(np.uint8)
# cv2.applyColorMap(intersection.astype(np.uint8),
#                                              colormap=cv2.COLORMAP_BONE)
cv2.imwrite("roi_385.jpg", pseudocolor_intersection)

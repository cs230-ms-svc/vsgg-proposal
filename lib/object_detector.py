import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from lib.funcs import assign_relations
from detectron2.utils.analysis import ROIAlign


MODEL="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
IMG_HEIGHT = 384
IMG_WIDTH  = 512

class detectron():
    def __init__(self, model_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
        # self.predictor = DefaultPredictor(self.cfg)
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(model_path)

    def partial(self, img_data, height, width):
        img_list = ImageList([torch.from_numpy(img) for img in img_data], [(height, width) for _ in img_data])
        features = self.model.backbone(img_list)
        proposals, _ = self.model.proposal_generator(img_list, features)
        instances, _ = self.model.roi_heads(img_list, features, proposals)
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        roi_heads = self.model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
        return features, proposals, roi_heads, instances, mask_features
        

class detector(nn.Module):
    def __init__(self, train, object_classes, use_SUPPLY):
        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.detectron = detectron("models")
    
    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        counter = 0
        counter_image = 0

        # create saved-bbox, labels, scores, features
        FINAL_BBOXES = torch.tensor([]).cuda(0)
        FINAL_LABELS = torch.tensor([], dtype=torch.int64).cuda(0)
        FINAL_SCORES = torch.tensor([]).cuda(0)
        FINAL_FEATURES = torch.tensor([]).cuda(0)
        FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)

        while counter < im_data.shape[0]:
            #compute 10 images in batch and  collect all frames data in the video
            if counter + 10 < im_data.shape[0]:
                inputs_data = im_data[counter:counter + 10]
                inputs_info = im_info[counter:counter + 10]
                inputs_gtboxes = gt_boxes[counter:counter + 10]
                inputs_numboxes = num_boxes[counter:counter + 10]

            else:
                inputs_data = im_data[counter:]
                inputs_info = im_info[counter:]
                inputs_gtboxes = gt_boxes[counter:]
                inputs_numboxes = num_boxes[counter:]

            rois, cls_prob, bbox_pred, base_feat, roi_features = self.detectron.partial(im_data, IMG_HEIGHT, IMG_WIDTH)

            SCORES = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            # bbox regression (class specific)
            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).cuda(0) \
                            + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).cuda(0)  # the first is normalize std, the second is mean
            box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
            PRED_BOXES = boxes

            #traverse frames
            for i in range(rois.shape[0]):
                # images in the batch
                scores = SCORES[i]
                pred_boxes = PRED_BOXES[i]

                for j in range(1, len(self.object_classes)):
                    # NMS according to obj categories
                    inds = torch.nonzero(scores[:, j] > 0.1).view(-1) #0.05 is score threshold
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]

                        final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(pred_boxes.shape[0], 1).cuda(0),
                                                pred_boxes), 1)
                        FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                        FINAL_LABELS = torch.cat((FINAL_LABELS, cls_dets), 0)
                        FINAL_SCORES = torch.cat((FINAL_SCORES, cls_scores), 0)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)

                counter_image += 1

            counter += 10
        FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)
        prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                        'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}
        
        if self.is_train:
            DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction, gt_annotation, assign_IOU_threshold=0.5)

            if self.use_SUPPLY:
                # supply the unfounded gt boxes by detector into the scene graph generation training
                FINAL_BBOXES_X = torch.tensor([]).cuda(0)
                FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).cuda(0)
                FINAL_SCORES_X = torch.tensor([]).cuda(0)
                FINAL_FEATURES_X = torch.tensor([]).cuda(0)
                assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                for i, j in enumerate(SUPPLY_RELATIONS):
                    if len(j) > 0:
                        unfound_gt_bboxes = torch.zeros([len(j), 5]).cuda(0)
                        unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64).cuda(0)
                        one_scores = torch.ones([len(j)], dtype=torch.float32).cuda(0)  # probability
                        for m, n in enumerate(j):
                            # if person box is missing or objects
                            if 'bbox' in n.keys():
                                unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']) * im_info[
                                    i, 2]  # don't forget scaling!
                                unfound_gt_classes[m] = n['class']
                            else:
                                # here happens always that IOU <0.5 but not unfounded
                                unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']) * im_info[
                                    i, 2]  # don't forget scaling!
                                unfound_gt_classes[m] = 1  # person class index

                        DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                        np.arange(
                                                                            start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                            stop=int(
                                                                                sum(FINAL_BBOXES[:, 0] == i)) + len(
                                                                                SUPPLY_RELATIONS[i]))), axis=0).astype(
                            'int64'))

                        GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                        # compute the features of unfound gt_boxes
                        pooled_feat = ROIAlign(FINAL_BASE_FEATURES[i].unsqueeze(0),
                                                                        unfound_gt_bboxes.cuda(0))
                        cls_prob = F.softmax(pooled_feat, 1)

                        unfound_gt_bboxes[:, 0] = i
                        unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                        FINAL_BBOXES_X = torch.cat(
                            (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                        FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],
                                                    unfound_gt_classes))  # final label is not gt!
                        FINAL_SCORES_X = torch.cat(
                            (FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                        FINAL_FEATURES_X = torch.cat(
                            (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                    else:
                        FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                        FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                        FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                        FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

            FINAL_DISTRIBUTIONS = torch.softmax(FINAL_FEATURES_X, dim=1)
            global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices

            im_idx = []  # which frame are the relations belong to
            pair = []
            a_rel = []
            s_rel = []
            c_rel = []
            for i, j in enumerate(DETECTOR_FOUND_IDX):

                for k, kk in enumerate(GT_RELATIONS[i]):
                    if 'person_bbox' in kk.keys():
                        kkk = k
                        break
                localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk])

                for m, n in enumerate(j):
                    if 'class' in GT_RELATIONS[i][m].keys():
                        im_idx.append(i)

                        pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)])])

                        a_rel.append(GT_RELATIONS[i][m]['attention_relationship'].tolist())
                        s_rel.append(GT_RELATIONS[i][m]['spatial_relationship'].tolist())
                        c_rel.append(GT_RELATIONS[i][m]['contacting_relationship'].tolist())

            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)
            union_boxes = torch.cat((im_idx[:, None],
                                        torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                        torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

            union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
            union_feat = ROIAlign(FINAL_BASE_FEATURES, union_boxes)

            pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
            spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

            entry = {'boxes': FINAL_BBOXES_X,
                        'labels': FINAL_LABELS_X,
                        'scores': FINAL_SCORES_X,
                        'distribution': FINAL_DISTRIBUTIONS,
                        'im_idx': im_idx,
                        'pair_idx': pair,
                        'features': FINAL_FEATURES_X,
                        'union_feat': union_feat,
                        'spatial_masks': spatial_masks,
                        'attention_gt': a_rel,
                        'spatial_gt': s_rel,
                        'contacting_gt': c_rel}

            return entry
        else:
            FINAL_DISTRIBUTIONS = torch.softmax(FINAL_SCORES, dim=1)
            FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)

            entry = {'boxes': FINAL_BBOXES,
                        'scores': FINAL_SCORES,
                        'distribution': FINAL_DISTRIBUTIONS,
                        'pred_labels': PRED_LABELS,
                        'features': FINAL_FEATURES,
                        'fmaps': FINAL_BASE_FEATURES,
                        'im_info': im_info[0, 2]}

            return entry        

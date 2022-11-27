from detectron2.structures.boxes import BoxMode
import torch, detectron2
import cv2
import pickle
from torch.utils.data import Dataset
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

MINI_DATASET_SIZE=1000
AG_DS_PATH="/home/cs230/code/action-genome-data/"

class AG(Dataset):
    def __init__(self, mode, datasize, data_path:str="", filter_nonperson_box_frame=True):
        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')
        self.records = []
        self.object_lookup = {}

        # collect the object classes
        # self.object_classes = ['__background__']
        self.object_classes = []
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.build_reverse_lookup_classes() # Do this before reassign

        # Commented to avoid confusion with labels
        # self.object_classes[9] = 'closet/cabinet'
        # self.object_classes[11] = 'cup/glass/bottle'
        # self.object_classes[23] = 'paper/notebook'
        # self.object_classes[24] = 'phone/camera'
        # self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        with open(root_path + 'annotations/person_bbox.pkl', 'rb') as f:
            person_bbox = pickle.load(f)
        f.close()
        with open(root_path+'annotations/object_bbox_and_relationship.pkl', 'rb') as f:
            object_bbox = pickle.load(f)
        f.close()

        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:MINI_DATASET_SIZE]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        self.frame_data_anno = []
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is at least one visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    self.parse_frame_data(person_bbox[i], object_bbox[i], i)
    
    def build_reverse_lookup_classes(self):
        for class_id, value in enumerate(self.object_classes):
            self.object_lookup[value] = class_id

    def parse_frame_data(self, person_bbox_data, obj_bbox_data, key):
        record = {}
        filename = os.path.join(self.frames_path, key)
        record["filename"] = filename
        record["image_id"] = key
        width, height = person_bbox_data["bbox_size"]
        record["height"] = height
        record["width"] = width

        person_bbox = person_bbox_data["bbox"]

        # Helper method
        def get_poly_from_bbox(bbox_data):
            px = (bbox_data[0] + 0.5, bbox_data[2] + 0.5)
            py = (bbox_data[1] + 0.5, bbox_data[3] + 0.5)
            box_height = px[1] - px[0]
            box_width = py[1] - py[0] 

            bbox_poly = [(px[0], py[0]), 
                                (px[0], py[0] + box_width),
                                (px[1], py[1]),
                                (px[1], py[1] - box_width)]
            return bbox_poly
        
        annos = []
        # Append person bbox data
        anno_person = {
            "bbox": person_bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": get_poly_from_bbox(person_bbox),
            "category_id": self.object_lookup["person"]
        }
        annos.append(anno_person)

        # Add objects bbox data
        for obj in obj_bbox_data:
            obj_bbox = obj["bbox"]
            anno_obj = {
                "bbox": obj_bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": get_poly_from_bbox(obj_bbox),
                "category_id": self.object_lookup[obj["class"]]
            }
            annos.append(anno_obj)
        record["annotations"] = annos
        self.records.append(record)

def create_datasets(data_path):
    AG_train = AG("train", MINI_DATASET_SIZE, data_path, filter_nonperson_box_frame=True)
    AG_test = AG("test", MINI_DATASET_SIZE, data_path, filter_nonperson_box_frame=True)

    return AG_train, AG_test

def register_datasets(dataset_dict):
    for mode, ds in dataset_dict.items():
        DatasetCatalog.register(f"action-genome-{mode}", lambda ds: ds.records)
        MetadataCatalog.get(f"action-genome-{mode}").set(thing_classes=ds.object_classes)

if __name__=="__main__":
    ag_train, ag_test = create_datasets(AG_DS_PATH)
    dataset_dict = {"train": ag_train, "test": ag_test}
    register_datasets(dataset_dict)

    # Create cfg
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "models"
    cfg.DATASETS.TRAIN = ("action-genome-train",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 32  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0005  # Learning Rate
    cfg.SOLVER.MAX_ITER = 1000   
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ag_train.object_classes)   # Const: 36
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

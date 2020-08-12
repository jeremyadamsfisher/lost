from typing import List, Callable
Point = List[float]
Label = str

from lost.pyapi import script
import json
import itertools as it
from collections import defaultdict
from pathlib import Path
import os
import uuid
import numpy as np
import pandas as pd
import cv2
import random

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


ENVS = ["lost-cv"]
ARGUMENTS = {
    "BATCH_SIZE": {
        "value": "1",
        "help": "Number of annots at a time"
    }
}


class RequestBatchOfMicroscopyToAnnotate(script.Script):
    def main(self):
        imgs_annotated_fp = Path(self.get_path("used.json"))
        try:
            with open(imgs_annotated_fp) as f:
                imgs_annotated = json.load(f)
        except FileNotFoundError:
            imgs_annotated = []
        self.logger.info(f"imgs_annotated={imgs_annotated}")
        imgs = it.chain(*[Path(ds.path).glob("*.tif") for ds in self.inp.datasources])
        imgs = [str(img.resolve()) for img in imgs]
        imgs_unannotated = [img for img in imgs if img not in imgs_annotated]
        if any(imgs_annotated):
            df = pd.read_csv(self.get_path('out.csv', context="pipe"))
            mask_predictor = self.detectron2_predictor_factory(df)
        else:
            mask_predictor = None
        annotask = next(iter(self.outp.anno_tasks))
        possible_labels = annotask.possible_label_df['idx'].values.tolist()
        for img_path in random.sample(imgs_unannotated, int(self.get_arg("BATCH_SIZE"))):
            imgs_annotated.append(img_path)
            if not mask_predictor:
                self.outp.request_annos(img_path=img_path)
            else:
                membrane_polygons = mask_predictor(cv2.imread(str(img_path)))
                self.outp.request_annos(
                    img_path=img_path,
                    annos=membrane_polygons,
                    anno_types=["polygon"] * len(membrane_polygons),
                    anno_labels=[random.sample(possible_labels, 1)]
                )
            self.logger.info("Requested annos for: {}".format(img_path))
        with open(imgs_annotated_fp, "w") as f:
            json.dump(imgs_annotated, f)
        if len(imgs_unannotated) == 0:
            self.break_loop()

    def detectron2_predictor_factory(self, lost_annotations=None) -> Callable:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.DEVICE="cpu"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # label assigned later
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
        cfg.OUTPUT_DIR = self.get_path("./detectron", ptype="rel")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.DATASETS.TRAIN = tuple(["train"])
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.WEIGHTS = self.get_path("model.pth", ptype="rel")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        DatasetCatalog.register("train", lambda d="train": self.lost2detectron_style_dataset(lost_annotations))
        MetadataCatalog.get("train").set(thing_classes=["na"])
        trainer = DefaultTrainer(cfg)
        trainer.train()
        predictor_ = DefaultPredictor(cfg)
        predictor_.resume_or_load(resume=False)
        def predict(img, predictor=predictor_) -> List[Point]:
            masks = predictor_(img)["instances"].get_fields()["pred_masks"]
            contours = []
            for mask in masks:
                X, Y = contour_polygon_from_mask(mask).T
                contour = list(zip(X, Y))
                contours.append(contour)
        return predict

    def lost2detectron_style_dataset(self,lost_annotations: pd.DataFrame) -> List[dict]:
        self.logger.info(f"lost_annotations.columns: {lost_annotations.columns}")
        self.logger.info(f"lost_annotations:\n{lost_annotations.head()}")
        annotations = lost_annotations[~lost_annotations["anno.anno_task_id"].isnull()]
        annotations = annotations.rename(columns={"anno.data": "polygon", "img.img_path": "img_fp"})[["polygon", "img_fp"]]
        img_fp2polygon = defaultdict(list)
        for _, row in annotations.iterrows():
            img_fp = os.path.join("/home", "lost", row.img_fp)
            polygon = json.loads(row.polygon)
            xs = np.array([pt["x"] for pt in polygon]).T
            ys = np.array([pt["y"] for pt in polygon]).T
            polygon = np.stack((xs,ys), axis=1)
            assert polygon.shape[0] == len(xs) == len(ys)
            assert polygon.shape[1] == 2
            img_fp2polygon[img_fp].append(polygon)
        self.logger.info(f"have annotations for: {list(img_fp2polygon.keys())}")
        detectron_dataset = []
        for img_fp, polygons in img_fp2polygon.items():
            img = cv2.imread(img_fp)
            if img is None:
                raise FileNotFoundError(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape
            detectron_dataset.append(dict(
                file_name=str(img_fp),
                image_id=str(uuid.uuid4()),
                height=height,
                width=width,
                annotations=[
                    {"bbox": bbox_from_polygon(polygon),
                     "bbox_mode": BoxMode.XYXY_ABS,
                     "segmentation": [polygon],
                     "category_id": 0}
                    for polygon in polygons
                ]
            ))
        return detectron_dataset


def contour_polygon_from_mask(binary_mask: np.ndarray):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour, *_ = contours
    return contour.reshape(-1,2)


def bbox_from_polygon(poly: np.ndarray):
    min_x = poly[:,0].min()
    min_y = poly[:,1].min()
    max_x = poly[:,0].max()
    max_y = poly[:,1].max()
    return [min_x, min_y, max_x, max_y]


if __name__ == "__main__":
    my_script = RequestBatchOfMicroscopyToAnnotate()

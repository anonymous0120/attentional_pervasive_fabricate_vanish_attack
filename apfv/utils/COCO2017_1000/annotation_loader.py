import os
import numpy as np

import pdb

def load_annotations(img_name_noext, img_size, coco_ann):
    """ Load annotations for an image_index.
    """
    ori_h = coco_ann.loadImgs(ids=[int(img_name_noext)])[0]['height']
    ori_w = coco_ann.loadImgs(ids=[int(img_name_noext)])[0]['width']
    ratio = (float(img_size[0]) / float(ori_h), float(img_size[1]) / float(ori_w))
    annotations_ids = coco_ann.getAnnIds(imgIds=[int(img_name_noext)])
    annotations     = {'classes': [], 'boxes': [], 'scores' : []}
    if len(annotations_ids) == 0:
            return annotations
    coco_annotations = coco_ann.loadAnns(annotations_ids)
    for idx, curt_ann in enumerate(coco_annotations):
        # some annotations have basically no width / height, skip them
        if curt_ann['bbox'][2] < 1 or curt_ann['bbox'][3] < 1:
            continue
        annotations['classes'].append(curt_ann['category_id'])
        curt_bbox = [curt_ann['bbox'][1], curt_ann['bbox'][0], curt_ann['bbox'][1] + curt_ann['bbox'][3], curt_ann['bbox'][0] + curt_ann['bbox'][2]]
        resized_bbox = [ratio[0] * curt_bbox[0], ratio[1] * curt_bbox[1], ratio[0] * curt_bbox[2], ratio[1] * curt_bbox[3]]
        annotations['boxes'].append(resized_bbox)
        annotations['scores'].append(1.0)
    return annotations
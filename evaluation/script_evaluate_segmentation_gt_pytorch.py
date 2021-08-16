import argparse
import cv2
import importlib
import json
import numpy as np
import os
import pickle
from PIL import Image
from pycocotools.coco import COCO
import sys
import torch
import torchvision
from tqdm import tqdm

from apfv21.models.deeplabv3plus.utils.metrics import Evaluator
from apfv21.utils.image_utils import load_image, save_image
from apfv21.utils.torch_utils import numpy_to_variable, variable_to_numpy
from apfv21.utils.COCO2017_1000.mask_loader import load_masks as load_coco_masks

import pdb


# CUDA_VISIBLE_DEVICES=1 python3 evaluation/script_evaluate_segmentation_gt_pytorch.py fcn_resnet50 voc --dataset-dir ~/workspace/data/AFV_experiments/voc/

# {voc_idx : coco_idx}
with open('apfv21/utils/VOC_AND_COCO91_CLASSES.pkl', 'rb') as f:
  VOC_AND_COCO91_CLASSES = pickle.load(f)
with open('apfv21/utils/VOC_AND_COCO80_CLASSES.pkl', 'rb') as f:
  VOC_AND_COCO80_CLASSES = pickle.load(f)

PICK_LIST = []
BAN_LIST = []


def parse_args(args):
  parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
  parser.add_argument('test_model', help='Model for testing AEs.', type=str)
  parser.add_argument('dataset_type', choices=['coco', 'voc'],
                      help='Dataset for testing AEs.', type=str)
  parser.add_argument('--dlv3p-backbone', type=str, default='resnet',
                      choices=['resnet', 'xception', 'drn', 'mobilenet'],
                      help='deeplabv3plus backbone name (default: resnet)')
  parser.add_argument('--dataset-dir', help='Dataset folder path.', 
                      default='/home/yantao/workspace/datasets/imagenet5000', type=str)
  return parser.parse_args(args)

def _convert_label(a_to_b):
  b_to_a = {}
  for key, val in a_to_b.items():
    b_to_a[val] = key + 1
  return b_to_a

def test(args):
  args_dic = vars(args)

  if args.dataset_type == 'voc':
    gt_dir = os.path.join(args.dataset_dir, '_segmentations')
      
  elif args.dataset_type == 'coco':
    gt_loader = COCO(os.path.join(args.dataset_dir, 'instances_val2017.json'))
    to_voc_21 = _convert_label(VOC_AND_COCO91_CLASSES)

  if args.test_model == 'deeplabv3plus':
    args_dic['num_classes'] = 21
    model = DeepLab(
        num_classes=21,
        backbone=args.dlv3p_backbone,
        output_stride=8,
        sync_bn=False,
        freeze_bn=True
    )
    img_size = (513, 513)
    model = model.cuda()
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    #img_transforms = None
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=img_mean, std=img_std)])
  else:
    args_dic['num_classes'] = 21
    source_lib = importlib.import_module("torchvision.models.segmentation")
    test_model_class = getattr(source_lib, args.test_model)
    model = test_model_class(pretrained=True).cuda().eval()
    img_size = (1024, 1024)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=img_mean, std=img_std)])

  evaluator = Evaluator(args.num_classes)

  test_folders = []
  for temp_folder in os.listdir(args.dataset_dir):
    if not os.path.isdir(os.path.join(args.dataset_dir, temp_folder)):
      continue 
    if temp_folder == 'imagenet_val_5000' or temp_folder == '.git' or \
        temp_folder == '_annotations' or temp_folder == '_segmentations':
      continue 
    if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
      continue
    if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
      continue
    test_folders.append(temp_folder)
  
  result_dict = {}
  for curt_folder in tqdm(test_folders):
    print('Folder : {0}'.format(curt_folder))
    evaluator.reset()
    for adv_name in tqdm(os.listdir(os.path.join(args.dataset_dir, curt_folder))):
      temp_image_name_noext = os.path.splitext(adv_name)[0]
      if args.dataset_type == 'voc':
        gt_path = os.path.join(gt_dir, temp_image_name_noext + '.png')

      if curt_folder == 'ori':
        adv_img_path = os.path.join(
            args.dataset_dir, curt_folder, temp_image_name_noext + '.jpg')
      else:
        adv_img_path = os.path.join(
            args.dataset_dir, curt_folder, temp_image_name_noext + '.png')

      if not os.path.exists(adv_img_path):
        print('File {0} not found.'.format(adv_name))
        continue

      if args.dataset_type == 'voc':
        mask_rgb = np.array(Image.open(gt_path))
        idx_255 = np.argwhere(mask_rgb == 255)
        for temp_idx_255 in idx_255:
          mask_rgb[temp_idx_255[0], temp_idx_255[1]] = 0
        output_ori = cv2.resize(mask_rgb, img_size, interpolation=cv2.INTER_NEAREST)
      elif args.dataset_type == 'coco':
        output_ori = load_coco_masks(
            temp_image_name_noext, img_size, gt_loader, to_voc=to_voc_21)
                  
      if img_transforms == None:
        image_adv_np = load_image(
            data_format='channels_first', 
            shape=img_size, 
            bounds=(0, 1), 
            abs_path=True, 
            fpath=adv_img_path
        )
        image_adv_var = numpy_to_variable(image_adv_np)
        with torch.no_grad():
          if args.test_model == 'deeplabv3plus':
            output_adv = model(image_adv_var)
      else:
        image_adv_var = img_transforms(
            Image.open(adv_img_path).convert('RGB')).unsqueeze_(axis=0).cuda()
        with torch.no_grad():
          if args.test_model == 'deeplabv3plus':
            output_adv = model(image_adv_var)
          else:
            output_adv = model(image_adv_var)['out']
      
      pred_ori = np.expand_dims(output_ori, axis=0)
      pred_adv = output_adv.data.cpu().numpy()
      pred_adv = np.argmax(pred_adv, axis=1)

      evaluator.add_batch(pred_ori, pred_adv)

    try:
      Acc = evaluator.Pixel_Accuracy()
      Acc_class = evaluator.Pixel_Accuracy_Class()
      mIoU = evaluator.Mean_Intersection_over_Union()
      FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    except:
      Acc = 0.
      Acc_class = 0.
      mIoU = 0.
      FWIoU = 0.

    result_str = \
        'Acc: {0:.04f}, Acc_class: {1:.04f}, mIoU: {2:.04f}, FWIoU: {3:.04f}'.format(
            Acc, Acc_class, mIoU, FWIoU)
    print(curt_folder, ' : ', result_str)
    result_dict[curt_folder] = result_str

    with open('temp_seg_results_gt_{0}_{1}.json'.format(
        args.test_model, args.dataset_type), 'w') as fout:
      json.dump(result_dict, fout, indent=2)
        

def main(args=None):
  # parse arguments
  if args is None:
    args = sys.argv[1:]
  args = parse_args(args)
  args_dic = vars(args)
  if args.dlv3p_backbone == 'resnet':
    args_dic['pretrained_path'] = \
        'models/deeplabv3plus/pretrained/deeplab-resnet.pth.tar'
  elif args.dlv3p_backbone == 'drn':
    args_dic['pretrained_path'] = \
        'models/deeplabv3plus/pretrained/deeplab-drn.pth.tar'
  elif args.dlv3p_backbone == 'mobile':
    args_dic['pretrained_path'] = \
        'models/deeplabv3plus/pretrained/deeplab-mobilenet.pth.tar'
  else:
    raise ValueError()

  test(args)

if __name__ == "__main__":
  main()
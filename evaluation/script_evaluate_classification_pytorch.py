import argparse
import importlib
import json
import numpy as np
import os
from PIL import Image
import sys
import shutil
import torchvision
import torch
from tqdm import tqdm

from apfv21.utils.image_utils import load_image, save_image
from apfv21.utils.torch_utils import numpy_to_variable, variable_to_numpy

import pdb   

# CUDA_VISIBLE_DEVICES=0 python3 evaluation/script_evaluate_classification_pytorch.py resnet50 --dataset-dir /home/yantao/workspace/data/AFV_experiments/imagenet/

PICK_LIST = []
BAN_LIST = []


def parse_args(args):
  """ Parse the arguments.
  """
  parser = argparse.ArgumentParser(
      description='Script for generating adversarial examples.')
  parser.add_argument('test_model', help='Model for testing AEs.', type=str)
  parser.add_argument('--dataset-dir', help='Dataset folder path.',
      default='/home/yantao/workspace/datasets/imagenet5000', type=str)

  return parser.parse_args()


def main(args=None):
  if args is None:
    args = sys.argv[1:]
  args = parse_args(args)
  args_dic = vars(args)

  with open('apfv21/utils/labels.txt','r') as inf:
    args_dic['imagenet_dict'] = eval(inf.read())

  input_dir = os.path.join(args.dataset_dir, 'ori')

  # if args.test_model == 'resnet50':
  #   test_model = torchvision.models.resnet50(pretrained=True).cuda()
  #   test_model.eval()
  # elif args.test_model == 'squeezenet1_1':
  #   test_model = torchvision.models.squeezenet1_1(pretrained=True).cuda()
  #   test_model.eval()
  # elif args.test_model == 'densenet121':
  #   test_model = torchvision.models.densenet121(pretrained=True).cuda()
  #   test_model.eval()
  # elif args.test_model == 'shufflenet_v2_x0_5':
  #   test_model = torchvision.models.shufflenet_v2_x0_5(pretrained=True).cuda()
  #   test_model.eval()
  # elif args.test_model == 'mobilenet_v3_large':
  #   test_model = torchvision.models.mobilenet_v3_large(pretrained=True).cuda()
  #   test_model.eval()
  # else:
  #   raise ValueError("Invalid test_model {}.".format(args.test_model))

  source_lib = importlib.import_module("torchvision.models")
  test_model_class = getattr(source_lib, args.test_model)
  test_model = test_model_class(pretrained=True).cuda()
  test_model.eval()

  test_folders = []
  for temp_folder in os.listdir(args.dataset_dir):
    if not os.path.isdir(os.path.join(args.dataset_dir, temp_folder)):
      continue 
    if temp_folder == 'imagenet_val_5000' or temp_folder == 'ori' or temp_folder == '.git':
      continue 
    if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
      continue
    if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
      continue
    test_folders.append(temp_folder)
  
  result_dict = {}
  for curt_folder in tqdm(test_folders):
    print('Folder : {0}'.format(curt_folder))
    correct_count = 0
    total_count = 0
    for image_name in tqdm(os.listdir(input_dir)):
      image_ori_path = os.path.join(input_dir, image_name)
      image_adv_path = os.path.join(args.dataset_dir, curt_folder, image_name)
      image_adv_path = os.path.splitext(image_adv_path)[0] + '.png'
      if not os.path.exists(image_adv_path):
        continue
          
      image_ori_np = load_image(
          data_format='channels_first', abs_path=True, fpath=image_ori_path)
      image_adv_np = load_image(
          data_format='channels_first', abs_path=True, fpath=image_adv_path)
      image_ori_var = numpy_to_variable(image_ori_np)
      image_adv_var = numpy_to_variable(image_adv_np)
      
      logits_ori = test_model(image_ori_var)
      logits_adv = test_model(image_adv_var)

      y_ori_var = logits_ori.argmax()
      y_adv_var = logits_adv.argmax()

      total_count += 1
      if y_ori_var == y_adv_var:
        correct_count += 1
    print('{0} samples are correctly labeled over {1} samples.'.format(
        correct_count, total_count))
    acc = float(correct_count) / float(total_count)
    print('Accuracy for {0} : {1}'.format(curt_folder, acc))
    result_dict[curt_folder] = str(acc)

  with open('temp_cls_results_{0}.json'.format(args.test_model), 'w') as fout:
    json.dump(result_dict, fout, indent=2)


if __name__ == '__main__':
    main()
    
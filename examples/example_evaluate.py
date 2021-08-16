""" Script for evaluating AE examples.
"""

import argparse
import importlib
import os
import shutil
import sys
import torch
import torchvision
from tqdm import tqdm

from apfv21.utils import img_utils, imagenet_utils, model_utils

import pdb


def evaluate_adv_example(args):
  target_model_type = args.target_model
  model_class = getattr(torchvision.models, args.target_model)
  model = model_class(pretrained=True).cuda()
  model.eval()

  img_mean, img_std = imagenet_utils.get_imagenet_normalize()
  torch_normalize = model_utils.Normalize(img_mean, img_std)

  img_names = os.listdir(args.benign_dir)
  acc_count = 0
  total_count = 0
  for img_name in tqdm(img_names):
    img_name_noext = os.path.splitext(img_name)[0]

    img_path_benign = os.path.join(args.benign_dir, img_name)
    img_benign_var = img_utils.load_img(img_path_benign).cuda()
    img_benign_var = torch_normalize(img_benign_var)
    pred_benign = torch.argmax(model(img_benign_var), axis=1)

    img_path_adv = os.path.join(args.adv_dir, img_name_noext + ".png")
    if not os.path.exists(img_path_adv):
      continue
    img_adv_var = img_utils.load_img(img_path_adv).cuda()
    img_adv_var = torch_normalize(img_adv_var)
    pred_adv = torch.argmax(model(img_adv_var), axis=1)

    if pred_benign.cpu().numpy()[0] == pred_adv.cpu().numpy()[0]:
      acc_count += 1
    total_count += 1
  accuracy = float(acc_count) / float(total_count)
  print("Evaluate path: ", args.adv_dir)
  print("Target Model: ", args.target_model)
  print("Accuracy: ", accuracy)
  print("{} over {}".format(acc_count, total_count))
  return


def parse_args(args):
  parser = argparse.ArgumentParser(description="PyTorch AE evaluator.")
  parser.add_argument(
      '--benign_dir',
      default="/home/yilan/yilan/workspace/datasets/icip21_images/ori/",
      type=str)
  parser.add_argument('--adv_dir', required=True, type=str)
  parser.add_argument('--target_model', default="vgg16", type=str)
  return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    evaluate_adv_example(args)

if __name__ == "__main__":
   main()
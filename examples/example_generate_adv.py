""" Script for generating AE examples.
"""

import argparse
import importlib
import numpy as np
import os
from PIL import Image
import shutil
import sys
import torch
from tqdm import tqdm

from apfv21.attacks.tidr import TIDR
from apfv21.attacks.afv import AFV
from apfv21.utils import img_utils, imagenet_utils

import pdb


ATTACKS_CFG = {
    "afv": {
        "decay_factor": 1.0,
        "prob": 1.0,
        "epsilon": 16./255,
        "steps": 80,
        "step_size": 1./255,
        "image_resize": 330,
        "dr_weight": 1.0,
        "ti_smoothing": True,
        "ti_kernel_radius": (20, 3),
        "random_start": False,
        "train_attention_model": False,
        "use_pretrained_foreground": True,
        "attention_resize": 256,
        "attention_weight_pth": "",
    },
    "dim": {
        "decay_factor": 1.0,
        "prob": 0.5,
        "epsilon": 16./255,
        "steps": 40,
        "step_size": 2./255,
        "image_resize": 330,
        "random_start": False,
    },
    "tidr": {
        "decay_factor": 1.0,
        "prob": 0.5,
        "epsilon": 16./255,
        "steps": 40,
        "step_size": 2./255,
        "image_resize": 330,
        "dr_weight": 0.0,
        "random_start": False,
    },
    "pgd": {
        "epsilon": 16/255.,
        "k": 40,
        "a": 2/255.,
    },
    "mifgsm": {
        "decay_factor": 1.0,
        "epsilon": 16./255,
        "steps": 40,
        "step_size": 2./255
    }
}

DR_LAYERS = {
  "vgg16": [12],
  "resnet152": [5],
  "inception_v3": [4],
  "mobilenet_v2": [8]
}


def serialize_config(cfg_dict: dict) -> str:
  key_list = list(cfg_dict.keys())
  key_list.sort()
  ret_str = ""
  for key in key_list:
    val = cfg_dict[key]
    if isinstance(val, float):
      val = "{0:.04f}".format(val)
    curt_str = "{}_{}".format(key, val)
    ret_str = ret_str + curt_str + "_"
  ret_str = ret_str.replace('.', 'p')
  ret_str = ret_str.replace('/', '_')
  ret_str = ret_str.replace('(', '_')
  ret_str = ret_str.replace(')', '_')
  ret_str = ret_str.replace(',', '_')
  ret_str = ret_str.replace(' ', '_')
  if len(ret_str) > 255:
    ret_str = ret_str[-128:]
  return ret_str[:-1]


def generate_adv_example(args):
  attack_config = ATTACKS_CFG[args.attack_method]
  suffix_str = "{}_{}_{}".format(
      args.source_model, args.attack_method, serialize_config(attack_config))
  output_folder = os.path.join(args.output_dir, suffix_str)
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  os.mkdir(output_folder)

  imgnet_label = imagenet_utils.load_imagenet_label_dict()

  source_lib = importlib.import_module(
      "apfv21.models." + args.source_model)
  source_model_class = getattr(source_lib, args.source_model.upper())
  source_model = source_model_class(is_normalize=True)

  attack_lib = importlib.import_module(
      os.path.join("apfv21.attacks." + args.attack_method))
  attacker_class = getattr(attack_lib, args.attack_method.upper())
  if args.attack_method == "afv":
    afv_config = ATTACKS_CFG["afv"]
    from apfv21.attacks.models.attention_model import ForegroundAttentionFCN as ForegroundAttention
    # Only one of |use_pretrained_foreground| and |train_attention_model|
    # can be turned on at one time.
    if afv_config['use_pretrained_foreground']:
      assert not afv_config["train_attention_model"]
    if afv_config['train_attention_model']:
      assert not afv_config["use_pretrained_foreground"] 

    attention_model = ForegroundAttention(
        1, 3,
        resize=(afv_config["attention_resize"],
                afv_config["attention_resize"]),
        use_pretrained_foreground=afv_config['use_pretrained_foreground']
    ).cuda() # num_classes, feature_channels

    if afv_config["train_attention_model"]:
      attention_optim = torch.optim.SGD(
          attention_model.parameters(), lr=1e-7, momentum=0.9)
      lr_scheduler = torch.optim.lr_scheduler.StepLR(
          attention_optim, step_size=10, gamma=0.2)
    else:
      attention_optim = None
      if not afv_config['use_pretrained_foreground']:
        assert afv_config["attention_weight_pth"] != ""
    if afv_config["attention_weight_pth"] != "":
      attention_model.load_state_dict(
          torch.load(afv_config["attention_weight_pth"]))
    attacker = attacker_class(source_model, attention_model, 
                              attention_optim, attack_config=afv_config)
  else:
    attacker = attacker_class(source_model, attack_config=attack_config)

  img_names = os.listdir(args.input_dir)
  success_count = 0
  total_count = 0
  for img_name in tqdm(img_names):
    img_name_noext = os.path.splitext(img_name)[0]
    img_path = os.path.join(args.input_dir, img_name)
    img_ori_var = img_utils.load_img(img_path).cuda()
    pred_ori = torch.argmax(source_model(img_ori_var)[1], dim=1)
    if isinstance(attacker, TIDR):
      img_adv_var = attacker(
          img_ori_var, pred_ori, internal=DR_LAYERS[args.source_model])
    elif isinstance(attacker, AFV):
      img_adv_var, attention_map_first, attention_map_last = attacker(
          img_ori_var, pred_ori, internal=DR_LAYERS[args.source_model])
    else:
      img_adv_var = attacker(img_ori_var, pred_ori)
    pred_adv = torch.argmax(source_model(img_adv_var.cuda())[1], dim=1)
    
    output_img = img_utils.save_img(
        img_adv_var, os.path.join(output_folder, img_name_noext + ".png"))
    
    # # Visualization for debuging. 
    # print("Ori: ", img_name, " , ", pred_ori, ":",
    #       imgnet_label[pred_ori.cpu().numpy()[0]])
    # print("Adv: ", img_name, " , ", pred_adv, ":",
    #       imgnet_label[pred_adv.cpu().numpy()[0]])
    # attention_img_first = Image.fromarray(
    #     (attention_map_first.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8))
    # attention_img_first.save("./temp_attention_first.png")
    # attention_img_last = Image.fromarray(
    #     (attention_map_last.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8))
    # attention_img_last.save("./temp_attention_last.png")
    # img_utils.save_img(img_adv_var, "./temp_adv.png")

    if imgnet_label[pred_ori.cpu().numpy()[0]] != \
        imgnet_label[pred_adv.cpu().numpy()[0]]:
      success_count += 1
    total_count += 1
    if total_count % 100 == 0 and args.attack_method == "afv" \
        and afv_config["train_attention_model"]:
      print("Saving attention model...")
      torch.save(
          attention_model.state_dict(),
          "./weights/attention_model_weights_{}.pth".format(total_count))
      lr_scheduler.step()

  if args.attack_method == "afv" and afv_config["train_attention_model"]:
    print("Saving attention model...")
    torch.save(
        attention_model.state_dict(),
        "./weights/attention_model_weights_{}.pth".format(total_count))
  success_rate = float(success_count) / float(total_count)
  print("Success rate: ", success_rate)
  print("{} over {}".format(success_count, total_count))
  return


def parse_args(args):
  parser = argparse.ArgumentParser(description="PyTorch AE generation.")
  parser.add_argument('--source_model',,
                      default="vgg16", type=str)
  parser.add_argument('--attack_method',,
                      default="afv", type=str)
  parser.add_argument('--input_dir', default="sample_images/", type=str)
  parser.add_argument('--output_dir', default="outputs/", type=str)
  return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    generate_adv_example(args)

if __name__ == "__main__":
   main()
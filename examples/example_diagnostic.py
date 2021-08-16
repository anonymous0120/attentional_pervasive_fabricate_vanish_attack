""" Script for diagnostic figures generation.
"""
import cv2
import numpy as np
import os
from PIL import Image
import shutil
import torch
from tqdm import tqdm

from apfv21.attacks.pgd import PGD
from apfv21.attacks.tidr import TIDR
from apfv21.attacks.afv import AFV
from apfv21.models.vgg16 import VGG16
from apfv21.models.resnet152 import RESNET152
from apfv21.utils import img_utils, image_utils, imagenet_utils

import pdb


def task1(output_folder):
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  os.mkdir(output_folder)

  imgnet_label = imagenet_utils.load_imagenet_label_dict()
  # Use vgg16 as surrogate
  vgg16 = VGG16()
  pgd_attack = PGD(vgg16)
  tidr_attack = TIDR(vgg16, dr_weight=10, steps=200)
  img_ori_var = img_utils.load_img("sample_images/dog.jpg").cuda()
  pred_ori = torch.argmax(vgg16(img_ori_var)[1], axis=1)
  print("Generating AEs. attack mtds: PGD, TIDR, surrogate: vgg16")
  img_adv_tidr_var = tidr_attack(img_ori_var, pred_ori, internal=[12])
  img_adv_pgd_var = pgd_attack(img_ori_var, pred_ori)
  # Use resnet152 as victim model
  resnet152 = RESNET152()
  internals = [4]
  print("Evaluating AEs. victim model: resnet152")
  layers_pgd, pred_pgd = resnet152(img_adv_pgd_var.cuda(), internals)
  layers_tidr, pred_tidr = resnet152(img_adv_tidr_var.cuda(), internals)
  # Show predictions
  logits_pgd_np = pred_pgd[0].detach().cpu().numpy()
  show_predictions(logits_pgd_np, imgnet_label, "pgd")
  logits_tidr_np = pred_tidr[0].detach().cpu().numpy()
  show_predictions(logits_tidr_np, imgnet_label, "tidr")
  print("Saving figures.")
  save_features(layers_pgd, output_folder, "pgd")
  save_features(layers_tidr, output_folder, "tidr")


def task2(output_folder):
  """Attention attack.
  """
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
  os.mkdir(output_folder)

  # input_folder = "/home/yantaolu/workspace/data/AFV_experiments/voc_ablation/"
  input_folder = "/home/yantaolu/workspace/data/AFV_experiments/voc/"
  ori_img_names = os.listdir(os.path.join(input_folder, "ori"))

  vgg16 = VGG16()
  pgd_attack = PGD(vgg16, epsilon=8/255.)
  from apfv21.attacks.models.attention_model import ForegroundAttentionFCN as ForegroundAttention
  attention_model = ForegroundAttention(
      1, 3,
      resize=(256, 256),
      use_pretrained_foreground=True
  ).cuda() # num_classes, feature_channels
  afv_config = {
      "decay_factor": 1.0,
      "prob": 0.5,
      "epsilon": 8./255,
      "steps": 100,
      "step_size": 1./255,
      "image_resize": 330,
      "dr_weight": 100.0,
      "ti_smoothing": True,
      "ti_kernel_radius": (20, 3),
      "random_start": False,
      "train_attention_model": False,
      "use_pretrained_foreground": True,
      "attention_resize": 256,
      "attention_weight_pth": "",
  }
  afv_attack = AFV(vgg16, attention_model, None, attack_config=afv_config)

  for ori_img_name in tqdm(ori_img_names):
    ori_img_name_noext = os.path.splitext(ori_img_name)[0]
    curt_img_path = os.path.join(input_folder, "ori", ori_img_name)
    img_ori_var = img_utils.load_img(curt_img_path).cuda()
    pred_ori = torch.argmax(vgg16(img_ori_var)[1], axis=1)
    hm_ori = attention_model(img_ori_var)

    img_adv_pgd_var = pgd_attack(img_ori_var, pred_ori)
    hm_pgd = attention_model(img_adv_pgd_var.cuda())

    img_adv_afv_var, hm_afv_first, hm_afv_last = afv_attack(
        img_ori_var, pred_ori, internal=[12])

    # Sem seg mask visualization
    import torchvision
    import torch.nn as nn
    semseg_model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=True, progress=True, num_classes=21).cuda()
    softmax = nn.Softmax(dim=1)
    objdet_model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=True).cuda()
    objdet_model.eval()

    for p in semseg_model.parameters():
      p.requires_grad = False
    heatmap_ori_dict = semseg_model(img_ori_var)
    heatmap_ori = softmax(heatmap_ori_dict['out'])
    heatmap_ori_np = heatmap_ori.detach().cpu().numpy()
    for idx in range(heatmap_ori_np.shape[1]):
      temp_heatmap_ori = heatmap_ori_np[0, idx]
      curt_seg_img = Image.fromarray(
        (temp_heatmap_ori * 255.).astype(np.uint8))
      curt_seg_img.save(os.path.join(
        output_folder, ori_img_name_noext + "_semseg_img_{}.png".format(idx)))

    # Visualization. 
    img_ori = Image.fromarray(
        np.transpose(
            (img_ori_var.detach().cpu().numpy()[0] * 255.).astype(np.uint8),
            (1, 2, 0)))
    img_ori.save(os.path.join(
        output_folder, ori_img_name_noext + "_ori_img.png"))
    attention_img_ori = Image.fromarray(
        (hm_ori.detach().cpu().numpy()[0, 0] * 255.).astype(np.uint8))
    attention_img_ori.save(os.path.join(
        output_folder, ori_img_name_noext + "_ori_hm.png"))

    from apfv21.utils.VOC2012_1000 import annotation_loader
    gt_path = os.path.join(input_folder, '_annotations', ori_img_name_noext + '.xml')
    gt_bbox = annotation_loader.load_annotations(gt_path, np.array(img_ori).shape[:2])
    
    gt_mask = np.array(Image.open(
        os.path.join(input_folder, '_segmentations', ori_img_name_noext + '.png')))
    idx_255 = np.argwhere(gt_mask == 255)
    for temp_idx_255 in idx_255:
      gt_mask[temp_idx_255[0], temp_idx_255[1]] = 0
    gt_mask = cv2.resize(gt_mask, np.array(img_ori).shape[:2], interpolation=cv2.INTER_NEAREST)
    image_utils.draw_masks(
        np.array(img_ori), gt_mask, 21, from_path=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_ori_mask.png"))
    image_utils.draw_boxes(
        os.path.join(
            output_folder,
            ori_img_name_noext + "_ori_mask.png"),
        gt_bbox,
        annotation_loader.NAME_LIST,
        from_path=True, show_text=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_ori_bboxmask.png"))
    
    attention_img_last = Image.fromarray(
        (hm_afv_last.detach().cpu().numpy()[0, 0] * 255.).astype(np.uint8))
    attention_img_last.save(os.path.join(
        output_folder, ori_img_name_noext + "_afv_hm.png"))
    attention_img_pgd = Image.fromarray(
        (hm_pgd.detach().cpu().numpy()[0, 0] * 255.).astype(np.uint8))
    attention_img_pgd.save(os.path.join(
        output_folder, ori_img_name_noext + "_pgd_hm.png"))

    img_utils.save_img(img_adv_afv_var, os.path.join(
        output_folder, ori_img_name_noext + "_afv_adv.png"))
    img_utils.save_img(img_adv_afv_var, os.path.join(
        output_folder, ori_img_name_noext + "_pgd_adv.png"))

    # Visualize PGD segmentation and detection
    heatmap_pgd = semseg_model(img_adv_pgd_var.cuda())['out']
    mask_pgd = np.argmax(heatmap_pgd.detach().cpu().numpy(), axis=1)
    image_utils.draw_masks(
        np.array(img_ori), mask_pgd[0], 21, from_path=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_pgd_mask.png"))
    bbox_pgd = objdet_model(img_adv_pgd_var.cuda())
    image_utils.draw_boxes(
        os.path.join(
            output_folder,
            ori_img_name_noext + "_pgd_mask.png"),
        gt_bbox,
        annotation_loader.NAME_LIST,
        from_path=True, show_text=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_pgd_bboxmask.png"))
    
    # Visualize AFV segmentation and detection
    heatmap_afv = semseg_model(img_adv_afv_var.cuda())['out']
    mask_afv = np.argmax(heatmap_afv.detach().cpu().numpy(), axis=1)
    image_utils.draw_masks(
        np.array(img_ori), mask_afv[0], 21, from_path=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_afv_mask.png"))
    bbox_afv = objdet_model(img_adv_afv_var.cuda())
    image_utils.draw_boxes(
        os.path.join(
            output_folder,
            ori_img_name_noext + "_afv_mask.png"),
        gt_bbox,
        annotation_loader.NAME_LIST,
        from_path=True, show_text=False,
        out_file=os.path.join(
            output_folder,
            ori_img_name_noext + "_afv_bboxmask.png"))


def show_predictions(logits: np.ndarray,
                     label_dict: dict, suffix: str) -> None:
  softmax_pgd = np.exp(logits)/sum(np.exp(logits))
  top_5_idx = softmax_pgd.argsort()[-5:][::-1]
  print("{}_prediction: ".format(suffix))
  for idx in top_5_idx:
    print("{0}: {1:.04f}".format(label_dict[idx], softmax_pgd[idx]))
  return


def save_features(layers: list, output_folder: str, suffix: str) -> None:
  for layer_idx, layer_var in enumerate(layers):
    layer = layer_var.detach().cpu().numpy()[0]
    layer_max = layer.max()
    for channel_idx, channel in enumerate(layer):
      channel_norm = ((channel / channel.max()) * 255.).astype(np.uint8)
      channel_img = cv2.applyColorMap(channel_norm, cv2.COLORMAP_JET)
      channel_img = cv2.resize(channel_img, (224, 224))
      filepath = os.path.join(
          output_folder,
          "l{0:02d}_c{1:03d}_{2}.png".format(layer_idx, channel_idx, suffix))
      cv2.imwrite(filepath, channel_img)


if __name__ == "__main__":
  output_folder = "temp_diagnostic"
  task2(output_folder)

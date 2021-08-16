import numpy as np
from PIL import Image
import torch
import torchvision

import pdb


def load_img(img_path: str, img_size: tuple=(224, 224)) -> torch.Tensor:
  img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size), 
    torchvision.transforms.ToTensor()])
  img_var = img_transforms(
      Image.open(img_path).convert('RGB')).unsqueeze_(dim=0)
  img_var = torch.autograd.Variable(img_var, requires_grad=True)
  img_var.retain_grad()
  return img_var


def save_img(img_var: torch.Tensor, save_path: str) -> None:
  img_np = np.transpose(img_var.detach().cpu().numpy()[0], (1, 2, 0))
  Image.fromarray((img_np * 255.).astype(np.uint8)).save(save_path)
  return
  
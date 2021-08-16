""" Script for attentional fabricate-vanish attack.
"""
import copy
import numpy as np
from torch.autograd import Variable
import torch
import scipy.stats as st
from scipy import ndimage
import warnings

import pdb
 

class AFV(object):
  """ Class of attentional fabricate-vanish attack.
  """
  def __init__(self, 
               victim_model: torch.nn.Module,
               attention_model: torch.nn.Module, 
               attention_optimizer: torch.optim.Optimizer=None,
               decay_factor: float=1.0, prob: float=0.5,
               epsilon: float=16./255, steps: int=100,
               step_size: float=2./255, image_resize: int=330,
               dr_weight: float=100.0, ti_smoothing: bool=False,
               ti_kernel_radius: tuple=(15, 3), random_start: bool=False,
               attack_config: dict=None):
    """
    Related Paper link: https://arxiv.org/pdf/1803.06978.pdf
    """
    self._victim_model = copy.deepcopy(victim_model)
    self._attention_model = attention_model
    self._attention_optimizer = attention_optimizer
    self._loss_fn = torch.nn.CrossEntropyLoss().cuda()
    if attack_config == None:
      self._epsilon = epsilon
      self._steps = steps
      self._step_size = step_size
      self._rand = random_start
      self._decay_factor = decay_factor
      self._prob = prob
      self._image_resize = image_resize
      self._dr_weight = dr_weight
      self._ti_smoothing = ti_smoothing
      self._ti_kernel_radius = ti_kernel_radius
    else:
      warnings.warn("Over-writting AFV pameters by |attack_config|.")
      self._epsilon = attack_config["epsilon"]
      self._steps = attack_config["steps"]
      self._step_size = attack_config["step_size"]
      self._rand = attack_config["random_start"]
      self._decay_factor = attack_config["decay_factor"]
      self._prob = attack_config["prob"]
      self._image_resize = attack_config["image_resize"]
      self._dr_weight = attack_config["dr_weight"]
      self._ti_smoothing = attack_config["ti_smoothing"]
      self._ti_kernel_radius = attack_config["ti_kernel_radius"]

    if self._ti_smoothing:
      assert len(self._ti_kernel_radius) == 2
      kernel = self.gkern(
          self._ti_kernel_radius[0],
          self._ti_kernel_radius[1]).astype(np.float32)
      self._stack_kernel = np.stack([kernel, kernel, kernel])

  def __call__(self,
               X_nat: torch.Tensor,
               y: torch.Tensor,
               internal: tuple=()) -> torch.Tensor:
    """
      Given examples (X_nat, y), returns adversarial
      examples within epsilon of X_nat in l_infinity norm.
    """
    X_nat_np = X_nat.detach().cpu().numpy()
    for p in self._victim_model.parameters():
      p.requires_grad = False
    
    self._victim_model.eval()
    if self._rand:
      X = X_nat_np + np.random.uniform(-self._epsilon,
                                        self._epsilon,
                                        X_nat_np.shape).astype('float32')
    else:
      X = np.copy(X_nat_np)
    
    momentum = 0.0
    attention_map = None
    attention_map_first = None
    attention_map_last = None
    for step_idx in range(self._steps):
      X_var = Variable(torch.from_numpy(X).cuda(),
                       requires_grad=True, volatile=False)
      y_var = y.cuda()

      # Calculate attention map.
      if self._attention_optimizer == None:
        self._attention_model.eval()
      else:
        self._attention_model.train()

      attention_map = self._attention_model(X_var)
      if step_idx == 0:
        attention_map_first = attention_map.clone().detach()
      elif step_idx == self._steps - 1:
        attention_map_last = attention_map.clone().detach()

      # Foreground processing
      X_fg_var = X_var * attention_map_first
      layers, _ = self._victim_model(X_fg_var, internal=internal)
      # Background processing
      X_bg_var = X_var * (1. - attention_map_first)
      rnd = np.random.rand()
      if rnd < self._prob:
        transformer = _tranform_resize_padding(
            X.shape[-2], X.shape[-1], self._image_resize,
            resize_back=True)
        X_trans_var = transformer(X_bg_var)
      else:
        X_trans_var = X_bg_var
      _, scores = self._victim_model(X_trans_var, internal=internal)

      # Calculate gradients for dispersion loss.
      loss_dr = 0.0
      for layer_idx, target_layer in enumerate(layers):
        temp_loss_dr = target_layer.var()
        loss_dr += temp_loss_dr
      # Calculate gradients for logit loss(TIDIM).
      loss_logit = -1 * self._loss_fn(scores, y_var)
      # Combine loss
      loss = loss_logit + self._dr_weight * loss_dr

      self._victim_model.zero_grad()
      if self._attention_optimizer != None:
        self._attention_optimizer.zero_grad()
      loss.backward()
      # train attention model if optimizer is not None.
      if self._attention_optimizer != None:
        self._attention_model.train()
        self._attention_optimizer.step()

      # Update adversarial image
      grad = X_var.grad.data.cpu().numpy()
      # Apply translation-invariant if |_ti_smoothing| flag is turned on.
      if self._ti_smoothing:
        grad = self.depthwise_conv2d(grad, self._stack_kernel)

      X_var.grad.zero_()
      velocity = grad / np.mean(np.absolute(grad), axis=(1, 2, 3))
      momentum = self._decay_factor * momentum + velocity

      X -= self._step_size * np.sign(momentum)
      X = np.clip(X, X_nat_np - self._epsilon, X_nat_np + self._epsilon)
      X = np.clip(X, 0, 1) # ensure valid pixel range
    return torch.from_numpy(X), attention_map_first, attention_map_last

  @staticmethod
  def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

  @staticmethod
  def depthwise_conv2d(in1, stack_kernel):
    ret = []
    for temp_in in in1:
      # numpy convolve operates differently to CNN conv, 
      # however they are the same when keernel is symetric.
      temp_out = ndimage.convolve(temp_in, stack_kernel, mode='constant')
      ret.append(temp_out)
    ret = np.array(ret)
    return ret


class _tranform_resize_padding(torch.nn.Module):
  def __init__(self, image_h, image_w, image_resize, resize_back=False):
    super(_tranform_resize_padding, self).__init__()
    self.shape = [image_h, image_w]
    self._image_resize = image_resize
    self.resize_back = resize_back

  def __call__(self, input_tensor):
    assert self.shape[0] < self._image_resize \
      and self.shape[1] < self._image_resize
    rnd = np.random.randint(self.shape[1], self._image_resize)
    input_upsample = torch.nn.functional.interpolate(
        input_tensor, size=(rnd, rnd), mode='nearest')
    h_rem = self._image_resize - rnd
    w_rem = self._image_resize - rnd
    pad_top = np.random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padder = torch.nn.ConstantPad2d(
        (pad_left, pad_right, pad_top, pad_bottom), 0.0)
    input_padded = padder(input_upsample)
    if self.resize_back:
      input_padded_resize = torch.nn.functional.interpolate(
          input_padded, size=self.shape, mode='nearest')
      return input_padded_resize
    else:
      return input_padded
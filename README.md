# Attentively Fabricate and Erase: A Pervasive Black-box Adversarial Attack with Cross-task Transferability
## Anonymous submission.

## Abstract

Recent works have shown that adversarial examples can transfer across, and attack different neural networks designed for the same tasks as well as networks designed for completely different tasks. This shows once more the extent of threat these malicious samples represent. Most state-of-the-art approaches highly rely on task-specific loss functions, making them less transferable across different networks, and much less effective across different tasks. In addition, it has been shown that denoising-based adaptive defense approaches provide promising performance against the aforementioned attacks. Hence, we propose a method to Attentively Fabricate and Erase (AF&E), which is a pervasive black-box attack with better transferability and attack effectiveness. The proposed AF\&E is designed to fabricate obstructive textures and erase benign informative features simultaneously. We treat the adversarial example transferability as a latent contribution for each layer of deep neural networks. The attack performance is maximized by balancing transferability and task-specific loss function via our proposed Foreground Attention Block (FAB). Comprehensive set of experiments on the ImageNet, VOC and MS COCO datasets show that the proposed AF&E attack is more effective and has better transferability compared to the state-of-the-art baselines. It causes significant degradation in the performance of the classification, detection and segmentation networks.

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/flow.PNG" width="100%" height="100%">

## Installation

0. The Pytorch version that is used for this work:
~~~
'torch==1.2.0',
'torchvision==0.4.0'
~~~

1. Download and Install APFV
~~~
git clone https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack.git
cd attentional_pervasive_fabricate_vanish_attack
pip install -e .
~~~

2. Example for AE generation
~~~
python examples/example_generate_adv.py --input_dir /path/to/original/images --output_dir /path/to/save/generated/images
~~~

3. Example for evaluation
~~~
python examples/example_evaluate.py --benign_dir /path/to/benign/images --adv_dir /path/to/adv/images
~~~

## Main Results

### Object detection on VOC and COCO

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/detection.PNG" width="100%" height="100%">

### Semantic segmentaion on VOC and COCO

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/segmentation.PNG" width="100%" height="100%">

### Classification on ImageNet

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/classification_robust.PNG" width="80%" height="80%">

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/classification_vanilla.PNG" width="100%" height="100%">

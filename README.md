# Attentional Ppervasive Fabricate Vanish Attack
## Anonymous repository for AAAI2022 submission.

## Abstract

Adversarial examples have emerged as increasingly severe threats for deep neural networks. Recent works have revealed that these malicious samples can transfer across, and effectively attack different neural networks models. The state-of-the-art attack methodologies leverage Fast Gradient Sign Method to generate obstructing textures, which can cause neural networks to make incorrect inferences. However, the over-reliance on task-specific loss functions makes the adversarial examples less transferable across different networks, and substantially weakens the attack effectiveness across different tasks, which is a property needed to have more practical, applicable and significant attacks. On the other hand, recent denoising-based adaptive defense approaches provide promising performance against aforementioned attacks. Therefore, to achieve better transferability and attack effectiveness, we propose a novel attack, referred to as the Attentional Pervasive Fabricate Vanish (APFV) attack, which is able to erase benign representative features and generate obstructive textures simultaneously. The proposed APFV attack treats the adversarial example transferability as a latent contribution for each layer of deep neural networks, and maximizes the attack performance by balancing transferability and task-specific loss function via our proposed Foreground Attention Block (FAB). The experimental results on ImageNet, VOC and MS COCO datasets show that the proposed APFV attack is more effective and has better transferability, compared to the state-of-the-art baselines, by causing significant drop and more degradation in the performance of the classification, detection and segmentation networks.

## Main Results

### Object detection on VOC and COCO

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/detection.PNG" width="100%" height="100%">

### Semantic segmentaion on VOC and COCO

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/segmentation.PNG" width="100%" height="100%">

### Classification on ImageNet

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/classification_robust.PNG" width="100%" height="100%">

<img src="https://github.com/anonymous0120/attentional_pervasive_fabricate_vanish_attack/blob/main/experimental_results/classification_vanilla.PNG" width="100%" height="100%">

## Installation



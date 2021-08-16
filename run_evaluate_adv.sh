#!/bin/bash

BENIGN_DIR=/path/to/benign/images
ADV_DIR=/path/to/adv/images
TARGET_MODEL=vgg16

python3 examples/example_evaluate.py --benign_dir  $BENIGN_DIR --adv_dir $ADV_DIR --target_model $TARGET_MODEL

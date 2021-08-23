#!/bin/bash

INPUT_DIR=/path/to/benign/images
OUTPUT_DIR=/path/to/output/images
ATTACK_METHOD=afv

for SOURCE_MODEL in vgg16 resnet152
do
  python3 examples/example_generate_adv.py --input_dir  $INPUT_DIR --output_dir $OUTPUT_DIR --source_model $SOURCE_MODEL --attack_method $ATTACK_METHOD
done

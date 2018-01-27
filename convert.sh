#!/usr/bin/env bash

python convert.py \
/home/melody/develop/caffe-segmentation/pspnet/models/pspnet101_VOC2012_deploy_t.prototxt \
--caffemodel=/home/melody/hdd1/data/pspnet101_VOC2012.caffemodel \
--data-output-path=pspnet.npy \
--code-output-path=pspnet.py \
2>&1 | tee log.txt
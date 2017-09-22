#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12 
EXAMPLE=/data/hjy1312/data/RESNET/cifar-10
DATA=/data/hjy1312/data/RESNET/cifar-10
TOOLS=/data/hjy1312/C3D-master/C3D-v1.1/build/tools
 
$TOOLS/compute_image_mean $EXAMPLE/cifar10_train_lmdb \
$DATA/mean.binaryproto 
echo "Done."


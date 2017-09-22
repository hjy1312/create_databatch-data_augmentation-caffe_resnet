#!/bin/bash
/data/hjy1312/Downloads/caffe-master/build/tools/caffe.bin train --solver=./res_net_solver_aug.prototxt --gpu=1 &>./resnet_aug.log&

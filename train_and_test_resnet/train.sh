#!/bin/bash
/data/hjy1312/Downloads/caffe-master/build/tools/caffe.bin train --solver=./res_net_solver.prototxt --gpu=1 &>./main1.log&

#caffe_root = "/data/hjy1312/Downloads/caffe-master/"
#import sys
#import pdb
#sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import tools
from caffe import layers as L, params as P, to_proto

def conv_BN_scale_relu(split, bottom, numout, kernelsize, stride, pad):
    conv = L.Convolution(bottom,kernel_size=kernelsize,stride=stride,
                         num_output=numout,pad=pad,bias_term=True,
                         weight_filler=dict(type = 'xavier'),   
                         bias_filler = dict(type = 'constant'), 
                         param = [dict(lr_mult = 1, decay_mult = 1), 
                                  dict(lr_mult = 2, decay_mult = 0)])
    if split == 'train':
        BN = L.BatchNorm(conv,batch_norm_param=dict(use_global_stats = False),
                         in_place=True,param=[dict(lr_mult=0,decay_mult=0),
                                              dict(lr_mult = 0, decay_mult = 0), 
                                              dict(lr_mult = 0, decay_mult = 0)])
    else:
        BN = L.BatchNorm(conv,batch_norm_param=dict(use_global_stats = True), 
                         in_place = True, param = [dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0), 
                                                   dict(lr_mult = 0, decay_mult = 0)])
    scale = L.Scale(BN, scale_param = dict(bias_term = True), in_place = True)  
    relu = L.ReLU(scale, in_place = True)
    return scale,relu

def ResNet_block(split,bottom,numout,kernelsize,stride,projection_stride,pad):
    if(projection_stride==1):
         scale0 = bottom
    else:
        scale0,relu0 = conv_BN_scale_relu(split,bottom,numout,1,projection_stride,0)#use 1*1kernel to make the shape of block's input and output the same
    scale1,relu1 = conv_BN_scale_relu(split,bottom,numout,kernelsize,projection_stride,pad)#if projection_stride=2,use to convolution and downsampling 
    scale2,relu2 = conv_BN_scale_relu(split,relu1,numout,kernelsize,stride,pad)
    wise = L.Eltwise(scale2,scale0,operation=P.Eltwise.SUM)
    wise_relu = L.ReLu(wise,in_place = True)
    return wise_relu 

def ResNet(split):
    train_file = "/data/hjy1312/data/RESNET/cifar-10/cifar10_train_lmdb"
    test_file = "/data/hjy1312/data/RESNET/cifar-10/cifar10_test_lmdb"
    #mean_file_pad = "/data/hjy1312/data/RESNET/cifar-10/mean_pad.binaryproto"
    mean_file = "/data/hjy1312/data/RESNET/cifar-10/mean.binaryproto"
    if split == 'train':
        data,labels = L.Data(source=train_file,backend=P.Data.LMDB,
			     batch_size=128,ntop=2,
                             transform_param = dict(mean_file = mean_file, 
                                                    crop_size = 28, 
                                                    mirror = True))
    else:
       data,labels = L.Data(source=test_file,backend=P.Data.LMDB,
			    batch_size=128,ntop=2,
                            transform_param = dict(mean_file = mean_file,
						   crop_size=28))
    repeat = 3
    #conv1_x
    scale,result = conv_BN_scale_relu(split,data,numout=16,kernelsize=3,stride=1,pad=1)
    #conv2_x
    for ii in range(repeat):
        projection_stride = 1
        result = ResNet_block(split,result,numout=16,kernelsize=3,stride=1,
			      projection_stride=projection_stride,pad=1)
    #conv3_x
    for ii in range(repeat):
        if ii==0:
            projection_stride = 2
        else:  
            projection_stride = 1
        result = ResNet_block(split,result,numout=32,kernelsize=3,stride=1,
			      projection_stride=projection_stride,pad=1)
  #conv4_x
    for ii in range(repeat):
        if ii==0:
            projection_stride = 2
        else:
            projection_stride = 1
        result = ResNet_block(split,result,numout=64,kernelsize=3,stride=1,
			      projection_stride=projection_stride,pad=1)

    #global pooling
    pool = L.Pooling(result, pool = P.Pooling.AVE, global_pooling = True)

    #FC
    IP = L.InnerProduct(pool, num_output = 10, 
                        weight_filler = dict(type = 'msra'), 
                        bias_filler = dict(type = 'constant'))
    #calculate accuracy
    acc = L.Accuracy(IP, labels)
    #softmax loss
    loss = L.SoftmaxWithLoss(IP, labels)
    #pdb.set_trace()
    return to_proto(acc, loss)
  
def make_net():
    
    with open(train_dir, 'w') as f:
        f.write(str(ResNet('train')))
        
    with open(test_dir, 'w') as f:
        f.write(str(ResNet('test')))

if __name__ == '__main__':
    
    train_dir = './train.prototxt'
    test_dir = './test.prototxt'
    make_net()
   
  

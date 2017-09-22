#-*- coding: UTF-8 -*-
import numpy as np
import caffe
import matplotlib.pyplot as plt

caffe.set_device(1)
caffe.set_mode_gpu()
mu = np.load('/data/hjy1312/data/RESNET/cifar-10/mean.npy')
mu = mu.mean(1).mean(1)
img = 'puppy.jpg'
model_def = 'deploy.prototxt'
model_prefix = '_iter_'
model_postfix = '.caffemodel'
caffe_model = model_prefix+str(int(66000))+model_postfix
labels_filename = 'labels.txt'
net = caffe.Net(model_def, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu) 
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
im = caffe.io.load_image(img)
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()
labels = np.loadtxt(labels_filename, str, delimiter='\t')
prob = net.blobs['Softmax1'].data[0].flatten()
order = prob.argsort()[-1]
plt.title(labels[order])
plt.imshow(im)
plt.axis('off')
plt.show()
print 'the class is',labels[order]
for layer_name,blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

from define_resnet import *
#this_dir = '/data/hjy1312/data/RESNET/ResNet-on-Cifar10-master/script/resnet_model'
train_dir = './train.prototxt'
test_dir = './test.prototxt'
def make_net():
  with open(train_dir,'w') as f:
    f.write(str(ResNet('train')))
  with open(test_dir,'w') as f:
    f.write(str(ResNet('test')))

if __name__ == '__main__':
  make_net()

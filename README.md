 This project is to bulid the 20-layer resnet based on cifar10 python version or your own data batch
 Firstly,create your own training and test batch with your pictures using tran_pic _to_mat,or you should down load the cifar10 python version.
 Secondly,do the data augmentation with data_augumentation,flip is not needed if you use mirror=True in your datalayer of train.prototxt
 Thirdly,transform the data to lmdb format with tran_matdata_to_lmdb
 Fourthly,define the resnet with build_resnet
 Fifthly,train and test the resnet with train_and_test_resnet
 At last,deploy the resnet with deploy_resnet

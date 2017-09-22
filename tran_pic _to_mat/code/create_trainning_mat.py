# -*- coding: UTF-8 -*-
import numpy as np
import glob
import random
import math
import os.path as osp
from scipy.io import savemat
from scipy.misc import imread, imresize, imsave,imshow

def img_preprocess(im_path):
    im = imread(im_path)
    im_resize = imresize(im,[32,32],'bilinear')
    #to deal with the 4 channel pictures
    if(im_resize.shape[2]>3):
        im_resize = im_resize[:,:,0:2]
    return im_resize

#to get the folder named by the class,for example,'cat'
#there must not be other folders 
def traversal_dir(root_path):
    list = []  
    if (osp.exists(root_path)):    
        files = glob.glob(root_path + '/*' )  
        #print files  
        for file in files:    
            if (osp.isdir(file)):   
                h = osp.split(file)  
                #print h[1]  
                list.append(h[1])  
        #print list
    return list

#the get the path postfix of the pictures,for example,'1.jpg'   
def traversal_img(img_path):
    list = []
    if (osp.exists(img_path)):    
        files1 = glob.glob(img_path + '/*.jpg' )
        files2 = glob.glob(img_path + '/*.png' )
        files = files1+files2  
        #print files  
        for file in files:       
            h = osp.split(file)    
            list.append(h[1])  
        #print list
    return list

#to transform pictures and store in an array
def get_data(root_path):
    labels_dict={'airplane':1,'automobile':2,'bird':3,
                 'cat':4,'deer':5,'dog':6,'frog':7,
                 'horse':8,'ship':9,'truck':10}

    data = np.zeros((1,32,32,3))
    i = 0
    labels = np.zeros(1)
    img_path_batch = traversal_dir(root_path)
    #the folder class_name include images with the same class
    for class_name in img_path_batch:
        img_path_postfix_batch = traversal_img(osp.join(root_path,class_name))
        for img_path_postfix in img_path_postfix_batch:
            img_path = osp.join(root_path,class_name,img_path_postfix)
            im_resize = img_preprocess(img_path)
            #print img_path
            #print im_resize.shape
            im_resize = im_resize.reshape(1,32,32,3)
            if(i==0):
                data[0] = im_resize
                labels[0] = labels_dict[class_name]
                i = i+1
            else:
	        data = np.concatenate((data,im_resize),axis = 0)
	 	labels = np.concatenate((labels,np.array([labels_dict[class_name]])),axis = 0)
    l = data.shape[0]
    indice = range(l)
    random.shuffle(indice)
    data = data[indice]
    labels = labels[indice]
    #print indice
    #print labels
    #print data.shape
    return data,labels

# to save the transform result to mat file in the form of dict
def create_mat(data,labels,batch_size,save_path):
    N = data.shape[0]
    #print N
    batch_num = int(math.floor(N/batch_size)+1)
    #print batch_num
    #create trainning data batch
    for i in xrange(batch_num-2):
        filename = 'data_batch_'+str(i+1)
        data_batch = data[0:int(batch_size*(i+1))]
        labels_batch = labels[0:int(batch_size*(i+1))]
        path = osp.join(save_path,filename)
        datadict = {'data':data_batch,'labels':labels_batch}
        savemat(path,datadict)
    
    #create testing data batch
    filename = 'test_batch'
    path = osp.join(save_path,filename)
    data_batch = data[int((batch_num-1)*batch_size):N]
    labels_batch = labels[int((batch_num-1)*batch_size):N]
    datadict = {'data':data_batch,'labels':labels_batch}
    savemat(path,datadict)


#based on the pictures you have,create your own trainning batch and testing batch
#store them into mat files in the format of dictionary
def main_do(root_path,save_path,batch_size):
    data,labels = get_data(root_path)
    create_mat(data,labels,batch_size,save_path)

if __name__ == '__main__':
    root_path = '.../'
    save_path = '.../save_mat/'
    batch_size = 30
    main_do(root_path,save_path,batch_size)


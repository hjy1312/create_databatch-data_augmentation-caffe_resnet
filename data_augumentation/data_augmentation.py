from data_utils import *
import numpy as np
import cv2
from scipy.io import savemat,loadmat
import cPickle
from scipy.misc import imread, imresize, imsave,imshow,imrotate

#x:data,y:labels
def get_rawdata(data_dir):
    x,y = load_CIFAR_batch(data_dir)   
    return x,y

def get_rawdata_from_mat(mat_path):
    datadict = loadmat(mat_path)
    x = datadict['data']
    y = datadict['labels']   
    return x,y

def rotate_data(x,y,theta):
    N = x.shape[0]
    x_rotate = np.zeros_like(x)
    y_rotate = np.zeros_like(y)
    for i in xrange (N):
        x_rotate[i] = imrotate(x[i],theta)
        y_rotate[i] = y[i]
    #x = np.concatenate((x,x_rotate),axis = 0)
    #y = np.concatenate((y,y_rotate),axis = 0)
    return x_rotate,y_rotate

def flip_data(x,y,direction):
    N = x.shape[0]
    x_flip = np.zeros_like(x)
    y_flip = np.zeros_like(y)
    for i in xrange (N):
        x_flip[i] = cv2.flip(x[i],0)
        y_flip[i] = y[i]
    #x = np.concatenate((x,x_flip),axis = 0)
    #y = np.concatenate((y,y_flip),axis = 0)
    return x_flip,y_flip

def translate_data(x,y,tx,ty):
    N = x.shape[0]
    #translation matrice,[[1,0,tx],[0,1,ty]],tx is the offset of the x axis,ty is the offset of the y axis
    M = np.float32([[1,0,tx],[0,1,ty]])
    N,rows,cols,channel = x.shape
    #translation
    x_tran = np.zeros_like(x)
    y_tran = np.zeros_like(y)
    for i in xrange (N):
        x_tran[i] = cv2.warpAffine(x[i],M,(cols,rows))
        y_tran[i] = y[i]
    #x = np.concatenate((x,x_tran),axis = 0)
    #y = np.concatenate((y,y_tran),axis = 0)
    return x_tran,y_tran

def gaussian_blur(x,y,kernel_size,sigma):
    N = x.shape[0]
    x_blur = np.zeros_like(x)
    y_blur = np.zeros_like(y)
    for i in xrange (N):
        x_blur[i] = cv2.GaussianBlur(x[i], kernel_size, sigma)
        y_blur[i] = y[i]
    #x = np.concatenate((x,x_blur),axis = 0)
    #y = np.concatenate((y,y_blur),axis = 0)
    return x_blur,y_blur

#to create a batch of augmentation data
def create_mat_data(x,y,data_dir):
    N = x.shape[0]
    #reshape to Nx3x32x32
    x = x.transpose(0,3,1,2)
    #reshape to Nx(3x32x32)
    x = x.reshape(N,-1)
    #tranform y from an array to a list
    #y = y.tolist()
    aug_filename = data_dir+'_aug'
    datadict = {'data':x,'labels':y}
    savemat(aug_filename,datadict)
    #with open (aug_filename,'wb') as f:
        #cPickle.dump(datadict,f) 

#to do the data augmentation,include rotating and translating
def transform_data(x,y):
    #rotate
    x_rotate,y_rotate = rotate_data(x,y,45)
    x1 = np.concatenate((x,x_rotate),axis = 0)
    y1 = np.concatenate((y,y_rotate),axis = 0)
    x_tran,y_tran = translate_data(x1,y1,10,0)
    x2 = np.concatenate((x1,x_tran),axis = 0)
    y2 = np.concatenate((y1,y_tran),axis = 0)
    return x2,y2

def main_do():
    filename_prefix = 'data_batch_'
    for i in xrange(5):
        data_dir = filename_prefix+str(i+1)
        x,y = get_rawdata(data_dir)
        x,y = transform_data(x,y)
        create_mat_data(x,y,data_dir)
    test_file_prefix = 'test_batch'
    x,y = get_rawdata(test_file_prefix)
    x,y = transform_data(x,y)
    create_mat_data(x,y,test_file_prefix)

if __name__ == '__main__':
    
    main_do()


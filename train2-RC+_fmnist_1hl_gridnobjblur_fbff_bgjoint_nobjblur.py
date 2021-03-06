from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('fMNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np
import scipy
from scipy.ndimage.interpolation import zoom
from random import shuffle

def batch_mod(x,y): # adds balanced dimming random images with None label
  x_1 = np.int(np.round((np.shape(x)[0])*10.5/10.))
  x1 = np.zeros([x_1,np.shape(x)[1]])
  x1[0:np.shape(x)[0],:] = x
  x1[np.shape(x)[0]:,:] = np.random.random([x_1-np.shape(x)[0],1])*np.random.random([x_1-np.shape(x)[0],np.shape(x)[1]])
  y1 = np.zeros([x_1,np.shape(y)[1]+1])
  y1[0:np.shape(x)[0],:-1] = y
  y1[np.shape(x)[0]:,-1] = 1
  return x1, y1

def gaussian_blur(imgd,sigma_max,size_h=40): # uniform distribution between 0 and sigma_max (which is in pixels)
    img_d = 0.*imgd
    for i in range(np.shape(imgd)[0]):
        blur_i = sigma_max*np.random.random(1)[0]
        img_d[i,:] = np.reshape(scipy.ndimage.filters.gaussian_filter(np.reshape(imgd[i,:],[size_h,size_h]),blur_i),[1,size_h*size_h])
    return img_d

def occlusion_distort_blocks(imgd,level_max,size_h=40): # uniform distribution between 0 and level_max (which is betwen 0 and 1), block knockout
    img_d = 0.*imgd
    for i in range(np.shape(imgd)[0]):
        occlude_i = np.int(np.floor(level_max*size_h*np.random.random(1)[0]))
        x_min = np.int(np.round((size_h-occlude_i)*np.random.random(1)[0]))
        y_min = np.int(np.round((size_h-occlude_i)*np.random.random(1)[0]))
        dum_1 = imgd[i,:].copy()
        dum_1 = np.reshape(dum_1,[size_h,size_h])
        dum_1[x_min:x_min+occlude_i,y_min:y_min+occlude_i] = 0.
        img_d[i,:] = np.reshape(dum_1,[1,size_h*size_h])
    return img_d

def batch_gen_specific(imgd,sigma_max,level_max,size_h=40): # 0.5 occlusion and blur
    img_d = 0.*imgd
    half_size = np.int(round(np.shape(imgd)[0]/2.))
    img_d[0:half_size,:] = gaussian_blur(imgd[0:half_size,:],sigma_max,size_h)
    img_d[half_size:,:] = occlusion_distort_blocks(imgd[half_size:,:],level_max,size_h)
    return img_d

def batch_mod_test(y): # adds None label
  y1 = np.zeros([np.shape(y)[0],np.shape(y)[1]+1])
  y1[:,:-1] = y
  return y1

def gen_random_pattern(n,p_mix,n_add_p): # p_mix blur+occlusion on random images, n_add_p percent of null and full images added
  n_h = np.int(np.round(n*p_mix))
  x = np.random.random([n*(1+2*n_add_p),1])*np.random.random([n*(1+2*n_add_p),784])
  y = np.zeros([n*(1+2*n_add_p),11])
  y[:,-1] = 1
  x[0:n_h,:] = batch_gen_specific(x[0:n_h,:],10,1)
  x[n_h:n_h+np.int(n*n_add_p),:] = np.ones([np.int(n*n_add_p),784])
  x[n_h+np.int(n*n_add_p):n_h+2*np.int(n*n_add_p),:] = 1e-4*np.random.random([np.int(n*n_add_p),784])
  return x,y

def random_pattern_gen(n,size_h=40):
  x = np.random.random([n,size_h*size_h])
  y = np.zeros([n,11])
  y[:,-1] = 1.
  for i in range(np.shape(x)[0]):
    dum0 = np.random.random(1)[0]
    dum1 = zoom(np.reshape(x[i,:],[size_h,size_h]),dum0)
    dum2_x = np.int((size_h-np.ceil(dum0*1.*size_h))*np.random.random(1)[0])
    dum2_y = np.int((size_h-np.ceil(dum0*1.*size_h))*np.random.random(1)[0])
    dum3 = np.zeros([size_h,size_h])
    dum3[dum2_x:dum2_x+np.shape(dum1)[0],dum2_y:dum2_y+np.shape(dum1)[1]] = dum1
    x[i,:] = np.reshape(dum3,[1,size_h*size_h])
  x = batch_gen_specific(x,9,1,size_h)
  return x,y

def random_pattern_gen_cue(n,size_h=40):
  x = np.random.random([n,size_h*size_h])
  y = np.zeros([n,11])
  y[:,-1] = 1.
  y_z_h = np.zeros([n,10])
  y_c_h = np.zeros([n,11])
  y_o_h = np.zeros([n,2])
  for i in range(np.shape(x)[0]):
    dum0 = np.random.random(1)[0]
    dum1 = zoom(np.reshape(x[i,:],[size_h,size_h]),dum0)
    dum2_x = np.int((size_h-np.ceil(dum0*1.*size_h))*np.random.random(1)[0])
    dum2_y = np.int((size_h-np.ceil(dum0*1.*size_h))*np.random.random(1)[0])
    dum3 = np.zeros([size_h,size_h])
    dum3[dum2_x:dum2_x+np.shape(dum1)[0],dum2_y:dum2_y+np.shape(dum1)[1]] = dum1
    x[i,:] = np.reshape(dum3,[1,size_h*size_h])
    y_z_h[i,np.random.randint(10)] = 1.
    y_o_h[i,1] = 1.
    y_c_h[i,-1] = 1.
    y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
  x = batch_gen_specific(x,9,1,size_h)
  return x,y,y_c_h,y_z_h,y_o_h

def gridify(imgd,labd,n,size_h=40): # n determines how many images will fit on the 2x2 grid
    imgd1 = imgd.copy()
    labd1 = labd.copy()
    labd2 = 0.*labd.copy()
    imgd2 = np.zeros([np.shape(imgd1)[0],size_h*size_h])
    size_h = 40
    for i in range(np.shape(imgd1)[0]):
        dum_h = np.zeros([size_h,size_h])
        order_h = np.arange(4)
        shuffle(order_h)
        order_h = order_h[0:n]
        if 0 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])
            dum_h[0:size_h/2,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 1 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                         
            dum_h[0:size_h/2,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 2 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                          
            dum_h[size_h/2:,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 3 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                          
            dum_h[size_h/2:,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        imgd2[i,:] = np.reshape(dum_h,size_h*size_h)
    for i in range(np.shape(imgd1)[0]):
        labdn = np.sum(labd2[i,:]>0)
        if labdn == 0:
          print(labdn)
        labd2[i,:] = labd2[i,:]/labdn
    return imgd2,labd2

def gridify_exact(imgd,labd,n,size_h=40): # n determines how many images will fit on the 2x2 grid
    imgd1 = imgd.copy()
    labd1 = labd.copy()
    labd2 = 0.*labd.copy()
    imgd2 = np.zeros([np.shape(imgd1)[0],size_h*size_h])
    size_h = 40
    for i in range(np.shape(imgd1)[0]):
        dum_h = np.zeros([size_h,size_h])
        order_h = np.arange(4)
        shuffle(order_h)
        order_h = order_h[0:n]
        ord_im = np.zeros([4,1])
        for j in range(4):
            if j == 0:
                ord_im[j,0] = np.random.randint(np.shape(imgd1)[0])
            else:
                dum1_h = ord_im[j-1,0]
                while dum1_h in ord_im[:j,0]:
                    dum1_h = np.random.randint(np.shape(imgd1)[0])
                    for k in range(j):
                        if np.argmax(labd1[np.int(dum1_h),:]) == np.argmax(labd1[np.int(ord_im[k,0]),:]):
                            dum1_h = ord_im[j-1]
                ord_im[j,0] = dum1_h
        #print ord_im
        if 0 in order_h:
            ind_h = np.int(ord_im[0])
            dum_h[0:size_h/2,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 1 in order_h:
            ind_h = np.int(ord_im[1])                        
            dum_h[0:size_h/2,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 2 in order_h:
            ind_h = np.int(ord_im[2])                         
            dum_h[size_h/2:,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 3 in order_h:
            ind_h = np.int(ord_im[3])                         
            dum_h[size_h/2:,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        imgd2[i,:] = np.reshape(dum_h,size_h*size_h)
    for i in range(np.shape(imgd1)[0]):
        labdn = np.sum(labd2[i,:]>0)
        if labdn == 0:
          print(labdn)
        labd2[i,:] = labd2[i,:]/labdn
    return imgd2,labd2

def accuracy_n(y1,y2): # n is no. of observations, y2 is true label
    n_h = np.shape(y1)[0]
    acc = np.zeros([n_h,1])
    for i in range(n_h):
        obj_h = np.sum(y2[i,:])
        if obj_h == 0:
            print(obj_h)
        y2_h = y2[i,:].copy()
        y1_h = y1[i,:].copy()
        obj_ind = y1_h.argsort()[-np.int(obj_h):][::-1]
        y1_h = 0.*y1_h
        y1_h[obj_ind] = 1.
        y2_h[y2_h>0] = 1.
        acc[i,0] = np.sum(np.multiply(y1_h,y2_h))*1./(obj_h*1.)
    acc = np.mean(acc)
    return acc

def gridify_stoc(imgd,labd,n,size_h=40): # n determines how many max images will fit on the 2x2 grid
    imgd1 = imgd.copy()
    labd1 = labd.copy()
    labd2 = 0.*labd.copy()
    imgd2 = np.zeros([np.shape(imgd1)[0],size_h*size_h])
    size_h = 40
    for i in range(np.shape(imgd1)[0]):
        dum_h = np.zeros([size_h,size_h])
        order_h = np.arange(4)
        shuffle(order_h)
        n_h1 = np.random.randint(1,n+1)
        order_h = order_h[0:n_h1]
        if 0 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])
            dum_h[0:size_h/2,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 1 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                         
            dum_h[0:size_h/2,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 2 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                          
            dum_h[size_h/2:,0:size_h/2] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        if 3 in order_h:
            ind_h = np.random.randint(np.shape(imgd1)[0])                          
            dum_h[size_h/2:,size_h/2:] = zoom(np.reshape(imgd1[ind_h,:],[28,28]),size_h*1./(2*1.*28))
            labd2[i,:] = labd2[i,:] + labd1[ind_h,:]
            labd2[labd2 > 0] = 1.
        imgd2[i,:] = np.reshape(dum_h,size_h*size_h)
    for i in range(np.shape(imgd1)[0]):
        labdn = np.sum(labd2[i,:]>0)
        if labdn == 0:
          print(labdn)
        labd2[i,:] = labd2[i,:]/labdn
    return imgd2,labd2

def y_ext_gen_cong_mix(labd): # labd[0,:] is 11d; 1 parts cong, 2 part unknown, y_z is 10d
  y_c_h = 0.*labd.copy()
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]): # matching congruent or unknown cue to congruent question (1/2 unknown)
    if np.random.random(1)[0] > 1./3.:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) == 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      if np.random.random(1)[0] > 1./2.:
        y_c_h[i,:10] = dum1_h
        y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      else:
        y_c_h[i,-1] = 1.
        y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,0] = 1.
    else: # matching unknown cue to incongruent question
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) > 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      y_c_h[i,-1] = 1.
      y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,1] = 1.
  return y_c_h, y_z_h, yo_h

def y_ext_gen_tog(labd): # cong 0.5 -> right, incong 0.5 -> wrong
  y_c_h = 0.*labd.copy()
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]): 
    if np.random.random(1)[0] > 0.5:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) == 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      y_c_h[i,:10] = dum1_h
      y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,0] = 1.
    else:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) > 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      y_c_h[i,:10] = dum1_h
      y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,1] = 1.
  return y_c_h, y_z_h, yo_h

def y_ext_gen_cong(labd):
  y_c_h = 0.*labd.copy()
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]): 
    dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
    dum1_h[0,np.random.randint(10)] = 1.
    while np.sum(np.multiply(dum1_h,labd[i,:10])) == 0:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
    y_z_h[i,:] = dum1_h
    y_c_h[i,:10] = dum1_h
    y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
    yo_h[i,0] = 1.
  return y_c_h, y_z_h, yo_h

def y_ext_gen_incong(labd): # labd[0,:] is 11d; 
  y_c_h = 0.*labd.copy()
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]): # matching incongruent cue to incongruent question
    dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
    dum1_h[0,np.random.randint(10)] = 1.
    while np.sum(np.multiply(dum1_h,labd[i,:10])) > 0:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
    y_z_h[i,:] = dum1_h
    y_c_h[i,:10] = dum1_h
    y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
    yo_h[i,0] = 1.
  return y_c_h, y_z_h, yo_h

def y_ext_gen_unk(labd):
  y_c_h = 0.*labd.copy()
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]): 
    if np.random.random(1)[0] > 1./2.:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) == 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      y_c_h[i,-1] = 1.
      y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,0] = 1.
    else:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) > 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      y_c_h[i,-1] = 1.
      y_c_h[i,:] = y_c_h[i,:]*0.99+0.01/11.
      yo_h[i,1] = 1.
  return y_c_h, y_z_h, yo_h

def y_ext_gen_neut(labd): # labd[0,:] is 11d;
  y_z_h = np.zeros([np.shape(labd)[0],10])
  yo_h = np.zeros([np.shape(labd)[0],2])
  for i in range(np.shape(labd)[0]):
    if np.random.random(1)[0] > 1./2.:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) == 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      yo_h[i,0] = 1.
    else:
      dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
      dum1_h[0,np.random.randint(10)] = 1.
      while np.sum(np.multiply(dum1_h,labd[i,:10])) > 0:
        dum1_h = np.zeros([1,np.shape(y_z_h)[1]])
        dum1_h[0,np.random.randint(10)] = 1.
      y_z_h[i,:] = dum1_h
      yo_h[i,1] = 1.
  return y_z_h, yo_h

def translate_images(x,zoomer): # zoomer = 22./28. 
  x1 = x.copy()
  x1 = 0.*x1
  for i in range(np.shape(x)[0]):
    dum1 = zoom(np.reshape(x[i,:],[28,28]),zoomer)
    dum2 = 0.*np.reshape(x[i,:],[28,28])
    dum_x = np.random.randint(np.int(np.floor(28.-zoomer*28.)))
    dum_y = np.random.randint(np.int(np.floor(28.-zoomer*28.)))
    dum2[dum_x:dum_x+np.int(np.floor(zoomer*28.)),dum_y:dum_y+np.int(np.floor(zoomer*28.))] = dum1
    x1[i,:] = np.reshape(dum2,[1,784])
  return x1

def inst_mixer(x11,y11,blur_h = 4.):
  dum1_h = np.arange(np.shape(x11)[0])
  shuffle(dum1_h)
  x11_h = np.zeros([np.shape(x11)[0],40*40])
  y11_h = y11.copy()
  x11_h[dum1_h[0:np.shape(x11)[0]/2],:], y11_h[dum1_h[0:np.shape(x11)[0]/2],:] = gridify_exact(x11[dum1_h[0:np.shape(x11)[0]/2],:],
    y11[dum1_h[0:np.shape(x11)[0]/2],:],1)
  x11_h[dum1_h[0:np.shape(x11)[0]/2],:] = gaussian_blur(x11_h[dum1_h[0:np.shape(x11)[0]/2],:],blur_h)
  dum_x, dum_y = gridify_exact(x11,y11,4)
  x11_h[dum1_h[np.shape(x11)[0]/2:],:], y11_h[dum1_h[np.shape(x11)[0]/2:],:] = dum_x[0:np.shape(x11)[0]/2,:], dum_y[0:np.shape(x11)[0]/2,:]
  return x11_h, y11_h

def weight_variable_conv(shape,name):
    initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
    return initial

def weight_variable_fc(shape,name):
    initial = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()) 
    return initial

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

cond_h = 1 # 1 for ffonly, 7 for ff+fb
max_blur = 4.

n_hl = 8 # neural capacity - 8,32,3072
n_m_hl = 200

x  = tf.placeholder(tf.float32, [None, 40*40], name='x')
y_ = tf.placeholder(tf.float32, [None, 2],  name='y_')
y_c = tf.placeholder(tf.float32, [None, 11],  name='y_c')
y_z = tf.placeholder(tf.float32, [None, 10],  name='y_z')
keep_prob  = tf.placeholder(tf.float32)

W_fc1 = weight_variable_fc([40*40, n_hl],'W_fc1')
b_fc1 = bias_variable([n_hl])
W_fc2 = weight_variable_fc([n_hl, 11],'W_fc2')
b_fc2 = bias_variable([11])

if cond_h == 1:
  saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2})
  temp = set(tf.all_variables())

attn_b = tf.placeholder(tf.float32)
attn_g = tf.placeholder(tf.float32)
keep_prob_f  = tf.placeholder(tf.float32)
fb_keep_prob_f  = tf.placeholder(tf.float32)

W_ff_h_f = weight_variable_fc([11, n_m_hl],'W_ff_h_f')
W_ff_h_z = weight_variable_fc([10, n_m_hl],'W_ff_h_z')
b_ff_h = bias_variable([n_m_hl])
W_ff = weight_variable_fc([n_m_hl, 2],'W_ff')
b_ff = bias_variable([2])

fb_W_ff_h_f = weight_variable_fc([11, n_m_hl],'fb_W_ff_h_f')
fb_W_ff_h_z = weight_variable_fc([10, n_m_hl],'fb_W_ff_h_z')
fb_b_ff_h = bias_variable([n_m_hl])
fb_W_ff = weight_variable_fc([n_m_hl, 2],'fb_W_ff')
fb_b_ff = bias_variable([2])

if cond_h == 7:
  saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2,
    "W_ff_h_f": W_ff_h_f , "W_ff_h_z": W_ff_h_z, "b_ff_h": b_ff_h, "W_ff": W_ff, "b_ff": b_ff,
    "fb_W_ff_h_f": fb_W_ff_h_f , "fb_W_ff_h_z": fb_W_ff_h_z, "fb_b_ff_h": fb_b_ff_h, "fb_W_ff": fb_W_ff, "fb_b_ff": fb_b_ff})
  temp = set(tf.all_variables())

W_bias_fc1 = weight_variable_fc([11, n_hl],'W_bias_fc1')
W_gain_fc1 = weight_variable_fc([11, n_hl],'W_gain_fc1')

h_gain_fc1 = attn_g*(tf.matmul(y_c, W_gain_fc1))
h_bias_fc1 = attn_b*(tf.matmul(y_c, W_bias_fc1))

h_fc1 = tf.multiply(tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1 + h_bias_fc1),1.+h_gain_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

h_ff_h = tf.nn.relu(tf.matmul(y, W_ff_h_f) + tf.matmul(y_z, W_ff_h_z) + b_ff_h)
y_ff = tf.nn.softmax(tf.matmul(tf.nn.dropout(h_ff_h,keep_prob_f), W_ff) + b_ff)

fb_h_ff_h = tf.nn.relu(tf.matmul(y, fb_W_ff_h_f) + tf.matmul(y_z, fb_W_ff_h_z) + fb_b_ff_h)
fb_y_ff = tf.nn.softmax(tf.matmul(tf.nn.dropout(fb_h_ff_h,fb_keep_prob_f), fb_W_ff) + fb_b_ff)

cross_entropy_ff = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(tf.clip_by_value(y_ff,1e-10,1.0)), reduction_indices=[1]))
correct_prediction_ff = tf.equal(tf.argmax(y_ff, 1), tf.argmax(y_, 1))
accuracy_ff = tf.reduce_mean(tf.cast(correct_prediction_ff, tf.float32), name='accuracy_ff')

cross_entropy_ff_fb = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(tf.clip_by_value(fb_y_ff,1e-10,1.0)), reduction_indices=[1]))
correct_prediction_ff_fb = tf.equal(tf.argmax(fb_y_ff, 1), tf.argmax(y_, 1))
accuracy_ff_fb = tf.reduce_mean(tf.cast(correct_prediction_ff_fb, tf.float32), name='accuracy_ff_fb')

h_fc1_1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
h_fc1_drop_1 = tf.nn.dropout(h_fc1_1, keep_prob)

y_1 = tf.nn.softmax(tf.matmul(h_fc1_drop_1, W_fc2) + b_fc2, name='y_1')

h_ff_h_1 = tf.nn.relu(tf.matmul(y_1, W_ff_h_f) + tf.matmul(y_z, W_ff_h_z) + b_ff_h)
y_ff_1 = tf.nn.softmax(tf.matmul(tf.nn.dropout(h_ff_h_1,keep_prob_f), W_ff) + b_ff)

cross_entropy_ff_1 = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(tf.clip_by_value(y_ff_1,1e-10,1.0)), reduction_indices=[1]))
correct_prediction_ff_1 = tf.equal(tf.argmax(y_ff_1, 1), tf.argmax(y_, 1))
accuracy_ff_1 = tf.reduce_mean(tf.cast(correct_prediction_ff_1, tf.float32), name='accuracy_ff_1')

if cond_h == 1: # 1 for rest, 0.35 for 8
  train_step_ff = tf.train.GradientDescentOptimizer(0.35).minimize(cross_entropy_ff,var_list=[W_ff_h_f,W_ff_h_z,b_ff_h,W_ff,b_ff])
  train_step_ff_fb = tf.train.GradientDescentOptimizer(0.35).minimize(cross_entropy_ff_fb,var_list=[fb_W_ff_h_f,fb_W_ff_h_z,fb_b_ff_h,fb_W_ff,fb_b_ff])
  tf.summary.scalar('cross_entropy_none_nobjblur_fb_ff_fb_nobjblur_ffonly_8', cross_entropy_ff_fb)
  tf.summary.scalar('accuracy_none_nobjblur_fb_ff_fb_nobjblur_ffonly_8', accuracy_ff_fb)
  merged = tf.summary.merge_all()
if cond_h == 7: # 1 for rest, 0.25 for 8
  train_step_ff_fb = tf.train.GradientDescentOptimizer(0.25).minimize(cross_entropy_ff_fb-cross_entropy_ff_1,var_list=[W_gain_fc1,W_bias_fc1])
  train_step_ff_fb1 = tf.train.GradientDescentOptimizer(0.25).minimize(cross_entropy_ff_fb,var_list=[W_gain_fc1,W_bias_fc1])
  tf.summary.scalar('cross_entropy_none_nobjblur_fb_ff_fb_nobjblur_fbonff_fp_tog_opt_8', cross_entropy_ff_fb-cross_entropy_ff_1)
  tf.summary.scalar('accuracy_none_nobjblur_fb_ff_fb_nobjblur_fbonff_fp_tog_opt_8', accuracy_ff_fb-accuracy_ff_1)
  merged = tf.summary.merge_all()

with tf.Session() as sess:

  train_writer = tf.summary.FileWriter('train1_2',sess.graph)
  val_writer = tf.summary.FileWriter('val1_2')

  #sess.run(tf.global_variables_initializer())
  if cond_h == 1: # ffonly
    saver.restore(sess,"./models_new/P3_fmnist_1hl_none_gridnobjblur_8.ckpt")
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    max_steps = 300001
  if cond_h == 7: #fbonff
    saver.restore(sess,"./models_new/P3_fmnist_1hl_none_gridnobjblur_fbff_nofp_ff_nofp_bgjoint_nobjblur_ffonly_8.ckpt")
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    max_steps = 350001

  for step in range(max_steps):

    batch_xs, batch_ys = mnist.train.next_batch(40)
    batch_xs = translate_images(batch_xs,24./28.)
    batch_ys = batch_mod_test(batch_ys)
    batch_xs, batch_ys = gridify_stoc(batch_xs,batch_ys,4)
    batch_xs = gaussian_blur(batch_xs,max_blur)

    if cond_h == 1:
      batch_yz_neut, batch_yo_neut = y_ext_gen_neut(batch_ys)
      batch_xs_r, batch_ys_r, batch_yc_r, batch_yz_r, batch_yo_r = random_pattern_gen_cue(40)
    if cond_h == 7:
      batch_yc_cong, batch_yz_cong, batch_yo_cong = y_ext_gen_cong(batch_ys)
      batch_yc_incong, batch_yz_incong, batch_yo_incong = y_ext_gen_incong(batch_ys)
      batch_yc_tog, batch_yz_tog, batch_yo_tog = y_ext_gen_tog(batch_ys)
      batch_yc_unk, batch_yz_unk, batch_yo_unk = y_ext_gen_unk(batch_ys)
      batch_yz_neut, batch_yo_neut = y_ext_gen_neut(batch_ys)
      batch_xs_r, batch_ys_r, batch_yc_r, batch_yz_r, batch_yo_r = random_pattern_gen_cue(60)

    if (step % 200) == 0:
      if cond_h == 1:
        print('Train_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
          y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Train_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
          y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
          y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
        train_writer.add_summary(summary, step)
        print('Random_t_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r,
          y_z: batch_yz_r, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Random_t_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r,
          y_z: batch_yz_r, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
      if cond_h == 7:
        print('Train_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_cong, y_c: batch_yc_cong,
          y_z: batch_yz_cong, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Train_ff_fb_FP: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_incong, y_c: batch_yc_incong,
          y_z: batch_yz_incong, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Train_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
          y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Train TP-FP_opt: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_cong, y_c: batch_yc_cong,
          y_z: batch_yz_cong, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}) - sess.run(accuracy_ff_fb
          , feed_dict={x: batch_xs, y_: batch_yo_incong, y_c: batch_yc_incong, y_z: batch_yz_incong
          , attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}) - sess.run(accuracy_ff_1, 
          feed_dict={x: batch_xs, y_: batch_yo_cong, y_c: batch_yc_cong, y_z: batch_yz_cong, attn_b: 1., attn_g: 1., keep_prob: 1., 
          keep_prob_f: 1., fb_keep_prob_f: 1.}) + sess.run(accuracy_ff_1, feed_dict={x: batch_xs, y_: batch_yo_incong, 
          y_c: batch_yc_incong, y_z: batch_yz_incong, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_yo_tog, y_c: batch_yc_tog,
          y_z: batch_yz_tog, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
        train_writer.add_summary(summary, step)
        print('Random_t_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r,
          y_z: batch_yz_r, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Random_t_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r,
          y_z: batch_yz_r, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))

    if (step % 1000) == 0:
      batch_xs_v, batch_ys_v, batch_yc_v, batch_yz_v, batch_yo_v = random_pattern_gen_cue(2000)
      batch_xsv, batch_ysv = mnist.validation.images, mnist.validation.labels
      batch_xsv = translate_images(batch_xsv,24./28.)
      batch_ysv = batch_mod_test(batch_ysv)
      batch_xsv, batch_ysv = gridify_stoc(batch_xsv,batch_ysv,4)
      batch_xsv = gaussian_blur(batch_xsv,max_blur)

      if cond_h == 1:
        batch_yz_neutv, batch_yo_neutv = y_ext_gen_neut(batch_ysv)
      if cond_h == 7:
        batch_yc_congv, batch_yz_congv, batch_yo_congv = y_ext_gen_cong(batch_ysv)
        batch_yc_togv, batch_yz_togv, batch_yo_togv = y_ext_gen_tog(batch_ysv)
        batch_yc_incongv, batch_yz_incongv, batch_yo_incongv = y_ext_gen_incong(batch_ysv)
        batch_yz_neutv, batch_yo_neutv = y_ext_gen_neut(batch_ysv)

      if cond_h == 1:
        print('Val_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xsv, y_: batch_yo_neutv, y_c: batch_mod_test(batch_yz_neutv), 
          y_z: batch_yz_neutv, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Val_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xsv, y_: batch_yo_neutv, y_c: batch_mod_test(batch_yz_neutv), 
          y_z: batch_yz_neutv, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        summary = sess.run(merged, feed_dict={x: batch_xsv, y_: batch_yo_neutv, y_c: batch_mod_test(batch_yz_neutv), 
          y_z: batch_yz_neutv, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
        val_writer.add_summary(summary, step)
        print('Random_v_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs_v, y_: batch_yo_v, y_c: batch_yc_v,
          y_z: batch_yz_v, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Random_v_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs_v, y_: batch_yo_v, y_c: batch_yc_v,
          y_z: batch_yz_v, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
      if cond_h == 7:
        print('Val_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xsv, y_: batch_yo_congv, y_c: batch_yc_congv,
          y_z: batch_yz_congv, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Val_ff_fb_FP: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xsv, y_: batch_yo_incongv, y_c: batch_yc_incongv,
          y_z: batch_yz_incongv, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Val_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xsv, y_: batch_yo_neutv, y_c: batch_mod_test(batch_yz_neutv), 
          y_z: batch_yz_neutv, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        summary = sess.run(merged, feed_dict={x: batch_xsv, y_: batch_yo_togv, y_c: batch_yc_togv,
          y_z: batch_yz_togv, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
        val_writer.add_summary(summary, step)
        print('Random_v_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs_v, y_: batch_yo_v, y_c: batch_yc_v,
          y_z: batch_yz_v, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
        print('Random_v_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs_v, y_: batch_yo_v, y_c: batch_yc_v,
          y_z: batch_yz_v, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))

    if cond_h == 1:
      sess.run(train_step_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), y_z: batch_yz_neut
        , attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 0.5, fb_keep_prob_f: 0.5})
      sess.run(train_step_ff_fb, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r, y_z: batch_yz_r
        , attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 0.5, fb_keep_prob_f: 0.5})
      sess.run(train_step_ff, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), y_z: batch_yz_neut
        , attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 0.5, fb_keep_prob_f: 0.5})
      sess.run(train_step_ff, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r, y_z: batch_yz_r
        , attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 0.5, fb_keep_prob_f: 0.5})
    if cond_h == 7:
      sess.run(train_step_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_tog, y_c: batch_yc_tog, y_z: batch_yz_tog
        , attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
      sess.run(train_step_ff_fb1, feed_dict={x: batch_xs, y_: batch_yo_unk, y_c: batch_yc_unk, y_z: batch_yz_unk
        , attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})
      sess.run(train_step_ff_fb1, feed_dict={x: batch_xs_r, y_: batch_yo_r, y_c: batch_yc_r, y_z: batch_yz_r
        , attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.})

  batch_xs, batch_ys = mnist.test.images, mnist.test.labels
  batch_xs = translate_images(batch_xs,24./28.)
  batch_ys = batch_mod_test(batch_ys)
  batch_xs, batch_ys = gridify_stoc(batch_xs,batch_ys,4)
  batch_xs = gaussian_blur(batch_xs,max_blur)

  if cond_h == 1:
    batch_yz_neut, batch_yo_neut = y_ext_gen_neut(batch_ys)
    print('Test_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
      y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
    print('Test_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
      y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
  if cond_h == 7:
    batch_yc_cong, batch_yz_cong, batch_yo_cong = y_ext_gen_cong(batch_ys)
    batch_yz_neut, batch_yo_neut = y_ext_gen_neut(batch_ys)
    print('Test_ff_fb: ',step, sess.run(accuracy_ff_fb, feed_dict={x: batch_xs, y_: batch_yo_cong, y_c: batch_yc_cong,
      y_z: batch_yz_cong, attn_b: 1., attn_g: 1., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))
    print('Test_ff: ',step, sess.run(accuracy_ff, feed_dict={x: batch_xs, y_: batch_yo_neut, y_c: batch_mod_test(batch_yz_neut), 
      y_z: batch_yz_neut, attn_b: 0., attn_g: 0., keep_prob: 1., keep_prob_f: 1., fb_keep_prob_f: 1.}))

  saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2,
    "W_bias_fc1": W_bias_fc1, "W_gain_fc1": W_gain_fc1,
    "W_ff_h_f": W_ff_h_f , "W_ff_h_z": W_ff_h_z, "b_ff_h": b_ff_h, "W_ff": W_ff, "b_ff": b_ff,
    "fb_W_ff_h_f": fb_W_ff_h_f , "fb_W_ff_h_z": fb_W_ff_h_z, "fb_b_ff_h": fb_b_ff_h, "fb_W_ff": fb_W_ff, 
    "fb_b_ff": fb_b_ff},write_version=tf.train.SaverDef.V1)

  if cond_h == 1:
    save_path = saver.save(sess, "./models_new/P3_fmnist_1hl_none_gridnobjblur_fbff_nofp_ff_nofp_bgjoint_nobjblur_ffonly_8.ckpt")
    print("Model saved in file: %s" % save_path)
  if cond_h == 7:
    save_path = saver.save(sess, "./models_new/P3_fmnist_1hl_none_gridnobjblur_fbff_nofp_ff_nofp_bgjoint_nobjblur_fbonff_tog_opt_8.ckpt")
    print("Model saved in file: %s" % save_path)
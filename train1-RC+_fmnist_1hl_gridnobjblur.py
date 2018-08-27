# source ~/tensorflow/bin/activate

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

max_blur = 4.
n_hl = 32 # neural capacity - 8,32,3072

x  = tf.placeholder(tf.float32, [None, 40*40], name='x')
y_ = tf.placeholder(tf.float32, [None, 11],  name='y_')

W_fc1 = weight_variable_fc([40*40, n_hl],'W_fc1')
b_fc1 = bias_variable([n_hl])
W_fc2 = weight_variable_fc([n_hl, 11],'W_fc2')
b_fc2 = bias_variable([11])

saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2})
temp = set(tf.all_variables())

h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

train_step = tf.train.GradientDescentOptimizer(5e-2).minimize(cross_entropy) 
#fMNIST - 1e-1 for 3072, .7e-1 for 32, .5e-1 for 16, 5e-2 for 8

tf.summary.scalar('cross_entropy_none_nobjblur_8', cross_entropy)
#tf.summary.scalar('accuracy_none_1obj', accuracy)
merged = tf.summary.merge_all()

with tf.Session() as sess:

  train_writer = tf.summary.FileWriter('train1',sess.graph)
  val_writer = tf.summary.FileWriter('val1')

  sess.run(tf.global_variables_initializer())

  max_steps = 200001 # 150001 for 3072, 200000 for 32, 200000 for 8
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(30)
    batch_xs = translate_images(batch_xs,24./28.)
    batch_ys = batch_mod_test(batch_ys)
    batch_xs, batch_ys = gridify_stoc(batch_xs,batch_ys,4)
    batch_xs = gaussian_blur(batch_xs,max_blur)
    batch_xs_r, batch_ys_r = random_pattern_gen(30)
    if (step % 100) == 0:
      y_h = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      print('Train: ',step, accuracy_n(y_h,batch_ys))
      summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      train_writer.add_summary(summary, step)
      y_h = sess.run(y, feed_dict={x: batch_xs_r, y_: batch_ys_r, keep_prob: 1.0})
      print('Random_t: ',step, accuracy_n(y_h,batch_ys_r))
    if (step % 500) == 0:
      batch_xs_v, batch_ys_v = random_pattern_gen(2000)
      batch_xsv, batch_ysv = mnist.validation.images, mnist.validation.labels
      batch_xsv = translate_images(batch_xsv,24./28.)
      batch_ysv = batch_mod_test(batch_ysv)
      batch_xsv, batch_ysv = gridify_stoc(batch_xsv,batch_ysv,4)
      batch_xsv = gaussian_blur(batch_xsv,max_blur)
      y_h = sess.run(y, feed_dict={x: batch_xsv, y_: batch_ysv, keep_prob: 1.0})
      print('Val: ',step, accuracy_n(y_h,batch_ysv))
      summary = sess.run(merged, feed_dict={x: batch_xsv, y_: batch_ysv, keep_prob: 1.0})
      val_writer.add_summary(summary, step)
      y_h = sess.run(y, feed_dict={x: batch_xs_v, y_: batch_ys_v, keep_prob: 1.0})
      print('Random_v: ',step, accuracy_n(y_h,batch_ys_v))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    sess.run(train_step, feed_dict={x: batch_xs_r, y_: batch_ys_r, keep_prob: 0.5})
  batch_xs, batch_ys = mnist.test.images, mnist.test.labels
  batch_xs = translate_images(batch_xs,24./28.)
  batch_ys = batch_mod_test(batch_ys)
  batch_xs, batch_ys = gridify_stoc(batch_xs,batch_ys,4)
  batch_xs = gaussian_blur(batch_xs,max_blur)
  y_h = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
  print(max_steps, accuracy_n(y_h,batch_ys))

  saver = tf.train.Saver({"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2},write_version=tf.train.SaverDef.V1)
  save_path = saver.save(sess, "./models_new/P3_fmnist_1hl_none_gridnobjblur_8.ckpt")
  print("Model saved in file: %s" % save_path)

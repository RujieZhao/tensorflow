



from skimage import io
#,transform
import matplotlib.pyplot as plt
import glob
import os
import tensorflow as tf
import numpy as np
import time
np.set_printoptions(threshold=np.inf)

path='/home/chaolu/SIDS/data'
path_1 = '/home/chaolu/SIDS/data/good' #997
path_2 = '/home/chaolu/SIDS/data/bad' #3525
#print('filenum:',len([lists for lists in os.listdir(path1) if os.path.isfile(os.path.join(path, lists))]))
#print('filenum:',len([lists for lists in os.listdir(path_2) if os.path.isfile(os.path.join(path_2, lists))]))
#Resize all pictures (w for width, h for height, c for color)
w=256
h=256
c=3


#Read the pictures
def read_img(path):
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    print('cate:',cate)
    imgs=[]
    labels=[]
    index = []
    for idx,folder in enumerate(cate):
        
        for im in glob.glob(folder+'/*.jpeg'):
            #print('reading the images:%s label: %s'%(im,idx))
            img=io.imread(im)
            #print('imgsize:', img.shape)
            #print('img:',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.show()
            #img_new=np.reshape(img,(256,256))
            index.append(im)
            #print(index)
            imgs.append(img)
            labels.append(idx)
        #print('imgs_shape:',np.array(imgs).shape)
        #print('imgs:',imgs)    
    return np.array(imgs,np.float32),np.array(labels,np.int32),np.array(index)

#print(len(read_img(path)))
data,label,index = read_img(path)
print('index_shape:',index.shape)
num_example=len(label)
print('num_example:',num_example)
arr=np.arange(num_example)
print('arr:',arr)
np.random.shuffle(arr)
print('arr_shuffle:',arr)
data=data[arr]
label=label[arr]
index=index[arr]
#plt.suptitle(index[0])
#plt.imshow(np.int32(data[0]), cmap=plt.get_cmap('seismic'))#gray_r
#plt.axis('off')
#plt.show()
#print('label:',label)
print(label.shape)

#Divide the whole data set into training set and testing set
ratio=0.785
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
print('x_val_type:',type(x_val))
print('x_val_shape:',x_val[0].shape)
y_val=label[s:]
print('y_val:',y_val[0])
print('y_val_shape:',y_val.shape)
z_val = index[s:]
print('z_val:',z_val[0])
print('z_val_shape:',z_val.shape)

#-----------------Create network----------------------
#placeholde r
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
#X=tf.placeholder(tf.float32,shape=[None,w,h,c])
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#1st conv layer（256>128)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


#2nd conv layer(128->64)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#3rd conv layer(64->32)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#4th conv layer(32->16)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4,[-1,16*16*128])

#Full connected layer
dense1 = tf.layers.dense(inputs=re1, 
                      units=1024,  
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

dense2= tf.layers.dense(inputs=dense1, 
                      units=512,   
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2, 
                        units=2, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------Network end---------------------------
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Define a function that select a batch for every epoch
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#Training and testing data
n_epoch=10
batch_size=64
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())

XX = range(0,n_epoch+1,1)
ACC = np.array([0.]*(n_epoch+1),'float32')
for epoch in range(1,n_epoch+1):
    start_time = time.time()
       
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    
    #validation
    #err, ac = sess.run([loss,acc], feed_dict={x: x_val, y_: y_val})
    #print('Accuracy:',ac)
      
    ACC[epoch] = ac*100  
    if epoch%100 == 0:
      plt.plot(XX, ACC,color='r',marker='o',markerfacecolor='blue',markersize=5)
      plt.yticks(np.linspace(0,100,21))
      plt.show()

for i in range(len(y_val)):
    result = sess.run(correct_prediction,feed_dict={x:np.reshape(x_val[i],[1,256,256,3]),y_:np.reshape(y_val[i],[1])})     
    if not result:
        #one_pic = np.reshape()
        #plt.suptitle(x)
        print('x_val[i]_shape:',x_val[i].shape)
        print('prediction:',sess.run(logits,feed_dict={x:np.reshape(x_val[i],[1,256,256,3]),y_:np.reshape(y_val[i],[1])}))
        print('truth value:',y_val[i])
        pic = np.int32(x_val[i])
        #pixel = np.matrix(pic,dtype= "float")
        plt.suptitle(z_val[i])
        plt.imshow(pic)
        plt.axis('off')
        plt.show()
    #print("   validation loss: %f" % (val_loss/ n_batch))
    #print("   validation acc: %f" % (val_acc/ n_batch))
sess.close()










import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.get_variable("w", shape=[3, 1])
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)
y_GD = tf.gradients(yhat,[w])
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)
opt = tf.train.AdamOptimizer(0.1)

train_op = opt.minimize(loss)
grad = opt.compute_gradients(loss,[w])
def generate_data():
  x_val = np.random.uniform(-10.0, 10.0, size=1)
  y_val = 5 * np.square(x_val) + 3
  return x_val, y_val
with tf.Session(config = config) as sess:
  sess.run(tf.global_variables_initializer())
  
  for _ in range(20):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    #print(loss_val)
    print("x_val",x_val)
    print('gd_loss:',sess.run(grad,feed_dict={x:x_val,y: y_val})[0])
  print(sess.run(y_GD,feed_dict={x:x_val}))
  
  print(sess.run([w]))






















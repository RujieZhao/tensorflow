
import tensorflow as tf 

a = tf.constant(0.)
b = 2*a
c = tf.stop_gradient(tf.constant(0.))
d = tf.stop_gradient(2 * a)
g = tf.gradients(a + b, [a, b], stop_gradients=[a,b])
g_1 = tf.gradients(a + b, [a, b])
g_2 = tf.gradients(c + d, [c, d])
with tf.Session() as sess:
	print(sess.run(g),sess.run(g_1),sess.run(g_2))






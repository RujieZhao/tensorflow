



import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Def custom square function using np.square instead of tf.square:

def square(x,y):
    square  = np.square(x+y)
    return square

def mysquare(x,y, name=None):
    
    with ops.op_scope([x,y], name, "Mysquare") as name:
        sqr_x = py_func(square,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=_MySquareGrad)  # <-- here's the call to the gradient
        return sqr_x[0]

# Actual gradient:
def _MySquareGrad(op, grad):
    x = op.inputs[0]
    y = op.inputs[1]
    return grad*x,grad*x1  # add a "small" error just to see the difference:

with tf.Session() as sess:
    x = tf.constant([1.])
    x1 = tf.constant([1.,2.,3.])
    y = mysquare(x,x1)
    sess.run(tf.global_variables_initializer())
    print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())





'''Code is from Chapter 9 of the book -Hands On Machine Learning-
'''

# Import Libraries
import tensorflow as tf


# Create your first graph
'''Code does not perform any computations.  It just creates a computation graph
 *To view this graph you need to create a tensorflow session.
'''
x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')
f = x*x*y + y + 2

# Initializing Tensorflow Sessions
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
print('Client session =>', sess)
result = sess.run(f)
print('Result => {}'.format(result))
sess.close()

# Alternative Approach 1 - Create A Session using 'With'
'Session is automatically closed without explicitly calling it'
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print('Result => {}'.format(result))
    

# Alternative Approach 2 - Create a Global Initializer
'Initialized all variables in the script'
init = tf.global_variables_initializer() # prepare unit node

with tf.Session() as sess:
    init.run() # initializes all variables
    result = f.eval()


# Alternative Approach 3 - Create an Interactive Session
'''Inter Session: automatically sets itself as the default session
   so there is no need to use the with block'''
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)










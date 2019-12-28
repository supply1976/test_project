import time
import tensorflow as tf

x_inputs_data = tf.random_normal([128, 1024], mean=0, stddev=1)
#y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis=1, keep_dims=True)>0, tf.int32)

# replace the placeholder by TF queue
with tf.variable_scope("queue"):
    q = tf.FIFOQueue(capacity=32, dtypes=tf.float32)
    enqueue_op = q.enqueue(x_inputs_data)
    numberOfThreads = 1
    qr = tf.train.QueueRunner(q, [enqueue_op]*numberOfThreads)
    tf.train.add_queue_runner(qr)
    X_inputs = q.dequeue()
    Y_true = tf.cast(tf.reduce_sum(X_inputs, axis=1, keep_dims=True)>0, tf.int32)

#with tf.variable_scope("placeholder"):
#    X_inputs = tf.placeholder(tf.float32, shape=[None, 1024])
#    Y_true = tf.placeholder(tf.int32, shape=[None, 1])

with tf.variable_scope("model"):
    w1 = tf.get_variable(name="w1", shape=[1024,1024], initializer=tf.random_normal_initializer(stddev=0.1))
    b1 = tf.get_variable(name="b1", shape=[1024], initializer=tf.constant_initializer(0.1))
    z1 = tf.matmul(X_inputs, w1) + b1
    y1 = tf.nn.relu(z1)
    
    w2 = tf.get_variable(name="w2", shape=[1024,1], initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable(name="b2", shape=[1], initializer=tf.constant_initializer(0.1))
    z2 = tf.matmul(y1, w2) + b2
    
with tf.variable_scope('Loss'):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(Y_true, tf.float32), z2)
    loss_op = tf.reduce_mean(losses)

with tf.variable_scope('Accuracy'):
    Y_pred = tf.cast(z2 > 0, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_true, Y_pred), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
    
train_op = tf.train.AdamOptimizer(0.01).minimize(loss_op)

startTime = time.time()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # add coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # check the accuracy before training...
    sess.run(accuracy)
    #x_inputs, y_inputs = sess.run([x_inputs_data, y_inputs_data])
    #sess.run(accuracy, feed_dict={X_inputs: x_inputs, Y_true: y_inputs})
    
    # training loop
    for i in range(5000):
        _, loss = sess.run([train_op, loss_op])
        #x_inputs, y_inputs = sess.run([x_inputs_data, y_inputs_data])
        #_, loss = sess.run([train_op, loss_op], feed_dict={X_inputs:x_inputs, Y_true:y_inputs})
        if i % 500 ==0:
            print "iter {}, loss={}".format(i, loss)
            
    # check the accuracy after training...
    sess.run(accuracy)
    #x_inputs, y_inputs = sess.run([x_inputs_data, y_inputs_data])
    #sess.run(accuracy, feed_dict={X_inputs: x_inputs, Y_true: y_inputs})
    coord.request_stop()
    coord.join(threads)
    
print "elapse time = {}".format(time.time()-startTime)
    
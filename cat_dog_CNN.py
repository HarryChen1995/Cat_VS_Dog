
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import process_image


train_image, train_label, test_image, test_label=process_image.read_data()


max_epochs=12000
LR=1e-5
batch_size=125
display_step=100

image_size = 64 
n_classes = 2
dropout = 0.75

x = tf.placeholder(tf.float32, [None, image_size,image_size,1])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)

def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w,strides=[1, 1, 1, 1],padding='SAME'),b))

def max_pool(img, k):
	return tf.nn.max_pool(img, ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')

wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

wc3 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
bc3 = tf.Variable(tf.random_normal([128]))

wc4 = tf.Variable(tf.random_normal([5, 5, 128, 256]))
bc4 = tf.Variable(tf.random_normal([256]))

wc5 = tf.Variable(tf.random_normal([5, 5, 256, 512]))
bc5 = tf.Variable(tf.random_normal([512]))



wd1 = tf.Variable(tf.random_normal([(image_size//32)*(image_size//32)*512, 1024]))

wout = tf.Variable(tf.random_normal([1024, n_classes]))
bd1 = tf.Variable(tf.random_normal([1024]))

tf.summary.histogram('wc1',wc1)
tf.summary.histogram('bc1',bc1)
tf.summary.histogram('wc2',wc2)
tf.summary.histogram('bc2',bc2)
tf.summary.histogram('wc3',wc3)
tf.summary.histogram('bc3',bc3)
tf.summary.histogram('wc4',wc4)
tf.summary.histogram('bc4',bc4)
tf.summary.histogram('wc5',wc5)
tf.summary.histogram('bc5',bc5)
tf.summary.histogram('wd1',wd1)
tf.summary.histogram('wout',wout)
tf.summary.histogram('bd1',bd1)



bout = tf.Variable(tf.random_normal([n_classes]))

conv1 = conv2d(x,wc1,bc1)

conv1 = max_pool(conv1, k=2)

conv2 = conv2d(conv1,wc2,bc2)

conv2 = max_pool(conv2, k=2)

conv3 = conv2d(conv2,wc3,bc3)

conv3 = max_pool(conv3, k=2)

conv4 = conv2d(conv3,wc4,bc4)

conv4 = max_pool(conv4, k=2)

conv5 = conv2d(conv4,wc5,bc5)

conv5 = max_pool(conv5, k=2)


dense1 = tf.reshape(conv5, [-1,wd1.get_shape().as_list()[0]])

dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1))

dense1 = tf.nn.dropout(dense1, keep_prob)

pred = tf.add(tf.matmul(dense1, wout), bout)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar("Loss",cost)
optimizer =tf.train.RMSPropOptimizer(learning_rate=LR).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy",accuracy)

init_op= tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    summary=tf.summary.merge_all()
    file_write=tf.summary.FileWriter("output",sess.graph)


    for epochs in range(max_epochs):
        index=np.random.choice(train_image.shape[0],size=batch_size)
        batch_image=train_image[index]
        batch_label=train_label[index]
        sess.run([optimizer],feed_dict={x:batch_image,y:batch_label,keep_prob:dropout})


        if epochs % display_step==0:
            train_loss,summary1=sess.run([cost,summary],feed_dict={x:batch_image,y:batch_label,keep_prob:dropout})
            test_accuarcy,summary2=sess.run([accuracy,summary],feed_dict={x:test_image,y:test_label,keep_prob:1.0})
            file_write.add_summary(summary1,global_step=epochs)
            file_write.add_summary(summary2,global_step=epochs)
            print("Training Loss: {}  Test Accuarcy: {}".format(train_loss,test_accuarcy))
        



    print("Finish Training......")
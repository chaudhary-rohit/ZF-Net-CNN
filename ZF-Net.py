import tensorflow as tf
import cv2

def Conv_Net(x, W, B):
    
    with tf.name_scope('Convolution_1') as scope:
        conv1 = tf.nn.conv2d(x, W['W1'], strides = (1,2,2,1), padding = 'VALID', name = 'conv1')
        conv1_b = tf.nn.bias_add(conv1, B['B1'], name = 'bias1')
        relu1 = tf.nn.relu(conv1_b, name = 'relu1')
        pool1 = tf.nn.max_pool(relu1, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID', name = 'm_pool1')
    
    with tf.name_scope('Convolution_2') as scope:
        conv2 = tf.nn.conv2d(pool1, W['W2'], strides = (1,2,2,1), padding = 'VALID', name = 'conv2')
        conv2_b = tf.nn.bias_add(conv2, B['B2'], name = 'bias2')
        pool2 = tf.nn.max_pool(conv2_b, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID', name = 'm_pool1')
        relu2 = tf.nn.relu(pool2, name = 'relu2')
    
    with tf.name_scope('Convolution_3') as scope:
        conv3 = tf.nn.conv2d(relu2, W['W3'], strides = (1,1,1,1), padding = 'SAME', name = 'conv3')
        conv3_b = tf.nn.bias_add(conv3, B['B3'], name = 'bias3')
        relu3 = tf.nn.relu(conv3_b, name = 'relu3')
      
    with tf.name_scope('Convolution_4') as scope:
        conv4 = tf.nn.conv2d(relu3, W['W4'], strides = (1,1,1,1), padding = 'SAME', name = 'conv4')
        conv4_b = tf.nn.bias_add(conv4, B['B4'], name = 'bias4')
        relu4 = tf.nn.relu(conv4_b, name = 'relu4')
    
    with tf.name_scope('Convolution_5') as scope:
        conv5 = tf.nn.conv2d(relu4, W['W5'], strides = (1,1,1,1), padding = 'SAME', name = 'conv5')
        conv5_b = tf.nn.bias_add(conv5, B['B5'], name = 'bias5')
        relu5 = tf.nn.relu(conv5_b, name = 'relu5') 
        pool5 = tf.nn.max_pool(relu5, ksize = (1,3,3,1), strides = (1,2,2,1), padding = 'VALID', name = 'm_pool1')

    with tf.name_scope('Fully_connected_1') as scope:
        fc_in = tf.reshape(pool5, shape = [-1, 256*5*5])
        fc1 = tf.matmul(fc_in, W['W6'], name = 'fc1')
        fc1_b = tf.add(fc1, B['B6'], name = 'bias6')
        relu6 = tf.nn.relu(fc1_b, name = 'relu6') 
        drop1 = tf.nn.dropout(relu6, rate = 0.5, name = 'drop-out1') 
   
    with tf.name_scope('Fully_connected_2') as scope:
        fc2 = tf.matmul(drop1, W['W7'], name = 'fc2')
        fc2 = tf.add(fc2, B['B7'], name = 'bias7')
        relu7 = tf.nn.relu(fc2, name = 'relu7') 
        drop2 = tf.nn.dropout(relu7, rate = 0.5, name = 'drop-out2') 
    
    with tf.name_scope('Fully_connected_2') as scope:
        fc3 = tf.matmul(drop2, W['W8'], name = 'fc3')
        fc3 = tf.add(fc3, B['B8'], name = 'bias8')
        out = tf.nn.relu(fc3, name = 'relu8') 
    
    return out

def train(b_X, b_Y, W, B): #train images received in batch
    
    with tf.name_scope('cross_entropy'):
        pred = Conv_Net(b_X, W, B)
    
    with tf.name_scope('training'):
        cross_E = tf.nn.softmax_cross_entropy_with_logits_v2(b_Y, pred, name = 'cross_entropy')
        mean_CE = tf.reduce_mean(cross_E, name = 'Loss_operation') 
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001, momentum=0.99)
        grads_and_vars = optimizer.compute_gradients(mean_CE)
        training_op = optimizer.apply_gradients(grads_and_vars, name='training_operation')
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(b_Y, 1), name='correct_prediction')
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            
# Dictionaries for weights, biases and labels
W = {
     'W1': tf.get_variable('w1', shape=(7,7,3,96), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W2': tf.get_variable('w2', shape=(5,5,96,256), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W3': tf.get_variable('w3', shape=(3,3,256,384), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W4': tf.get_variable('w4', shape=(3,3,384,384), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W5': tf.get_variable('w5', shape=(3,3,384,256), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W6': tf.get_variable('w6', shape=(256*5*5,4096), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W7': tf.get_variable('w7', shape=(4096,4096), initializer=tf.random_normal_initializer(stddev=0.01)),
     'W8': tf.get_variable('w8', shape=(4096,15), initializer=tf.random_normal_initializer(stddev=0.01))
     }
B = {
     'B1': tf.get_variable('b1', shape=(96), initializer=tf.constant_initializer(0)),
     'B2': tf.get_variable('b2', shape=(256), initializer=tf.constant_initializer(0)),
     'B3': tf.get_variable('b3', shape=(384), initializer=tf.constant_initializer(0)),
     'B4': tf.get_variable('b4', shape=(384), initializer=tf.constant_initializer(0)),
     'B5': tf.get_variable('b5', shape=(256), initializer=tf.constant_initializer(0)),
     'B6': tf.get_variable('b6', shape=(4096), initializer=tf.constant_initializer(0)),
     'B7': tf.get_variable('b7', shape=(4096), initializer=tf.constant_initializer(0)),
     'B8': tf.get_variable('b8', shape=(15), initializer=tf.constant_initializer(0))
    }

def single_test(image, W, B):
    
    image1 = cv2.resize(image, (227,227))
    image2 = image1.astype('float32')
    image3 = image2.reshape((1,227,227,3))
    pred = Conv_Net(image3, W, B)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(pred)
       
def test(X, Y, W, B):
    
    pred = Conv_Net(X, W, B)
    
    with tf.Session() as sess:
        print('Evaluating dataset...')
        sess.run(tf.global_variables_initializer())
        test_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1), name='correct_prediction')
        test_accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32), name='accuracy_operation')
        print('Test Accuracy = {:.3f}'.format(test_accuracy))
    
def save_model(sess, filename):
    
    saver = tf.train.saver()
    saver.save(sess, filename)
    
def restore_model(sess, checkpoint_dir):
   
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    

    
    
       
    
    

    




























    


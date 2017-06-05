## Importing required libraries
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from ipywidgets import FloatProgress
from IPython.display import display
import time
import os
from tqdm import tqdm

## Declaring constants
IMAGE_DEPTH = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
BATCH_SIZE = 100
N_EPOCHS = 10
LEARNING_RATE = 1e-4

def _conv_layer(prev_layer,kernel_size,num_filters,layer_name):
    with tf.variable_scope(layer_name) as scope:
        k_size = kernel_size + [num_filters]
        w = tf.get_variable(name="weights",shape=k_size,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(name="biases",shape=[num_filters],initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(prev_layer,w,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(conv+b,name=scope.name)
        return relu

def _max_pool(prev_layer,layer_name):
    with tf.variable_scope(layer_name) as scope:
        max_pool_ = tf.nn.max_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return max_pool_

def _fully_connected(prev_layer,n_output,layer_name):
    with tf.variable_scope(layer_name) as scope:
        shape = prev_layer.get_shape().as_list()
        flattened = shape[1]*shape[2]*shape[3]
        w = tf.get_variable(name="weights",shape=[flattened,n_output],initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.random_normal_initializer())
        flat = tf.reshape(prev_layer,[-1,flattened])
        full = tf.nn.relu(tf.matmul(flat,w) + b)
        return full

def _create_dropout(prev_layer,layer_name):
    with tf.variable_scope(layer_name) as scope:
        keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(prev_layer,keep_prob=keep_prob)
        return drop,keep_prob

def _create_softmax(prev_layer,n_outputs,layer_name):
    with tf.variable_scope(layer_name) as scope:
        n_input = prev_layer.shape[1]
        w = tf.get_variable(name="weights",shape=[n_input,n_outputs],initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases",shape=[n_outputs],initializer=tf.random_normal_initializer())
        out = tf.nn.softmax(tf.matmul(prev_layer,w)+b)
        return out

def _create_loss(correct_preds,output_preds):
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(-tf.reduce_sum(correct_preds*tf.log(output_preds),reduction_indices=[1]))
        return loss

def _output_accuracy(correct_preds,output_preds):
    with tf.name_scope("accuracy") as scope:
        correct_predictions = tf.equal(tf.arg_max(output_preds,1),tf.arg_max(correct_preds,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
        return accuracy

def _create_summaries(model):
    tf.summary.scalar("loss",model['loss'])
    tf.summary.histogram("loss",model['loss'])
    summary_op = tf.summary.merge_all()
    return summary_op

def train(model,data,n_training_size,image,y,keep_prob):
    init = tf.global_variables_initializer()
    n_batches = n_training_size/BATCH_SIZE
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('graphs/cnn',sess.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print 'Restoring checkpoint...'
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(N_EPOCHS):
            epoch_loss = 0
            epoch_accuracy = 0
            for j in tqdm(range(n_batches)):
                x_batch,y_batch = data.train.next_batch(100)
                x_batch = np.reshape(x_batch,(100,28,28,1))
                _,l,acc, summary = sess.run([model['optimizer'],model['loss'],model['accuracy'],model['summary_op']],
                                            feed_dict={image:x_batch,y:y_batch,keep_prob:0.5})
                epoch_loss += l
                epoch_accuracy += acc
            writer.add_summary(summary,global_step=i)
            if i % 5 == 0:
                saver.save(sess, 'checkpoints/cnn', i)
            print 'Epoch: {}\tLoss: {}\tAccuracy: {}'.format(i,epoch_loss,epoch_accuracy/n_batches)

def main():
    ## Initialising the input layer
    with tf.variable_scope("input") as scope:
        image = tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH],name="image")
        y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="label")
    
    ## Reading in the data and defining constants
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    n_training_size,n_features = mnist.train.images.shape
    n_test_size = mnist.test.images.shape[0]

    ## Initialising the model parameters
    model = {}
    model['conv1'] = _conv_layer(image,[5,5,1],32,'conv1')
    model['max_pool1'] = _max_pool(model['conv1'],'max_pool1')
    model['conv2'] = _conv_layer(model['max_pool1'],[5,5,32],64,'conv2')
    model['max_pool2'] = _max_pool(model['conv2'],'max_pool2')
    model['fully_connected'] = _fully_connected(model['max_pool2'],1024,'fully_connected')
    model['dropout'],keep_prob = _create_dropout(model['fully_connected'],'dropout')
    model['softmax'] = _create_softmax(model['dropout'],10,'softmax')
    model['loss'] = _create_loss(y, model['softmax'])
    model['accuracy'] = _output_accuracy(y, model['softmax'])
    model['global_step'] = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
    model['optimizer'] = tf.train.AdamOptimizer(LEARNING_RATE).minimize(model['loss'],global_step = model['global_step'])
    model['summary_op'] = _create_summaries(model)
    print 'Model initialised....'
    ## Passing the initialised model to train() method
    train(model,mnist,n_training_size,image,y,keep_prob)

if __name__ == '__main__':
    main()

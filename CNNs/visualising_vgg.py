## Visualising the activation maps of the VGG network

import numpy as np
import tensorflow as tf
import scipy.io
from scipy import misc
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_DEPTH = 3
path = '../Assignment_2/style_transfer/imagenet-vgg-verydeep-19.mat'

def _weights(vgg_layers,layer,expected_layer_name):
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)

def _conv_relu(vgg_layers, prev_layer, layer, expected_layer_name):
    with tf.variable_scope(expected_layer_name) as scope:

        W,b = _weights(vgg_layers, layer, expected_layer_name)
        W = tf.constant(W)
        b = tf.constant(b)
        conv = tf.nn.conv2d(prev_layer,W,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(conv + b,name=scope.name)
        return relu

def _avg_pool(prev_layer,layer_name):
    with tf.variable_scope(layer_name) as scope:
        avgpool = tf.nn.avg_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return avgpool

def get_activation_map(graph, input_image, image_tensor,layer):
    input_image = np.reshape(input_image, [1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])
    assert input_image.shape == (1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH)
    with tf.Session() as sess:
        activation = sess.run([graph[layer]],feed_dict={image_tensor:input_image})
    return activation

def main():
    
    ## Loading the graph
    print 'Loading the graph..'
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    graph = {}
    with tf.variable_scope("placeholders") as scope:
        image = tf.placeholder(shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH],dtype=tf.float32,name="Input")
    
    ## Assembling the graph
    print 'Assembling the graph...'
    graph['conv1_1'] = _conv_relu(vgg_layers,image,0,'conv1_1')
    graph['conv1_2'] = _conv_relu(vgg_layers,graph['conv1_1'],2,'conv1_2')
    graph['avg_pool1'] = _avg_pool(graph['conv1_2'],'avg_pool1')
    graph['conv2_1'] = _conv_relu(vgg_layers,graph['avg_pool1'],5,'conv2_1')
    graph['conv2_2'] = _conv_relu(vgg_layers,graph['conv2_1'],7,'conv2_2')
    graph['avg_pool2'] = _avg_pool(graph['conv2_2'],'avg_pool2')

    ## Reading in the Image
    print 'Reading in the image...'
    img = misc.imread("car.jpg")
    print img.shape

    ## Getting the activation map
    layer = str(raw_input('Enter the layer name: '))
    if layer == '':
        layer = 'conv1_1'
    print 'Getting Activation map'
    activation = get_activation_map(graph, img, image, layer)
    _,_,x,y,z = np.asarray(activation).shape
    activation = np.asarray(activation).reshape((x,y,z))
    print activation.shape
    
    print 'Saving activations for {}'.format(layer)
    directory = 'activations_{}'.format(layer)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
        for i in range(1,activation.shape[2]+1):
            filename = os.path.join(directory,"conv1_1_{}.jpg".format(i))
            misc.imsave(filename,activation[:,:,i-1])

if __name__ == "__main__":
    main()
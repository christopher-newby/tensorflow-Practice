'''
Created on Jul 13, 2017

This file creates a multi-perceptron neural network using tensorflow
It trains on a data set made in Making_Data, and classifies a set of points
then plots the results

@author: Chris Newby
'''



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to get rid of the warnings about faster performance from tensorflow


import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 100

n_data_points = 10000
batch_size = 100    # needs to be a divisor of n_data_points for an epoch to actually use all the data

display_step = 10 # number of epochs to wait before showing convergence results

# Network Parameters
n_hidden_layers= 2
n_neurons_per_layer = 4

n_inputs = 2
n_classes = 2

# tf Graph input and output
x_in = tf.placeholder("float", [None, n_inputs])
y_out = tf.placeholder("float", [None, n_classes])


# Create mlp
def multilayer_perceptron(x, n_hidden, weights, biases):
    # here I'm making a mpl of n_hidden layers + an output layer
    # returns the completed diagram
    # uses RELU activation for internal and linear for output
    
    # first layer
    layer_hold = tf.add(tf.matmul(x,weights[0]),biases[0])
    layer_hold = tf.nn.relu(layer_hold)
    
    # hidden layers
    for i in range(1,n_hidden):
        layer_hold = tf.add(tf.matmul(layer_hold,weights[i]),biases[i])
        layer_hold = tf.nn.relu(layer_hold)
    
    # output layer (linear for now)
    layer_out = tf.add( tf.matmul( layer_hold, weights[ len(weights)-1 ] ) , biases[ len(biases)-1 ] )
    
    return layer_out

# Store layers weight & bias
# first layer
weights = [tf.Variable(tf.random_normal(shape = [n_inputs,n_neurons_per_layer],stddev=0.01 ) )]
biases = [tf.Variable( tf.constant( value = 0.1, shape = [n_neurons_per_layer] ) )]

# rest of hidden layers
for _ in range(n_hidden_layers-1):
    weights.append( tf.Variable( tf.random_normal( shape = [n_neurons_per_layer,n_neurons_per_layer],stddev=0.01 ) ) )
    biases.append(tf.Variable( tf.constant( value = 0.1, shape = [n_neurons_per_layer] ) ))
    
# output layer
weights.append( tf.Variable( tf.random_normal(shape = [n_neurons_per_layer,n_classes],stddev=0.01) ) )
biases.append(tf.Variable( tf.constant( value = 0.1, shape = [n_classes] ) ))


# Construct model
prediction = multilayer_perceptron(x_in, n_hidden_layers, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_out))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


#function to create the batches (from lasagne tutorial on mnist data)
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# creating and storing the data
import Checking_Folder
import Making_Data

data_dir = 'Data'

Checking_Folder.Folder_Check(data_dir)

data_filename = 'Sample Data.dat'
Making_Data.CreateData(data_dir, data_filename, n_data_points)
dat_in, dat_class = Making_Data.Getting_Data(data_dir, data_filename)

data_testdata_name = 'Test Data.dat'
n_test = 1000
Making_Data.CreateData(data_dir, data_testdata_name, n_test)
test_in, test_class = Making_Data.Getting_Data(data_dir, data_testdata_name)

# this is going to be the grid I classify later:
n_class_points = 50

unclass_data = []
unclass_data_classification = []

for i in range(n_class_points+1):
    x_h = 2.0*i/float(n_class_points) - 1.0
    for j in range(n_class_points+1):
        y_h = 2.0*j/float(n_class_points) - 1.0
        unclass_data.append([x_h,y_h])
        unclass_data_classification.append([0,0])
        

# this changes the training data to be easier to analyze
dat_groomed = []
for elem in dat_in:
    dat_groomed.append([elem[0]**2,elem[1]**2])
    
# changes the testing data in the same way
test_groomed = []
for elem in test_in:
    test_groomed.append([elem[0]**2,elem[1]**2])
    
# changes the soon-to-be classified data in the same way as the trained data
unclass_groomed = []
for elem in unclass_data:
    unclass_groomed.append([elem[0]**2,elem[1]**2])
    
# array of the classifications of the unclass_data set
output_array = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    print('Beginning optimization.')
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_data_points/batch_size)
        
        # loop through batches
        for batch in iterate_minibatches(dat_groomed, dat_class, batch_size):
            batch_x, batch_y = batch
            
            # run optimization on batch
            _, c = sess.run([optimizer, cost], feed_dict={x_in: batch_x, y_out: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            
        if (epoch+1) % display_step == 0:
            print('Epoch: {:5}, cost = {:.9f}'.format(epoch+1,avg_cost))
    print('Optimization Finished!\n')

    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_out, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    acc = accuracy.eval({x_in: test_groomed, y_out: test_class})
    print('Accuracy with test data = {:.3f}'.format(acc) )
    
    # now for the unclassified data:
    hold = tf.argmax(prediction,1)
    output_array = sess.run(hold,feed_dict={x_in: unclass_groomed})


# making the classification array for the unclassified data
for i in range(len(output_array)):
    unclass_data_classification[i][output_array[i]] = 1

# plotting the two data sets (sample input and classified output):
import Plotter

#Plotter.Plot_Data(dat_in, dat_class, 'Sample Data')
#Plotter.Plot_Data(unclass_data, unclass_data_classification, 'Classified Data')
Plotter.Plot_two_data(dat_in,dat_class,unclass_data,unclass_data_classification)














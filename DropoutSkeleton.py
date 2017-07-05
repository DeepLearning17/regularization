#https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/3_regularization.ipynb

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

pickle_file = 'notMNIST.pickle' #generated notMINST.pickle file-> https://github.com/StryxZilla/noob/blob/master/notMNIST.pickle
    #code for generating the notMNIST.pickle file-> https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/1_notmnist.ipynb

#***********************DATASET HANDLING************************

#Loading the dataset
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('------Before dataset reshaping-----')
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


#Reshaping the dataset
'''
Reformat into a shape that's more adapted to the models we're going to train:
data as a flat matrix,
labels as float 1-hot encodings.
'''

batch_size = 128
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 100
beta_regul = 1e-3

graph = tf.Graph()
with graph.as_default():

   #### Insert code for initializing train, validation, test and regularization data.
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  # Use validation and test data as constants
  '''
  tf_train_dataset = 
  tf_train_labels = 
  tf_valid_dataset = 
  tf_test_dataset = 
  global_step = 
  '''
  
  ### Insert code for weights and biases (weights1, biases1, weights2, biases2)
  '''
  weights1 =
  biases1 = 
  weights2 = 
  biases2 = 
  weights3 = 
  biases3 = 
  '''
  
  # Training computation.
  ### Insert code for calculating loss using relu, softmax logits
  '''
  lay1_train = 
  lay2_train = 
  logits = 
  loss = 
  '''
  
  # Optimizer.
  ### Insert code for optimizing the training using GradientDescentOptimizer with learning_rate as parameter
  '''
  learning_rate = 
  optimizer = 
  '''
  
  ### Insert code for predictions for the training, validation, and test data using softmax and relu
  '''
  train_prediction = 
  lay1_valid = 
  lay2_valid = 
  valid_prediction = 
  lay1_test = 
  lay2_test = 
  test_prediction = 
  '''
  
  num_steps = 9001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    
	### Insert code for randomized offset
    '''
	offset = 
    '''
	
	# Generate a minibatch.
    ### Insert code to generate minibatch from train dataset using offset and batch_size
	'''
	batch_data = 
    batch_labels = 
    '''
	
	### Insert code to prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    '''
	feed_dict = 
    '''
	
	# print accuracy prediction for minibatch and validation sets
	if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  
  # show final accuracy on test set
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

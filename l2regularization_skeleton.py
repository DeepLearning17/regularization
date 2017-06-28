
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
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('------After dataset reshaping-----')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


#*******************L2 REGULARIZATION IN 1-LAYER NEURAL NETWORK*******************
'''
L2 amounts to adding a penalty on the norm of the weights to the loss. 
In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). 
The right amount of regularization should improve your validation / test accuracy.
'''

batch_size = 128
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():
  #### Insert code for initializing train, validation, test and regularization data.
      # This is input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch. 
      # Use validation, test data and beta regularization as constants.
  
  
  ### Insert code for weights and biases

  
  ### Insert code for calculating loss using relu, softmax logits
  
  
  ### Insert code for optimizing the training using GradientDescentOptimizer
  
  
  ### Insert code for predictions for the training, validation, and test data using softmax and relu



num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('------Validation accuracy-----')
  for step in range(num_steps):

    ### Insert code for randomized offset
   
    
    ### Insert code to generate minibatch from train dataset using offset and batch_size

   
    ### Insert code to prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
    
    ### Insert code to print accuracy prediction for minibatch and validation sets
    

  ### Insert code to show final accuracy on test set


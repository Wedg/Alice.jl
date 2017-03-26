# Alice.jl

A deep learning package built in Julia.

The package is not yet registered so to download, run the command:
```jlcon
Pkg.clone("https://github.com/Wedg/Alice.jl.git")
```

## Demos

There are 3 demos that display almost all the functionality. All the data for each one is contained in the repository.

#### MNIST
![](demo/mnist/mnist.jpg)  
Predict the digit label (0 to 9) of each image.  
Feedforward Neural Network and Convolutional Neural Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/mnist/Demo_MNIST_28x28.html) (view in browser) or 
[ipynb file](demo/mnist/Demo_MNIST_28x28.ipynb) (run in Jupyter)

#### STL10 (reduced)
![](demo/stl10/stl10_8.jpg)  
Predict the object in each image from 4 classes (airplane, car, cat, dog).  
Part 1 - Sparse Autoencoder - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/stl10/Demo_STL10_A_Sparse_Autoencoder.html) (view in browser) or 
[ipynb file](demo/stl10/Demo_STL10_A_Sparse_Autoencoder.ipynb) (run in Jupyter)  
Part 2 - Convolutional Neural Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/stl10/Demo_STL10_B_Convolution_and_Pooling.html) (view in browser) or 
[ipynb file](demo/stl10/Demo_STL10_B_Convolution_and_Pooling.ipynb) (run in Jupyter) 

#### Word Embedding
NLP (Natural Language Processing) example. Learn feature representations of words through learning to predict the next word in a given sequence of words.  
Word Embedding Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/ngrams/Demo_Word_Embedding.html) (view in browser) or 
[ipynb file](demo/ngrams/Demo_Word_Embedding.ipynb) (run in Jupyter)

## Documentation
There are a number of types/structs defined that are used to build a neural network. There is a data container to place all training, validation and test sets in. And there are a number of network layers (input, hidden and output). The current available list is:

Data container:
- Data

Input Layer:
- InputLayer

Hidden Layers:
- FullyConnectedLayer
- WordEmbeddingLayer
- SparseEncoderLayer
- ConvolutionLayer
- MeanPoolLayer
- MaxPoolLayer

Output Layers:
- LinearOutputLayer
- MultiLinearOutputLayer
- LogisticOutputLayer
- SoftmaxOutputLayer

## Building a Neural Network
#### Step 1 - Place data in a data container
There are 4 functions available:  
(Here, `X_train` is an array of training data, `y_train` is an array of target data for the training set, `X_val` is an array of validation data, `y_val` is an array of target data for the validation set, `X_test` is an array of test data, and `y_test` is an array of target data for the test set.)  
- `data(X_train)`
- `data(X_train, y_train)`
- `data(X_train, y_train, X_val, y_val)`
- `data(X_train, y_train, X_val, y_val, X_test, y_test)`

Note that a reference to the original data is used as opposed to a copy for better memory management. So if the data is changed that will also change the data in this data container.
